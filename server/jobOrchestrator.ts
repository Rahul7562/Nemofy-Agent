import crypto from 'node:crypto';
import type {
  ChatMessageRecord,
  ChatThreadRecord,
  ContextUsageSnapshot,
  FileChangeRecord,
  ToolEventRecord,
  WorkspaceAndConfigStore,
} from './configStore.ts';

export interface ActivityFeedEntry {
  id: string;
  phase: string;
  text: string;
  timestamp?: string;
}

export interface ActiveTaskEntry {
  id: string;
  kind: string;
  label: string;
  elapsed_seconds: number;
  expected_seconds?: number;
}

export interface JobActivitySnapshot {
  jobId: string;
  status: string;
  currentPhase: string;
  lastEvent: string;
  elapsedSeconds: number;
  turnCount: number;
  thinking: string | null;
  activityFeed: ActivityFeedEntry[];
  toolEvents: ToolEventRecord[];
  fileChanges: FileChangeRecord[];
  contextUsage: ContextUsageSnapshot | null;
  activeTasks: ActiveTaskEntry[];
}

export interface ActiveBackgroundJobView {
  jobId: string;
  threadId: string;
  status: string;
  activity: JobActivitySnapshot;
  createdAt: number;
}

export interface JobPollResponse {
  ok: boolean;
  status: number;
  data: Record<string, unknown>;
}

export type AgentCoreCaller = (
  endpoint: string,
  options?: { method?: string; body?: unknown; timeoutMs?: number },
) => Promise<JobPollResponse>;

const TERMINAL_STATUSES = new Set(['completed', 'awaiting_approval', 'error', 'stalled', 'cancelled']);

export function isTerminalJobStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export function extractActivityFromJobPayload(
  jobId: string,
  data: Record<string, unknown>,
): JobActivitySnapshot {
  const resultObj =
    data.result && typeof data.result === 'object'
      ? (data.result as Record<string, unknown>)
      : null;
  const rawThinking =
    typeof data.thinking === 'string' && data.thinking.trim()
      ? data.thinking.trim()
      : typeof resultObj?.thinking === 'string' && resultObj.thinking.trim()
        ? resultObj.thinking.trim()
        : null;
  const toolEvents = Array.isArray(data.tool_events)
    ? (data.tool_events as ToolEventRecord[])
    : Array.isArray(resultObj?.tool_events)
      ? (resultObj.tool_events as ToolEventRecord[])
      : [];
  const fileChanges = Array.isArray(data.file_changes)
    ? (data.file_changes as FileChangeRecord[])
    : Array.isArray(resultObj?.file_changes)
      ? (resultObj.file_changes as FileChangeRecord[])
      : [];
  const activityFeed = Array.isArray(data.activity_feed)
    ? (data.activity_feed as ActivityFeedEntry[])
    : [];
  const contextUsage =
    data.context_usage && typeof data.context_usage === 'object'
      ? (data.context_usage as ContextUsageSnapshot)
      : resultObj?.context_usage && typeof resultObj.context_usage === 'object'
        ? (resultObj.context_usage as ContextUsageSnapshot)
        : null;
  const activeTasks = Array.isArray(data.active_tasks)
    ? (data.active_tasks as ActiveTaskEntry[])
    : [];

  return {
    jobId,
    status: String(data.status || 'running'),
    currentPhase: String(data.current_phase || 'thinking'),
    lastEvent: String(data.last_event || 'Nemo is working...'),
    elapsedSeconds: typeof data.elapsed_seconds === 'number' ? data.elapsed_seconds : 0,
    turnCount: typeof data.turn_count === 'number' ? data.turn_count : 0,
    thinking: rawThinking,
    activityFeed,
    toolEvents,
    fileChanges,
    contextUsage,
    activeTasks,
  };
}

interface TrackedBackgroundJob {
  jobId: string;
  threadId: string;
  createdAt: number;
  finalized: boolean;
  finalizedMessage: ChatMessageRecord | null;
  activity: JobActivitySnapshot;
  pollTimer: ReturnType<typeof setTimeout> | null;
  pollingInFlight: boolean;
  consecutiveErrors: number;
}

/**
 * Server-side background job orchestrator.
 *
 * Independently monitors Agent-Core jobs and persists results, tool events, file changes,
 * pending approvals, and context usage to threads.json — even if the user navigates to Settings,
 * switches chat threads, refreshes the page, or closes the browser tab completely.
 */
export class ChatJobOrchestrator {
  private readonly store: Pick<WorkspaceAndConfigStore, 'listThreads' | 'updateThreads'>;
  private readonly callAgentCore: AgentCoreCaller;
  private readonly pollIntervalMs: number;
  private readonly pruneAfterMs: number;
  private readonly maxConsecutiveErrors: number;
  private readonly jobs = new Map<string, TrackedBackgroundJob>();
  private readonly activeJobByThread = new Map<string, string>();
  private readonly launchingThreads = new Set<string>();
  private readonly deletedThreads = new Set<string>();
  private closed = false;

  constructor(options: {
    store: Pick<WorkspaceAndConfigStore, 'listThreads' | 'updateThreads'>;
    callAgentCore: AgentCoreCaller;
    pollIntervalMs?: number;
    pruneAfterMs?: number;
    maxConsecutiveErrors?: number;
  }) {
    this.store = options.store;
    this.callAgentCore = options.callAgentCore;
    this.pollIntervalMs = Math.max(50, options.pollIntervalMs ?? 600);
    this.pruneAfterMs = Math.max(5_000, options.pruneAfterMs ?? 30 * 60 * 1000);
    this.maxConsecutiveErrors = Math.max(3, options.maxConsecutiveErrors ?? 20);
  }

  pruneOldJobs(): void {
    const cutoff = Date.now() - this.pruneAfterMs;
    for (const [jid, job] of this.jobs.entries()) {
      if (job.finalized && job.createdAt < cutoff) {
        if (job.pollTimer) {
          clearTimeout(job.pollTimer);
          job.pollTimer = null;
        }
        this.jobs.delete(jid);
      }
    }
  }

  getActiveJobForThread(threadId: string): ActiveBackgroundJobView | null {
    const jobId = this.activeJobByThread.get(threadId);
    if (!jobId) return null;
    const job = this.jobs.get(jobId);
    if (!job || job.finalized) {
      this.activeJobByThread.delete(threadId);
      return null;
    }
    return {
      jobId: job.jobId,
      threadId: job.threadId,
      status: job.activity.status,
      activity: job.activity,
      createdAt: job.createdAt,
    };
  }

  /**
   * Synchronously reserves exclusive launch rights for a thread across async awaits
   * (config enrichment, secret sync, Agent-Core HTTP POST). Returns false if another
   * job is already active or launching for this thread.
   */
  tryAcquireThreadLaunch(threadId: string): boolean {
    if (!threadId) return true;
    if (this.getActiveJobForThread(threadId) || this.launchingThreads.has(threadId)) {
      return false;
    }
    this.launchingThreads.add(threadId);
    this.deletedThreads.delete(threadId);
    return true;
  }

  releaseThreadLaunch(threadId: string): void {
    if (!threadId) return;
    this.launchingThreads.delete(threadId);
  }

  isThreadLaunching(threadId: string): boolean {
    return this.launchingThreads.has(threadId);
  }

  markThreadDeleted(threadId: string): void {
    if (!threadId) return;
    this.deletedThreads.add(threadId);
  }

  clearThreadDeleted(threadId: string): void {
    if (!threadId) return;
    this.deletedThreads.delete(threadId);
  }

  isThreadDeleted(threadId: string): boolean {
    return this.deletedThreads.has(threadId);
  }

  getActiveJobs(): ActiveBackgroundJobView[] {
    this.pruneOldJobs();
    const out: ActiveBackgroundJobView[] = [];
    for (const job of this.jobs.values()) {
      if (!job.finalized) {
        out.push({
          jobId: job.jobId,
          threadId: job.threadId,
          status: job.activity.status,
          activity: job.activity,
          createdAt: job.createdAt,
        });
      }
    }
    return out;
  }

  getTrackedJob(jobId: string): {
    jobId: string;
    threadId: string;
    finalized: boolean;
    activity: JobActivitySnapshot;
    finalizedMessage: ChatMessageRecord | null;
  } | null {
    const job = this.jobs.get(jobId);
    if (!job) return null;
    return {
      jobId: job.jobId,
      threadId: job.threadId,
      finalized: job.finalized,
      activity: job.activity,
      finalizedMessage: job.finalizedMessage,
    };
  }

  registerJob(params: {
    jobId: string;
    threadId: string;
    initialPayload: Record<string, unknown>;
  }): JobActivitySnapshot {
    this.pruneOldJobs();
    const activity = extractActivityFromJobPayload(params.jobId, params.initialPayload);
    const existing = this.jobs.get(params.jobId);
    if (existing) {
      existing.activity = activity;
      return activity;
    }

    const tracked: TrackedBackgroundJob = {
      jobId: params.jobId,
      threadId: params.threadId,
      createdAt: Date.now(),
      finalized: false,
      finalizedMessage: null,
      activity,
      pollTimer: null,
      pollingInFlight: false,
      consecutiveErrors: 0,
    };

    this.jobs.set(params.jobId, tracked);
    this.activeJobByThread.set(params.threadId, params.jobId);

    if (isTerminalJobStatus(activity.status)) {
      this.finalizeJob(params.jobId, params.initialPayload);
    } else if (!this.closed) {
      this.scheduleNextPoll(params.jobId, this.pollIntervalMs);
    }

    return activity;
  }

  private scheduleNextPoll(jobId: string, delayMs: number): void {
    if (this.closed) return;
    const job = this.jobs.get(jobId);
    if (!job || job.finalized) return;
    if (job.pollTimer) {
      clearTimeout(job.pollTimer);
      job.pollTimer = null;
    }
    job.pollTimer = setTimeout(() => {
      job.pollTimer = null;
      void this.backgroundTick(jobId);
    }, delayMs);
    if (typeof job.pollTimer.unref === 'function') {
      job.pollTimer.unref();
    }
  }

  private async backgroundTick(jobId: string): Promise<void> {
    if (this.closed) return;
    const job = this.jobs.get(jobId);
    if (!job || job.finalized || job.pollingInFlight) return;

    job.pollingInFlight = true;
    try {
      const resp = await this.callAgentCore(`/agent/jobs/${encodeURIComponent(jobId)}?wait_ms=250`, {
        timeoutMs: 12_000,
      });

      if (!resp.ok) {
        if (resp.status === 404) {
          this.finalizeJob(jobId, {
            status: 'error',
            current_phase: 'error',
            last_event: 'Background job was not found on Agent-Core.',
            error: String(resp.data.detail || 'Background job expired or disappeared on Agent-Core.'),
          });
          return;
        }
        job.consecutiveErrors += 1;
        if (job.consecutiveErrors >= this.maxConsecutiveErrors) {
          this.finalizeJob(jobId, {
            status: 'error',
            current_phase: 'error',
            last_event: 'Lost connection to Agent-Core during background execution.',
            error: String(resp.data.detail || 'Agent-Core became unreachable.'),
          });
          return;
        }
        this.scheduleNextPoll(jobId, Math.min(3_000, this.pollIntervalMs * 2));
        return;
      }

      job.consecutiveErrors = 0;
      const activity = extractActivityFromJobPayload(jobId, resp.data);
      job.activity = activity;

      if (isTerminalJobStatus(activity.status)) {
        this.finalizeJob(jobId, resp.data);
        return;
      }

      this.scheduleNextPoll(jobId, this.pollIntervalMs);
    } catch (err) {
      job.consecutiveErrors += 1;
      if (job.consecutiveErrors >= this.maxConsecutiveErrors) {
        this.finalizeJob(jobId, {
          status: 'error',
          current_phase: 'error',
          last_event: 'Background polling error.',
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
      this.scheduleNextPoll(jobId, Math.min(3_000, this.pollIntervalMs * 2));
    } finally {
      job.pollingInFlight = false;
    }
  }

  /**
   * Client-facing poll handler.
   * Returns finalized state immediately if background monitoring already persisted the job,
   * otherwise fetches fresh status from Agent-Core and idempotently finalizes on terminal states.
   */
  async pollJobForClient(
    jobId: string,
    waitMs = 300,
  ): Promise<
    | {
        ok: true;
        done: boolean;
        activity: JobActivitySnapshot;
        thread?: ChatThreadRecord | null;
        message?: ChatMessageRecord | null;
      }
    | { ok: false; status: number; error: string }
  > {
    const tracked = this.jobs.get(jobId);
    if (tracked && tracked.finalized) {
      const thread = this.store.listThreads().find((t) => t.id === tracked.threadId) || null;
      return {
        ok: true,
        done: true,
        activity: tracked.activity,
        thread,
        message: tracked.finalizedMessage,
      };
    }

    const clampedWait = Math.min(2000, Math.max(0, Number(waitMs) || 300));
    const poll = await this.callAgentCore(
      `/agent/jobs/${encodeURIComponent(jobId)}?wait_ms=${clampedWait}`,
      { timeoutMs: 15_000 },
    );

    if (!poll.ok) {
      return {
        ok: false,
        status: poll.status,
        error: String(poll.data.detail || 'Unable to poll live agent job.'),
      };
    }

    const data = poll.data;
    const activity = extractActivityFromJobPayload(jobId, data);
    if (tracked) {
      tracked.activity = activity;
      tracked.consecutiveErrors = 0;
    }

    if (!isTerminalJobStatus(activity.status)) {
      return { ok: true, done: false, activity };
    }

    const finalized = this.finalizeJob(jobId, data);
    return {
      ok: true,
      done: true,
      activity: finalized.activity,
      thread: finalized.thread,
      message: finalized.message,
    };
  }

  async cancelJob(jobId: string): Promise<{
    ok: true;
    activity: JobActivitySnapshot;
    thread: ChatThreadRecord | null;
    message: ChatMessageRecord | null;
  }> {
    const cancelResp = await this.callAgentCore(`/agent/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
      timeoutMs: 10_000,
    });

    const data = cancelResp.data || {};
    const cancelledPayload: Record<string, unknown> = {
      ...data,
      status: data.status || 'cancelled',
      current_phase: data.current_phase || 'cancelled',
      last_event: data.last_event || 'Stopped by user.',
    };

    const finalized = this.finalizeJob(jobId, cancelledPayload, { forceCancelledText: true });
    return {
      ok: true,
      activity: finalized.activity,
      thread: finalized.thread,
      message: finalized.message,
    };
  }

  async cancelThreadJob(threadId: string): Promise<void> {
    const activeJobId = this.activeJobByThread.get(threadId);
    if (!activeJobId) return;
    await this.cancelJob(activeJobId);
  }

  /**
   * Idempotently finalizes a terminal job and persists the assistant message,
   * pendingApproval, toolEvents, fileChanges, and contextUsage into threads.json.
   *
   * Safe against:
   * - Concurrent calls (background loop + client poll + cancel) -> writes at most once.
   * - Deleted threads -> never resurrects a thread that was deleted while running.
   */
  finalizeJob(
    jobId: string,
    data: Record<string, unknown>,
    options: { forceCancelledText?: boolean } = {},
  ): {
    activity: JobActivitySnapshot;
    thread: ChatThreadRecord | null;
    message: ChatMessageRecord | null;
  } {
    const tracked = this.jobs.get(jobId);
    const activity = extractActivityFromJobPayload(jobId, data);
    const targetThreadId = tracked?.threadId || String(data.thread_id || '');

    if (tracked && tracked.finalized) {
      const existingThread = targetThreadId
        ? this.store.listThreads().find((t) => t.id === targetThreadId) || null
        : null;
      return {
        activity: tracked.activity,
        thread: existingThread,
        message: tracked.finalizedMessage,
      };
    }

    if (tracked) {
      tracked.finalized = true;
      tracked.activity = activity;
      if (tracked.pollTimer) {
        clearTimeout(tracked.pollTimer);
        tracked.pollTimer = null;
      }
    }

    if (targetThreadId && this.activeJobByThread.get(targetThreadId) === jobId) {
      this.activeJobByThread.delete(targetThreadId);
    }

    if (!targetThreadId) {
      return { activity, thread: null, message: null };
    }

    const replyNow = new Date().toISOString();
    const resultObj =
      data.result && typeof data.result === 'object'
        ? (data.result as Record<string, unknown>)
        : {};
    const pendingApproval =
      activity.status === 'cancelled'
        ? null
        : ((resultObj.pending_approval as Record<string, unknown> | null) || null);
    const errText =
      typeof data.error === 'string' && data.error.trim()
        ? data.error.trim()
        : typeof resultObj.error === 'string' && resultObj.error.trim()
          ? resultObj.error.trim()
          : null;

    const finalText = options.forceCancelledText
      ? 'Stopped by user.'
      : String(resultObj.final_response || '').trim() ||
        (activity.status === 'awaiting_approval'
          ? 'Execution paused for your approval before running a sensitive tool action.'
          : activity.status === 'cancelled'
            ? 'Stopped by user.'
            : errText
              ? `Agent-Core could not complete this run: ${errText}`
              : activity.status === 'stalled'
                ? 'Agent-Core stopped because the task became unresponsive with no active background work.'
                : 'Task completed.');

    const assistantMsg: ChatMessageRecord = {
      id: `msg-${crypto.randomBytes(6).toString('hex')}`,
      role: 'assistant',
      content: finalText,
      timestamp: replyNow,
      plan: Array.isArray(resultObj.plan)
        ? (resultObj.plan as ChatThreadRecord['messages'][number]['plan'])
        : [],
      activeSkills: Array.isArray(resultObj.active_skills)
        ? (resultObj.active_skills as string[])
        : [],
      subagentReports: Array.isArray(resultObj.subagent_reports)
        ? (resultObj.subagent_reports as Array<Record<string, unknown>>)
        : [],
      activeModels: resultObj.active_models as ChatThreadRecord['messages'][number]['activeModels'],
      thinking: activity.thinking,
      toolEvents: activity.toolEvents,
      fileChanges: activity.fileChanges,
      pendingApproval,
      status: activity.status,
    };

    const updatedThread = this.store.updateThreads((freshThreads) => {
      const target = freshThreads.find((t) => t.id === targetThreadId);
      // Do NOT resurrect a thread that was deleted while the background job was running
      if (!target) return null;
      target.messages.push(assistantMsg);
      target.pendingApproval = pendingApproval;
      if (activity.contextUsage) {
        target.contextUsage = activity.contextUsage;
      }
      target.updatedAt = replyNow;
      return target;
    });

    if (tracked) {
      tracked.finalizedMessage = updatedThread ? assistantMsg : null;
    }

    return {
      activity,
      thread: updatedThread,
      message: updatedThread ? assistantMsg : null,
    };
  }

  shutdown(): void {
    this.closed = true;
    for (const job of this.jobs.values()) {
      if (job.pollTimer) {
        clearTimeout(job.pollTimer);
        job.pollTimer = null;
      }
    }
  }
}
