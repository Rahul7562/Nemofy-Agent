import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import {
  MAX_CONSECUTIVE_POLL_ERRORS,
  computePollRetryDelayMs,
  shouldRetryPollError,
} from '../lib/activityUtils.ts';
import {
  cancelChatJob,
  fetchActiveChatJobs,
  pollChatJob,
  startChatJob,
  startResumeApprovalJob,
  uploadChatMedia,
} from '../lib/apiClient.ts';
import {
  createInitialExecutionState,
  executionReducer,
  getRunningThreadIds,
  getThreadDraft,
  getThreadRun,
  resolveDraftKey,
} from '../lib/executionState.ts';
import { classifyFileModality } from '../lib/llmPresets.ts';
import type {
  ChatAttachment,
  ChatThread,
  ExecutionAccessMode,
  LiveAgentActivity,
} from '../types/agent.ts';

function cleanThreadTitle(raw: string): string {
  return raw
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[*_~`#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 56);
}

interface UseAgentExecutionOptions {
  authenticated: boolean;
  onCreateThread: (title?: string) => Promise<ChatThread | null>;
  onUpsertThread: (thread: ChatThread, activate?: boolean) => void;
  onUpdateThreadById: (id: string, updater: (thread: ChatThread) => ChatThread) => void;
  onWorkspaceFilesChanged?: () => void;
}

export interface SendMessageParams {
  threadId: string | null;
  prompt: string;
  attachments?: ChatAttachment[];
  reasoningEffort: string;
  accessMode: ExecutionAccessMode;
  hasPrimaryModel: boolean;
}

export interface SubmitApprovalParams {
  threadId: string;
  decision: 'approve' | 'reject' | 'edit';
  toolName?: string;
  editedArgs?: Record<string, unknown>;
  accessMode: ExecutionAccessMode;
}

export function useAgentExecution({
  authenticated,
  onCreateThread,
  onUpsertThread,
  onUpdateThreadById,
  onWorkspaceFilesChanged,
}: UseAgentExecutionOptions) {
  const [state, dispatch] = useReducer(executionReducer, undefined, createInitialExecutionState);

  const pollControllersRef = useRef<Map<string, AbortController>>(new Map());
  const stopRequestedRef = useRef<Set<string>>(new Set());
  const activeJobByThreadRef = useRef<Map<string, string>>(new Map());
  const deletedThreadsRef = useRef<Set<string>>(new Set());
  const sendingLockRef = useRef<Set<string>>(new Set());
  const unmountedRef = useRef(false);

  const onUpsertThreadRef = useRef(onUpsertThread);
  onUpsertThreadRef.current = onUpsertThread;
  const onUpdateThreadByIdRef = useRef(onUpdateThreadById);
  onUpdateThreadByIdRef.current = onUpdateThreadById;
  const onWorkspaceFilesChangedRef = useRef(onWorkspaceFilesChanged);
  onWorkspaceFilesChangedRef.current = onWorkspaceFilesChanged;

  const runningThreadIds = useMemo(() => getRunningThreadIds(state), [state]);
  const hasRunningJobs = runningThreadIds.size > 0;

  // Smooth elapsed timer ticker across all running threads
  useEffect(() => {
    if (!hasRunningJobs) return;
    const timer = window.setInterval(() => {
      dispatch({ type: 'TICK_ELAPSED', nowMs: Date.now() });
    }, 350);
    return () => window.clearInterval(timer);
  }, [hasRunningJobs]);

  const pollJobForThread = useCallback(async (threadId: string, jobId: string) => {
    // Cancel any previous client poller for this exact thread before attaching a new one
    pollControllersRef.current.get(threadId)?.abort();
    const controller = new AbortController();
    pollControllersRef.current.set(threadId, controller);
    activeJobByThreadRef.current.set(threadId, jobId);

    let consecutivePollErrors = 0;

    try {
      while (!unmountedRef.current && !stopRequestedRef.current.has(threadId)) {
        if (deletedThreadsRef.current.has(threadId)) break;
        if (activeJobByThreadRef.current.get(threadId) !== jobId) break;

        try {
          const polled = await pollChatJob(jobId, 300, controller.signal);
          consecutivePollErrors = 0;

          if (
            unmountedRef.current ||
            deletedThreadsRef.current.has(threadId) ||
            activeJobByThreadRef.current.get(threadId) !== jobId
          ) {
            break;
          }

          if (polled.activity) {
            dispatch({
              type: 'UPDATE_ACTIVITY',
              threadId,
              jobId,
              liveActivity: polled.activity,
            });
          }

          if (polled.done) {
            if (polled.thread && !deletedThreadsRef.current.has(threadId)) {
              // NEVER hijack activeThreadId when a background thread completes
              onUpsertThreadRef.current(polled.thread, false);
              const lastMsg = polled.thread.messages[polled.thread.messages.length - 1];
              if (lastMsg?.fileChanges && lastMsg.fileChanges.length > 0) {
                onWorkspaceFilesChangedRef.current?.();
              }
            }
            dispatch({
              type: 'FINISH_RUN',
              threadId,
              jobId,
              error:
                polled.activity?.status === 'error' && !stopRequestedRef.current.has(threadId)
                  ? polled.activity.lastEvent || 'Agent execution failed.'
                  : null,
            });
            return;
          }
        } catch (pollErr) {
          const isAbort =
            unmountedRef.current ||
            stopRequestedRef.current.has(threadId) ||
            deletedThreadsRef.current.has(threadId) ||
            (pollErr instanceof DOMException && pollErr.name === 'AbortError') ||
            (pollErr instanceof Error && pollErr.name === 'AbortError');
          consecutivePollErrors += 1;
          if (!shouldRetryPollError(consecutivePollErrors, MAX_CONSECUTIVE_POLL_ERRORS, isAbort)) {
            if (!isAbort) {
              dispatch({
                type: 'FINISH_RUN',
                threadId,
                jobId,
                error: pollErr instanceof Error ? pollErr.message : 'Lost connection while polling agent.',
              });
            }
            return;
          }
          const delayMs = computePollRetryDelayMs(consecutivePollErrors);
          await new Promise<void>((resolve) => {
            const timer = window.setTimeout(resolve, delayMs);
            controller.signal.addEventListener(
              'abort',
              () => {
                window.clearTimeout(timer);
                resolve();
              },
              { once: true },
            );
          });
        }
      }
    } finally {
      if (pollControllersRef.current.get(threadId) === controller) {
        pollControllersRef.current.delete(threadId);
      }
      if (activeJobByThreadRef.current.get(threadId) === jobId) {
        activeJobByThreadRef.current.delete(threadId);
      }
      if (!unmountedRef.current) {
        dispatch({ type: 'FINISH_RUN', threadId, jobId });
      }
    }
  }, []);

  // Rehydrate any server-side running background jobs after login / browser reload
  useEffect(() => {
    unmountedRef.current = false;

    if (!authenticated) {
      for (const controller of pollControllersRef.current.values()) {
        controller.abort();
      }
      pollControllersRef.current.clear();
      activeJobByThreadRef.current.clear();
      stopRequestedRef.current.clear();
      deletedThreadsRef.current.clear();
      sendingLockRef.current.clear();
      dispatch({ type: 'RESET_ALL' });
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const activeJobs = await fetchActiveChatJobs();
        if (cancelled || unmountedRef.current) return;
        for (const job of activeJobs) {
          if (job.status !== 'running' || !job.threadId || !job.jobId) continue;
          if (deletedThreadsRef.current.has(job.threadId)) continue;
          if (activeJobByThreadRef.current.get(job.threadId) === job.jobId) continue;

          const fallbackActivity: LiveAgentActivity = job.activity ?? {
            jobId: job.jobId,
            status: 'running',
            currentPhase: 'thinking',
            lastEvent: 'Continuing background task',
            elapsedSeconds: 0,
            turnCount: 1,
            thinking: null,
            activityFeed: [
              {
                id: `reattach-${job.jobId}`,
                phase: 'thinking',
                text: 'Continuing background task',
              },
            ],
            toolEvents: [],
            fileChanges: [],
          };
          const startedAtMs =
            typeof job.createdAt === 'number' && Number.isFinite(job.createdAt)
              ? job.createdAt
              : Date.now() - Math.round((fallbackActivity.elapsedSeconds || 0) * 1000);

          dispatch({
            type: 'START_RUN',
            threadId: job.threadId,
            isResume: fallbackActivity.currentPhase === 'resuming_approval',
            jobId: job.jobId,
            liveActivity: fallbackActivity,
            startedAtMs,
          });
          void pollJobForThread(job.threadId, job.jobId);
        }
      } catch {
        // Non-blocking on startup
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [authenticated, pollJobForThread]);

  // Clean up client pollers only when the entire App unmounts (server keeps executing jobs!)
  useEffect(() => {
    return () => {
      unmountedRef.current = true;
      for (const controller of pollControllersRef.current.values()) {
        controller.abort();
      }
      pollControllersRef.current.clear();
    };
  }, []);

  const setThreadPrompt = useCallback((threadId: string | null, prompt: string) => {
    dispatch({
      type: 'SET_DRAFT_PROMPT',
      draftKey: resolveDraftKey(threadId),
      prompt,
    });
  }, []);

  const clearThreadError = useCallback((threadId: string | null) => {
    if (!threadId) return;
    dispatch({ type: 'SET_THREAD_ERROR', threadId, error: null });
  }, []);

  const removeAttachmentFromThread = useCallback((threadId: string | null, attachmentId: string) => {
    dispatch({
      type: 'REMOVE_ATTACHMENT',
      draftKey: resolveDraftKey(threadId),
      attachmentId,
    });
  }, []);

  const attachFilesToThread = useCallback(
    async (threadId: string | null, fileList: FileList | File[]) => {
      const files = Array.from(fileList || []);
      const draftKey = resolveDraftKey(threadId);
      const currentRun = getThreadRun(state, threadId);
      const currentDraft = getThreadDraft(state, threadId);
      if (files.length === 0 || currentRun.sending) return;

      const errTarget = threadId || '__unassigned__';
      dispatch({ type: 'SET_THREAD_ERROR', threadId: errTarget, error: null });

      const remainingSlots = Math.max(0, 6 - currentDraft.pendingAttachments.length);
      if (remainingSlots === 0) {
        dispatch({
          type: 'SET_THREAD_ERROR',
          threadId: errTarget,
          error: 'You can attach up to 6 media files per message.',
        });
        return;
      }

      const toProcess = files.slice(0, remainingSlots);
      const validFiles: File[] = [];
      for (const file of toProcess) {
        const modality = classifyFileModality(file.type, file.name);
        if (!modality) {
          dispatch({
            type: 'SET_THREAD_ERROR',
            threadId: errTarget,
            error: `Unsupported file "${file.name}". Attach PNG, JPG, WebP, GIF, MP4, WebM, MOV, MP3, WAV, OGG, FLAC, M4A, or AAC.`,
          });
          continue;
        }
        validFiles.push(file);
      }
      if (validFiles.length === 0) return;

      const names = validFiles.map((f) => f.name);
      dispatch({ type: 'ADD_UPLOADING_FILES', draftKey, names });

      try {
        const uploaded: ChatAttachment[] = [];
        for (const file of validFiles) {
          try {
            const att = await uploadChatMedia(file, threadId || undefined);
            uploaded.push(att);
          } catch (err) {
            dispatch({
              type: 'SET_THREAD_ERROR',
              threadId: errTarget,
              error: err instanceof Error ? err.message : `Failed to upload ${file.name}.`,
            });
          }
        }
        if (uploaded.length > 0) {
          dispatch({ type: 'ADD_ATTACHMENTS', draftKey, attachments: uploaded });
        }
      } finally {
        dispatch({ type: 'REMOVE_UPLOADING_FILES', draftKey, names });
      }
    },
    [state],
  );

  const sendMessage = useCallback(
    async ({
      threadId,
      prompt,
      attachments,
      reasoningEffort,
      accessMode,
      hasPrimaryModel,
    }: SendMessageParams) => {
      const sourceDraftKey = resolveDraftKey(threadId);
      const draft = getThreadDraft(state, threadId);
      const clean = prompt.trim();
      const attachmentsToSend = attachments ?? draft.pendingAttachments;

      if ((!clean && attachmentsToSend.length === 0) || draft.uploadingFiles.length > 0) {
        return;
      }

      const lockKey = threadId || '__creating__';
      if (
        sendingLockRef.current.has(lockKey) ||
        (threadId && (getThreadRun(state, threadId).sending || activeJobByThreadRef.current.has(threadId)))
      ) {
        return;
      }

      if (!hasPrimaryModel) {
        const errTarget = threadId || '__unassigned__';
        dispatch({
          type: 'SET_THREAD_ERROR',
          threadId: errTarget,
          error:
            'No Primary model selected yet. Choose your preferred Primary model in the header or Settings → LLM Config before sending.',
        });
        return;
      }

      sendingLockRef.current.add(lockKey);
      let lockedTargetId: string | null = null;

      try {
      const displayPrompt =
        clean ||
        `Analyze the attached ${Array.from(new Set(attachmentsToSend.map((a) => a.kind))).join(', ')} file(s).`;

      let targetId = threadId;
      const derivedTitle =
        cleanThreadTitle(clean || attachmentsToSend[0]?.name || '') || 'New Agent Session';

      if (!targetId) {
        const created = await onCreateThread(derivedTitle);
        if (!created) {
          return;
        }
        targetId = created.id;
        lockedTargetId = targetId;
        sendingLockRef.current.add(lockedTargetId);
      }

      deletedThreadsRef.current.delete(targetId);
      stopRequestedRef.current.delete(targetId);

      // Clear source draft immediately
      dispatch({ type: 'CLEAR_DRAFT', draftKey: sourceDraftKey });
      if (sourceDraftKey !== targetId) {
        dispatch({ type: 'CLEAR_DRAFT', draftKey: targetId });
      }

      const optimisticId = `temp-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      let committedOnServer = false;

      onUpdateThreadByIdRef.current(targetId, (t) => ({
        ...t,
        title: t.messages.length === 0 ? derivedTitle : t.title,
        updatedAt: new Date().toISOString(),
        messages: [
          ...t.messages,
          {
            id: optimisticId,
            role: 'user',
            content: displayPrompt,
            timestamp: new Date().toISOString(),
            attachments: attachmentsToSend.length > 0 ? attachmentsToSend : undefined,
          },
        ],
      }));

      const initialActivity: LiveAgentActivity = {
        jobId: 'starting',
        status: 'running',
        currentPhase: 'thinking',
        lastEvent:
          attachmentsToSend.length > 0
            ? 'Routing attached media and reviewing your request'
            : 'Reviewing your request and choosing the best approach',
        elapsedSeconds: 0,
        turnCount: 1,
        thinking: null,
        activityFeed: [
          {
            id: 'init-1',
            phase: 'thinking',
            text:
              attachmentsToSend.length > 0
                ? 'Routing attached media and reviewing your request'
                : 'Reviewing your request and choosing the best approach',
          },
        ],
        toolEvents: [],
        fileChanges: [],
      };

      dispatch({
        type: 'START_RUN',
        threadId: targetId,
        isResume: false,
        liveActivity: initialActivity,
        startedAtMs: Date.now(),
      });

      try {
        const started = await startChatJob(
          targetId,
          clean,
          reasoningEffort,
          undefined,
          attachmentsToSend.length > 0 ? attachmentsToSend : undefined,
          accessMode,
        );
        committedOnServer = true;

        if (!deletedThreadsRef.current.has(targetId)) {
          // Do NOT force-activate targetId — user may have switched threads while startChatJob was in-flight
          onUpsertThreadRef.current(started.thread, false);
        }

        if (!started.jobId) {
          dispatch({ type: 'FINISH_RUN', threadId: targetId });
          return;
        }

        if (stopRequestedRef.current.has(targetId) || deletedThreadsRef.current.has(targetId)) {
          try {
            const stopped = await cancelChatJob(started.jobId);
            if (stopped.thread && !deletedThreadsRef.current.has(targetId)) {
              onUpsertThreadRef.current(stopped.thread, false);
            }
          } catch {
            // ignore
          }
          dispatch({ type: 'FINISH_RUN', threadId: targetId, jobId: started.jobId });
          return;
        }

        dispatch({
          type: 'ATTACH_JOB',
          threadId: targetId,
          jobId: started.jobId,
          liveActivity: started.activity || initialActivity,
        });

        // Fire-and-forget background poller (does not block switching sessions)
        void pollJobForThread(targetId, started.jobId);
      } catch (err) {
        if (!committedOnServer && !deletedThreadsRef.current.has(targetId)) {
          onUpdateThreadByIdRef.current(targetId, (t) => ({
            ...t,
            messages: t.messages.filter((m) => m.id !== optimisticId),
          }));
          dispatch({
            type: 'RESTORE_DRAFT',
            draftKey: targetId,
            prompt: clean,
            attachments: attachmentsToSend,
          });
        }
        const message = err instanceof Error ? err.message : 'Failed to execute task.';
        dispatch({
          type: 'FINISH_RUN',
          threadId: targetId,
          error: stopRequestedRef.current.has(targetId) ? null : message,
        });
      }
      } finally {
        sendingLockRef.current.delete(lockKey);
        if (lockedTargetId) {
          sendingLockRef.current.delete(lockedTargetId);
        }
      }
    },
    [state, onCreateThread, pollJobForThread],
  );

  const submitApprovalDecision = useCallback(
    async ({ threadId, decision, toolName, editedArgs, accessMode }: SubmitApprovalParams) => {
      if (getThreadRun(state, threadId).sending || activeJobByThreadRef.current.has(threadId)) {
        return;
      }

      deletedThreadsRef.current.delete(threadId);
      stopRequestedRef.current.delete(threadId);

      const toolLabel = toolName || 'approved tool';
      const actionLabel =
        decision === 'reject'
          ? `Rejecting ${toolLabel}`
          : decision === 'edit'
            ? `Applying edited arguments for ${toolLabel}`
            : `Running approved tool: ${toolLabel}`;

      const initialActivity: LiveAgentActivity = {
        jobId: 'resuming',
        status: 'running',
        currentPhase: 'resuming_approval',
        lastEvent: actionLabel,
        elapsedSeconds: 0,
        turnCount: 1,
        thinking: null,
        activityFeed: [
          {
            id: 'resume-init-1',
            phase: 'resuming_approval',
            text: actionLabel,
          },
        ],
        toolEvents: [],
        fileChanges: [],
      };

      dispatch({
        type: 'START_RUN',
        threadId,
        isResume: true,
        liveActivity: initialActivity,
        startedAtMs: Date.now(),
      });

      try {
        const started = await startResumeApprovalJob(
          threadId,
          decision,
          editedArgs,
          undefined,
          undefined,
          accessMode,
        );

        if (started.thread && !deletedThreadsRef.current.has(threadId)) {
          onUpsertThreadRef.current(started.thread, false);
        }

        if (!started.jobId) {
          dispatch({ type: 'FINISH_RUN', threadId });
          return;
        }

        if (stopRequestedRef.current.has(threadId) || deletedThreadsRef.current.has(threadId)) {
          try {
            const stopped = await cancelChatJob(started.jobId);
            if (stopped.thread && !deletedThreadsRef.current.has(threadId)) {
              onUpsertThreadRef.current(stopped.thread, false);
            }
          } catch {
            // ignore
          }
          dispatch({ type: 'FINISH_RUN', threadId, jobId: started.jobId });
          return;
        }

        dispatch({
          type: 'ATTACH_JOB',
          threadId,
          jobId: started.jobId,
          liveActivity: started.activity || initialActivity,
        });

        void pollJobForThread(threadId, started.jobId);
      } catch (err) {
        dispatch({
          type: 'FINISH_RUN',
          threadId,
          error: err instanceof Error ? err.message : 'Unable to submit approval decision.',
        });
      }
    },
    [state, pollJobForThread],
  );

  const stopThread = useCallback(async (threadId: string) => {
    stopRequestedRef.current.add(threadId);
    dispatch({ type: 'MARK_STOPPING', threadId });

    pollControllersRef.current.get(threadId)?.abort();
    pollControllersRef.current.delete(threadId);

    const jobId = activeJobByThreadRef.current.get(threadId);
    if (jobId) {
      try {
        const res = await cancelChatJob(jobId);
        if (res.thread && !deletedThreadsRef.current.has(threadId)) {
          onUpsertThreadRef.current(res.thread, false);
        }
      } catch {
        // Ignore network errors during cancel
      }
      activeJobByThreadRef.current.delete(threadId);
    }

    dispatch({ type: 'FINISH_RUN', threadId, error: null });
  }, []);

  const cancelAndClearThread = useCallback(async (threadId: string) => {
    deletedThreadsRef.current.add(threadId);
    stopRequestedRef.current.add(threadId);

    pollControllersRef.current.get(threadId)?.abort();
    pollControllersRef.current.delete(threadId);

    const jobId = activeJobByThreadRef.current.get(threadId);
    if (jobId) {
      activeJobByThreadRef.current.delete(threadId);
      try {
        await cancelChatJob(jobId);
      } catch {
        // ignore
      }
    }

    dispatch({ type: 'CLEAR_THREAD_STATE', threadId });
  }, []);

  const readThreadRun = useCallback(
    (threadId: string | null | undefined) => {
      if (!threadId) {
        return getThreadRun(state, '__unassigned__');
      }
      return getThreadRun(state, threadId);
    },
    [state],
  );

  const readThreadDraft = useCallback(
    (threadId: string | null | undefined) => getThreadDraft(state, threadId),
    [state],
  );

  return {
    runningThreadIds,
    getThreadRun: readThreadRun,
    getThreadDraft: readThreadDraft,
    setThreadPrompt,
    clearThreadError,
    attachFilesToThread,
    removeAttachmentFromThread,
    sendMessage,
    submitApprovalDecision,
    stopThread,
    cancelAndClearThread,
  };
}
