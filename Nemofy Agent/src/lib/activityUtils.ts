import type { FileChangeItem, ToolEventItem } from '../types/agent.ts';

export interface ActivitySummary {
  hasThinking: boolean;
  toolCount: number;
  runningToolCount: number;
  fileCreated: number;
  fileModified: number;
  fileDeleted: number;
  dirCreated: number;
  dirDeleted: number;
  totalFileChanges: number;
  hasAnyActivity: boolean;
  summaryParts: string[];
}

export interface ActivityUserPrefs {
  liveExpanded: boolean;
  showThinkingSection: boolean;
  showToolsSection: boolean;
  showFilesSection: boolean;
}

export const ACTIVITY_PREFS_STORAGE_KEY = 'nemofy_activity_prefs_v1';

export const DEFAULT_ACTIVITY_PREFS: ActivityUserPrefs = {
  liveExpanded: true,
  showThinkingSection: true,
  showToolsSection: true,
  showFilesSection: true,
};

export function formatElapsedSeconds(seconds?: number | null): string {
  const safe = typeof seconds === 'number' && Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  if (safe < 60) {
    return `${safe.toFixed(1)}s`;
  }
  const mins = Math.floor(safe / 60);
  const rem = Math.round(safe % 60);
  return `${mins}m ${rem}s`;
}

export function formatPhaseLabel(phase?: string | null, lastEvent?: string | null): string {
  const cleanEvent = (lastEvent || '').trim();
  const cleanPhase = (phase || '').trim().toLowerCase();
  if (
    cleanEvent &&
    cleanEvent !== cleanPhase &&
    cleanEvent !== 'LangGraph job started' &&
    cleanEvent !== 'Job created; waiting to start.'
  ) {
    return cleanEvent;
  }
  if (cleanPhase === 'thinking' || cleanPhase === 'plan_start') {
    return 'Analyzing your request';
  }
  if (cleanPhase === 'planning_tool') {
    return 'Preparing tool call';
  }
  if (cleanPhase === 'responding') {
    return 'Drafting response';
  }
  if (cleanPhase === 'act_tool_start') {
    return 'Running tool';
  }
  if (cleanPhase === 'act_tool_done') {
    return 'Tool finished';
  }
  if (cleanPhase === 'observe') {
    return 'Reviewing tool results';
  }
  if (cleanPhase === 'reflect') {
    return 'Deciding next step';
  }
  if (cleanPhase === 'plan_llm_done') {
    return 'Preparing next step';
  }
  if (cleanPhase === 'rate_limited') {
    return 'Rate limited (HTTP 429) — retrying every 3s';
  }
  if (cleanPhase === 'background_task') {
    return 'Waiting for background task';
  }
  if (cleanPhase === 'resuming_approval') {
    return 'Applying your approval decision';
  }
  if (cleanPhase === 'compacting_context') {
    return 'Compacting conversation context';
  }
  if (cleanPhase === 'subagent_reason') {
    return 'Sub-agent reasoning';
  }
  if (cleanPhase === 'starting' || cleanPhase === 'staged') {
    return 'Starting Nemo';
  }
  return 'Nemo is working';
}

export const MAX_CONSECUTIVE_POLL_ERRORS = 5;

export function shouldRetryPollError(
  consecutiveErrors: number,
  maxRetries = MAX_CONSECUTIVE_POLL_ERRORS,
  isAbort = false,
): boolean {
  if (isAbort) return false;
  return consecutiveErrors > 0 && consecutiveErrors <= maxRetries;
}

export function computePollRetryDelayMs(attempt: number): number {
  const safeAttempt = Math.max(1, Math.floor(attempt));
  return Math.min(4000, 800 + (safeAttempt - 1) * 700);
}

export function extractLatestThoughtLine(thinking?: string | null): string | null {
  if (!thinking) return null;
  const lines = thinking
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length === 0) return null;
  const latest = lines[lines.length - 1];
  return latest.length > 110 ? `${latest.slice(0, 107)}...` : latest;
}

export function dedupeFileChanges(changes?: FileChangeItem[] | null): FileChangeItem[] {
  if (!Array.isArray(changes) || changes.length === 0) return [];
  const out: FileChangeItem[] = [];
  for (const item of changes) {
    if (!item || !item.path) continue;
    const prev = out[out.length - 1];
    if (
      prev &&
      prev.kind === item.kind &&
      prev.action === item.action &&
      prev.path === item.path &&
      prev.tool === item.tool
    ) {
      continue;
    }
    out.push(item);
  }
  return out;
}

export function summarizeActivity(input: {
  thinking?: string | null;
  toolEvents?: ToolEventItem[] | null;
  fileChanges?: FileChangeItem[] | null;
}): ActivitySummary {
  const hasThinking = Boolean(input.thinking && input.thinking.trim().length > 0);
  const tools = Array.isArray(input.toolEvents) ? input.toolEvents : [];
  const changes = dedupeFileChanges(input.fileChanges);

  let runningToolCount = 0;
  for (const t of tools) {
    if (t.status === 'running') runningToolCount += 1;
  }

  let fileCreated = 0;
  let fileModified = 0;
  let fileDeleted = 0;
  let dirCreated = 0;
  let dirDeleted = 0;

  for (const ch of changes) {
    if (ch.kind === 'dir') {
      if (ch.action === 'created') dirCreated += 1;
      else if (ch.action === 'deleted') dirDeleted += 1;
    } else {
      if (ch.action === 'created') fileCreated += 1;
      else if (ch.action === 'modified') fileModified += 1;
      else if (ch.action === 'deleted') fileDeleted += 1;
    }
  }

  const summaryParts: string[] = [];
  if (hasThinking) {
    summaryParts.push('Thinking');
  }
  if (tools.length > 0) {
    summaryParts.push(`${tools.length} ${tools.length === 1 ? 'tool' : 'tools'}`);
  }

  const fsTokens: string[] = [];
  if (dirCreated > 0) fsTokens.push(`+${dirCreated} ${dirCreated === 1 ? 'folder' : 'folders'}`);
  if (fileCreated > 0) fsTokens.push(`+${fileCreated} ${fileCreated === 1 ? 'file' : 'files'}`);
  if (fileModified > 0) fsTokens.push(`~${fileModified} edited`);
  if (fileDeleted > 0) fsTokens.push(`-${fileDeleted} ${fileDeleted === 1 ? 'file' : 'files'}`);
  if (dirDeleted > 0) fsTokens.push(`-${dirDeleted} ${dirDeleted === 1 ? 'folder' : 'folders'}`);

  if (fsTokens.length > 0) {
    summaryParts.push(fsTokens.join(', '));
  }

  return {
    hasThinking,
    toolCount: tools.length,
    runningToolCount,
    fileCreated,
    fileModified,
    fileDeleted,
    dirCreated,
    dirDeleted,
    totalFileChanges: changes.length,
    hasAnyActivity: hasThinking || tools.length > 0 || changes.length > 0,
    summaryParts,
  };
}

export function parseActivityPrefs(raw: string | null): ActivityUserPrefs {
  if (!raw) return { ...DEFAULT_ACTIVITY_PREFS };
  try {
    const parsed = JSON.parse(raw) as Partial<ActivityUserPrefs>;
    return {
      liveExpanded:
        typeof parsed.liveExpanded === 'boolean'
          ? parsed.liveExpanded
          : DEFAULT_ACTIVITY_PREFS.liveExpanded,
      showThinkingSection:
        typeof parsed.showThinkingSection === 'boolean'
          ? parsed.showThinkingSection
          : DEFAULT_ACTIVITY_PREFS.showThinkingSection,
      showToolsSection:
        typeof parsed.showToolsSection === 'boolean'
          ? parsed.showToolsSection
          : DEFAULT_ACTIVITY_PREFS.showToolsSection,
      showFilesSection:
        typeof parsed.showFilesSection === 'boolean'
          ? parsed.showFilesSection
          : DEFAULT_ACTIVITY_PREFS.showFilesSection,
    };
  } catch {
    return { ...DEFAULT_ACTIVITY_PREFS };
  }
}

export function readActivityPrefs(): ActivityUserPrefs {
  if (typeof window === 'undefined') return { ...DEFAULT_ACTIVITY_PREFS };
  try {
    return parseActivityPrefs(window.localStorage.getItem(ACTIVITY_PREFS_STORAGE_KEY));
  } catch {
    return { ...DEFAULT_ACTIVITY_PREFS };
  }
}

export function writeActivityPrefs(prefs: ActivityUserPrefs): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(ACTIVITY_PREFS_STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // ignore storage errors
  }
}
