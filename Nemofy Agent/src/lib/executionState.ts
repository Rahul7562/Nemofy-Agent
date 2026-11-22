import type { ChatAttachment, LiveAgentActivity } from '../types/agent.ts';

export const DRAFT_UNASSIGNED_KEY = '__new__';

export interface ThreadRunState {
  sending: boolean;
  stopping: boolean;
  resuming: boolean;
  jobId: string | null;
  startedAtMs: number | null;
  liveActivity: LiveAgentActivity | null;
  error: string | null;
}

export interface ThreadDraftState {
  prompt: string;
  pendingAttachments: ChatAttachment[];
  uploadingFiles: string[];
}

export interface ExecutionStoreState {
  runsByThread: Record<string, ThreadRunState>;
  draftsByThread: Record<string, ThreadDraftState>;
}

export const EMPTY_THREAD_RUN: ThreadRunState = Object.freeze({
  sending: false,
  stopping: false,
  resuming: false,
  jobId: null,
  startedAtMs: null,
  liveActivity: null,
  error: null,
});

export const EMPTY_THREAD_DRAFT: ThreadDraftState = Object.freeze({
  prompt: '',
  pendingAttachments: [],
  uploadingFiles: [],
});

export function createInitialExecutionState(): ExecutionStoreState {
  return {
    runsByThread: {},
    draftsByThread: {},
  };
}

export function resolveDraftKey(threadId: string | null | undefined): string {
  return threadId && threadId.trim() ? threadId : DRAFT_UNASSIGNED_KEY;
}

export function getThreadRun(
  state: ExecutionStoreState,
  threadId: string | null | undefined,
): ThreadRunState {
  if (!threadId) return EMPTY_THREAD_RUN;
  return state.runsByThread[threadId] ?? EMPTY_THREAD_RUN;
}

export function getThreadDraft(
  state: ExecutionStoreState,
  threadId: string | null | undefined,
): ThreadDraftState {
  const key = resolveDraftKey(threadId);
  return state.draftsByThread[key] ?? EMPTY_THREAD_DRAFT;
}

export function getRunningThreadIds(state: ExecutionStoreState): Set<string> {
  const ids = new Set<string>();
  for (const [threadId, run] of Object.entries(state.runsByThread)) {
    if (run.sending) {
      ids.add(threadId);
    }
  }
  return ids;
}

export type ExecutionAction =
  | {
      type: 'START_RUN';
      threadId: string;
      isResume?: boolean;
      jobId?: string | null;
      liveActivity: LiveAgentActivity;
      startedAtMs: number;
    }
  | {
      type: 'ATTACH_JOB';
      threadId: string;
      jobId: string;
      liveActivity?: LiveAgentActivity | null;
    }
  | {
      type: 'UPDATE_ACTIVITY';
      threadId: string;
      jobId: string;
      liveActivity: LiveAgentActivity;
    }
  | {
      type: 'TICK_ELAPSED';
      nowMs: number;
    }
  | {
      type: 'MARK_STOPPING';
      threadId: string;
    }
  | {
      type: 'FINISH_RUN';
      threadId: string;
      jobId?: string | null;
      error?: string | null;
    }
  | {
      type: 'SET_THREAD_ERROR';
      threadId: string;
      error: string | null;
    }
  | {
      type: 'CLEAR_THREAD_STATE';
      threadId: string;
    }
  | {
      type: 'SET_DRAFT_PROMPT';
      draftKey: string;
      prompt: string;
    }
  | {
      type: 'ADD_UPLOADING_FILES';
      draftKey: string;
      names: string[];
    }
  | {
      type: 'REMOVE_UPLOADING_FILES';
      draftKey: string;
      names: string[];
    }
  | {
      type: 'ADD_ATTACHMENTS';
      draftKey: string;
      attachments: ChatAttachment[];
    }
  | {
      type: 'REMOVE_ATTACHMENT';
      draftKey: string;
      attachmentId: string;
    }
  | {
      type: 'CLEAR_DRAFT';
      draftKey: string;
    }
  | {
      type: 'RESTORE_DRAFT';
      draftKey: string;
      prompt: string;
      attachments?: ChatAttachment[];
    }
  | {
      type: 'RESET_ALL';
    };

function getOrCreateRun(state: ExecutionStoreState, threadId: string): ThreadRunState {
  return state.runsByThread[threadId] ?? { ...EMPTY_THREAD_RUN };
}

function getOrCreateDraft(state: ExecutionStoreState, draftKey: string): ThreadDraftState {
  return (
    state.draftsByThread[draftKey] ?? {
      prompt: '',
      pendingAttachments: [],
      uploadingFiles: [],
    }
  );
}

export function executionReducer(
  state: ExecutionStoreState,
  action: ExecutionAction,
): ExecutionStoreState {
  switch (action.type) {
    case 'START_RUN': {
      return {
        ...state,
        runsByThread: {
          ...state.runsByThread,
          [action.threadId]: {
            sending: true,
            stopping: false,
            resuming: Boolean(action.isResume),
            jobId: action.jobId ?? null,
            startedAtMs: action.startedAtMs,
            liveActivity: action.liveActivity,
            error: null,
          },
        },
      };
    }

    case 'ATTACH_JOB': {
      const existing = getOrCreateRun(state, action.threadId);
      if (!existing.sending) return state;
      return {
        ...state,
        runsByThread: {
          ...state.runsByThread,
          [action.threadId]: {
            ...existing,
            jobId: action.jobId,
            liveActivity: action.liveActivity ?? existing.liveActivity,
          },
        },
      };
    }

    case 'UPDATE_ACTIVITY': {
      const existing = state.runsByThread[action.threadId];
      if (!existing || !existing.sending) return state;
      if (existing.jobId && existing.jobId !== action.jobId) {
        // Ignore stale updates from a superseded job
        return state;
      }
      return {
        ...state,
        runsByThread: {
          ...state.runsByThread,
          [action.threadId]: {
            ...existing,
            jobId: action.jobId,
            liveActivity: action.liveActivity,
          },
        },
      };
    }

    case 'TICK_ELAPSED': {
      let changed = false;
      const nextRuns: Record<string, ThreadRunState> = { ...state.runsByThread };
      for (const [threadId, run] of Object.entries(state.runsByThread)) {
        if (!run.sending || !run.liveActivity || run.startedAtMs === null) continue;
        const localElapsed = Math.max(0, (action.nowMs - run.startedAtMs) / 1000);
        if (localElapsed > run.liveActivity.elapsedSeconds) {
          nextRuns[threadId] = {
            ...run,
            liveActivity: {
              ...run.liveActivity,
              elapsedSeconds: localElapsed,
            },
          };
          changed = true;
        }
      }
      return changed ? { ...state, runsByThread: nextRuns } : state;
    }

    case 'MARK_STOPPING': {
      const existing = state.runsByThread[action.threadId];
      if (!existing || !existing.sending) return state;
      return {
        ...state,
        runsByThread: {
          ...state.runsByThread,
          [action.threadId]: {
            ...existing,
            stopping: true,
          },
        },
      };
    }

    case 'FINISH_RUN': {
      const existing = state.runsByThread[action.threadId];
      if (!existing) {
        if (!action.error) return state;
        return {
          ...state,
          runsByThread: {
            ...state.runsByThread,
            [action.threadId]: {
              ...EMPTY_THREAD_RUN,
              error: action.error,
            },
          },
        };
      }
      // If a newer job superseded this one, do not clobber the newer run
      if (action.jobId && existing.jobId && existing.jobId !== action.jobId) {
        return state;
      }
      return {
        ...state,
        runsByThread: {
          ...state.runsByThread,
          [action.threadId]: {
            sending: false,
            stopping: false,
            resuming: false,
            jobId: null,
            startedAtMs: null,
            liveActivity: null,
            error: action.error !== undefined ? action.error : existing.error,
          },
        },
      };
    }

    case 'SET_THREAD_ERROR': {
      const existing = getOrCreateRun(state, action.threadId);
      return {
        ...state,
        runsByThread: {
          ...state.runsByThread,
          [action.threadId]: {
            ...existing,
            error: action.error,
          },
        },
      };
    }

    case 'CLEAR_THREAD_STATE': {
      const nextRuns = { ...state.runsByThread };
      const nextDrafts = { ...state.draftsByThread };
      delete nextRuns[action.threadId];
      delete nextDrafts[action.threadId];
      return {
        runsByThread: nextRuns,
        draftsByThread: nextDrafts,
      };
    }

    case 'SET_DRAFT_PROMPT': {
      const existing = getOrCreateDraft(state, action.draftKey);
      return {
        ...state,
        draftsByThread: {
          ...state.draftsByThread,
          [action.draftKey]: {
            ...existing,
            prompt: action.prompt,
          },
        },
      };
    }

    case 'ADD_UPLOADING_FILES': {
      const existing = getOrCreateDraft(state, action.draftKey);
      return {
        ...state,
        draftsByThread: {
          ...state.draftsByThread,
          [action.draftKey]: {
            ...existing,
            uploadingFiles: [...existing.uploadingFiles, ...action.names],
          },
        },
      };
    }

    case 'REMOVE_UPLOADING_FILES': {
      const existing = getOrCreateDraft(state, action.draftKey);
      const remaining = [...existing.uploadingFiles];
      for (const name of action.names) {
        const idx = remaining.indexOf(name);
        if (idx >= 0) remaining.splice(idx, 1);
      }
      return {
        ...state,
        draftsByThread: {
          ...state.draftsByThread,
          [action.draftKey]: {
            ...existing,
            uploadingFiles: remaining,
          },
        },
      };
    }

    case 'ADD_ATTACHMENTS': {
      const existing = getOrCreateDraft(state, action.draftKey);
      return {
        ...state,
        draftsByThread: {
          ...state.draftsByThread,
          [action.draftKey]: {
            ...existing,
            pendingAttachments: [...existing.pendingAttachments, ...action.attachments].slice(0, 6),
          },
        },
      };
    }

    case 'REMOVE_ATTACHMENT': {
      const existing = getOrCreateDraft(state, action.draftKey);
      return {
        ...state,
        draftsByThread: {
          ...state.draftsByThread,
          [action.draftKey]: {
            ...existing,
            pendingAttachments: existing.pendingAttachments.filter(
              (a) => a.id !== action.attachmentId,
            ),
          },
        },
      };
    }

    case 'CLEAR_DRAFT': {
      const existing = getOrCreateDraft(state, action.draftKey);
      return {
        ...state,
        draftsByThread: {
          ...state.draftsByThread,
          [action.draftKey]: {
            ...existing,
            prompt: '',
            pendingAttachments: [],
          },
        },
      };
    }

    case 'RESTORE_DRAFT': {
      const existing = getOrCreateDraft(state, action.draftKey);
      return {
        ...state,
        draftsByThread: {
          ...state.draftsByThread,
          [action.draftKey]: {
            ...existing,
            prompt: existing.prompt || action.prompt,
            pendingAttachments:
              existing.pendingAttachments.length > 0
                ? existing.pendingAttachments
                : action.attachments || [],
          },
        },
      };
    }

    case 'RESET_ALL': {
      return createInitialExecutionState();
    }

    default:
      return state;
  }
}
