import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createInitialExecutionState,
  executionReducer,
  getRunningThreadIds,
  getThreadDraft,
  getThreadRun,
} from './executionState.ts';
import type { LiveAgentActivity } from '../types/agent.ts';

function makeActivity(jobId: string, lastEvent: string, elapsedSeconds = 0): LiveAgentActivity {
  return {
    jobId,
    status: 'running',
    currentPhase: 'thinking',
    lastEvent,
    elapsedSeconds,
    turnCount: 1,
    thinking: null,
    activityFeed: [{ id: `${jobId}-1`, phase: 'thinking', text: lastEvent }],
    toolEvents: [],
    fileChanges: [],
  };
}

test('executionReducer isolates concurrent runs, activity, errors, and stopping strictly per thread', () => {
  let state = createInitialExecutionState();

  // Start run in Thread A
  state = executionReducer(state, {
    type: 'START_RUN',
    threadId: 'thread-a',
    jobId: 'job-a',
    liveActivity: makeActivity('job-a', 'Working on A'),
    startedAtMs: 1000,
  });

  // Thread B remains idle and completely usable while A is running
  assert.equal(getThreadRun(state, 'thread-a').sending, true);
  assert.equal(getThreadRun(state, 'thread-b').sending, false);
  assert.equal(getThreadRun(state, 'thread-b').liveActivity, null);

  // Start concurrent run in Thread B
  state = executionReducer(state, {
    type: 'START_RUN',
    threadId: 'thread-b',
    jobId: 'job-b',
    liveActivity: makeActivity('job-b', 'Working on B'),
    startedAtMs: 2000,
  });

  assert.deepEqual(Array.from(getRunningThreadIds(state)).sort(), ['thread-a', 'thread-b']);

  // Update activity on A must not affect B
  state = executionReducer(state, {
    type: 'UPDATE_ACTIVITY',
    threadId: 'thread-a',
    jobId: 'job-a',
    liveActivity: makeActivity('job-a', 'Running tool in A', 4.5),
  });
  assert.equal(getThreadRun(state, 'thread-a').liveActivity?.lastEvent, 'Running tool in A');
  assert.equal(getThreadRun(state, 'thread-b').liveActivity?.lastEvent, 'Working on B');

  // Stale jobId update on A is ignored
  state = executionReducer(state, {
    type: 'UPDATE_ACTIVITY',
    threadId: 'thread-a',
    jobId: 'job-stale',
    liveActivity: makeActivity('job-stale', 'Stale event'),
  });
  assert.equal(getThreadRun(state, 'thread-a').liveActivity?.lastEvent, 'Running tool in A');

  // Stopping Thread A does NOT stop Thread B
  state = executionReducer(state, { type: 'MARK_STOPPING', threadId: 'thread-a' });
  assert.equal(getThreadRun(state, 'thread-a').stopping, true);
  assert.equal(getThreadRun(state, 'thread-b').stopping, false);

  state = executionReducer(state, { type: 'FINISH_RUN', threadId: 'thread-a', jobId: 'job-a', error: null });
  assert.equal(getThreadRun(state, 'thread-a').sending, false);
  assert.equal(getThreadRun(state, 'thread-b').sending, true);
  assert.deepEqual(Array.from(getRunningThreadIds(state)), ['thread-b']);

  // Error on Thread A does not leak to Thread B or C
  state = executionReducer(state, { type: 'SET_THREAD_ERROR', threadId: 'thread-a', error: 'A failed' });
  assert.equal(getThreadRun(state, 'thread-a').error, 'A failed');
  assert.equal(getThreadRun(state, 'thread-b').error, null);
  assert.equal(getThreadRun(state, 'thread-c').error, null);
});

test('executionReducer preserves independent drafts across rapid session switches', () => {
  let state = createInitialExecutionState();

  state = executionReducer(state, {
    type: 'SET_DRAFT_PROMPT',
    draftKey: 'thread-a',
    prompt: 'Draft for A',
  });
  state = executionReducer(state, {
    type: 'SET_DRAFT_PROMPT',
    draftKey: 'thread-b',
    prompt: 'Draft for B',
  });
  state = executionReducer(state, {
    type: 'SET_DRAFT_PROMPT',
    draftKey: 'thread-c',
    prompt: '',
  });

  assert.equal(getThreadDraft(state, 'thread-a').prompt, 'Draft for A');
  assert.equal(getThreadDraft(state, 'thread-b').prompt, 'Draft for B');
  assert.equal(getThreadDraft(state, 'thread-c').prompt, '');

  // Clearing A's draft on send does not touch B
  state = executionReducer(state, { type: 'CLEAR_DRAFT', draftKey: 'thread-a' });
  assert.equal(getThreadDraft(state, 'thread-a').prompt, '');
  assert.equal(getThreadDraft(state, 'thread-b').prompt, 'Draft for B');

  // Deleting thread B clears only thread B
  state = executionReducer(state, { type: 'CLEAR_THREAD_STATE', threadId: 'thread-b' });
  assert.equal(getThreadDraft(state, 'thread-b').prompt, '');
});

test('TICK_ELAPSED advances elapsedSeconds independently for all running threads', () => {
  let state = createInitialExecutionState();
  state = executionReducer(state, {
    type: 'START_RUN',
    threadId: 'thread-a',
    jobId: 'job-a',
    liveActivity: makeActivity('job-a', 'A', 0),
    startedAtMs: 1000,
  });
  state = executionReducer(state, {
    type: 'START_RUN',
    threadId: 'thread-b',
    jobId: 'job-b',
    liveActivity: makeActivity('job-b', 'B', 10),
    startedAtMs: 5000,
  });

  state = executionReducer(state, { type: 'TICK_ELAPSED', nowMs: 6500 });
  assert.equal(getThreadRun(state, 'thread-a').liveActivity?.elapsedSeconds, 5.5);
  // Thread B already had 10s elapsed from server, so local 1.5s should not decrease it
  assert.equal(getThreadRun(state, 'thread-b').liveActivity?.elapsedSeconds, 10);
});
