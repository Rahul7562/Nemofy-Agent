import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_ACTIVITY_PREFS,
  MAX_CONSECUTIVE_POLL_ERRORS,
  computePollRetryDelayMs,
  dedupeFileChanges,
  extractLatestThoughtLine,
  formatElapsedSeconds,
  formatPhaseLabel,
  parseActivityPrefs,
  shouldRetryPollError,
  summarizeActivity,
} from './activityUtils.ts';

test('formatElapsedSeconds formats sub-minute and multi-minute durations cleanly', () => {
  assert.equal(formatElapsedSeconds(0), '0.0s');
  assert.equal(formatElapsedSeconds(3.46), '3.5s');
  assert.equal(formatElapsedSeconds(65), '1m 5s');
  assert.equal(formatElapsedSeconds(-2), '0.0s');
});

test('formatPhaseLabel shows concrete live action details or clean phase fallbacks', () => {
  assert.equal(formatPhaseLabel('thinking'), 'Analyzing your request');
  assert.equal(formatPhaseLabel('observe'), 'Reviewing tool results');
  assert.equal(formatPhaseLabel('responding'), 'Drafting response');
  assert.equal(
    formatPhaseLabel('act_tool_start', 'Running write_file (path=notes.md)'),
    'Running write_file (path=notes.md)',
  );
  assert.equal(
    formatPhaseLabel('thinking', 'Checking existing workspace files first...'),
    'Checking existing workspace files first...',
  );
});

test('extractLatestThoughtLine extracts the latest meaningful reasoning line', () => {
  assert.equal(extractLatestThoughtLine(null), null);
  assert.equal(
    extractLatestThoughtLine('First inspect /workspace.\n\nNow write summary.txt with results.'),
    'Now write summary.txt with results.',
  );
});

test('dedupeFileChanges removes identical consecutive duplicates while preserving create->modify->delete sequence', () => {
  const list = dedupeFileChanges([
    { kind: 'dir', action: 'created', path: 'demo', tool: 'write_file' },
    { kind: 'dir', action: 'created', path: 'demo', tool: 'write_file' },
    { kind: 'file', action: 'created', path: 'demo/a.txt', tool: 'write_file' },
    { kind: 'file', action: 'modified', path: 'demo/a.txt', tool: 'edit_file' },
    { kind: 'file', action: 'deleted', path: 'demo/a.txt', tool: 'bash_exec' },
  ]);
  assert.equal(list.length, 4);
  assert.deepEqual(
    list.map((c) => `${c.kind}:${c.action}:${c.path}`),
    ['dir:created:demo', 'file:created:demo/a.txt', 'file:modified:demo/a.txt', 'file:deleted:demo/a.txt'],
  );
});

test('summarizeActivity aggregates thinking, tools, and file/dir changes into readable summary parts', () => {
  const summary = summarizeActivity({
    thinking: 'Let me create a directory, write a file, and remove an old temp file.',
    toolEvents: [
      { id: '1', tool: 'write_file', status: 'ok' },
      { id: '2', tool: 'bash_exec', status: 'running' },
    ],
    fileChanges: [
      { kind: 'dir', action: 'created', path: 'reports' },
      { kind: 'file', action: 'created', path: 'reports/q1.md' },
      { kind: 'file', action: 'modified', path: 'reports/q1.md' },
      { kind: 'file', action: 'deleted', path: 'reports/old.tmp' },
      { kind: 'dir', action: 'deleted', path: 'scratch' },
    ],
  });

  assert.equal(summary.hasThinking, true);
  assert.equal(summary.toolCount, 2);
  assert.equal(summary.runningToolCount, 1);
  assert.equal(summary.dirCreated, 1);
  assert.equal(summary.fileCreated, 1);
  assert.equal(summary.fileModified, 1);
  assert.equal(summary.fileDeleted, 1);
  assert.equal(summary.dirDeleted, 1);
  assert.equal(summary.totalFileChanges, 5);
  assert.equal(summary.hasAnyActivity, true);
  assert.deepEqual(summary.summaryParts, [
    'Thinking',
    '2 tools',
    '+1 folder, +1 file, ~1 edited, -1 file, -1 folder',
  ]);
});

test('summarizeActivity returns hasAnyActivity=false when empty', () => {
  const empty = summarizeActivity({ thinking: '   ', toolEvents: [], fileChanges: [] });
  assert.equal(empty.hasAnyActivity, false);
  assert.deepEqual(empty.summaryParts, []);
});

test('parseActivityPrefs falls back safely on invalid JSON and preserves valid user preferences', () => {
  assert.deepEqual(parseActivityPrefs(null), DEFAULT_ACTIVITY_PREFS);
  assert.deepEqual(parseActivityPrefs('{bad-json'), DEFAULT_ACTIVITY_PREFS);
  assert.deepEqual(
    parseActivityPrefs(JSON.stringify({ liveExpanded: false, showToolsSection: false })),
    {
      liveExpanded: false,
      showThinkingSection: true,
      showToolsSection: false,
      showFilesSection: true,
    },
  );
});

test('formatPhaseLabel covers rate_limited, background_task, and resuming_approval phases', () => {
  assert.equal(formatPhaseLabel('rate_limited'), 'Rate limited (HTTP 429) — retrying every 3s');
  assert.equal(formatPhaseLabel('background_task'), 'Waiting for background task');
  assert.equal(formatPhaseLabel('resuming_approval'), 'Applying your approval decision');
  assert.equal(formatPhaseLabel('compacting_context'), 'Compacting conversation context');
  assert.equal(
    formatPhaseLabel('rate_limited', 'Rate limit (429) — retrying in 3s (attempt 2)'),
    'Rate limit (429) — retrying in 3s (attempt 2)',
  );
});

test('shouldRetryPollError and computePollRetryDelayMs bound retries and skip aborts', () => {
  assert.equal(shouldRetryPollError(1, MAX_CONSECUTIVE_POLL_ERRORS, false), true);
  assert.equal(shouldRetryPollError(MAX_CONSECUTIVE_POLL_ERRORS, MAX_CONSECUTIVE_POLL_ERRORS, false), true);
  assert.equal(shouldRetryPollError(MAX_CONSECUTIVE_POLL_ERRORS + 1, MAX_CONSECUTIVE_POLL_ERRORS, false), false);
  assert.equal(shouldRetryPollError(1, MAX_CONSECUTIVE_POLL_ERRORS, true), false);
  assert.equal(computePollRetryDelayMs(1), 800);
  assert.equal(computePollRetryDelayMs(2), 1500);
  assert.equal(computePollRetryDelayMs(10), 4000);
});
