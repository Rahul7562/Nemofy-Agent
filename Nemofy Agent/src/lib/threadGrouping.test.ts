import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatThread } from '../types/agent.ts';
import {
  classifyThreadDateBucket,
  filterThreadsByQuery,
  groupThreadsByDate,
  resolveThreadTimestampMs,
} from './threadGrouping.ts';

function makeThread(id: string, title: string, updatedAt: string, createdAt = updatedAt): ChatThread {
  return {
    id,
    title,
    createdAt,
    updatedAt,
    messages: [],
  };
}

test('resolveThreadTimestampMs prefers updatedAt and falls back to createdAt', () => {
  const withBoth = makeThread('1', 'A', '2026-11-22T10:00:00.000Z', '2026-11-10T10:00:00.000Z');
  assert.equal(resolveThreadTimestampMs(withBoth), Date.parse('2026-11-22T10:00:00.000Z'));

  const fallback = makeThread('2', 'B', '', '2026-11-15T08:30:00.000Z');
  assert.equal(resolveThreadTimestampMs(fallback), Date.parse('2026-11-15T08:30:00.000Z'));

  const invalid = makeThread('3', 'C', 'not-a-date', 'also-bad');
  assert.equal(resolveThreadTimestampMs(invalid), null);
});

test('groupThreadsByDate buckets threads into Today, Yesterday, Previous 7 Days, and Older', () => {
  const now = new Date(2026, 10, 22, 15, 0, 0); // Nov 22, 2026 15:00 local

  const tTodayEarly = makeThread('t1', 'Morning check', new Date(2026, 10, 22, 8, 0, 0).toISOString());
  const tTodayLate = makeThread('t2', 'Afternoon plan', new Date(2026, 10, 22, 14, 30, 0).toISOString());
  const tYesterday = makeThread('t3', 'Yesterday audit', new Date(2026, 10, 21, 20, 0, 0).toISOString());
  const tFourDaysAgo = makeThread('t4', 'Mid-week refactor', new Date(2026, 10, 18, 12, 0, 0).toISOString());
  const tSevenDaysAgo = makeThread('t5', 'Last week notes', new Date(2026, 10, 15, 9, 0, 0).toISOString());
  const tOlder = makeThread('t6', 'Archive spec', new Date(2026, 10, 10, 9, 0, 0).toISOString());
  const tInvalid = makeThread('t7', 'Unknown date', '', '');

  const groups = groupThreadsByDate(
    [tOlder, tTodayEarly, tSevenDaysAgo, tYesterday, tInvalid, tTodayLate, tFourDaysAgo],
    now,
  );

  assert.deepEqual(
    groups.map((g) => g.label),
    ['Today', 'Yesterday', 'Previous 7 Days', 'Older'],
  );

  assert.deepEqual(
    groups[0].threads.map((t) => t.id),
    ['t2', 't1'],
  );
  assert.deepEqual(
    groups[1].threads.map((t) => t.id),
    ['t3'],
  );
  assert.deepEqual(
    groups[2].threads.map((t) => t.id),
    ['t4', 't5'],
  );
  assert.deepEqual(
    groups[3].threads.map((t) => t.id),
    ['t6', 't7'],
  );
});

test('groupThreadsByDate omits empty groups and classifies future timestamps as Today', () => {
  const now = new Date(2026, 10, 22, 12, 0, 0);
  const future = new Date(2026, 10, 23, 2, 0, 0).getTime();
  assert.equal(classifyThreadDateBucket(future, now), 'Today');

  const groups = groupThreadsByDate([
    makeThread('only-old', 'Old thread', new Date(2026, 9, 1, 10, 0, 0).toISOString()),
  ], now);

  assert.equal(groups.length, 1);
  assert.equal(groups[0].label, 'Older');
});

test('filterThreadsByQuery performs case-insensitive title matching', () => {
  const threads = [
    makeThread('1', 'Inspect Workspace Files', '2026-11-22T10:00:00.000Z'),
    makeThread('2', 'SaaS Launch Checklist', '2026-11-22T09:00:00.000Z'),
    makeThread('3', '', '2026-11-22T08:00:00.000Z'),
  ];

  assert.equal(filterThreadsByQuery(threads, '').length, 3);
  assert.deepEqual(
    filterThreadsByQuery(threads, 'workspace').map((t) => t.id),
    ['1'],
  );
  assert.deepEqual(
    filterThreadsByQuery(threads, 'UNTITLED').map((t) => t.id),
    ['3'],
  );
  assert.equal(filterThreadsByQuery(threads, 'nonexistent').length, 0);
});
