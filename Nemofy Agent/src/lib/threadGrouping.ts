import type { ChatThread } from '../types/agent.ts';

export type ThreadDateBucket = 'Today' | 'Yesterday' | 'Previous 7 Days' | 'Older';

export interface ThreadDateGroup {
  label: ThreadDateBucket;
  threads: ChatThread[];
}

const BUCKET_ORDER: ThreadDateBucket[] = [
  'Today',
  'Yesterday',
  'Previous 7 Days',
  'Older',
];

function startOfLocalDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

/**
 * Resolves the canonical timestamp (ms since epoch) for a thread using
 * `updatedAt` with fallback to `createdAt`. Returns `null` when missing/invalid.
 */
export function resolveThreadTimestampMs(thread: Pick<ChatThread, 'updatedAt' | 'createdAt'>): number | null {
  const candidates = [thread.updatedAt, thread.createdAt];
  for (const raw of candidates) {
    if (!raw || typeof raw !== 'string') continue;
    const parsed = Date.parse(raw);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

/**
 * Classifies a timestamp into Today, Yesterday, Previous 7 Days, or Older
 * relative to the local calendar day of `now`.
 */
export function classifyThreadDateBucket(timestampMs: number | null, now: Date = new Date()): ThreadDateBucket {
  if (timestampMs === null || !Number.isFinite(timestampMs)) {
    return 'Older';
  }
  const todayStart = startOfLocalDay(now).getTime();
  const threadDayStart = startOfLocalDay(new Date(timestampMs)).getTime();
  const msPerDay = 24 * 60 * 60 * 1000;
  const dayDiff = Math.round((todayStart - threadDayStart) / msPerDay);

  if (dayDiff <= 0) return 'Today';
  if (dayDiff === 1) return 'Yesterday';
  if (dayDiff <= 7) return 'Previous 7 Days';
  return 'Older';
}

/**
 * Filters threads client-side by title (case-insensitive substring match).
 */
export function filterThreadsByQuery(threads: ChatThread[], query: string): ChatThread[] {
  const clean = query.trim().toLowerCase();
  if (!clean) return threads;
  return threads.filter((t) => (t.title || 'Untitled session').toLowerCase().includes(clean));
}

/**
 * Groups threads into ordered, non-empty calendar buckets (`Today`, `Yesterday`,
 * `Previous 7 Days`, `Older`), sorting newest-first within each group.
 */
export function groupThreadsByDate(
  threads: ChatThread[],
  now: Date = new Date(),
): ThreadDateGroup[] {
  const buckets = new Map<ThreadDateBucket, Array<{ thread: ChatThread; ts: number }>>();
  for (const label of BUCKET_ORDER) {
    buckets.set(label, []);
  }

  for (const thread of threads) {
    const ts = resolveThreadTimestampMs(thread);
    const bucket = classifyThreadDateBucket(ts, now);
    buckets.get(bucket)!.push({ thread, ts: ts ?? -Infinity });
  }

  const groups: ThreadDateGroup[] = [];
  for (const label of BUCKET_ORDER) {
    const items = buckets.get(label)!;
    if (items.length === 0) continue;
    items.sort((a, b) => b.ts - a.ts);
    groups.push({
      label,
      threads: items.map((entry) => entry.thread),
    });
  }
  return groups;
}
