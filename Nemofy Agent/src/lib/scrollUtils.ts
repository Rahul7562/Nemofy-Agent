/**
 * Pure scroll-pinning and auto-growing textarea utilities.
 * Keeps UI scroll behavior predictable and unit-testable.
 */

export interface ScrollMetrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

export interface AutoGrowResult {
  heightPx: number;
  isCapped: boolean;
}

/**
 * Returns true when a scroll container is within `thresholdPx` of its bottom edge
 * (or when its content does not exceed the visible viewport).
 */
export function isNearBottom(metrics: ScrollMetrics, thresholdPx = 64): boolean {
  const { scrollTop, scrollHeight, clientHeight } = metrics;
  if (scrollHeight <= clientHeight) return true;
  const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
  return distanceFromBottom <= Math.max(0, thresholdPx);
}

/**
 * Updates whether a scroll container should remain pinned to the bottom after a
 * scroll event. Scrolling upward immediately unpins so incoming updates never
 * fight or drag the user back down; scrolling back within `thresholdPx` of the
 * bottom re-pins.
 */
export function resolveScrollPinState(
  metrics: ScrollMetrics,
  previousScrollTop: number,
  wasPinned: boolean,
  thresholdPx = 48,
): boolean {
  if (metrics.scrollHeight <= metrics.clientHeight) {
    return true;
  }
  const scrolledUp = metrics.scrollTop < previousScrollTop - 2;
  if (scrolledUp) {
    return false;
  }
  if (isNearBottom(metrics, thresholdPx)) {
    return true;
  }
  return wasPinned && isNearBottom(metrics, thresholdPx);
}

/**
 * Computes the vertical pixel height for an auto-expanding chat textarea.
 * Clamps between `minHeightPx` and `maxHeightPx` and reports whether the
 * content reached the cap.
 */
export function computeAutoGrowHeight(
  scrollHeight: number,
  minHeightPx = 36,
  maxHeightPx = 220,
): AutoGrowResult {
  const safeMin = Math.max(1, Math.round(minHeightPx));
  const safeMax = Math.max(safeMin, Math.round(maxHeightPx));
  const measured = Number.isFinite(scrollHeight) ? Math.ceil(scrollHeight) : safeMin;
  if (measured <= safeMin) {
    return { heightPx: safeMin, isCapped: false };
  }
  if (measured >= safeMax) {
    return { heightPx: safeMax, isCapped: true };
  }
  return { heightPx: measured, isCapped: false };
}
