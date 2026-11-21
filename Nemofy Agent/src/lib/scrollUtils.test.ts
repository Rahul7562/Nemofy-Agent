import assert from 'node:assert/strict';
import test from 'node:test';
import {
  computeAutoGrowHeight,
  isNearBottom,
  resolveScrollPinState,
} from './scrollUtils.ts';

test('isNearBottom returns true when content fits without scrolling', () => {
  assert.equal(isNearBottom({ scrollTop: 0, scrollHeight: 400, clientHeight: 500 }), true);
  assert.equal(isNearBottom({ scrollTop: 0, scrollHeight: 500, clientHeight: 500 }), true);
});

test('isNearBottom returns true within threshold and false when scrolled up', () => {
  assert.equal(isNearBottom({ scrollTop: 950, scrollHeight: 1500, clientHeight: 500 }, 64), true);
  assert.equal(isNearBottom({ scrollTop: 936, scrollHeight: 1500, clientHeight: 500 }, 64), true);
  assert.equal(isNearBottom({ scrollTop: 800, scrollHeight: 1500, clientHeight: 500 }, 64), false);
  assert.equal(isNearBottom({ scrollTop: 0, scrollHeight: 1500, clientHeight: 500 }, 64), false);
});

test('resolveScrollPinState immediately unpins when user scrolls upward', () => {
  // Even if still fairly close to bottom, explicit upward scroll unpins
  const pinned = resolveScrollPinState(
    { scrollTop: 980, scrollHeight: 1500, clientHeight: 500 },
    1000,
    true,
    48,
  );
  assert.equal(pinned, false);
});

test('resolveScrollPinState stays unpinned mid-list and re-pins at bottom', () => {
  const stillMid = resolveScrollPinState(
    { scrollTop: 650, scrollHeight: 1500, clientHeight: 500 },
    600,
    false,
    48,
  );
  assert.equal(stillMid, false);

  const reachedBottom = resolveScrollPinState(
    { scrollTop: 970, scrollHeight: 1500, clientHeight: 500 },
    900,
    false,
    48,
  );
  assert.equal(reachedBottom, true);
});

test('computeAutoGrowHeight expands smoothly between min and max height', () => {
  assert.deepEqual(computeAutoGrowHeight(20, 36, 220), { heightPx: 36, isCapped: false });
  assert.deepEqual(computeAutoGrowHeight(36, 36, 220), { heightPx: 36, isCapped: false });
  assert.deepEqual(computeAutoGrowHeight(96.2, 36, 220), { heightPx: 97, isCapped: false });
  assert.deepEqual(computeAutoGrowHeight(220, 36, 220), { heightPx: 220, isCapped: true });
  assert.deepEqual(computeAutoGrowHeight(480, 36, 220), { heightPx: 220, isCapped: true });
});
