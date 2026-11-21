import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatThinkingBadge,
  getThinkingProfile,
  normalizeModelSlug,
  normalizeThinkingLevel,
} from './thinkingModes';

test('normalizeModelSlug strips provider and gateway prefixes cleanly', () => {
  assert.equal(normalizeModelSlug('openai/gpt-5.2'), 'gpt-5.2');
  assert.equal(normalizeModelSlug('deepseek-ai/DeepSeek-V4.1-Flash'), 'deepseek-v4.1-flash');
  assert.equal(normalizeModelSlug('accounts/fireworks/models/deepseek-r1'), 'deepseek-r1');
});

test('getThinkingProfile adapts options per model family (max vs xhigh vs standard)', () => {
  // DeepSeek V4.1 Flash (River AI): off, low, high, max (no medium, no xhigh)
  const deepseekV4 = getThinkingProfile('river_ai', 'deepseek-ai/DeepSeek-V4.1-Flash');
  assert.equal(deepseekV4.supported, true);
  assert.deepEqual(
    deepseekV4.options.map((o) => o.value),
    ['off', 'low', 'high', 'max'],
  );

  // OpenAI GPT-5.2: off, low, medium, high, xhigh (no max)
  const gpt52 = getThinkingProfile('openai', 'gpt-5.2');
  assert.equal(gpt52.supported, true);
  assert.deepEqual(
    gpt52.options.map((o) => o.value),
    ['off', 'low', 'medium', 'high', 'xhigh'],
  );

  // Anthropic Claude Opus 4.6 (direct or via OpenRouter): off, low, medium, high, max (no xhigh)
  const claude46 = getThinkingProfile('openrouter', 'anthropic/claude-opus-4-6');
  assert.equal(claude46.supported, true);
  assert.deepEqual(
    claude46.options.map((o) => o.value),
    ['off', 'low', 'medium', 'high', 'max'],
  );

  // OpenAI o3-mini: low, medium, high (cannot disable)
  const o3Mini = getThinkingProfile('openai', 'o3-mini');
  assert.equal(o3Mini.supported, true);
  assert.equal(o3Mini.canDisable, false);
  assert.deepEqual(
    o3Mini.options.map((o) => o.value),
    ['low', 'medium', 'high'],
  );

  // Grok 3 mini: low, high only
  const grokMini = getThinkingProfile('openrouter', 'x-ai/grok-3-mini');
  assert.deepEqual(
    grokMini.options.map((o) => o.value),
    ['low', 'high'],
  );

  // GPT-4o: standard model, thinking not supported
  const gpt4o = getThinkingProfile('openai', 'gpt-4o');
  assert.equal(gpt4o.supported, false);
  assert.deepEqual(
    gpt4o.options.map((o) => o.value),
    ['off'],
  );
});

test('normalizeThinkingLevel adapts seamlessly across models with max, xhigh, or no thinking', () => {
  const deepseekV4 = getThinkingProfile('river_ai', 'deepseek-ai/DeepSeek-V4.1-Flash');
  const gpt52 = getThinkingProfile('openai', 'gpt-5.2');
  const claude46 = getThinkingProfile('anthropic', 'claude-opus-4-6');
  const o3Mini = getThinkingProfile('openai', 'o3-mini');
  const gpt4o = getThinkingProfile('openai', 'gpt-4o');

  // max <-> xhigh equivalence
  assert.equal(normalizeThinkingLevel(gpt52, 'max'), 'xhigh');
  assert.equal(normalizeThinkingLevel(claude46, 'xhigh'), 'max');
  assert.equal(normalizeThinkingLevel(deepseekV4, 'xhigh'), 'max');

  // medium maps cleanly to high on DeepSeek-V4.1-Flash (which supports low/high/max)
  assert.equal(normalizeThinkingLevel(deepseekV4, 'medium'), 'high');

  // xhigh / max cap cleanly at high on o3-mini, and off maps to low
  assert.equal(normalizeThinkingLevel(o3Mini, 'xhigh'), 'high');
  assert.equal(normalizeThinkingLevel(o3Mini, 'max'), 'high');
  assert.equal(normalizeThinkingLevel(o3Mini, 'off'), 'low');

  // Non-thinking model always resolves to off
  assert.equal(normalizeThinkingLevel(gpt4o, 'high'), 'off');
  assert.equal(normalizeThinkingLevel(gpt4o, 'max'), 'off');

  // Badge formatting
  assert.equal(formatThinkingBadge('off'), null);
  assert.equal(formatThinkingBadge('xhigh'), 'XHigh');
  assert.equal(formatThinkingBadge('max'), 'Max');
});
