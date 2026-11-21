import assert from 'node:assert/strict';
import test from 'node:test';
import {
  LLM_PROVIDER_PRESETS,
  applyProviderPreset,
  canonicalizeProviderId,
  filterProviderModels,
  formatContextLength,
  getProviderPreset,
  mergeProviderModels,
  validateLLMConfig,
} from './llmPresets.ts';
import { getThinkingProfile } from './thinkingModes.ts';

test('NVIDIA NIM provider preset is registered with expected endpoint and models', () => {
  const nvidia = getProviderPreset('nvidia');
  assert.ok(nvidia);
  assert.equal(nvidia.label, 'NVIDIA NIM');
  assert.equal(nvidia.baseUrl, 'https://integrate.api.nvidia.com/v1');
  assert.deepEqual(nvidia.envVars, ['NVIDIA_API_KEY']);
  assert.ok(nvidia.defaultModels.includes('nvidia/nemotron-3-ultra-550b-a55b'));
  assert.equal(canonicalizeProviderId('nvidia_nim'), 'nvidia');
  assert.ok(LLM_PROVIDER_PRESETS.some((p) => p.id === 'nvidia'));
});

test('applyProviderPreset updates base_url, model, and thinking level cleanly across providers', () => {
  const initial = {
    provider: 'river_ai',
    model: 'deepseek-ai/DeepSeek-V4.1-Flash',
    base_url: 'https://api.river.ai/v1',
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'max',
  };

  const switchedNvidia = applyProviderPreset(initial, 'nvidia');
  assert.equal(switchedNvidia.provider, 'nvidia');
  assert.equal(switchedNvidia.base_url, 'https://integrate.api.nvidia.com/v1');
  assert.equal(switchedNvidia.model, 'nvidia/nemotron-3-ultra-550b-a55b');
  assert.equal(switchedNvidia.reasoning_effort, 'high');

  const switchedCustom = applyProviderPreset(switchedNvidia, 'custom');
  assert.equal(switchedCustom.provider, 'custom');
  assert.equal(switchedCustom.base_url, '');
  assert.equal(switchedCustom.model, '');

  const customWithUrl = {
    ...switchedCustom,
    base_url: 'https://internal.llm/v1',
    model: 'my-model',
  };
  const keepCustom = applyProviderPreset(customWithUrl, 'custom');
  assert.equal(keepCustom.base_url, 'https://internal.llm/v1');
  assert.equal(keepCustom.model, 'my-model');
});

test('NVIDIA Nemotron thinking profile is enabled while standard Llama-3 remains standard', () => {
  const nemotron = getThinkingProfile('nvidia', 'nvidia/llama-3.3-nemotron-super-49b-v1');
  assert.equal(nemotron.supported, true);
  assert.ok(nemotron.options.some((o) => o.value === 'high'));

  const llama = getThinkingProfile('nvidia', 'meta/llama-3.3-70b-instruct');
  assert.equal(llama.supported, false);
});

test('mergeProviderModels and filterProviderModels support live catalogs, presets, and search', () => {
  const live = [
    { id: 'nvidia/nemotron-3-ultra-550b-a55b:free', name: 'Nemotron 3 Ultra', is_free: true, context_length: 131072, source: 'live' as const },
    { id: 'openai/gpt-5.2', name: 'GPT-5.2', is_free: false, context_length: 400000, source: 'live' as const },
  ];

  const merged = mergeProviderModels(live, ['fallback/model'], 'custom/my-model');
  assert.equal(merged[0].id, 'custom/my-model');
  assert.equal(merged.length, 3);

  const fallbackMerged = mergeProviderModels([], ['nvidia/llama-3.3-nemotron-super-49b-v1'], '');
  assert.equal(fallbackMerged.length, 1);
  assert.equal(fallbackMerged[0].source, 'preset');

  const filtered = filterProviderModels(merged, 'nemotron free');
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].id, 'nvidia/nemotron-3-ultra-550b-a55b:free');

  assert.equal(formatContextLength(131072), '131k ctx');
  assert.equal(formatContextLength(1_000_000), '1M ctx');
  assert.equal(formatContextLength(null), null);
});

test('validateLLMConfig catches missing URLs and accepts valid NVIDIA config', () => {
  const valid = validateLLMConfig({
    primary: {
      provider: 'nvidia',
      model: 'nvidia/llama-3.3-nemotron-super-49b-v1',
      base_url: 'https://integrate.api.nvidia.com/v1',
      temperature: 0.2,
      max_tokens: 4096,
      reasoning_effort: 'high',
    },
    worker: {
      provider: 'river_ai',
      model: 'deepseek-ai/DeepSeek-V4.1-Flash',
      base_url: 'https://api.river.ai/v1',
      temperature: 0.1,
      max_tokens: 2048,
      reasoning_effort: 'off',
    },
    max_turns: 30,
  });
  assert.equal(valid.valid, true);

  const invalid = validateLLMConfig({
    primary: {
      provider: 'custom',
      model: 'my-model',
      base_url: '',
      temperature: 0.2,
      max_tokens: 4096,
      reasoning_effort: 'off',
    },
    worker: {
      provider: 'river_ai',
      model: 'deepseek-ai/DeepSeek-V4.1-Flash',
      base_url: 'https://api.river.ai/v1',
      temperature: 0.1,
      max_tokens: 2048,
      reasoning_effort: 'off',
    },
    max_turns: 30,
  });
  assert.equal(invalid.valid, false);
});
