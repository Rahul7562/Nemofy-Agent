import assert from 'node:assert/strict';
import test from 'node:test';
import {
  LLM_PROVIDER_PRESETS,
  applyProviderPreset,
  canonicalizeProviderId,
  classifyFileModality,
  detectModelInputModalities,
  filterProviderModels,
  formatContextLength,
  formatFileSize,
  getDefaultAccessoriesConfig,
  getProviderPreset,
  mergeProviderModels,
  normalizeAccessoriesConfig,
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

  // Max execution turns 1..180 boundary validation
  const baseValid = {
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
  };
  assert.equal(validateLLMConfig({ ...baseValid, max_turns: 1 }).valid, true);
  assert.equal(validateLLMConfig({ ...baseValid, max_turns: 180 }).valid, true);
  assert.equal(validateLLMConfig({ ...baseValid, max_turns: 0 }).valid, false);
  assert.equal(validateLLMConfig({ ...baseValid, max_turns: -1 }).valid, false);
  assert.equal(validateLLMConfig({ ...baseValid, max_turns: 181 }).valid, false);
  assert.equal(validateLLMConfig({ ...baseValid, max_turns: 12.5 }).valid, false);
});

test('detectModelInputModalities matches DeepSeek-V4.1-Flash, Nemotron Omni, and text-only models', () => {
  const ds = detectModelInputModalities('river_ai', 'deepseek-ai/DeepSeek-V4.1-Flash');
  assert.equal(ds.has('text'), true);
  assert.equal(ds.has('image'), true);
  assert.equal(ds.has('audio'), false);
  assert.equal(ds.has('video'), false);

  const omni = detectModelInputModalities(
    'openrouter',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
  );
  assert.equal(omni.has('text'), true);
  assert.equal(omni.has('image'), true);
  assert.equal(omni.has('audio'), true);
  assert.equal(omni.has('video'), true);

  const r1 = detectModelInputModalities('deepseek', 'deepseek-r1');
  assert.equal(r1.has('image'), false);
  assert.equal(r1.has('audio'), false);
  assert.equal(r1.has('video'), false);
});

test('classifyFileModality, formatFileSize, and normalizeAccessoriesConfig behave cleanly', () => {
  assert.equal(classifyFileModality('image/png', 'pic.png'), 'image');
  assert.equal(classifyFileModality('video/mp4', 'demo.mp4'), 'video');
  assert.equal(classifyFileModality('audio/wav', 'voice.wav'), 'audio');
  assert.equal(classifyFileModality('image/svg+xml', 'bad.svg'), null);
  assert.equal(classifyFileModality('text/plain', 'notes.txt'), null);

  assert.equal(formatFileSize(512), '512 B');
  assert.equal(formatFileSize(2048), '2.0 KB');
  assert.equal(formatFileSize(5 * 1024 * 1024), '5.0 MB');

  const defaults = getDefaultAccessoriesConfig();
  assert.equal(defaults.video.provider, 'openrouter');
  assert.equal(defaults.video.model, 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free');
  const normalized = normalizeAccessoriesConfig({ audio: { enabled: false } });
  assert.equal(normalized.audio.enabled, false);
  assert.equal(normalized.video.enabled, true);
});
