import assert from 'node:assert/strict';
import test from 'node:test';
import { applyProviderPreset, getProviderPreset, validateLLMConfig } from './llmPresets.ts';

test('applyProviderPreset populates base URL and suggested model', () => {
  const base = {
    provider: 'river_ai',
    model: 'deepseek-ai/DeepSeek-V4.1-Flash',
    base_url: 'https://api.river.ai/v1',
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'off',
  };
  const next = applyProviderPreset(base, 'openai');
  assert.equal(next.provider, 'openai');
  assert.equal(next.base_url, 'https://api.openai.com/v1');
  assert.equal(next.model, 'gpt-4o');
  assert.ok(getProviderPreset('ollama')?.requiresApiKey === false);
});

test('validateLLMConfig checks both main agent and subagent settings', () => {
  const validConfig = {
    primary: {
      provider: 'river_ai',
      model: 'deepseek-ai/DeepSeek-V4.1-Flash',
      base_url: 'https://api.river.ai/v1',
      temperature: 0.2,
      max_tokens: 4096,
      reasoning_effort: 'off',
    },
    worker: {
      provider: 'openai',
      model: 'gpt-4o-mini',
      base_url: 'https://api.openai.com/v1',
      temperature: 0.1,
      max_tokens: 2048,
      reasoning_effort: 'low',
    },
    max_turns: 30,
  };
  assert.equal(validateLLMConfig(validConfig).valid, true);

  const badUrl = {
    ...validConfig,
    worker: { ...validConfig.worker, base_url: 'not-a-url' },
  };
  assert.equal(validateLLMConfig(badUrl).valid, false);

  const badTemp = {
    ...validConfig,
    primary: { ...validConfig.primary, temperature: 3.5 },
  };
  assert.equal(validateLLMConfig(badTemp).valid, false);
});
