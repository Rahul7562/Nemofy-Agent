import test from 'node:test';
import assert from 'node:assert/strict';
import {
  listProviderModelsDirect,
  parseModelsPayload,
  probeProviderModelDirect,
  resolveProviderApiKey,
} from './providerCatalog.ts';

test('resolveProviderApiKey isolates named providers from PRIMARY_API_KEY', () => {
  const env = {
    PRIMARY_API_KEY: 'sk-primary-only',
    RIVER_AI_API: 'sk-river-123',
    NVIDIA_API_KEY: '',
  };
  assert.equal(resolveProviderApiKey('river_ai', 'primary', env), 'sk-river-123');
  assert.equal(resolveProviderApiKey('nvidia', 'primary', env), null);
  assert.equal(resolveProviderApiKey('openrouter', 'primary', env), null);
  assert.equal(resolveProviderApiKey('custom', 'primary', env), 'sk-primary-only');
  assert.equal(resolveProviderApiKey('nvidia', 'primary', env, 'nvapi-draft'), 'nvapi-draft');
});

test('parseModelsPayload handles OpenAI/NVIDIA/OpenRouter/Gemini/Together/Ollama shapes and filters non-chat models', () => {
  const nvidiaModels = parseModelsPayload(
    {
      data: [
        { id: 'nvidia/llama-3.3-nemotron-super-49b-v1', owned_by: 'nvidia' },
        { id: 'nvidia/nv-embedqa-e5-v5', owned_by: 'nvidia' },
        { id: 'deepseek-ai/deepseek-r1', owned_by: 'deepseek-ai' },
      ],
    },
    'nvidia',
  );
  assert.deepEqual(
    nvidiaModels.map((m) => m.id),
    ['deepseek-ai/deepseek-r1', 'nvidia/llama-3.3-nemotron-super-49b-v1'],
  );

  const geminiModels = parseModelsPayload(
    {
      data: [
        { id: 'models/gemini-2.5-pro', inputTokenLimit: 1048576, outputTokenLimit: 65536 },
        { id: 'models/text-embedding-004' },
      ],
    },
    'gemini',
  );
  assert.equal(geminiModels.length, 1);
  assert.equal(geminiModels[0].id, 'gemini-2.5-pro');
  assert.equal(geminiModels[0].context_length, 1048576);
  assert.equal(geminiModels[0].max_output_tokens, 65536);

  const openRouterModels = parseModelsPayload(
    {
      data: [
        {
          id: 'openai/gpt-5.2',
          name: 'OpenAI GPT-5.2',
          context_length: 400000,
          top_provider: { max_completion_tokens: 128000 },
          pricing: { prompt: '0.00001', completion: '0.00003' },
        },
        {
          id: 'deepseek/deepseek-r1:free',
          name: 'DeepSeek R1 Free',
          pricing: { prompt: '0', completion: '0' },
        },
      ],
    },
    'openrouter',
  );
  assert.equal(openRouterModels.find((m) => m.id === 'deepseek/deepseek-r1:free')?.is_free, true);
  assert.equal(openRouterModels.find((m) => m.id === 'openai/gpt-5.2')?.is_free, false);
  assert.equal(openRouterModels.find((m) => m.id === 'openai/gpt-5.2')?.context_length, 400000);
  assert.equal(openRouterModels.find((m) => m.id === 'openai/gpt-5.2')?.max_output_tokens, 128000);
});

test('probeProviderModelDirect rejects missing key for cloud providers instead of returning false Connected', async () => {
  const persisted = {
    provider: 'river_ai',
    model: 'deepseek-ai/DeepSeek-V4.1-Flash',
    base_url: 'https://api.river.ai/v1',
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'high',
  };

  let fetchCalled = false;
  const mockFetch = (async () => {
    fetchCalled = true;
    return new Response(JSON.stringify({ data: [] }), { status: 200 });
  }) as unknown as typeof fetch;

  const res = await probeProviderModelDirect({
    role: 'primary',
    override: { provider: 'nvidia', model: 'nvidia/llama-3.3-nemotron-super-49b-v1' },
    persistedRoleConfig: persisted,
    env: { RIVER_AI_API: 'sk-river' },
    fetchImpl: mockFetch,
  });

  assert.equal(res.ok, false);
  assert.equal(fetchCalled, false);
  assert.match(res.error || '', /NVIDIA_API_KEY/);
});

test('probeProviderModelDirect verifies NVIDIA auth via /chat/completions and fails on 401', async () => {
  const persisted = {
    provider: 'nvidia',
    model: 'nvidia/llama-3.3-nemotron-super-49b-v1',
    base_url: 'https://integrate.api.nvidia.com/v1',
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'medium',
  };

  const calls: string[] = [];
  const mockFetch = (async (input: string | URL) => {
    const url = String(input);
    calls.push(url);
    if (url.endsWith('/models')) {
      return new Response(
        JSON.stringify({ data: [{ id: 'nvidia/llama-3.3-nemotron-super-49b-v1' }] }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    }
    return new Response(JSON.stringify({ detail: 'Invalid API key' }), {
      status: 401,
      headers: { 'content-type': 'application/json' },
    });
  }) as unknown as typeof fetch;

  const res = await probeProviderModelDirect({
    role: 'primary',
    override: persisted,
    persistedRoleConfig: persisted,
    apiKey: 'nvapi-bad-key',
    env: {},
    fetchImpl: mockFetch,
  });

  assert.equal(res.ok, false);
  assert.equal(res.authenticated, false);
  assert.equal(res.model_listed, true);
  assert.equal(calls.length, 2);
  assert.match(calls[1], /\/chat\/completions$/);
});

test('listProviderModelsDirect returns live models cleanly from /models', async () => {
  const mockFetch = (async () =>
    new Response(
      JSON.stringify({
        data: [
          { id: 'deepseek-ai/DeepSeek-V4.1-Flash' },
          { id: 'GLM-5.3' },
        ],
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    )) as unknown as typeof fetch;

  const res = await listProviderModelsDirect({
    provider: 'river_ai',
    env: { RIVER_AI_API: 'sk-test' },
    fetchImpl: mockFetch,
  });

  assert.equal(res.ok, true);
  assert.equal(res.source, 'live');
  assert.equal(res.count, 2);
  assert.deepEqual(
    res.models.map((m) => m.id),
    ['deepseek-ai/DeepSeek-V4.1-Flash', 'GLM-5.3'],
  );
});
