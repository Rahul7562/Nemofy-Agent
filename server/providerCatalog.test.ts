import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildReferenceModelIndex,
  clearReferenceModelCatalogCache,
  listProviderModelsDirect,
  lookupReferenceModelMetadata,
  parseModelsPayload,
  probeProviderModelDirect,
  resolveModelMetadataFromCatalog,
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

test('providerCatalog blocks internal/metadata SSRF URLs and prevents stored vault key exfiltration to non-canonical hosts (BUG-3)', async () => {
  let called = false;
  const mockFetch = (async () => {
    called = true;
    return new Response(JSON.stringify({ data: [] }), { status: 200 });
  }) as unknown as typeof fetch;

  const metadataRes = await listProviderModelsDirect({
    provider: 'openai',
    base_url: 'http://169.254.169.254/latest/meta-data',
    env: { OPENAI_API_KEY: 'sk-vault-secret' },
    fetchImpl: mockFetch,
  });
  assert.equal(metadataRes.ok, false);
  assert.equal(called, false);
  assert.match(metadataRes.error || '', /private|internal|blocked|metadata/i);

  const loopbackCloudRes = await listProviderModelsDirect({
    provider: 'openai',
    base_url: 'http://127.0.0.1:8080/v1',
    env: { OPENAI_API_KEY: 'sk-vault-secret' },
    fetchImpl: mockFetch,
  });
  assert.equal(loopbackCloudRes.ok, false);
  assert.equal(called, false);

  const nonCanonicalList = await listProviderModelsDirect({
    provider: 'openai',
    base_url: 'https://93.184.216.34/v1',
    env: { OPENAI_API_KEY: 'sk-vault-secret' },
    fetchImpl: mockFetch,
  });
  assert.equal(nonCanonicalList.ok, false);
  assert.equal(called, false);
  assert.match(nonCanonicalList.error || '', /canonical|explicit API key/i);

  const persisted = {
    provider: 'openai',
    model: 'gpt-4o',
    base_url: 'https://api.openai.com/v1',
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'medium',
  };
  const exfilProbe = await probeProviderModelDirect({
    role: 'primary',
    override: { provider: 'openai', base_url: 'https://93.184.216.34/v1', model: 'gpt-4o' },
    persistedRoleConfig: persisted,
    env: { OPENAI_API_KEY: 'sk-vault-secret' },
    fetchImpl: mockFetch,
  });
  assert.equal(exfilProbe.ok, false);
  assert.equal(called, false);
  assert.match(exfilProbe.error || '', /canonical|explicit API key/i);
});

test('parseModelsPayload extracts max_model_len, context_size, and nested limits across vLLM and custom routers', () => {
  const models = parseModelsPayload(
    {
      data: [
        { id: 'vllm/qwen3-coder-30b', max_model_len: 262144, max_completion_tokens: 65536 },
        { id: 'custom/router-a', limits: { context: 524288, output: 32768 } },
        { id: 'custom/router-b', context_size: 196608, maxOutputTokens: 16384 },
      ],
    },
    'vllm',
  );

  const vllmModel = models.find((m) => m.id === 'vllm/qwen3-coder-30b');
  assert.equal(vllmModel?.context_length, 262144);
  assert.equal(vllmModel?.max_output_tokens, 65536);

  const routerA = models.find((m) => m.id === 'custom/router-a');
  assert.equal(routerA?.context_length, 524288);
  assert.equal(routerA?.max_output_tokens, 32768);

  const routerB = models.find((m) => m.id === 'custom/router-b');
  assert.equal(routerB?.context_length, 196608);
  assert.equal(routerB?.max_output_tokens, 16384);
});

test('listProviderModelsDirect queries Google Gemini native /v1beta/models for official inputTokenLimit and outputTokenLimit', async () => {
  const seenUrls: string[] = [];
  const mockFetch = (async (input: string | URL) => {
    const url = String(input);
    seenUrls.push(url);
    if (url.includes('/v1beta/models?pageSize=200')) {
      return new Response(
        JSON.stringify({
          models: [
            {
              name: 'models/gemini-2.5-pro',
              displayName: 'Gemini 2.5 Pro',
              inputTokenLimit: 1048576,
              outputTokenLimit: 65536,
              supportedGenerationMethods: ['generateContent', 'countTokens'],
            },
            {
              name: 'models/embedding-001',
              displayName: 'Embedding 001',
              inputTokenLimit: 2048,
              supportedGenerationMethods: ['embedContent'],
            },
          ],
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    }
    return new Response(JSON.stringify({ data: [] }), { status: 404 });
  }) as unknown as typeof fetch;

  const res = await listProviderModelsDirect({
    provider: 'gemini',
    env: { GEMINI_API_KEY: 'AIza-test-key' },
    fetchImpl: mockFetch,
  });

  assert.equal(res.ok, true);
  assert.equal(res.count, 1);
  assert.equal(res.models[0].id, 'gemini-2.5-pro');
  assert.equal(res.models[0].context_length, 1048576);
  assert.equal(res.models[0].max_output_tokens, 65536);
  assert.ok(seenUrls[0].includes('/v1beta/models?pageSize=200'));
});

test('listProviderModelsDirect enriches Ollama local models via /api/show model_info context_length', async () => {
  const mockFetch = (async (input: string | URL) => {
    const url = String(input);
    if (url.endsWith('/v1/models')) {
      return new Response(
        JSON.stringify({ data: [{ id: 'my-local-llama:8b' }] }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    }
    if (url.endsWith('/api/show')) {
      return new Response(
        JSON.stringify({
          model_info: { 'llama.context_length': 131072 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    }
    return new Response('Not found', { status: 404 });
  }) as unknown as typeof fetch;

  const res = await listProviderModelsDirect({
    provider: 'ollama',
    fetchImpl: mockFetch,
  });

  assert.equal(res.ok, true);
  assert.equal(res.models[0].id, 'my-local-llama:8b');
  assert.equal(res.models[0].context_length, 131072);
});

test('cross-provider reference catalog enriches NVIDIA/Anthropic/OpenAI/River models without overwriting provider values', async () => {
  clearReferenceModelCatalogCache();
  const referenceIndex = buildReferenceModelIndex([
    { id: 'minimax/minimax-m3.1-flash', context_length: 1_048_576, max_output_tokens: 524_288 },
    { id: 'mistralai/mistral-large', context_length: 128_000, max_output_tokens: 102_400 },
    { id: 'mistralai/mistral-large-4-0', context_length: 1_048_576, max_output_tokens: 262_144 },
    { id: 'moonshotai/kimi-k3.1-pro', context_length: 1_048_576, max_output_tokens: 943_718 },
    { id: 'anthropic/claude-opus-4.6', context_length: 1_000_000, max_output_tokens: 128_000 },
    { id: 'anthropic/claude-opus-4.6:batch', context_length: 200_000, max_output_tokens: 32_000 },
    { id: 'qwen/qwen3.8-27b', context_length: 1_024_000, max_output_tokens: 131_072 },
    { id: 'z-ai/glm-5.3', context_length: 1_048_576, max_output_tokens: 131_072 },
    { id: 'deepseek/deepseek-v4.1-flash', context_length: 1_048_576, max_output_tokens: 393_216 },
  ]);

  // :batch variants must never pollute canonical lookups
  assert.equal(lookupReferenceModelMetadata('claude-opus-4-6', referenceIndex)?.context_length, 1_000_000);
  assert.equal(lookupReferenceModelMetadata('Qwen/Qwen3.8-27B-FP8', referenceIndex)?.context_length, 1_024_000);
  assert.equal(lookupReferenceModelMetadata('GLM-5.3', referenceIndex)?.context_length, 1_048_576);
  assert.equal(lookupReferenceModelMetadata('deepseek-ai/DeepSeek-V4.1-Flash', referenceIndex)?.context_length, 1_048_576);

  const mockFetch = (async () =>
    new Response(
      JSON.stringify({
        data: [
          { id: 'minimaxai/minimax-m3.1-flash' },
          { id: 'mistralai/mistral-large-4-1t-a52b' },
          // Provider explicitly reports a custom 512k context window: MUST NOT be overwritten by reference 1M!
          { id: 'moonshotai/kimi-k3.1-pro', context_window: 524_288 },
          // Unknown custom model must NOT be stamped with a fake 128k context_length
          { id: 'acme/proprietary-unknown-v1' },
        ],
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    )) as unknown as typeof fetch;

  const res = await listProviderModelsDirect({
    provider: 'nvidia',
    env: { NVIDIA_API_KEY: 'nvapi-test' },
    fetchImpl: mockFetch,
    referenceIndex,
  });

  assert.equal(res.ok, true);
  const byId = Object.fromEntries(res.models.map((m) => [m.id, m]));
  assert.equal(byId['minimaxai/minimax-m3.1-flash'].context_length, 1_048_576);
  assert.equal(byId['minimaxai/minimax-m3.1-flash'].max_output_tokens, 524_288);
  assert.equal(byId['mistralai/mistral-large-4-1t-a52b'].context_length, 1_048_576);
  assert.equal(byId['moonshotai/kimi-k3.1-pro'].context_length, 524_288);
  assert.equal(byId['moonshotai/kimi-k3.1-pro'].max_output_tokens, 524_288); // clamped to provider's 524,288 window!
  assert.equal(byId['acme/proprietary-unknown-v1'].context_length, undefined);

  // resolveModelMetadataFromCatalog fills missing role metadata cleanly
  const resolved = await resolveModelMetadataFromCatalog({
    provider: 'anthropic',
    model: 'claude-opus-4-6',
    referenceIndex,
  });
  assert.equal(resolved.context_window, 1_000_000);
  assert.equal(resolved.max_output_tokens, 128_000);
});
