import { normalizeProviderId, type RoleLLMConfig } from './configStore.ts';

export interface ProviderModelCatalogItem {
  id: string;
  name?: string;
  owned_by?: string;
  context_length?: number | null;
  is_free?: boolean;
  source?: 'live' | 'preset';
}

export interface ProviderModelsResult {
  ok: boolean;
  provider: string;
  base_url: string;
  source: 'live' | 'preset' | 'error';
  models: ProviderModelCatalogItem[];
  count: number;
  error: string | null;
}

export interface ProviderProbeResult {
  ok: boolean;
  role: 'primary' | 'worker';
  provider: string;
  model: string;
  base_url: string;
  reachable: boolean;
  authenticated: boolean;
  model_listed: boolean | null;
  status_code?: number | null;
  error?: string | null;
}

export const PROVIDER_CATALOG_DEFAULTS: Record<
  string,
  { baseUrl: string; envVars: string[]; requiresKey: boolean }
> = {
  river_ai: { baseUrl: 'https://api.river.ai/v1', envVars: ['RIVER_AI_API'], requiresKey: true },
  openai: { baseUrl: 'https://api.openai.com/v1', envVars: ['OPENAI_API_KEY'], requiresKey: true },
  openrouter: { baseUrl: 'https://openrouter.ai/api/v1', envVars: ['OPENROUTER_API_KEY'], requiresKey: true },
  nvidia: { baseUrl: 'https://integrate.api.nvidia.com/v1', envVars: ['NVIDIA_API_KEY'], requiresKey: true },
  gemini: {
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    envVars: ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
    requiresKey: true,
  },
  deepseek: { baseUrl: 'https://api.deepseek.com/v1', envVars: ['DEEPSEEK_API_KEY'], requiresKey: true },
  groq: { baseUrl: 'https://api.groq.com/openai/v1', envVars: ['GROQ_API_KEY'], requiresKey: true },
  anthropic: { baseUrl: 'https://api.anthropic.com/v1', envVars: ['ANTHROPIC_API_KEY'], requiresKey: true },
  together: { baseUrl: 'https://api.together.xyz/v1', envVars: ['TOGETHER_API_KEY'], requiresKey: true },
  fireworks: { baseUrl: 'https://api.fireworks.ai/inference/v1', envVars: ['FIREWORKS_API_KEY'], requiresKey: true },
  ollama: { baseUrl: 'http://127.0.0.1:11434/v1', envVars: ['OLLAMA_API_KEY'], requiresKey: false },
  vllm: { baseUrl: 'http://127.0.0.1:8000/v1', envVars: ['VLLM_API_KEY'], requiresKey: false },
  custom: { baseUrl: '', envVars: ['PRIMARY_API_KEY', 'WORKER_API_KEY'], requiresKey: true },
};

const NON_CHAT_MODEL_MARKERS = [
  'embedding',
  'text-embedding',
  'embed-',
  'rerank',
  'moderation',
  'whisper',
  'tts-',
  'dall-e',
  'imagen',
  'stable-diffusion',
  'flux-',
  'babbage',
  'davinci',
  'nv-embed',
  'nv-rerank',
  'audio-transcribe',
  'omni-moderation',
];

export function resolveProviderApiKey(
  provider: string,
  role: 'primary' | 'worker',
  env: Record<string, string | undefined> = process.env,
  explicitKey?: string,
): string | null {
  const explicit = (explicitKey || '').trim();
  if (explicit) return explicit;

  const prov = normalizeProviderId(provider);
  const info = PROVIDER_CATALOG_DEFAULTS[prov];
  if (prov !== 'custom' && info) {
    for (const envVar of info.envVars) {
      const val = (env[envVar] || '').trim();
      if (val) return val;
    }
    return null;
  }

  const roleVar = role === 'worker' ? 'WORKER_API_KEY' : 'PRIMARY_API_KEY';
  const fallbackVar = role === 'worker' ? 'PRIMARY_API_KEY' : 'WORKER_API_KEY';
  const val = (env[roleVar] || env[fallbackVar] || '').trim();
  return val || null;
}

export function isNonChatModelId(modelId: string): boolean {
  const lower = modelId.trim().toLowerCase();
  return NON_CHAT_MODEL_MARKERS.some((marker) => lower.includes(marker));
}

export function parseModelsPayload(payload: unknown, provider: string): ProviderModelCatalogItem[] {
  let rawItems: unknown[] = [];
  if (Array.isArray(payload)) {
    rawItems = payload;
  } else if (payload && typeof payload === 'object') {
    const obj = payload as Record<string, unknown>;
    if (Array.isArray(obj.data)) {
      rawItems = obj.data;
    } else if (Array.isArray(obj.models)) {
      rawItems = obj.models;
    }
  }

  const prov = normalizeProviderId(provider);
  const seen = new Set<string>();
  const models: ProviderModelCatalogItem[] = [];

  for (const item of rawItems) {
    if (!item || typeof item !== 'object') continue;
    const record = item as Record<string, unknown>;
    let rawId = String(record.id ?? record.name ?? record.model ?? '').trim();
    if (!rawId) continue;
    if (prov === 'gemini' && rawId.startsWith('models/')) {
      rawId = rawId.slice('models/'.length);
    }
    if (isNonChatModelId(rawId) || seen.has(rawId)) continue;
    seen.add(rawId);

    const displayName =
      typeof record.display_name === 'string' && record.display_name.trim()
        ? record.display_name.trim()
        : typeof record.name === 'string' && record.name.trim() && record.name.trim() !== rawId
          ? record.name.trim()
          : undefined;

    const ownedBy =
      typeof record.owned_by === 'string' && record.owned_by.trim()
        ? record.owned_by.trim()
        : undefined;

    let contextLength: number | null = null;
    for (const field of ['context_length', 'context_window', 'max_context_length', 'inputTokenLimit']) {
      const candidate = Number(record[field]);
      if (Number.isFinite(candidate) && candidate > 0) {
        contextLength = Math.round(candidate);
        break;
      }
    }

    let isFree = rawId.toLowerCase().endsWith(':free');
    if (!isFree && record.pricing && typeof record.pricing === 'object') {
      const pricing = record.pricing as Record<string, unknown>;
      const promptCost = Number(pricing.prompt);
      const completionCost = Number(pricing.completion);
      if (Number.isFinite(promptCost) && Number.isFinite(completionCost) && promptCost === 0 && completionCost === 0) {
        isFree = true;
      }
    }

    models.push({
      id: rawId,
      ...(displayName ? { name: displayName } : {}),
      ...(ownedBy ? { owned_by: ownedBy } : {}),
      ...(contextLength ? { context_length: contextLength } : {}),
      is_free: isFree,
      source: 'live',
    });
  }

  models.sort((a, b) => a.id.toLowerCase().localeCompare(b.id.toLowerCase()));
  return models;
}

function buildProviderHeaders(provider: string, apiKey: string | null): Record<string, string> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  const prov = normalizeProviderId(provider);
  if (apiKey) {
    if (prov === 'anthropic') {
      headers['x-api-key'] = apiKey;
      headers['anthropic-version'] = '2023-06-01';
    } else {
      headers.Authorization = `Bearer ${apiKey}`;
    }
  }
  if (prov === 'openrouter') {
    headers['HTTP-Referer'] = 'https://nemofy.tech';
    headers['X-Title'] = 'Nemofy Agent';
  }
  return headers;
}

function extractErrorDetail(statusCode: number, bodyText: string, provider: string): string {
  const trimmed = bodyText.trim();
  if (trimmed) {
    try {
      const parsed = JSON.parse(trimmed) as Record<string, unknown>;
      if (parsed.error && typeof parsed.error === 'object') {
        const errObj = parsed.error as Record<string, unknown>;
        if (typeof errObj.message === 'string' && errObj.message.trim()) {
          return errObj.message.trim().slice(0, 240);
        }
      }
      if (typeof parsed.error === 'string' && parsed.error.trim()) {
        return parsed.error.trim().slice(0, 240);
      }
      if (typeof parsed.detail === 'string' && parsed.detail.trim()) {
        return parsed.detail.trim().slice(0, 240);
      }
      if (typeof parsed.message === 'string' && parsed.message.trim()) {
        return parsed.message.trim().slice(0, 240);
      }
    } catch {
      // plain text body
    }
  }
  if (statusCode === 401 || statusCode === 403) {
    const expected = PROVIDER_CATALOG_DEFAULTS[provider]?.envVars.join(' / ') || 'API key';
    return `Authentication failed (HTTP ${statusCode}). Provide a valid ${expected}.`;
  }
  return `Provider '${provider}' returned HTTP ${statusCode}.`;
}

export async function listProviderModelsDirect(params: {
  provider: string;
  base_url?: string;
  api_key?: string;
  role?: 'primary' | 'worker';
  env?: Record<string, string | undefined>;
  fetchImpl?: typeof fetch;
}): Promise<ProviderModelsResult> {
  const prov = normalizeProviderId(params.provider);
  const role = params.role === 'worker' ? 'worker' : 'primary';
  const defaults = PROVIDER_CATALOG_DEFAULTS[prov];
  const baseUrl = (params.base_url || defaults?.baseUrl || '').trim().replace(/\/+$/, '');
  const fetchFn = params.fetchImpl || fetch;

  if (!baseUrl || !/^https?:\/\//i.test(baseUrl)) {
    return {
      ok: false,
      provider: prov,
      base_url: baseUrl,
      source: 'error',
      models: [],
      count: 0,
      error: `Base URL is required to fetch models for '${prov}'.`,
    };
  }

  const apiKey = resolveProviderApiKey(prov, role, params.env || process.env, params.api_key);
  const headers = buildProviderHeaders(prov, apiKey);

  try {
    const resp = await fetchFn(`${baseUrl}/models`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(12_000),
    });

    if (resp.ok) {
      const payload = (await resp.json()) as unknown;
      const models = parseModelsPayload(payload, prov);
      return {
        ok: models.length > 0,
        provider: prov,
        base_url: baseUrl,
        source: models.length > 0 ? 'live' : 'preset',
        models,
        count: models.length,
        error: models.length > 0 ? null : `Endpoint returned 0 chat models at ${baseUrl}/models.`,
      };
    }

    // Ollama native /api/tags fallback
    if (prov === 'ollama' && resp.status === 404) {
      const rootUrl = baseUrl.endsWith('/v1') ? baseUrl.slice(0, -3) : baseUrl;
      const tagsResp = await fetchFn(`${rootUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(8_000),
      });
      if (tagsResp.ok) {
        const tagsPayload = (await tagsResp.json()) as unknown;
        const models = parseModelsPayload(tagsPayload, prov);
        return {
          ok: models.length > 0,
          provider: prov,
          base_url: baseUrl,
          source: models.length > 0 ? 'live' : 'preset',
          models,
          count: models.length,
          error: models.length > 0 ? null : 'Ollama is reachable, but no local models are pulled yet.',
        };
      }
    }

    const bodyText = await resp.text().catch(() => '');
    const expectedVars = defaults?.envVars.join(' / ') || 'API key';
    const errorMsg =
      (resp.status === 401 || resp.status === 403) && !apiKey && defaults?.requiresKey !== false
        ? `Add your ${expectedVars} to query live ${prov} models.`
        : extractErrorDetail(resp.status, bodyText, prov);

    return {
      ok: false,
      provider: prov,
      base_url: baseUrl,
      source: 'error',
      models: [],
      count: 0,
      error: errorMsg,
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      provider: prov,
      base_url: baseUrl,
      source: 'error',
      models: [],
      count: 0,
      error: `Could not reach ${baseUrl}/models: ${msg}`,
    };
  }
}

export async function probeProviderModelDirect(params: {
  role: 'primary' | 'worker';
  override?: Partial<RoleLLMConfig> & { api_key?: string };
  persistedRoleConfig: RoleLLMConfig;
  apiKey?: string;
  env?: Record<string, string | undefined>;
  fetchImpl?: typeof fetch;
}): Promise<ProviderProbeResult> {
  const role = params.role === 'worker' ? 'worker' : 'primary';
  const ov = params.override || {};
  const prov = normalizeProviderId(ov.provider || params.persistedRoleConfig.provider);
  const defaults = PROVIDER_CATALOG_DEFAULTS[prov];
  const baseUrl = (
    (typeof ov.base_url === 'string' && ov.base_url.trim()) ||
    defaults?.baseUrl ||
    params.persistedRoleConfig.base_url ||
    ''
  )
    .trim()
    .replace(/\/+$/, '');
  const model = ((typeof ov.model === 'string' && ov.model.trim()) || params.persistedRoleConfig.model || '').trim();
  const explicitKey = (params.apiKey || ov.api_key || '').trim();
  const apiKey = resolveProviderApiKey(prov, role, params.env || process.env, explicitKey);
  const fetchFn = params.fetchImpl || fetch;

  const baseResult: ProviderProbeResult = {
    ok: false,
    role,
    provider: prov,
    model,
    base_url: baseUrl,
    reachable: false,
    authenticated: false,
    model_listed: null,
    status_code: null,
    error: null,
  };

  if (!baseUrl || !/^https?:\/\//i.test(baseUrl)) {
    return { ...baseResult, error: `A valid http(s) Base URL is required for '${prov}'.` };
  }
  if (!model) {
    return { ...baseResult, error: 'Model ID must not be empty.' };
  }

  const requiresKey = defaults ? defaults.requiresKey : true;
  if (requiresKey && !apiKey) {
    const envVar =
      prov === 'custom'
        ? role === 'worker'
          ? 'WORKER_API_KEY'
          : 'PRIMARY_API_KEY'
        : defaults?.envVars[0] || 'API key';
    return {
      ...baseResult,
      error: `Missing API key for '${prov}'. Paste ${envVar} above (or save it to .env) before testing.`,
    };
  }

  const headers = buildProviderHeaders(prov, apiKey);

  try {
    // 1. Query /models to check reachability and whether the selected model is listed
    const modelsResp = await fetchFn(`${baseUrl}/models`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(12_000),
    });

    baseResult.reachable = true;
    baseResult.status_code = modelsResp.status;

    if (modelsResp.status === 401 || modelsResp.status === 403) {
      const bodyText = await modelsResp.text().catch(() => '');
      return {
        ...baseResult,
        authenticated: false,
        error: extractErrorDetail(modelsResp.status, bodyText, prov),
      };
    }

    if (modelsResp.ok) {
      const payload = (await modelsResp.json().catch(() => null)) as unknown;
      const catalog = parseModelsPayload(payload, prov);
      if (catalog.length > 0) {
        baseResult.model_listed = catalog.some((m) => m.id === model);
      }
    }

    // 2. Providers whose /models endpoint is public (OpenRouter, NVIDIA NIM): verify actual key auth!
    if (prov === 'openrouter' && apiKey) {
      const authResp = await fetchFn(`${baseUrl}/auth/key`, {
        method: 'GET',
        headers,
        signal: AbortSignal.timeout(10_000),
      });
      baseResult.status_code = authResp.status;
      if (!authResp.ok) {
        const bodyText = await authResp.text().catch(() => '');
        return {
          ...baseResult,
          authenticated: false,
          error: extractErrorDetail(authResp.status, bodyText, prov),
        };
      }
      return {
        ...baseResult,
        ok: true,
        authenticated: true,
        error: null,
      };
    }

    if (prov === 'nvidia' && apiKey) {
      const authResp = await fetchFn(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: 'ping' }],
          max_tokens: 1,
          temperature: 0,
        }),
        signal: AbortSignal.timeout(12_000),
      });
      baseResult.status_code = authResp.status;
      if (authResp.status === 401 || authResp.status === 403) {
        const bodyText = await authResp.text().catch(() => '');
        return {
          ...baseResult,
          authenticated: false,
          error: extractErrorDetail(authResp.status, bodyText, prov),
        };
      }
      if (authResp.status === 404) {
        return {
          ...baseResult,
          authenticated: true,
          model_listed: false,
          error: `NVIDIA NIM key accepted, but model '${model}' was not found (HTTP 404). Pick a model from the dropdown.`,
        };
      }
      if (authResp.status >= 400 && authResp.status !== 429) {
        const bodyText = await authResp.text().catch(() => '');
        return {
          ...baseResult,
          authenticated: false,
          error: extractErrorDetail(authResp.status, bodyText, prov),
        };
      }
      return {
        ...baseResult,
        ok: true,
        authenticated: true,
        error: null,
      };
    }

    if (!modelsResp.ok) {
      const bodyText = await modelsResp.text().catch(() => '');
      return {
        ...baseResult,
        authenticated: false,
        error: extractErrorDetail(modelsResp.status, bodyText, prov),
      };
    }

    return {
      ...baseResult,
      ok: true,
      authenticated: true,
      error: null,
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      ...baseResult,
      reachable: false,
      error: `Cannot reach ${baseUrl}: ${msg}`,
    };
  }
}
