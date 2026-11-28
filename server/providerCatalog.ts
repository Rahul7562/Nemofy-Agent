import dns from 'node:dns/promises';
import net from 'node:net';
import {
  deriveCustomProviderSecretKey,
  isCustomProviderId,
  normalizeProviderId,
  type CustomProviderConfig,
  type CustomProviderProtocol,
  type RoleLLMConfig,
} from './configStore.ts';
import { resolveModelLimits } from '../Nemofy Agent/src/lib/contextUtils.ts';

function isBlockedIpv4(ip: string): boolean {
  const parts = ip.split('.').map((p) => Number(p));
  if (parts.length !== 4 || parts.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) {
    return true;
  }
  const [a, b] = parts;
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a >= 224) return true;
  return false;
}

export function isBlockedIpAddress(rawIp: string): boolean {
  const ip = rawIp.trim().toLowerCase().replace(/^\[|\]$/g, '');
  if (!ip) return true;
  if (net.isIPv4(ip)) return isBlockedIpv4(ip);
  if (net.isIPv6(ip)) {
    if (ip === '::' || ip === '::1') return true;
    if (ip.startsWith('::ffff:')) {
      const mapped = ip.slice('::ffff:'.length);
      if (net.isIPv4(mapped)) return isBlockedIpv4(mapped);
    }
    const firstHextet = parseInt(ip.split(':')[0] || '0', 16);
    if ((firstHextet & 0xfe00) === 0xfc00) return true; // fc00::/7 unique local
    if ((firstHextet & 0xffc0) === 0xfe80) return true; // fe80::/10 link-local
    if ((firstHextet & 0xff00) === 0xff00) return true; // ff00::/8 multicast
    return false;
  }
  return false;
}

function isLoopbackHost(hostname: string): boolean {
  const lower = hostname.trim().toLowerCase().replace(/^\[|\]$/g, '');
  return lower === 'localhost' || lower === '127.0.0.1' || lower === '::1';
}

export async function validateOutboundProviderUrl(
  rawUrl: string,
  options: { allowLoopback?: boolean; resolveDns?: boolean } = {},
): Promise<{ ok: true; url: URL } | { ok: false; error: string }> {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return { ok: false, error: 'A valid http(s) Base URL is required.' };
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    return { ok: false, error: 'Only http:// and https:// URLs are allowed.' };
  }
  if (parsed.username || parsed.password) {
    return { ok: false, error: 'Embedded credentials in Base URL are not permitted.' };
  }

  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');
  if (!host) {
    return { ok: false, error: 'Base URL hostname is missing.' };
  }

  if (options.allowLoopback && isLoopbackHost(host)) {
    return { ok: true, url: parsed };
  }

  if (
    host === 'localhost' ||
    host.endsWith('.localhost') ||
    host.endsWith('.local') ||
    host.endsWith('.internal') ||
    host === 'metadata.google.internal'
  ) {
    return { ok: false, error: `Blocked restricted or local host '${host}'.` };
  }

  if (net.isIP(host)) {
    if (isBlockedIpAddress(host)) {
      return { ok: false, error: `Blocked private, loopback, or metadata IP '${host}'.` };
    }
    return { ok: true, url: parsed };
  }

  if (options.resolveDns !== false) {
    try {
      const records = await dns.lookup(host, { all: true });
      if (records.length === 0) {
        return { ok: false, error: `Unable to resolve host '${host}'.` };
      }
      for (const rec of records) {
        if (isBlockedIpAddress(rec.address)) {
          return {
            ok: false,
            error: `Blocked host '${host}' because it resolves to a private, loopback, or metadata address.`,
          };
        }
      }
    } catch {
      return { ok: false, error: `Unable to resolve host '${host}'.` };
    }
  }

  return { ok: true, url: parsed };
}

function isStoredKeyAllowedForBaseUrl(
  provider: string,
  baseUrl: string,
  hasExplicitKey: boolean,
): boolean {
  if (hasExplicitKey) return true;
  if (isCustomProviderId(provider)) return true;
  const defaults = PROVIDER_CATALOG_DEFAULTS[provider];
  if (!defaults || !defaults.baseUrl) return true;
  try {
    return new URL(baseUrl).origin === new URL(defaults.baseUrl).origin;
  } catch {
    return false;
  }
}

export interface ProviderModelCatalogItem {
  id: string;
  name?: string;
  owned_by?: string;
  context_length?: number | null;
  max_output_tokens?: number | null;
  is_free?: boolean;
  input_modalities?: string[];
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
  if (prov.startsWith('custom_')) {
    const customEnvVar = deriveCustomProviderSecretKey(prov);
    const customVal = (env[customEnvVar] || '').trim();
    if (customVal) return customVal;
    if (prov === 'custom_legacy') {
      const legacyVal = (env.PRIMARY_API_KEY || env.WORKER_API_KEY || '').trim();
      if (legacyVal) return legacyVal;
    }
    return null;
  }

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

export interface ReferenceModelMetadata {
  context_length?: number;
  max_output_tokens?: number;
  input_modalities?: string[];
}

export interface ReferenceModelIndex {
  exact: Map<string, ReferenceModelMetadata>;
  normalizedOrgSlug: Map<string, ReferenceModelMetadata>;
  bareSlug: Map<string, ReferenceModelMetadata>;
  strippedSlug: Map<string, ReferenceModelMetadata>;
}

const ORG_PREFIX_ALIASES: Record<string, string> = {
  'minimaxai': 'minimax',
  'deepseek-ai': 'deepseek',
  'meta': 'meta-llama',
  'zhipuai': 'z-ai',
  'zai': 'z-ai',
  'xai': 'x-ai',
  'mistral': 'mistralai',
  'moonshot': 'moonshotai',
  'nvidia-nim': 'nvidia',
  'google-deepmind': 'google',
};

const MODEL_SLUG_ALIASES: Record<string, string[]> = {
  'deepseek-chat': ['deepseek/deepseek-chat', 'deepseek/deepseek-v3'],
  'deepseek-reasoner': ['deepseek/deepseek-r1', 'deepseek/deepseek-reasoner'],
  'mistral-large-4': ['mistralai/mistral-large-4-0'],
  'gemini-2.5-pro-latest': ['google/gemini-2.5-pro'],
  'gemini-2.5-flash-latest': ['google/gemini-2.5-flash'],
  'gpt-4o-latest': ['openai/gpt-4o'],
  'claude-sonnet-4-5': ['anthropic/claude-sonnet-4.5'],
  'claude-opus-4-6': ['anthropic/claude-opus-4.6'],
};

const REFERENCE_CATALOG_TTL_MS = 6 * 60 * 60 * 1000;
const LIVE_CATALOG_TTL_MS = 10 * 60 * 1000;
const OPENROUTER_PUBLIC_MODELS_URL = 'https://openrouter.ai/api/v1/models';

let cachedReferenceIndex: { index: ReferenceModelIndex; fetchedAt: number } | null = null;
let inFlightReferencePromise: Promise<ReferenceModelIndex | null> | null = null;

const liveCatalogCache = new Map<string, { models: ProviderModelCatalogItem[]; fetchedAt: number }>();

function liveCatalogCacheKey(provider: string, baseUrl: string): string {
  return `${normalizeProviderId(provider)}::${(baseUrl || '').trim().replace(/\/+$/, '').toLowerCase()}`;
}

export function cacheProviderModels(
  provider: string,
  baseUrl: string,
  models: ProviderModelCatalogItem[],
): void {
  if (!Array.isArray(models) || models.length === 0) return;
  liveCatalogCache.set(liveCatalogCacheKey(provider, baseUrl), {
    models,
    fetchedAt: Date.now(),
  });
}

export function getCachedProviderModels(
  provider: string,
  baseUrl?: string,
): ProviderModelCatalogItem[] | null {
  const prov = normalizeProviderId(provider);
  const cleanBase = (baseUrl || PROVIDER_CATALOG_DEFAULTS[prov]?.baseUrl || '').trim().replace(/\/+$/, '');
  const exact = liveCatalogCache.get(liveCatalogCacheKey(prov, cleanBase));
  if (exact && Date.now() - exact.fetchedAt < LIVE_CATALOG_TTL_MS) {
    return exact.models;
  }
  // Fallback to any fresh cached catalog for the same provider when baseUrl was omitted
  if (!baseUrl) {
    for (const [k, entry] of liveCatalogCache.entries()) {
      if (k.startsWith(`${prov}::`) && Date.now() - entry.fetchedAt < LIVE_CATALOG_TTL_MS) {
        return entry.models;
      }
    }
  }
  return null;
}

export function clearLiveCatalogCache(): void {
  liveCatalogCache.clear();
}

function normalizeVersionSeparators(input: string): string {
  return input
    .replace(/v(\d+)p(\d+)/g, '$1.$2')
    .replace(/(\d)-(\d)(?=$|[-_/:])/g, '$1.$2');
}

function normalizeProviderOrg(org: string): string {
  const clean = org.trim().toLowerCase();
  return ORG_PREFIX_ALIASES[clean] || clean;
}

function stripDeploymentSuffixes(bareSlug: string): string {
  let s = bareSlug.toLowerCase();
  // Strip repeated deployment / quantization / parameter / date suffixes
  const patterns: RegExp[] = [
    /-(?:fp8|fp16|bf16|int4|int8|awq|gptq|gguf|nvfp4)$/,
    /-(?:turbo|versatile|instant|preview|latest|exp|experimental|beta|alpha)$/,
    /-(?:instruct|it|chat)$/,
    /-20\d{6}$/,
    /-(?:0[1-9]|1[0-2])\d{2}$/,
    /-\d+(?:\.\d+)?[tb]-a\d+(?:\.\d+)?b$/,
    /-\d+(?:\.\d+)?b$/,
  ];
  let changed = true;
  while (changed) {
    changed = false;
    for (const pat of patterns) {
      const next = s.replace(pat, '');
      if (next !== s && next.length >= 4) {
        s = next;
        changed = true;
      }
    }
  }
  return s;
}

function normalizeOrgAndBare(rawModelId: string): {
  exactLower: string;
  tag: string;
  org: string | null;
  bare: string;
  normalizedBare: string;
  strippedBare: string;
} {
  let exactLower = rawModelId.trim().toLowerCase();
  exactLower = exactLower
    .replace(/^accounts\/[^/]+\/models\//, '')
    .replace(/^publishers\/[^/]+\/models\//, '')
    .replace(/^models\//, '');

  let baseId = exactLower;
  let tag = '';
  const colonIdx = exactLower.indexOf(':');
  if (colonIdx > 0) {
    baseId = exactLower.slice(0, colonIdx);
    tag = exactLower.slice(colonIdx + 1);
  }

  const slashIdx = baseId.indexOf('/');
  const org = slashIdx > 0 ? normalizeProviderOrg(baseId.slice(0, slashIdx)) : null;
  const bare = slashIdx > 0 ? baseId.slice(slashIdx + 1) : baseId;
  const normalizedBare = normalizeVersionSeparators(bare);
  const strippedBare = stripDeploymentSuffixes(normalizedBare);

  return { exactLower, tag, org, bare, normalizedBare, strippedBare };
}

export function buildReferenceModelIndex(models: ProviderModelCatalogItem[]): ReferenceModelIndex {
  const exact = new Map<string, ReferenceModelMetadata>();
  const normalizedOrgSlug = new Map<string, ReferenceModelMetadata>();
  const bareSlug = new Map<string, ReferenceModelMetadata>();
  const strippedSlug = new Map<string, ReferenceModelMetadata>();

  // Pass 1: Non-batch, non-free canonical models take highest priority
  // Pass 2: :free models fill gaps if no paid canonical model existed
  const sorted = [...models].sort((a, b) => {
    const aLower = a.id.toLowerCase();
    const bLower = b.id.toLowerCase();
    const aRank = aLower.includes(':batch') ? 3 : aLower.endsWith(':free') ? 2 : 1;
    const bRank = bLower.includes(':batch') ? 3 : bLower.endsWith(':free') ? 2 : 1;
    return aRank - bRank;
  });

  for (const item of sorted) {
    const hasCtx = typeof item.context_length === 'number' && item.context_length > 0;
    const hasOut = typeof item.max_output_tokens === 'number' && item.max_output_tokens > 0;
    if (!hasCtx && !hasOut) continue;

    const meta: ReferenceModelMetadata = {
      ...(hasCtx ? { context_length: Math.round(item.context_length!) } : {}),
      ...(hasOut ? { max_output_tokens: Math.round(item.max_output_tokens!) } : {}),
      ...(item.input_modalities?.length ? { input_modalities: item.input_modalities } : {}),
    };

    const parts = normalizeOrgAndBare(item.id);
    if (!exact.has(parts.exactLower)) {
      exact.set(parts.exactLower, meta);
    }

    // Never let :batch variants overwrite canonical non-batch lookups
    if (parts.tag === 'batch') continue;

    if (parts.org) {
      const orgKey = `${parts.org}/${parts.normalizedBare}`;
      if (!normalizedOrgSlug.has(orgKey)) {
        normalizedOrgSlug.set(orgKey, meta);
      }
      const orgStrippedKey = `${parts.org}/${parts.strippedBare}`;
      if (!normalizedOrgSlug.has(orgStrippedKey)) {
        normalizedOrgSlug.set(orgStrippedKey, meta);
      }
    }

    if (!bareSlug.has(parts.bare)) {
      bareSlug.set(parts.bare, meta);
    }
    if (!bareSlug.has(parts.normalizedBare)) {
      bareSlug.set(parts.normalizedBare, meta);
    }
    if (parts.strippedBare.length >= 4 && !strippedSlug.has(parts.strippedBare)) {
      strippedSlug.set(parts.strippedBare, meta);
    }
  }

  return { exact, normalizedOrgSlug, bareSlug, strippedSlug };
}

export function lookupReferenceModelMetadata(
  modelId: string,
  index: ReferenceModelIndex | null | undefined,
): ReferenceModelMetadata | null {
  if (!index || !modelId || !modelId.trim()) return null;
  const parts = normalizeOrgAndBare(modelId);

  // 1. Exact ID match (preserves :free vs paid distinction when exact match exists)
  const exactHit = index.exact.get(parts.exactLower);
  if (exactHit) return exactHit;

  // 2. Explicit known cross-provider aliases
  for (const candidateKey of [parts.bare, parts.normalizedBare, parts.strippedBare]) {
    const aliases = MODEL_SLUG_ALIASES[candidateKey];
    if (aliases) {
      for (const alias of aliases) {
        const aliasHit = index.exact.get(alias) || index.normalizedOrgSlug.get(alias);
        if (aliasHit) return aliasHit;
      }
    }
  }

  // 3. Normalized org + slug match (e.g. minimaxai/minimax-m3.1-flash -> minimax/minimax-m3.1-flash)
  if (parts.org) {
    const orgExact = index.normalizedOrgSlug.get(`${parts.org}/${parts.normalizedBare}`);
    if (orgExact) return orgExact;
    const orgStripped = index.normalizedOrgSlug.get(`${parts.org}/${parts.strippedBare}`);
    if (orgStripped) return orgStripped;
  }

  // 4. Bare slug match (e.g. claude-opus-4-6 -> anthropic/claude-opus-4.6, GLM-5.3 -> z-ai/glm-5.3)
  const bareHit = index.bareSlug.get(parts.bare) || index.bareSlug.get(parts.normalizedBare);
  if (bareHit) return bareHit;

  // 5. Stripped suffix match (e.g. Qwen/Qwen3.8-27B-FP8 -> qwen/qwen3.8-27b, mistral-large-4-1t-a52b -> mistral-large-4)
  if (parts.strippedBare.length >= 4) {
    const strippedHit =
      index.bareSlug.get(parts.strippedBare) ||
      index.strippedSlug.get(parts.strippedBare);
    if (strippedHit) return strippedHit;

    // Also try prefix match on stripped slug for versioned variants like mistral-large-4 -> mistral-large-4-0
    for (const [key, val] of index.bareSlug.entries()) {
      if (key.startsWith(`${parts.strippedBare}-`) || key.startsWith(`${parts.strippedBare}.`)) {
        return val;
      }
    }
  }

  return null;
}

export function seedReferenceModelCatalog(models: ProviderModelCatalogItem[]): void {
  if (!Array.isArray(models) || models.length === 0) return;
  cachedReferenceIndex = {
    index: buildReferenceModelIndex(models),
    fetchedAt: Date.now(),
  };
}

export function clearReferenceModelCatalogCache(): void {
  cachedReferenceIndex = null;
  inFlightReferencePromise = null;
}

export async function getReferenceModelCatalog(
  fetchFn: typeof fetch = fetch,
  timeoutMs = 5_000,
): Promise<ReferenceModelIndex | null> {
  if (cachedReferenceIndex && Date.now() - cachedReferenceIndex.fetchedAt < REFERENCE_CATALOG_TTL_MS) {
    return cachedReferenceIndex.index;
  }
  if (inFlightReferencePromise) {
    return inFlightReferencePromise;
  }

  inFlightReferencePromise = (async () => {
    try {
      const resp = await fetchFn(OPENROUTER_PUBLIC_MODELS_URL, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
          'HTTP-Referer': 'https://nemofy.tech',
          'X-Title': 'Nemofy Agent',
        },
        redirect: 'error',
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!resp.ok) {
        return cachedReferenceIndex?.index || null;
      }
      const payload = (await resp.json()) as unknown;
      const models = parseModelsPayload(payload, 'openrouter');
      if (models.length === 0) {
        return cachedReferenceIndex?.index || null;
      }
      const index = buildReferenceModelIndex(models);
      cachedReferenceIndex = { index, fetchedAt: Date.now() };
      return index;
    } catch {
      return cachedReferenceIndex?.index || null;
    } finally {
      inFlightReferencePromise = null;
    }
  })();

  return inFlightReferencePromise;
}

export function enrichModelsWithReferenceMetadata(
  models: ProviderModelCatalogItem[],
  provider: string,
  referenceIndex: ReferenceModelIndex | null,
): ProviderModelCatalogItem[] {
  const prov = normalizeProviderId(provider);
  const skipCloudHeuristics = isCustomProviderId(prov) || prov === 'ollama' || prov === 'vllm';
  return models.map((item) => {
    let contextLength =
      typeof item.context_length === 'number' && item.context_length > 0
        ? Math.round(item.context_length)
        : null;
    let maxOutputTokens =
      typeof item.max_output_tokens === 'number' && item.max_output_tokens > 0
        ? Math.round(item.max_output_tokens)
        : null;
    let inputModalities = item.input_modalities;

    if (!skipCloudHeuristics) {
      // 1. Check provider-specific registry first (handles same model differing across providers)
      if (!contextLength || !maxOutputTokens) {
        const provResolved = resolveModelLimits(prov, item.id, contextLength, maxOutputTokens);
        if (!contextLength && provResolved.contextWindowSource === 'provider_registry' && provResolved.contextWindow) {
          contextLength = provResolved.contextWindow;
        }
        if (!maxOutputTokens && provResolved.maxOutputTokensSource === 'provider_registry' && provResolved.maxOutputTokens) {
          maxOutputTokens = provResolved.maxOutputTokens;
        }
      }

      // 2. Fill missing values from live reference catalog (never overwriting provider-reported values)
      if (!contextLength || !maxOutputTokens || !inputModalities?.length) {
        const refMeta = lookupReferenceModelMetadata(item.id, referenceIndex);
        if (refMeta) {
          if (!contextLength && refMeta.context_length) {
            contextLength = refMeta.context_length;
          }
          if (!maxOutputTokens && refMeta.max_output_tokens) {
            maxOutputTokens = refMeta.max_output_tokens;
          }
          if (!inputModalities?.length && refMeta.input_modalities?.length) {
            inputModalities = refMeta.input_modalities;
          }
        }
      }

      // 3. Fill remaining gaps from built-in model registry ONLY when matched to a known model family
      if (!contextLength || !maxOutputTokens) {
        const resolved = resolveModelLimits(prov, item.id, contextLength, maxOutputTokens);
        if (
          !contextLength &&
          (resolved.contextWindowSource === 'provider_registry' || resolved.contextWindowSource === 'model_registry') &&
          resolved.contextWindow
        ) {
          contextLength = resolved.contextWindow;
        }
        if (
          !maxOutputTokens &&
          (resolved.maxOutputTokensSource === 'provider_registry' || resolved.maxOutputTokensSource === 'model_registry') &&
          resolved.maxOutputTokens
        ) {
          maxOutputTokens = resolved.maxOutputTokens;
        }
      }
    }

    if (contextLength && maxOutputTokens && maxOutputTokens > contextLength) {
      maxOutputTokens = contextLength;
    }

    return {
      ...item,
      ...(contextLength ? { context_length: contextLength } : {}),
      ...(maxOutputTokens ? { max_output_tokens: maxOutputTokens } : {}),
      ...(inputModalities?.length ? { input_modalities: inputModalities } : {}),
    };
  });
}

export type ModelMetadataSource =
  | 'manual'
  | 'custom_provider_config'
  | 'provider_catalog'
  | 'provider_registry'
  | 'reference_catalog'
  | 'model_registry'
  | 'unknown';

export interface UnifiedModelResolutionResult {
  resolved: boolean;
  provider: string;
  model: string;
  matched_model: string | null;
  context_window: number | null;
  max_output_tokens: number | null;
  context_window_source: ModelMetadataSource;
  max_output_tokens_source: ModelMetadataSource;
}

function findMatchingCatalogItem(
  modelId: string,
  catalog: ProviderModelCatalogItem[],
): ProviderModelCatalogItem | null {
  if (!modelId.trim() || catalog.length === 0) return null;
  const queryParts = normalizeOrgAndBare(modelId);
  // 1. Exact case-insensitive match
  for (const item of catalog) {
    const itemParts = normalizeOrgAndBare(item.id);
    if (itemParts.exactLower === queryParts.exactLower) return item;
  }
  // 2. Normalized org + slug match
  if (queryParts.org) {
    const targetOrgKey = `${queryParts.org}/${queryParts.normalizedBare}`;
    for (const item of catalog) {
      const itemParts = normalizeOrgAndBare(item.id);
      if (itemParts.org && `${itemParts.org}/${itemParts.normalizedBare}` === targetOrgKey) {
        return item;
      }
    }
  }
  // 3. Bare slug match
  for (const item of catalog) {
    const itemParts = normalizeOrgAndBare(item.id);
    if (
      itemParts.bare === queryParts.bare ||
      itemParts.normalizedBare === queryParts.normalizedBare
    ) {
      return item;
    }
  }
  // 4. Stripped suffix match
  if (queryParts.strippedBare.length >= 4) {
    for (const item of catalog) {
      const itemParts = normalizeOrgAndBare(item.id);
      if (itemParts.strippedBare === queryParts.strippedBare) {
        return item;
      }
    }
  }
  return null;
}

/**
 * Unified provider-aware model metadata resolver used for BOTH manually entered model IDs and dropdown selections.
 * Resolution priority:
 *  1. Manual context-window override ('manual')
 *  2. Custom provider per-model configuration ('custom_provider_config')
 *  3. Explicit catalog values or live provider catalog ('provider_catalog')
 *  4. Provider-specific verified registry ('provider_registry') for built-in cloud providers
 *  5. OpenRouter reference metadata catalog ('reference_catalog') for built-in cloud providers
 *  6. Built-in verified model registry ('model_registry') for built-in cloud providers
 *  7. Explicit 'unknown' state (context_window: null) — NEVER stamps a fabricated default or guesses for local/custom endpoints.
 */
export async function resolveModelMetadataUnified(params: {
  provider: string;
  model: string;
  base_url?: string;
  api_key?: string;
  protocol?: CustomProviderProtocol;
  customProvider?: CustomProviderConfig | null;
  role?: 'primary' | 'worker';
  env?: Record<string, string | undefined>;
  manualContextWindow?: number | null;
  explicitContextWindow?: number | null;
  explicitMaxOutputTokens?: number | null;
  liveModels?: ProviderModelCatalogItem[];
  referenceIndex?: ReferenceModelIndex | null;
  allowLiveFetch?: boolean;
  allowReferenceFetch?: boolean;
  forceRefresh?: boolean;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}): Promise<UnifiedModelResolutionResult> {
  const prov = normalizeProviderId(params.provider);
  const cleanModel = (params.model || '').trim();
  const isUnverifiedCustomOrLocal = isCustomProviderId(prov) || prov === 'ollama' || prov === 'vllm';
  const effectiveBaseUrl = (params.base_url || params.customProvider?.base_url || '').trim().replace(/\/+$/, '');

  if (!cleanModel) {
    return {
      resolved: false,
      provider: prov,
      model: '',
      matched_model: null,
      context_window: null,
      max_output_tokens: null,
      context_window_source: 'unknown',
      max_output_tokens_source: 'unknown',
    };
  }

  let matchedModel: string | null = null;
  let contextWindow: number | null = null;
  let contextWindowSource: ModelMetadataSource = 'unknown';

  // 0. Manual override always takes highest precedence
  if (
    typeof params.manualContextWindow === 'number' &&
    Number.isFinite(params.manualContextWindow) &&
    params.manualContextWindow >= 512
  ) {
    contextWindow = Math.round(params.manualContextWindow);
    contextWindowSource = 'manual';
  }

  let maxOutputTokens: number | null =
    typeof params.explicitMaxOutputTokens === 'number' &&
    Number.isFinite(params.explicitMaxOutputTokens) &&
    params.explicitMaxOutputTokens >= 256
      ? Math.round(params.explicitMaxOutputTokens)
      : null;
  let maxOutputTokensSource: ModelMetadataSource = maxOutputTokens ? 'provider_catalog' : 'unknown';

  // 0.5. Custom provider per-model configuration metadata
  if (params.customProvider && Array.isArray(params.customProvider.models)) {
    const customMatch = params.customProvider.models.find(
      (m) => m.id.trim().toLowerCase() === cleanModel.toLowerCase(),
    );
    if (customMatch) {
      matchedModel = customMatch.id;
      if (!contextWindow && typeof customMatch.context_window === 'number' && customMatch.context_window >= 512) {
        contextWindow = Math.round(customMatch.context_window);
        contextWindowSource = 'custom_provider_config';
      }
      if (!maxOutputTokens && typeof customMatch.max_output_tokens === 'number' && customMatch.max_output_tokens >= 256) {
        maxOutputTokens = Math.round(customMatch.max_output_tokens);
        maxOutputTokensSource = 'custom_provider_config';
      }
    }
  }

  if (
    !contextWindow &&
    typeof params.explicitContextWindow === 'number' &&
    Number.isFinite(params.explicitContextWindow) &&
    params.explicitContextWindow >= 512
  ) {
    contextWindow = Math.round(params.explicitContextWindow);
    contextWindowSource = 'provider_catalog';
  }

  const applyCatalogMatch = (models: ProviderModelCatalogItem[] | null | undefined): void => {
    if (!models || models.length === 0) return;
    const hit = findMatchingCatalogItem(cleanModel, models);
    if (!hit) return;
    matchedModel = matchedModel || hit.id;
    if (!contextWindow && typeof hit.context_length === 'number' && hit.context_length >= 512) {
      contextWindow = Math.round(hit.context_length);
      contextWindowSource = 'provider_catalog';
    }
    if (!maxOutputTokens && typeof hit.max_output_tokens === 'number' && hit.max_output_tokens >= 256) {
      maxOutputTokens = Math.round(hit.max_output_tokens);
      maxOutputTokensSource = 'provider_catalog';
    }
  };

  // 1. Check explicitly provided liveModels or cached live catalog for this provider (skip cache on forceRefresh)
  if (!contextWindow || !maxOutputTokens) {
    applyCatalogMatch(params.liveModels);
  }
  if ((!contextWindow || !maxOutputTokens) && !params.forceRefresh) {
    applyCatalogMatch(getCachedProviderModels(prov, effectiveBaseUrl || undefined));
  }

  // 2. If still missing context window and live fetch is permitted, query the provider's live catalog
  if ((!contextWindow || !maxOutputTokens) && params.allowLiveFetch) {
    const defaults = PROVIDER_CATALOG_DEFAULTS[prov];
    const resolvedBase = (effectiveBaseUrl || defaults?.baseUrl || '').trim().replace(/\/+$/, '');
    const apiKey = resolveProviderApiKey(
      prov,
      params.role === 'worker' ? 'worker' : 'primary',
      params.env || process.env,
      params.api_key,
    );
    const canQueryWithoutKey =
      defaults?.requiresKey === false || prov === 'openrouter' || isCustomProviderId(prov);
    if (resolvedBase && (apiKey || canQueryWithoutKey)) {
      const liveResult = await listProviderModelsDirect({
        provider: prov,
        base_url: resolvedBase,
        api_key: params.api_key,
        protocol: params.protocol || params.customProvider?.protocol,
        customProvider: params.customProvider,
        role: params.role === 'worker' ? 'worker' : 'primary',
        env: params.env,
        fetchImpl: params.fetchImpl,
        referenceIndex: params.referenceIndex,
        enableReferenceEnrichment: params.allowReferenceFetch,
        forceRefresh: params.forceRefresh,
      });
      if (liveResult.ok && liveResult.models.length > 0) {
        applyCatalogMatch(liveResult.models);
      }
    }
  }

  if (!isUnverifiedCustomOrLocal) {
    // 3. Provider-specific verified registry first (handles same model name having different limits per provider)
    if (!contextWindow || !maxOutputTokens) {
      const provReg = resolveModelLimits(prov, cleanModel, contextWindow, maxOutputTokens);
      if (!contextWindow && provReg.contextWindowSource === 'provider_registry' && provReg.contextWindow) {
        contextWindow = provReg.contextWindow;
        contextWindowSource = 'provider_registry';
      }
      if (!maxOutputTokens && provReg.maxOutputTokensSource === 'provider_registry' && provReg.maxOutputTokens) {
        maxOutputTokens = provReg.maxOutputTokens;
        maxOutputTokensSource = 'provider_registry';
      }
    }

    // 4. Reference catalog lookup (supports cross-provider aliases & normalized slugs)
    if (!contextWindow || !maxOutputTokens) {
      let refIndex: ReferenceModelIndex | null = null;
      if (params.referenceIndex !== undefined) {
        refIndex = params.referenceIndex;
      } else if (params.allowReferenceFetch !== false) {
        if (params.forceRefresh) {
          clearReferenceModelCatalogCache();
        }
        refIndex = await getReferenceModelCatalog(params.fetchImpl || fetch, params.timeoutMs ?? 2_500);
      }
      const refMeta = lookupReferenceModelMetadata(cleanModel, refIndex);
      if (refMeta) {
        if (!contextWindow && refMeta.context_length && refMeta.context_length >= 512) {
          contextWindow = Math.round(refMeta.context_length);
          contextWindowSource = 'reference_catalog';
        }
        if (!maxOutputTokens && refMeta.max_output_tokens && refMeta.max_output_tokens >= 256) {
          maxOutputTokens = Math.round(refMeta.max_output_tokens);
          maxOutputTokensSource = 'reference_catalog';
        }
      }
    }

    // 5. Built-in verified model registry (only accepts genuine matches, never 128K default)
    if (!contextWindow || !maxOutputTokens) {
      const reg = resolveModelLimits(prov, cleanModel, contextWindow, maxOutputTokens);
      if (
        !contextWindow &&
        (reg.contextWindowSource === 'provider_registry' || reg.contextWindowSource === 'model_registry') &&
        reg.contextWindow
      ) {
        contextWindow = reg.contextWindow;
        contextWindowSource = reg.contextWindowSource;
      }
      if (
        !maxOutputTokens &&
        (reg.maxOutputTokensSource === 'provider_registry' || reg.maxOutputTokensSource === 'model_registry') &&
        reg.maxOutputTokens
      ) {
        maxOutputTokens = reg.maxOutputTokens;
        maxOutputTokensSource = reg.maxOutputTokensSource;
      }
    }
  }

  if (contextWindow && maxOutputTokens && maxOutputTokens > contextWindow) {
    maxOutputTokens = contextWindow;
  }

  return {
    resolved: contextWindow !== null,
    provider: prov,
    model: cleanModel,
    matched_model: matchedModel,
    context_window: contextWindow,
    max_output_tokens: maxOutputTokens,
    context_window_source: contextWindowSource,
    max_output_tokens_source: maxOutputTokensSource,
  };
}

export async function resolveModelMetadataFromCatalog(params: {
  provider: string;
  model: string;
  base_url?: string;
  api_key?: string;
  env?: Record<string, string | undefined>;
  explicitContextWindow?: number | null;
  explicitMaxOutputTokens?: number | null;
  manualContextWindow?: number | null;
  customProvider?: CustomProviderConfig | null;
  forceRefresh?: boolean;
  referenceIndex?: ReferenceModelIndex | null;
  allowLiveFetch?: boolean;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}): Promise<{
  context_window?: number;
  max_output_tokens?: number;
  context_window_source?: string;
  max_output_tokens_source?: string;
}> {
  const unified = await resolveModelMetadataUnified({
    provider: params.provider,
    model: params.model,
    base_url: params.base_url,
    api_key: params.api_key,
    env: params.env,
    explicitContextWindow: params.explicitContextWindow,
    explicitMaxOutputTokens: params.explicitMaxOutputTokens,
    manualContextWindow: params.manualContextWindow,
    customProvider: params.customProvider,
    forceRefresh: params.forceRefresh,
    referenceIndex: params.referenceIndex,
    allowLiveFetch: params.allowLiveFetch ?? false,
    fetchImpl: params.fetchImpl,
    timeoutMs: params.timeoutMs,
  });

  return {
    ...(unified.context_window ? { context_window: unified.context_window } : {}),
    ...(unified.max_output_tokens ? { max_output_tokens: unified.max_output_tokens } : {}),
    ...(unified.context_window_source ? { context_window_source: unified.context_window_source } : {}),
    ...(unified.max_output_tokens_source ? { max_output_tokens_source: unified.max_output_tokens_source } : {}),
  };
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

    // Google native /v1beta/models includes supportedGenerationMethods: skip non-chat models (embeddings, etc.)
    if (prov === 'gemini' && Array.isArray(record.supportedGenerationMethods)) {
      const methods = record.supportedGenerationMethods.map((m) => String(m).trim());
      if (methods.length > 0 && !methods.includes('generateContent')) {
        continue;
      }
    }

    let rawId = String(record.id ?? record.name ?? record.model ?? '').trim();
    if (!rawId) continue;
    if (prov === 'gemini' && rawId.startsWith('models/')) {
      rawId = rawId.slice('models/'.length);
    }
    if (isNonChatModelId(rawId) || seen.has(rawId)) continue;
    seen.add(rawId);

    const displayName =
      typeof record.displayName === 'string' && record.displayName.trim()
        ? record.displayName.trim()
        : typeof record.display_name === 'string' && record.display_name.trim()
          ? record.display_name.trim()
          : typeof record.name === 'string' && record.name.trim() && record.name.trim() !== rawId && !record.name.trim().startsWith('models/')
            ? record.name.trim()
            : undefined;

    const ownedBy =
      typeof record.owned_by === 'string' && record.owned_by.trim()
        ? record.owned_by.trim()
        : undefined;

    let contextLength: number | null = null;
    const topProvider =
      record.top_provider && typeof record.top_provider === 'object'
        ? (record.top_provider as Record<string, unknown>)
        : null;
    const limits =
      record.limits && typeof record.limits === 'object'
        ? (record.limits as Record<string, unknown>)
        : null;
    for (const candidateRaw of [
      record.context_length,
      record.context_window,
      record.contextWindow,
      record.max_context_length,
      record.max_model_len,
      record.context_size,
      record.max_input_tokens,
      record.max_context_tokens,
      record.n_ctx,
      record.inputTokenLimit,
      record.input_token_limit,
      topProvider?.context_length,
      limits?.context,
      limits?.input,
      limits?.context_length,
      limits?.max_input_tokens,
    ]) {
      const candidate = Number(candidateRaw);
      if (Number.isFinite(candidate) && candidate > 0) {
        contextLength = Math.round(candidate);
        break;
      }
    }

    let maxOutputTokens: number | null = null;
    for (const candidateRaw of [
      record.max_output_tokens,
      record.maxOutputTokens,
      record.max_completion_tokens,
      record.maxCompletionTokens,
      record.outputTokenLimit,
      record.output_token_limit,
      topProvider?.max_completion_tokens,
      topProvider?.max_output_tokens,
      limits?.output,
      limits?.max_output_tokens,
      limits?.max_completion_tokens,
    ]) {
      const candidate = Number(candidateRaw);
      if (Number.isFinite(candidate) && candidate > 0) {
        maxOutputTokens = Math.round(candidate);
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

    let inputModalities: string[] | undefined;
    const arch = record.architecture as Record<string, unknown> | undefined;
    const rawMods = Array.isArray(arch?.input_modalities)
      ? arch.input_modalities
      : Array.isArray(record.input_modalities)
        ? record.input_modalities
        : null;
    if (rawMods) {
      const cleanedMods = rawMods
        .map((m) => String(m).trim().toLowerCase())
        .filter((m) => m === 'text' || m === 'image' || m === 'audio' || m === 'video');
      if (cleanedMods.length > 0) {
        inputModalities = Array.from(new Set(cleanedMods));
      }
    }

    models.push({
      id: rawId,
      ...(displayName ? { name: displayName } : {}),
      ...(ownedBy ? { owned_by: ownedBy } : {}),
      ...(contextLength ? { context_length: contextLength } : {}),
      ...(maxOutputTokens ? { max_output_tokens: maxOutputTokens } : {}),
      ...(inputModalities ? { input_modalities: inputModalities } : {}),
      is_free: isFree,
      source: 'live',
    });
  }

  models.sort((a, b) => a.id.toLowerCase().localeCompare(b.id.toLowerCase()));
  return models;
}

function buildProviderHeaders(
  provider: string,
  apiKey: string | null,
  protocol?: CustomProviderProtocol,
): Record<string, string> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  const prov = normalizeProviderId(provider);
  const useAnthropicHeaders = prov === 'anthropic' || protocol === 'anthropic_compatible';
  if (apiKey) {
    if (useAnthropicHeaders) {
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

async function fetchGeminiNativeModels(
  baseUrl: string,
  apiKey: string | null,
  fetchFn: typeof fetch,
): Promise<ProviderModelCatalogItem[] | null> {
  if (!apiKey) return null;
  let parsed: URL;
  try {
    parsed = new URL(baseUrl);
  } catch {
    return null;
  }
  if (parsed.hostname.toLowerCase() !== 'generativelanguage.googleapis.com') {
    return null;
  }
  try {
    const nativeUrl = `${parsed.origin}/v1beta/models?pageSize=200`;
    const resp = await fetchFn(nativeUrl, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
        'x-goog-api-key': apiKey,
      },
      redirect: 'error',
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return null;
    const payload = (await resp.json()) as unknown;
    const models = parseModelsPayload(payload, 'gemini');
    return models.length > 0 ? models : null;
  } catch {
    return null;
  }
}

async function enrichOllamaContextLengths(
  models: ProviderModelCatalogItem[],
  baseUrl: string,
  fetchFn: typeof fetch,
): Promise<ProviderModelCatalogItem[]> {
  const missing = models.filter((m) => !m.context_length).slice(0, 20);
  if (missing.length === 0) return models;

  const rootUrl = baseUrl.endsWith('/v1') ? baseUrl.slice(0, -3) : baseUrl;
  const ctxById = new Map<string, number>();

  await Promise.allSettled(
    missing.map(async (m) => {
      try {
        const resp = await fetchFn(`${rootUrl}/api/show`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ name: m.id }),
          redirect: 'error',
          signal: AbortSignal.timeout(2_500),
        });
        if (!resp.ok) return;
        const data = (await resp.json()) as Record<string, unknown>;
        const modelInfo = data.model_info as Record<string, unknown> | undefined;
        if (modelInfo && typeof modelInfo === 'object') {
          for (const [k, v] of Object.entries(modelInfo)) {
            if (k.toLowerCase().endsWith('context_length')) {
              const num = Number(v);
              if (Number.isFinite(num) && num >= 512) {
                ctxById.set(m.id, Math.round(num));
                return;
              }
            }
          }
        }
        if (typeof data.parameters === 'string') {
          const match = data.parameters.match(/num_ctx\s+(\d+)/i);
          if (match) {
            const num = Number(match[1]);
            if (Number.isFinite(num) && num >= 512) {
              ctxById.set(m.id, Math.round(num));
            }
          }
        }
      } catch {
        // ignore per-model timeout
      }
    }),
  );

  if (ctxById.size === 0) return models;
  return models.map((m) => {
    const found = ctxById.get(m.id);
    return found && !m.context_length ? { ...m, context_length: found } : m;
  });
}

function buildConfiguredCustomModels(
  customProvider?: CustomProviderConfig | null,
): ProviderModelCatalogItem[] {
  if (!customProvider || !Array.isArray(customProvider.models)) return [];
  return customProvider.models
    .filter((m) => m && typeof m.id === 'string' && m.id.trim())
    .map((m) => ({
      id: m.id.trim(),
      ...(m.name ? { name: m.name } : {}),
      ...(typeof m.context_window === 'number' && m.context_window >= 512
        ? { context_length: Math.round(m.context_window) }
        : {}),
      ...(typeof m.max_output_tokens === 'number' && m.max_output_tokens >= 256
        ? { max_output_tokens: Math.round(m.max_output_tokens) }
        : {}),
      source: 'preset' as const,
    }));
}

function mergeWithCustomConfiguredModels(
  liveModels: ProviderModelCatalogItem[],
  customProvider?: CustomProviderConfig | null,
): ProviderModelCatalogItem[] {
  const configured = buildConfiguredCustomModels(customProvider);
  if (configured.length === 0) return liveModels;
  const byId = new Map<string, ProviderModelCatalogItem>();
  for (const item of liveModels) {
    byId.set(item.id.toLowerCase(), item);
  }
  for (const cfg of configured) {
    const key = cfg.id.toLowerCase();
    const existing = byId.get(key);
    if (existing) {
      byId.set(key, {
        ...existing,
        ...(cfg.name ? { name: cfg.name } : {}),
        ...(cfg.context_length ? { context_length: cfg.context_length } : {}),
        ...(cfg.max_output_tokens ? { max_output_tokens: cfg.max_output_tokens } : {}),
      });
    } else {
      byId.set(key, cfg);
    }
  }
  return Array.from(byId.values()).sort((a, b) => a.id.toLowerCase().localeCompare(b.id.toLowerCase()));
}

export async function listProviderModelsDirect(params: {
  provider: string;
  base_url?: string;
  api_key?: string;
  protocol?: CustomProviderProtocol;
  customProvider?: CustomProviderConfig | null;
  role?: 'primary' | 'worker';
  env?: Record<string, string | undefined>;
  fetchImpl?: typeof fetch;
  referenceIndex?: ReferenceModelIndex | null;
  enableReferenceEnrichment?: boolean;
  forceRefresh?: boolean;
}): Promise<ProviderModelsResult> {
  const prov = normalizeProviderId(params.provider);
  const role = params.role === 'worker' ? 'worker' : 'primary';
  const defaults = PROVIDER_CATALOG_DEFAULTS[prov];
  const baseUrl = (params.base_url || params.customProvider?.base_url || defaults?.baseUrl || '')
    .trim()
    .replace(/\/+$/, '');
  const protocol = params.protocol || params.customProvider?.protocol;
  const fetchFn = params.fetchImpl || fetch;
  const configuredCustomModels = buildConfiguredCustomModels(params.customProvider);

  if (params.forceRefresh && baseUrl) {
    liveCatalogCache.delete(liveCatalogCacheKey(prov, baseUrl));
  }

  if (!baseUrl || !/^https?:\/\//i.test(baseUrl)) {
    if (configuredCustomModels.length > 0) {
      return {
        ok: true,
        provider: prov,
        base_url: baseUrl,
        source: 'preset',
        models: configuredCustomModels,
        count: configuredCustomModels.length,
        error: null,
      };
    }
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

  const allowLoopback =
    defaults?.requiresKey === false || Boolean(params.customProvider && prov.startsWith('custom_'));
  const urlCheck = await validateOutboundProviderUrl(baseUrl, {
    allowLoopback,
    resolveDns: !params.fetchImpl,
  });
  if (!urlCheck.ok) {
    return {
      ok: false,
      provider: prov,
      base_url: baseUrl,
      source: 'error',
      models: [],
      count: 0,
      error: urlCheck.error,
    };
  }

  const hasExplicitKey = Boolean((params.api_key || '').trim());
  if (!isStoredKeyAllowedForBaseUrl(prov, baseUrl, hasExplicitKey)) {
    return {
      ok: false,
      provider: prov,
      base_url: baseUrl,
      source: 'error',
      models: [],
      count: 0,
      error: `Custom Base URL does not match the canonical '${prov}' host. Provide an explicit API key or use the default provider URL.`,
    };
  }

  const apiKey = resolveProviderApiKey(prov, role, params.env || process.env, params.api_key);
  const headers = buildProviderHeaders(prov, apiKey, protocol);

  const finalizeModels = async (rawModels: ProviderModelCatalogItem[]): Promise<ProviderModelCatalogItem[]> => {
    if (prov === 'openrouter' && rawModels.length > 0) {
      seedReferenceModelCatalog(rawModels);
    }

    let working = rawModels;
    if (prov === 'ollama' && working.length > 0) {
      working = await enrichOllamaContextLengths(working, baseUrl, fetchFn);
    }

    const skipRef = isCustomProviderId(prov) || prov === 'ollama' || prov === 'vllm';
    const needsRef = !skipRef && working.some((m) => !m.context_length || !m.max_output_tokens);
    let refIndex: ReferenceModelIndex | null = null;
    if (params.referenceIndex !== undefined) {
      refIndex = params.referenceIndex;
    } else if (
      needsRef &&
      prov !== 'openrouter' &&
      (params.enableReferenceEnrichment === true || !params.fetchImpl)
    ) {
      if (params.forceRefresh) {
        clearReferenceModelCatalogCache();
      }
      refIndex = await getReferenceModelCatalog(fetchFn, 4_000);
    }

    const enriched = enrichModelsWithReferenceMetadata(working, prov, refIndex);
    return mergeWithCustomConfiguredModels(enriched, params.customProvider);
  };

  try {
    // Google Gemini native /v1beta/models returns official inputTokenLimit & outputTokenLimit
    if (prov === 'gemini' && apiKey) {
      const nativeGeminiModels = await fetchGeminiNativeModels(baseUrl, apiKey, fetchFn);
      if (nativeGeminiModels && nativeGeminiModels.length > 0) {
        const enriched = await finalizeModels(nativeGeminiModels);
        cacheProviderModels(prov, baseUrl, enriched);
        return {
          ok: true,
          provider: prov,
          base_url: baseUrl,
          source: 'live',
          models: enriched,
          count: enriched.length,
          error: null,
        };
      }
    }

    const resp = await fetchFn(`${baseUrl}/models`, {
      method: 'GET',
      headers,
      redirect: 'error',
      signal: AbortSignal.timeout(12_000),
    });

    if (resp.ok) {
      const payload = (await resp.json()) as unknown;
      const parsed = parseModelsPayload(payload, prov);
      const models = await finalizeModels(parsed);
      if (models.length > 0) {
        cacheProviderModels(prov, baseUrl, models);
      }
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
        redirect: 'error',
        signal: AbortSignal.timeout(8_000),
      });
      if (tagsResp.ok) {
        const tagsPayload = (await tagsResp.json()) as unknown;
        const parsed = parseModelsPayload(tagsPayload, prov);
        const models = await finalizeModels(parsed);
        if (models.length > 0) {
          cacheProviderModels(prov, baseUrl, models);
        }
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

    if (configuredCustomModels.length > 0) {
      return {
        ok: true,
        provider: prov,
        base_url: baseUrl,
        source: 'preset',
        models: configuredCustomModels,
        count: configuredCustomModels.length,
        error: null,
      };
    }

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
    if (configuredCustomModels.length > 0) {
      return {
        ok: true,
        provider: prov,
        base_url: baseUrl,
        source: 'preset',
        models: configuredCustomModels,
        count: configuredCustomModels.length,
        error: null,
      };
    }
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
  override?: Partial<RoleLLMConfig> & { api_key?: string; protocol?: CustomProviderProtocol };
  persistedRoleConfig: RoleLLMConfig;
  customProvider?: CustomProviderConfig | null;
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
    params.customProvider?.base_url ||
    defaults?.baseUrl ||
    params.persistedRoleConfig.base_url ||
    ''
  )
    .trim()
    .replace(/\/+$/, '');
  const protocol = ov.protocol || params.customProvider?.protocol;
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
  const allowLoopback =
    defaults?.requiresKey === false || Boolean(params.customProvider && prov.startsWith('custom_'));
  const urlCheck = await validateOutboundProviderUrl(baseUrl, {
    allowLoopback,
    resolveDns: !params.fetchImpl,
  });
  if (!urlCheck.ok) {
    return { ...baseResult, error: urlCheck.error };
  }
  if (!isStoredKeyAllowedForBaseUrl(prov, baseUrl, Boolean(explicitKey))) {
    return {
      ...baseResult,
      error: `Custom Base URL does not match the canonical '${prov}' host. Provide an explicit API key or use the default provider URL.`,
    };
  }
  if (!model) {
    return { ...baseResult, error: 'Model ID must not be empty.' };
  }

  const requiresKey = prov.startsWith('custom_') ? false : defaults ? defaults.requiresKey : true;
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

  const headers = buildProviderHeaders(prov, apiKey, protocol);

  try {
    // 1. Query /models to check reachability and whether the selected model is listed
    const modelsResp = await fetchFn(`${baseUrl}/models`, {
      method: 'GET',
      headers,
      redirect: 'error',
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
        redirect: 'error',
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
        redirect: 'error',
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
