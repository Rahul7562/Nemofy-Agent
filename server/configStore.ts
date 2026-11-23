import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

export type MediaModality = 'image' | 'video' | 'audio';
export type ExecutionAccessMode = 'default' | 'full_access';

export const MIN_MAX_TURNS = 1;
export const MAX_MAX_TURNS = 180;

export function normalizeAccessMode(raw: unknown): ExecutionAccessMode {
  return raw === 'full_access' ? 'full_access' : 'default';
}

export function validateMaxTurns(
  raw: unknown,
): { ok: true; value: number } | { ok: false; error: string } {
  if (typeof raw === 'boolean' || raw === null || raw === undefined) {
    return {
      ok: false,
      error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
    };
  }
  let parsed: number;
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!/^-?\d+$/.test(trimmed)) {
      return {
        ok: false,
        error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
      };
    }
    parsed = Number(trimmed);
  } else if (typeof raw === 'number') {
    parsed = raw;
  } else {
    return {
      ok: false,
      error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
    };
  }

  if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
    return {
      ok: false,
      error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
    };
  }
  if (parsed < MIN_MAX_TURNS || parsed > MAX_MAX_TURNS) {
    return {
      ok: false,
      error: `Max execution turns must be between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS} (received ${parsed}).`,
    };
  }
  return { ok: true, value: parsed };
}

export type CustomProviderProtocol = 'openai_compatible' | 'anthropic_compatible';

export interface CustomProviderModelConfig {
  id: string;
  name?: string;
  context_window?: number | null;
  max_output_tokens?: number | null;
}

export interface CustomProviderConfig {
  id: string;
  name: string;
  base_url: string;
  protocol: CustomProviderProtocol;
  enabled: boolean;
  models: CustomProviderModelConfig[];
  timeout_seconds?: number | null;
  created_at: string;
  updated_at: string;
}

export interface PlanningTimeoutConfig {
  connect_timeout_seconds: number;
  read_timeout_seconds: number;
  max_retries: number;
  fallback_on_timeout: boolean;
  fallback_provider?: string | null;
  fallback_model?: string | null;
}

export interface ThreadSessionConfig {
  provider: string;
  model: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  reasoning_effort: string;
  context_window?: number | null;
  context_window_source?: string | null;
  manual_context_window?: number | null;
  max_output_tokens?: number | null;
  access_mode: ExecutionAccessMode;
}

export interface RoleLLMConfig {
  provider: string;
  model: string;
  base_url: string;
  temperature: number;
  max_tokens: number;
  reasoning_effort: string;
  context_window?: number | null;
  context_window_source?: string | null;
  manual_context_window?: number | null;
  max_output_tokens?: number | null;
}

export interface ContextCategorySnapshot {
  key: string;
  label: string;
  description: string;
  tokens: number;
  percent: number;
  share_of_used: number;
}

export interface ContextUsageSnapshot {
  model: string;
  provider: string;
  context_window: number | null;
  context_window_source: string;
  detection_mode?: 'auto' | 'manual' | 'unknown';
  is_manual_override?: boolean;
  manual_context_window?: number | null;
  max_output_tokens?: number | null;
  max_output_tokens_source?: string;
  configured_max_tokens?: number | null;
  reserved_output_tokens?: number;
  safety_buffer_tokens?: number;
  usable_input_window?: number | null;
  remaining_input_budget?: number | null;
  input_budget_percent?: number | null;
  used_tokens: number;
  free_tokens: number | null;
  usage_percent: number | null;
  auto_compact_threshold_percent: number;
  classification: {
    level: 'optimal' | 'moderate' | 'high' | 'critical' | 'unknown';
    label: string;
    description: string;
  };
  categories: ContextCategorySnapshot[];
  compacted_count: number;
  last_compacted_at: string | null;
  tokens_saved_last_compaction: number;
  calibrated_from_provider?: boolean;
}

export interface AccessoryLLMConfig extends RoleLLMConfig {
  enabled: boolean;
}

export interface AccessoriesConfig {
  image: AccessoryLLMConfig;
  video: AccessoryLLMConfig;
  audio: AccessoryLLMConfig;
}

export interface PersistedLLMConfig {
  primary: RoleLLMConfig;
  worker: RoleLLMConfig;
  accessories: AccessoriesConfig;
  max_turns: number;
  access_mode: ExecutionAccessMode;
  allow_unsafe_local_bash: boolean;
  firecrawl_api_url?: string;
  custom_providers: CustomProviderConfig[];
  planning_timeout: PlanningTimeoutConfig;
}

export interface ChatAttachmentRecord {
  id: string;
  name: string;
  kind: MediaModality;
  mimeType: string;
  size: number;
  url: string;
  storageKey: string;
}

export interface WorkspaceNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  size?: number;
  modifiedAt?: string;
  children?: WorkspaceNode[];
  hasChildren?: boolean;
  childCount?: number;
  truncated?: boolean;
}

export interface SkillDocument {
  name: string;
  description: string;
  filename: string;
  content: string;
}

export interface MCPServerItem {
  name: string;
  url: string;
  transport: 'streamable_http' | 'sse';
  enabled: boolean;
  timeout_seconds: number;
}

export interface ToolEventRecord {
  id: string;
  tool: string;
  args_summary?: string;
  status: 'running' | 'ok' | 'error' | 'rejected';
  output_summary?: string;
  started_at?: string;
  finished_at?: string | null;
}

export interface FileChangeRecord {
  kind: 'file' | 'dir';
  action: 'created' | 'modified' | 'deleted';
  path: string;
  tool?: string;
  timestamp?: string;
}

export interface ChatMessageRecord {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  plan?: Array<{
    step_id: string;
    description: string;
    status: string;
    tool_or_action: string;
    result: string;
  }>;
  activeSkills?: string[];
  subagentReports?: Array<Record<string, unknown>>;
  activeModels?: {
    primary?: { provider?: string; model?: string; reasoning_effort?: string | null };
    worker?: { provider?: string; model?: string; reasoning_effort?: string | null };
  };
  thinking?: string | null;
  toolEvents?: ToolEventRecord[];
  fileChanges?: FileChangeRecord[];
  attachments?: ChatAttachmentRecord[];
  pendingApproval?: Record<string, unknown> | null;
  status?: string;
}

export interface ChatThreadRecord {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessageRecord[];
  pendingApproval?: Record<string, unknown> | null;
  contextUsage?: ContextUsageSnapshot | null;
  sessionConfig?: ThreadSessionConfig;
}

const PROVIDER_DEFAULTS: Record<string, { baseUrl: string; envVars: string[]; requiresKey: boolean }> = {
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

const PROVIDER_ALIASES: Record<string, string> = {
  river: 'river_ai',
  google: 'gemini',
  nvidia_nim: 'nvidia',
  nim: 'nvidia',
  local: 'ollama',
};

export function isCustomProviderId(provider?: string | null): boolean {
  const clean = (provider || '').trim().toLowerCase();
  return clean === 'custom' || clean.startsWith('custom_');
}

export function deriveCustomProviderSecretKey(providerId: string): string {
  const safe = (providerId || '')
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64);
  return `CUSTOM_PROVIDER_${safe || 'CUSTOM'}_API_KEY`;
}

export function slugifyCustomProviderId(name: string, existingIds?: Iterable<string>): string {
  const taken = new Set(Array.from(existingIds || []).map((id) => id.toLowerCase()));
  const baseSlug = (name || 'provider')
    .trim()
    .toLowerCase()
    .replace(/^custom[_-]+/, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 36) || 'provider';
  let candidate = `custom_${baseSlug}`;
  let idx = 2;
  while (taken.has(candidate)) {
    candidate = `custom_${baseSlug.slice(0, 30)}_${idx}`;
    idx += 1;
  }
  return candidate;
}

export function getDefaultAllowUnsafeLocalBash(env: Record<string, string | undefined> = process.env): boolean {
  const raw = (env.ALLOW_UNSAFE_LOCAL_BASH || '').trim().toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on';
}

export function getDefaultPlanningTimeoutConfig(
  env: Record<string, string | undefined> = process.env,
): PlanningTimeoutConfig {
  const rawConnect = Number(env.MODEL_CONNECT_TIMEOUT_SECONDS || '20');
  const rawRead = Number(env.MODEL_READ_TIMEOUT_SECONDS || env.MODEL_TIMEOUT_SECONDS || '120');
  const rawRetries = Number(env.MODEL_MAX_RETRIES || '2');
  const rawFallback = (env.MODEL_FALLBACK_ON_TIMEOUT || '').trim().toLowerCase();
  return {
    connect_timeout_seconds:
      Number.isFinite(rawConnect) && rawConnect >= 3 && rawConnect <= 120 ? Math.round(rawConnect) : 20,
    read_timeout_seconds:
      Number.isFinite(rawRead) && rawRead >= 15 && rawRead <= 600 ? Math.round(rawRead) : 120,
    max_retries:
      Number.isFinite(rawRetries) && rawRetries >= 0 && rawRetries <= 5 ? Math.round(rawRetries) : 2,
    fallback_on_timeout: rawFallback === '1' || rawFallback === 'true' || rawFallback === 'yes',
    fallback_provider: (env.MODEL_FALLBACK_PROVIDER || '').trim() || null,
    fallback_model: (env.MODEL_FALLBACK_MODEL || '').trim() || null,
  };
}

function normalizePlanningTimeoutConfig(
  raw: Partial<PlanningTimeoutConfig> | undefined,
  fallback: PlanningTimeoutConfig,
): PlanningTimeoutConfig {
  const connect = Number(raw?.connect_timeout_seconds);
  const read = Number(raw?.read_timeout_seconds);
  const retries = Number(raw?.max_retries);
  return {
    connect_timeout_seconds:
      Number.isFinite(connect) && connect >= 3 && connect <= 120
        ? Math.round(connect)
        : fallback.connect_timeout_seconds,
    read_timeout_seconds:
      Number.isFinite(read) && read >= 15 && read <= 600
        ? Math.round(read)
        : fallback.read_timeout_seconds,
    max_retries:
      Number.isFinite(retries) && retries >= 0 && retries <= 5
        ? Math.round(retries)
        : fallback.max_retries,
    fallback_on_timeout:
      typeof raw?.fallback_on_timeout === 'boolean'
        ? raw.fallback_on_timeout
        : fallback.fallback_on_timeout,
    fallback_provider:
      raw && 'fallback_provider' in raw
        ? (typeof raw.fallback_provider === 'string' && raw.fallback_provider.trim()
            ? normalizeProviderId(raw.fallback_provider)
            : null)
        : fallback.fallback_provider ?? null,
    fallback_model:
      raw && 'fallback_model' in raw
        ? (typeof raw.fallback_model === 'string' && raw.fallback_model.trim()
            ? raw.fallback_model.trim()
            : null)
        : fallback.fallback_model ?? null,
  };
}

function normalizeCustomProviderModels(rawModels: unknown): CustomProviderModelConfig[] {
  if (!Array.isArray(rawModels)) return [];
  const seen = new Set<string>();
  const out: CustomProviderModelConfig[] = [];
  for (const item of rawModels) {
    if (typeof item === 'string') {
      const id = item.trim();
      if (!id || seen.has(id)) continue;
      seen.add(id);
      out.push({ id });
      continue;
    }
    if (!item || typeof item !== 'object') continue;
    const rec = item as Partial<CustomProviderModelConfig>;
    const id = typeof rec.id === 'string' ? rec.id.trim() : '';
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const ctx = Number(rec.context_window);
    const maxOut = Number(rec.max_output_tokens);
    out.push({
      id,
      ...(typeof rec.name === 'string' && rec.name.trim() ? { name: rec.name.trim() } : {}),
      ...(Number.isFinite(ctx) && ctx >= 512 ? { context_window: Math.round(ctx) } : {}),
      ...(Number.isFinite(maxOut) && maxOut >= 256 ? { max_output_tokens: Math.round(maxOut) } : {}),
    });
  }
  return out;
}

function normalizeCustomProvidersList(rawList: unknown): CustomProviderConfig[] {
  if (!Array.isArray(rawList)) return [];
  const now = new Date().toISOString();
  const seen = new Set<string>();
  const out: CustomProviderConfig[] = [];
  for (const item of rawList) {
    if (!item || typeof item !== 'object') continue;
    const rec = item as Partial<CustomProviderConfig>;
    const rawName = typeof rec.name === 'string' ? rec.name.trim() : '';
    const rawId = typeof rec.id === 'string' ? rec.id.trim().toLowerCase() : '';
    const id =
      rawId && /^custom_[a-z0-9][a-z0-9_-]{0,48}$/.test(rawId)
        ? rawId
        : slugifyCustomProviderId(rawName || 'provider', seen);
    if (seen.has(id)) continue;
    seen.add(id);
    const base_url = typeof rec.base_url === 'string' ? rec.base_url.trim().replace(/\/+$/, '') : '';
    if (!base_url) continue;
    const protocol: CustomProviderProtocol =
      rec.protocol === 'anthropic_compatible' ? 'anthropic_compatible' : 'openai_compatible';
    const rawTimeout = Number(rec.timeout_seconds);
    out.push({
      id,
      name: rawName || id.replace(/^custom_/, '').replace(/_/g, ' '),
      base_url,
      protocol,
      enabled: rec.enabled !== false,
      models: normalizeCustomProviderModels(rec.models),
      ...(Number.isFinite(rawTimeout) && rawTimeout >= 10 && rawTimeout <= 600
        ? { timeout_seconds: Math.round(rawTimeout) }
        : {}),
      created_at: typeof rec.created_at === 'string' && rec.created_at ? rec.created_at : now,
      updated_at: typeof rec.updated_at === 'string' && rec.updated_at ? rec.updated_at : now,
    });
  }
  return out;
}

export function normalizeProviderId(provider?: string | null): string {
  const clean = (provider || '').trim().toLowerCase();
  if (!clean) return 'river_ai';
  return PROVIDER_ALIASES[clean] || clean;
}

function normalizeRoleConfig(
  raw: Partial<RoleLLMConfig> | undefined,
  fallback: RoleLLMConfig,
  options: { allowEmptyModel?: boolean; customProviders?: CustomProviderConfig[] } = {},
): RoleLLMConfig {
  const provider = normalizeProviderId(raw?.provider || fallback.provider);
  const customMatch = (options.customProviders || []).find((cp) => cp.id === provider);
  const presetBase = customMatch?.base_url || PROVIDER_DEFAULTS[provider]?.baseUrl || '';
  const rawBase = typeof raw?.base_url === 'string' ? raw.base_url.trim().replace(/\/+$/, '') : '';
  const base_url = rawBase || presetBase || fallback.base_url;
  const hasExplicitModel = typeof raw?.model === 'string';
  const trimmedModel = hasExplicitModel ? raw!.model!.trim() : '';
  const model =
    trimmedModel ||
    (options.allowEmptyModel && hasExplicitModel ? '' : fallback.model);
  const rawTemp = Number(raw?.temperature);
  const temperature = Number.isFinite(rawTemp) && rawTemp >= 0 && rawTemp <= 2 ? rawTemp : fallback.temperature;
  const rawTokens = Number(raw?.max_tokens);
  const max_tokens =
    Number.isFinite(rawTokens) && rawTokens >= 64 && rawTokens <= 20_000_000
      ? Math.round(rawTokens)
      : fallback.max_tokens;
  const reasoning_effort =
    typeof raw?.reasoning_effort === 'string' && raw.reasoning_effort.trim()
      ? raw.reasoning_effort.trim().toLowerCase()
      : fallback.reasoning_effort;

  const sameModelAndProvider = model === fallback.model && provider === fallback.provider;

  let manual_context_window: number | null | undefined;
  if (raw && 'manual_context_window' in raw) {
    if (raw.manual_context_window === null || raw.manual_context_window === undefined) {
      manual_context_window = null;
    } else {
      const rawManual = Number(raw.manual_context_window);
      manual_context_window = Number.isFinite(rawManual) && rawManual >= 512 ? Math.round(rawManual) : null;
    }
  } else if (
    typeof fallback.manual_context_window === 'number' &&
    Number.isFinite(fallback.manual_context_window) &&
    fallback.manual_context_window >= 512 &&
    sameModelAndProvider
  ) {
    manual_context_window = Math.round(fallback.manual_context_window);
  }

  let context_window: number | null | undefined;
  if (raw && 'context_window' in raw) {
    const rawCtx = Number(raw.context_window);
    context_window = Number.isFinite(rawCtx) && rawCtx >= 512 ? Math.round(rawCtx) : null;
  } else if (
    typeof fallback.context_window === 'number' &&
    Number.isFinite(fallback.context_window) &&
    fallback.context_window >= 512 &&
    sameModelAndProvider
  ) {
    context_window = Math.round(fallback.context_window);
  }

  let context_window_source: string | null | undefined;
  if (manual_context_window) {
    context_window = manual_context_window;
    context_window_source = 'manual';
  } else if (raw && 'context_window_source' in raw) {
    context_window_source =
      typeof raw.context_window_source === 'string' && raw.context_window_source.trim()
        ? raw.context_window_source.trim()
        : null;
  } else if (sameModelAndProvider && fallback.context_window_source) {
    context_window_source = fallback.context_window_source;
  }

  let max_output_tokens: number | null | undefined;
  if (raw && 'max_output_tokens' in raw) {
    const rawOut = Number(raw.max_output_tokens);
    max_output_tokens = Number.isFinite(rawOut) && rawOut >= 256 ? Math.round(rawOut) : null;
  } else if (
    typeof fallback.max_output_tokens === 'number' &&
    Number.isFinite(fallback.max_output_tokens) &&
    fallback.max_output_tokens >= 256 &&
    sameModelAndProvider
  ) {
    max_output_tokens = Math.round(fallback.max_output_tokens);
  }

  return {
    provider,
    model,
    base_url,
    temperature,
    max_tokens,
    reasoning_effort,
    ...(context_window ? { context_window } : {}),
    ...(context_window_source ? { context_window_source } : {}),
    ...(manual_context_window ? { manual_context_window } : {}),
    ...(max_output_tokens ? { max_output_tokens } : {}),
  };
}

export const DEFAULT_ACCESSORY_MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free';

export function getDefaultAccessoriesConfig(): AccessoriesConfig {
  const base: AccessoryLLMConfig = {
    enabled: true,
    provider: 'openrouter',
    model: DEFAULT_ACCESSORY_MODEL,
    base_url: 'https://openrouter.ai/api/v1',
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'low',
  };
  return {
    image: { ...base },
    video: { ...base },
    audio: { ...base },
  };
}

function normalizeAccessoryConfig(
  raw: Partial<AccessoryLLMConfig> | undefined,
  fallback: AccessoryLLMConfig,
): AccessoryLLMConfig {
  const roleNormalized = normalizeRoleConfig(raw, fallback);
  return {
    ...roleNormalized,
    enabled: typeof raw?.enabled === 'boolean' ? raw.enabled : fallback.enabled,
  };
}

const ALLOWED_MEDIA_MIME: Record<string, { kind: MediaModality; ext: string }> = {
  'image/png': { kind: 'image', ext: '.png' },
  'image/jpeg': { kind: 'image', ext: '.jpg' },
  'image/jpg': { kind: 'image', ext: '.jpg' },
  'image/webp': { kind: 'image', ext: '.webp' },
  'image/gif': { kind: 'image', ext: '.gif' },
  'image/bmp': { kind: 'image', ext: '.bmp' },
  'image/avif': { kind: 'image', ext: '.avif' },
  'audio/wav': { kind: 'audio', ext: '.wav' },
  'audio/x-wav': { kind: 'audio', ext: '.wav' },
  'audio/wave': { kind: 'audio', ext: '.wav' },
  'audio/mpeg': { kind: 'audio', ext: '.mp3' },
  'audio/mp3': { kind: 'audio', ext: '.mp3' },
  'audio/ogg': { kind: 'audio', ext: '.ogg' },
  'audio/opus': { kind: 'audio', ext: '.ogg' },
  'audio/flac': { kind: 'audio', ext: '.flac' },
  'audio/x-flac': { kind: 'audio', ext: '.flac' },
  'audio/mp4': { kind: 'audio', ext: '.m4a' },
  'audio/m4a': { kind: 'audio', ext: '.m4a' },
  'audio/x-m4a': { kind: 'audio', ext: '.m4a' },
  'audio/aac': { kind: 'audio', ext: '.aac' },
  'audio/webm': { kind: 'audio', ext: '.webm' },
  'video/mp4': { kind: 'video', ext: '.mp4' },
  'video/webm': { kind: 'video', ext: '.webm' },
  'video/quicktime': { kind: 'video', ext: '.mov' },
  'video/x-matroska': { kind: 'video', ext: '.mkv' },
  'video/ogg': { kind: 'video', ext: '.ogv' },
};

const EXT_TO_MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.avif': 'image/avif',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
  '.webm': 'video/webm',
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.ogv': 'video/ogg',
};

const MAX_UPLOAD_BYTES: Record<MediaModality, number> = {
  image: 15 * 1024 * 1024,
  audio: 25 * 1024 * 1024,
  video: 40 * 1024 * 1024,
};

export function maskEnvSecret(value?: string | null): string | null {
  const clean = (value || '').trim();
  if (!clean) return null;
  if (clean.length <= 8) return '****';
  return `${clean.slice(0, 4)}...${clean.slice(-4)}`;
}

function formatEnvValue(raw: string): string {
  const clean = raw.replace(/\r?\n/g, '').trim();
  if (!clean) return '';
  if (/[\s#'"`$\\]/.test(clean)) {
    return `"${clean.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
  }
  return clean;
}

export function upsertEnvContent(existingContent: string, updates: Record<string, string>): string {
  const remaining = new Map<string, string>();
  for (const [k, v] of Object.entries(updates)) {
    const cleanKey = k.trim();
    if (!/^[A-Z][A-Z0-9_]*$/.test(cleanKey)) continue;
    remaining.set(cleanKey, v);
  }

  const lines = existingContent ? existingContent.split(/\r?\n/) : [];
  const nextLines: string[] = [];

  for (const line of lines) {
    const match = line.match(/^\s*([A-Z][A-Z0-9_]*)\s*=/);
    if (match && remaining.has(match[1])) {
      const key = match[1];
      const nextVal = remaining.get(key) || '';
      remaining.delete(key);
      nextLines.push(`${key}=${formatEnvValue(nextVal)}`);
    } else {
      nextLines.push(line);
    }
  }

  // Remove trailing blank lines before appending new keys cleanly
  while (nextLines.length > 0 && nextLines[nextLines.length - 1] === '') {
    nextLines.pop();
  }

  for (const [key, val] of remaining.entries()) {
    nextLines.push(`${key}=${formatEnvValue(val)}`);
  }

  return nextLines.join('\n') + '\n';
}

export class WorkspaceAndConfigStore {
  readonly rootDir: string;
  readonly workspaceDir: string;
  readonly skillsDir: string;
  readonly mcpConfigPath: string;
  readonly llmConfigPath: string;
  readonly threadsPath: string;
  readonly uploadsDir: string;
  readonly envFilePath: string;

  constructor(rootDir?: string) {
    this.rootDir = path.resolve(rootDir || process.cwd());
    this.workspaceDir = path.resolve(process.env.WORKSPACE_DIR || path.join(this.rootDir, 'workspace'));
    this.skillsDir = path.resolve(process.env.SKILLS_DIR || path.join(this.rootDir, 'Skills'));
    this.mcpConfigPath = path.resolve(process.env.MCP_CONFIG_PATH || path.join(this.rootDir, 'MCPs', 'servers.json'));
    this.llmConfigPath = path.resolve(process.env.LLM_CONFIG_PATH || path.join(this.rootDir, 'data', 'llm_config.json'));
    this.threadsPath = path.resolve(path.join(this.rootDir, 'data', 'threads.json'));
    this.uploadsDir = path.resolve(process.env.NEMOFY_UPLOADS_DIR || path.join(this.rootDir, 'data', 'uploads'));
    this.envFilePath = path.resolve(path.join(this.rootDir, '.env'));

    fs.mkdirSync(this.workspaceDir, { recursive: true });
    fs.mkdirSync(this.skillsDir, { recursive: true });
    fs.mkdirSync(path.dirname(this.llmConfigPath), { recursive: true });
    fs.mkdirSync(path.dirname(this.mcpConfigPath), { recursive: true });
    fs.mkdirSync(this.uploadsDir, { recursive: true });
  }

  // ---------------------------------------------------------------------------
  // Multimodal Media Uploads (confined strictly inside ./data/uploads)
  // ---------------------------------------------------------------------------
  saveUploadedMedia(params: {
    threadId: string;
    filename: string;
    mimeType: string;
    buffer: Buffer;
  }): ChatAttachmentRecord & { localPath: string } {
    const cleanMime = (params.mimeType || '').toLowerCase().split(';')[0].trim();
    const extFromName = path.extname(params.filename || '').toLowerCase();
    const resolvedMime =
      ALLOWED_MEDIA_MIME[cleanMime]
        ? cleanMime
        : extFromName && EXT_TO_MIME[extFromName] && ALLOWED_MEDIA_MIME[EXT_TO_MIME[extFromName]]
          ? EXT_TO_MIME[extFromName]
          : '';
    const mediaSpec = resolvedMime ? ALLOWED_MEDIA_MIME[resolvedMime] : undefined;
    if (!mediaSpec) {
      throw new Error(
        `Unsupported media type '${cleanMime || extFromName || 'unknown'}'. Allowed formats: PNG, JPEG, WebP, GIF, MP3, WAV, OGG, FLAC, M4A, WebM, MP4, MOV.`,
      );
    }
    if (!params.buffer || params.buffer.length === 0) {
      throw new Error('Uploaded media file is empty.');
    }
    const maxBytes = MAX_UPLOAD_BYTES[mediaSpec.kind];
    if (params.buffer.length > maxBytes) {
      throw new Error(
        `${mediaSpec.kind.toUpperCase()} file exceeds the ${Math.round(maxBytes / (1024 * 1024))} MB limit.`,
      );
    }

    const safeThread = (params.threadId || 'draft')
      .replace(/[^a-zA-Z0-9_-]/g, '')
      .slice(0, 80) || 'draft';
    const safeDisplayName =
      path.basename(params.filename || `upload${mediaSpec.ext}`).replace(/[^\w.\- ()]/g, '_').slice(0, 120) ||
      `upload${mediaSpec.ext}`;
    const id = `att-${crypto.randomBytes(6).toString('hex')}`;
    const storedFilename = `${id}${mediaSpec.ext}`;
    const threadDir = path.resolve(this.uploadsDir, safeThread);
    if (!threadDir.startsWith(this.uploadsDir + path.sep)) {
      throw new Error('Invalid upload directory.');
    }
    fs.mkdirSync(threadDir, { recursive: true });
    const fullPath = path.resolve(threadDir, storedFilename);
    if (!fullPath.startsWith(threadDir + path.sep)) {
      throw new Error('Invalid upload target path.');
    }
    fs.writeFileSync(fullPath, params.buffer);

    const storageKey = `${safeThread}/${storedFilename}`;
    return {
      id,
      name: safeDisplayName,
      kind: mediaSpec.kind,
      mimeType: resolvedMime,
      size: params.buffer.length,
      url: `/api/media/${encodeURIComponent(safeThread)}/${encodeURIComponent(storedFilename)}`,
      storageKey,
      localPath: fullPath,
    };
  }

  resolveUploadedMediaPath(
    threadId: string,
    filename: string,
  ): { fullPath: string; mimeType: string } | null {
    const safeThread = (threadId || '').trim();
    const safeFile = (filename || '').trim();
    if (
      !/^[a-zA-Z0-9_-]{1,80}$/.test(safeThread) ||
      !/^[a-zA-Z0-9_.-]{1,120}$/.test(safeFile) ||
      safeFile.includes('..')
    ) {
      return null;
    }
    const fullPath = path.resolve(this.uploadsDir, safeThread, safeFile);
    if (!fullPath.startsWith(this.uploadsDir + path.sep)) {
      return null;
    }
    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
      return null;
    }
    const ext = path.extname(safeFile).toLowerCase();
    const mimeType = EXT_TO_MIME[ext];
    if (!mimeType) return null;
    return { fullPath, mimeType };
  }

  resolveAttachmentLocalPath(storageKey: string): string | null {
    const parts = (storageKey || '').split('/');
    if (parts.length !== 2) return null;
    const resolved = this.resolveUploadedMediaPath(parts[0], parts[1]);
    return resolved ? resolved.fullPath : null;
  }

  // ---------------------------------------------------------------------------
  // Workspace Filesystem (confined strictly inside ./workspace)
  // ---------------------------------------------------------------------------
  private getWorkspaceRealDir(): string {
    fs.mkdirSync(this.workspaceDir, { recursive: true });
    return fs.realpathSync(this.workspaceDir);
  }

  resolveWorkspacePath(relPath: string, allowRoot = false): string {
    const wsReal = this.getWorkspaceRealDir();
    const clean = (relPath || '').trim().replace(/^\/workspace\/?/, '').replace(/^\/+/, '');
    if (!clean || clean === '.') {
      if (allowRoot) return wsReal;
      throw new Error('A file or folder path inside /workspace is required.');
    }
    const segments = clean.split(/[/\\]+/).filter(Boolean);
    if (clean.includes('\0') || segments.includes('..')) {
      throw new Error('Path traversal outside /workspace is not allowed.');
    }
    const target = path.resolve(wsReal, ...segments);
    if (target !== wsReal && !target.startsWith(wsReal + path.sep)) {
      throw new Error('Resolved path escapes /workspace.');
    }

    // Walk every path segment to reject symbolic links escaping /workspace
    let current = wsReal;
    for (const seg of segments) {
      current = path.join(current, seg);
      if (fs.existsSync(current)) {
        const lstat = fs.lstatSync(current);
        if (lstat.isSymbolicLink()) {
          const realTarget = fs.realpathSync(current);
          if (realTarget !== wsReal && !realTarget.startsWith(wsReal + path.sep)) {
            throw new Error('Symbolic link target escapes /workspace.');
          }
        }
      } else {
        // Also catch broken symlinks (existsSync is false, lstatSync succeeds)
        try {
          const lstat = fs.lstatSync(current);
          if (lstat.isSymbolicLink()) {
            throw new Error('Symbolic link target escapes /workspace.');
          }
        } catch (err) {
          if (err instanceof Error && err.message.includes('Symbolic link')) {
            throw err;
          }
        }
        break;
      }
    }

    if (fs.existsSync(target)) {
      const realResolved = fs.realpathSync(target);
      if (realResolved !== wsReal && !realResolved.startsWith(wsReal + path.sep)) {
        throw new Error('Resolved path escapes /workspace.');
      }
      return realResolved;
    }

    return target;
  }

  private static readonly IGNORED_TREE_DIRS = new Set([
    '.git',
    '__pycache__',
    '.cache',
    'node_modules',
    '.venv',
    'venv',
    '.pytest_cache',
    '.mypy_cache',
  ]);

  private static readonly MAX_READABLE_FILE_BYTES = 1_500_000;

  listWorkspaceTree(
    dir = this.workspaceDir,
    relPrefix = '',
    options: { maxDepth?: number; maxEntriesPerDir?: number; maxTotalNodes?: number } = {},
  ): WorkspaceNode[] {
    const maxDepth = options.maxDepth ?? 2;
    const maxEntriesPerDir = options.maxEntriesPerDir ?? 250;
    const budget = { remaining: options.maxTotalNodes ?? 400 };

    const walk = (currentDir: string, prefix: string, depth: number): WorkspaceNode[] => {
      if (!fs.existsSync(currentDir) || budget.remaining <= 0) return [];
      let rawEntries: fs.Dirent[];
      try {
        rawEntries = fs.readdirSync(currentDir, { withFileTypes: true });
      } catch {
        return [];
      }

      const filtered = rawEntries.filter(
        (e) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(e.name),
      );
      filtered.sort((a, b) => {
        const aDir = a.isDirectory();
        const bDir = b.isDirectory();
        if (aDir !== bDir) return aDir ? -1 : 1;
        return a.name.localeCompare(b.name);
      });

      const nodes: WorkspaceNode[] = [];
      for (const entry of filtered.slice(0, maxEntriesPerDir)) {
        if (budget.remaining <= 0) break;
        const fullPath = path.join(currentDir, entry.name);
        const relPath = prefix ? `${prefix}/${entry.name}` : entry.name;

        if (entry.isDirectory()) {
          budget.remaining -= 1;
          let childCount = 0;
          try {
            childCount = fs
              .readdirSync(fullPath)
              .filter((n) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(n)).length;
          } catch {
            childCount = 0;
          }

          if (depth < maxDepth && budget.remaining > 0 && childCount > 0) {
            const children = walk(fullPath, relPath, depth + 1);
            nodes.push({
              name: entry.name,
              path: relPath,
              type: 'folder',
              childCount,
              hasChildren: childCount > 0,
              truncated: childCount > children.length,
              children,
            });
          } else {
            nodes.push({
              name: entry.name,
              path: relPath,
              type: 'folder',
              childCount,
              hasChildren: childCount > 0,
              truncated: childCount > 0,
            });
          }
        } else if (entry.isFile()) {
          try {
            const stat = fs.statSync(fullPath);
            budget.remaining -= 1;
            nodes.push({
              name: entry.name,
              path: relPath,
              type: 'file',
              size: stat.size,
              modifiedAt: stat.mtime.toISOString(),
            });
          } catch {
            // Skip unreadable or vanished entries safely
          }
        }
      }
      return nodes;
    };

    return walk(dir, relPrefix, 0);
  }

  listWorkspaceDirectory(
    relDir = '',
    options: { maxEntries?: number } = {},
  ): { entries: WorkspaceNode[]; totalCount: number; truncated: boolean } {
    const clean = (relDir || '').trim().replace(/^\/workspace\/?/, '').replace(/^\/+/, '');
    const targetDir = clean ? this.resolveWorkspacePath(clean) : this.workspaceDir;
    if (!fs.existsSync(targetDir)) {
      throw new Error(`Directory not found: ${relDir}`);
    }
    const stat = fs.statSync(targetDir);
    if (!stat.isDirectory()) {
      throw new Error(`Path is not a directory: ${relDir}`);
    }
    const maxEntries = Math.max(10, Math.min(500, options.maxEntries ?? 250));
    const rawEntries = fs
      .readdirSync(targetDir, { withFileTypes: true })
      .filter((e) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(e.name));

    rawEntries.sort((a, b) => {
      const aDir = a.isDirectory();
      const bDir = b.isDirectory();
      if (aDir !== bDir) return aDir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    const entries: WorkspaceNode[] = [];
    for (const entry of rawEntries.slice(0, maxEntries)) {
      const fullPath = path.join(targetDir, entry.name);
      const relPath = clean ? `${clean}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        let childCount = 0;
        try {
          childCount = fs
            .readdirSync(fullPath)
            .filter((n) => !WorkspaceAndConfigStore.IGNORED_TREE_DIRS.has(n)).length;
        } catch {
          childCount = 0;
        }
        entries.push({
          name: entry.name,
          path: relPath,
          type: 'folder',
          childCount,
          hasChildren: childCount > 0,
          truncated: childCount > 0,
        });
      } else if (entry.isFile()) {
        try {
          const fstat = fs.statSync(fullPath);
          entries.push({
            name: entry.name,
            path: relPath,
            type: 'file',
            size: fstat.size,
            modifiedAt: fstat.mtime.toISOString(),
          });
        } catch {
          // ignore
        }
      }
    }

    return {
      entries,
      totalCount: rawEntries.length,
      truncated: rawEntries.length > entries.length,
    };
  }

  readWorkspaceFile(relPath: string): { path: string; content: string; size: number; modifiedAt: string } {
    const full = this.resolveWorkspacePath(relPath);
    if (!fs.existsSync(full)) throw new Error(`File not found: ${relPath}`);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) throw new Error(`Path is a folder, not a file: ${relPath}`);
    if (stat.size > WorkspaceAndConfigStore.MAX_READABLE_FILE_BYTES) {
      const mb = (stat.size / (1024 * 1024)).toFixed(2);
      throw new Error(
        `File is too large to open in the inline editor (${mb} MB, limit 1.5 MB). Ask Nemo to inspect or summarize it instead.`,
      );
    }
    const rawBuf = fs.readFileSync(full);
    const probeLen = Math.min(rawBuf.length, 4096);
    for (let i = 0; i < probeLen; i++) {
      if (rawBuf[i] === 0) {
        throw new Error(
          `Binary file cannot be edited as plain text (${path.basename(full)}).`,
        );
      }
    }
    const content = rawBuf.toString('utf-8');
    return {
      path: path.relative(this.workspaceDir, full).replace(/\\/g, '/'),
      content,
      size: stat.size,
      modifiedAt: stat.mtime.toISOString(),
    };
  }

  writeWorkspaceFile(
    relPath: string,
    content: string,
    options: { createOnly?: boolean } = {},
  ): { path: string; size: number; modifiedAt: string } {
    const full = this.resolveWorkspacePath(relPath);
    const wsReal = this.getWorkspaceRealDir();
    fs.mkdirSync(path.dirname(full), { recursive: true });
    // Re-verify parent realpath after mkdirSync before writing
    const parentReal = fs.realpathSync(path.dirname(full));
    if (parentReal !== wsReal && !parentReal.startsWith(wsReal + path.sep)) {
      throw new Error('Resolved parent path escapes /workspace.');
    }
    if (options.createOnly) {
      try {
        fs.writeFileSync(full, content, { encoding: 'utf-8', flag: 'wx' });
      } catch (err) {
        if ((err as NodeJS.ErrnoException)?.code === 'EEXIST') {
          const conflictErr = new Error(`File already exists: /workspace/${relPath}`) as NodeJS.ErrnoException;
          conflictErr.code = 'EEXIST';
          throw conflictErr;
        }
        throw err;
      }
    } else {
      fs.writeFileSync(full, content, 'utf-8');
    }
    const stat = fs.statSync(full);
    return {
      path: path.relative(wsReal, full).replace(/\\/g, '/'),
      size: stat.size,
      modifiedAt: stat.mtime.toISOString(),
    };
  }

  createWorkspaceFolder(relPath: string): { path: string } {
    const full = this.resolveWorkspacePath(relPath);
    fs.mkdirSync(full, { recursive: true });
    return { path: path.relative(this.workspaceDir, full).replace(/\\/g, '/') };
  }

  deleteWorkspaceEntry(relPath: string): void {
    const full = this.resolveWorkspacePath(relPath);
    if (!fs.existsSync(full)) throw new Error(`Entry not found: ${relPath}`);
    fs.rmSync(full, { recursive: true, force: true });
  }

  // ---------------------------------------------------------------------------
  // LLM Config (Main Agent & Sub-Agent)
  // ---------------------------------------------------------------------------
  syncEnvFromFile(): void {
    if (!fs.existsSync(this.envFilePath)) return;
    try {
      const content = fs.readFileSync(this.envFilePath, 'utf-8');
      for (const rawLine of content.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) continue;
        const eqIdx = line.indexOf('=');
        if (eqIdx <= 0) continue;
        const key = line.slice(0, eqIdx).trim();
        if (!/^[A-Z][A-Z0-9_]*$/.test(key)) continue;
        let val = line.slice(eqIdx + 1).trim();
        if (
          (val.startsWith('"') && val.endsWith('"')) ||
          (val.startsWith("'") && val.endsWith("'"))
        ) {
          val = val.slice(1, -1);
        }
        if (val) {
          process.env[key] = val;
        }
      }
    } catch {
      // ignore read errors
    }
  }

  getProviderAvailability(
    profileSecrets?: Record<string, string>,
  ): Record<
    string,
    {
      base_url: string;
      env_keys: string[];
      primary_env_var: string;
      api_key_configured: boolean;
      masked_key: string | null;
      requires_api_key: boolean;
      storage?: 'profile_vault' | 'env';
      is_custom?: boolean;
      name?: string;
      enabled?: boolean;
      protocol?: CustomProviderProtocol;
      models?: CustomProviderModelConfig[];
    }
  > {
    if (!profileSecrets) {
      this.syncEnvFromFile();
    }
    const sourceEnv: Record<string, string | undefined> = profileSecrets || process.env;
    const storage: 'profile_vault' | 'env' = profileSecrets ? 'profile_vault' : 'env';
    const out: Record<
      string,
      {
        base_url: string;
        env_keys: string[];
        primary_env_var: string;
        api_key_configured: boolean;
        masked_key: string | null;
        requires_api_key: boolean;
        storage?: 'profile_vault' | 'env';
        is_custom?: boolean;
        name?: string;
        enabled?: boolean;
        protocol?: CustomProviderProtocol;
        models?: CustomProviderModelConfig[];
      }
    > = {};
    for (const [key, info] of Object.entries(PROVIDER_DEFAULTS)) {
      let foundValue = '';
      for (const ev of info.envVars) {
        const val = (sourceEnv[ev] || '').trim();
        if (val) {
          foundValue = val;
          break;
        }
      }
      const configured = !info.requiresKey || Boolean(foundValue);
      out[key] = {
        base_url: info.baseUrl,
        env_keys: info.envVars,
        primary_env_var: info.envVars[0] || 'PRIMARY_API_KEY',
        api_key_configured: configured,
        masked_key: maskEnvSecret(foundValue),
        requires_api_key: info.requiresKey,
        storage,
      };
    }

    // Include all user-defined custom providers
    const customProviders = this.getLLMConfig().custom_providers || [];
    for (const cp of customProviders) {
      const envVar = deriveCustomProviderSecretKey(cp.id);
      const val = (sourceEnv[envVar] || (cp.id === 'custom_legacy' ? sourceEnv.PRIMARY_API_KEY : '') || '').trim();
      out[cp.id] = {
        base_url: cp.base_url,
        env_keys: [envVar],
        primary_env_var: envVar,
        api_key_configured: Boolean(val),
        masked_key: maskEnvSecret(val),
        requires_api_key: false,
        storage,
        is_custom: true,
        name: cp.name,
        enabled: cp.enabled,
        protocol: cp.protocol,
        models: cp.models,
      };
    }

    return out;
  }

  getEnvVarForProvider(provider: string, role: 'primary' | 'worker'): string {
    const normalized = normalizeProviderId(provider);
    if (normalized.startsWith('custom_')) {
      return deriveCustomProviderSecretKey(normalized);
    }
    if (normalized === 'custom') {
      return role === 'worker' ? 'WORKER_API_KEY' : 'PRIMARY_API_KEY';
    }
    return PROVIDER_DEFAULTS[normalized]?.envVars[0] || (role === 'worker' ? 'WORKER_API_KEY' : 'PRIMARY_API_KEY');
  }

  saveEnvUpdates(updates: Record<string, string>): string[] {
    const cleanUpdates: Record<string, string> = {};
    for (const [k, v] of Object.entries(updates)) {
      const cleanKey = k.trim();
      const cleanVal = (v || '').trim();
      if (!/^[A-Z][A-Z0-9_]*$/.test(cleanKey) || !cleanVal) continue;
      cleanUpdates[cleanKey] = cleanVal;
    }
    const keys = Object.keys(cleanUpdates);
    if (keys.length === 0) return [];

    const existing = fs.existsSync(this.envFilePath)
      ? fs.readFileSync(this.envFilePath, 'utf-8')
      : '';
    const nextContent = upsertEnvContent(existing, cleanUpdates);
    fs.writeFileSync(this.envFilePath, nextContent, { encoding: 'utf-8', mode: 0o600 });

    for (const [k, v] of Object.entries(cleanUpdates)) {
      process.env[k] = v;
    }
    return keys;
  }

  getLLMConfig(): PersistedLLMConfig {
    const defaultPrimaryProvider = normalizeProviderId(process.env.PRIMARY_PROVIDER || 'river_ai');
    const defaultWorkerProvider = normalizeProviderId(process.env.WORKER_PROVIDER || defaultPrimaryProvider);
    const defaults: PersistedLLMConfig = {
      primary: {
        provider: defaultPrimaryProvider,
        model: process.env.PRIMARY_MODEL || 'deepseek-ai/DeepSeek-V4.1-Flash',
        base_url:
          process.env.PRIMARY_BASE_URL ||
          PROVIDER_DEFAULTS[defaultPrimaryProvider]?.baseUrl ||
          'https://api.river.ai/v1',
        temperature: Number(process.env.PRIMARY_TEMPERATURE || '0.2'),
        max_tokens: Number(process.env.PRIMARY_MAX_TOKENS || '16384'),
        reasoning_effort: process.env.PRIMARY_REASONING_EFFORT || 'off',
      },
      worker: {
        provider: defaultWorkerProvider,
        model: process.env.WORKER_MODEL || 'deepseek-ai/DeepSeek-V4.1-Flash',
        base_url:
          process.env.WORKER_BASE_URL ||
          PROVIDER_DEFAULTS[defaultWorkerProvider]?.baseUrl ||
          'https://api.river.ai/v1',
        temperature: Number(process.env.WORKER_TEMPERATURE || '0.1'),
        max_tokens: Number(process.env.WORKER_MAX_TOKENS || '8192'),
        reasoning_effort: process.env.WORKER_REASONING_EFFORT || 'off',
      },
      accessories: getDefaultAccessoriesConfig(),
      max_turns: Math.max(
        MIN_MAX_TURNS,
        Math.min(MAX_MAX_TURNS, Math.round(Number(process.env.MAX_TURNS || '30') || 30)),
      ),
      access_mode: normalizeAccessMode(process.env.EXECUTION_ACCESS_MODE),
      allow_unsafe_local_bash: getDefaultAllowUnsafeLocalBash(process.env),
      firecrawl_api_url: (process.env.FIRECRAWL_API_URL || '').trim().replace(/\/+$/, ''),
      custom_providers: [],
      planning_timeout: getDefaultPlanningTimeoutConfig(process.env),
    };

    if (!fs.existsSync(this.llmConfigPath)) {
      return defaults;
    }

    try {
      const parsed = JSON.parse(fs.readFileSync(this.llmConfigPath, 'utf-8')) as Partial<PersistedLLMConfig>;
      let customProviders = normalizeCustomProvidersList(parsed.custom_providers);
      // Seamless migration of legacy single 'custom' provider if base_url was configured
      if (
        customProviders.length === 0 &&
        parsed.primary?.provider === 'custom' &&
        typeof parsed.primary.base_url === 'string' &&
        parsed.primary.base_url.trim()
      ) {
        const now = new Date().toISOString();
        customProviders = [
          {
            id: 'custom_legacy',
            name: 'Legacy Custom Provider',
            base_url: parsed.primary.base_url.trim().replace(/\/+$/, ''),
            protocol: 'openai_compatible',
            enabled: true,
            models: parsed.primary.model?.trim()
              ? [
                  {
                    id: parsed.primary.model.trim(),
                    ...(parsed.primary.context_window ? { context_window: parsed.primary.context_window } : {}),
                    ...(parsed.primary.max_output_tokens ? { max_output_tokens: parsed.primary.max_output_tokens } : {}),
                  },
                ]
              : [],
            created_at: now,
            updated_at: now,
          },
        ];
      }
      return {
        primary: normalizeRoleConfig(parsed.primary, defaults.primary, {
          allowEmptyModel: true,
          customProviders,
        }),
        worker: normalizeRoleConfig(parsed.worker, defaults.worker, { customProviders }),
        accessories: {
          image: normalizeAccessoryConfig(parsed.accessories?.image, defaults.accessories.image),
          video: normalizeAccessoryConfig(parsed.accessories?.video, defaults.accessories.video),
          audio: normalizeAccessoryConfig(parsed.accessories?.audio, defaults.accessories.audio),
        },
        max_turns:
          typeof parsed.max_turns === 'number' && Number.isFinite(parsed.max_turns)
            ? Math.max(MIN_MAX_TURNS, Math.min(MAX_MAX_TURNS, Math.round(parsed.max_turns)))
            : defaults.max_turns,
        access_mode:
          parsed.access_mode !== undefined
            ? normalizeAccessMode(parsed.access_mode)
            : defaults.access_mode,
        allow_unsafe_local_bash:
          typeof parsed.allow_unsafe_local_bash === 'boolean'
            ? parsed.allow_unsafe_local_bash
            : defaults.allow_unsafe_local_bash,
        firecrawl_api_url:
          typeof parsed.firecrawl_api_url === 'string'
            ? parsed.firecrawl_api_url.trim().replace(/\/+$/, '')
            : defaults.firecrawl_api_url || '',
        custom_providers: customProviders,
        planning_timeout: normalizePlanningTimeoutConfig(parsed.planning_timeout, defaults.planning_timeout),
      };
    } catch {
      return defaults;
    }
  }

  saveLLMConfig(
    next: Partial<Omit<PersistedLLMConfig, 'accessories' | 'planning_timeout'>> & {
      accessories?: Partial<{
        image: Partial<AccessoryLLMConfig>;
        video: Partial<AccessoryLLMConfig>;
        audio: Partial<AccessoryLLMConfig>;
      }>;
      planning_timeout?: Partial<PlanningTimeoutConfig>;
    },
  ): PersistedLLMConfig {
    const current = this.getLLMConfig();
    const customProviders =
      next.custom_providers !== undefined
        ? normalizeCustomProvidersList(next.custom_providers)
        : current.custom_providers;
    const merged: PersistedLLMConfig = {
      primary: normalizeRoleConfig(next.primary, current.primary, {
        allowEmptyModel: true,
        customProviders,
      }),
      worker: normalizeRoleConfig(next.worker, current.worker, { customProviders }),
      accessories: {
        image: normalizeAccessoryConfig(next.accessories?.image, current.accessories.image),
        video: normalizeAccessoryConfig(next.accessories?.video, current.accessories.video),
        audio: normalizeAccessoryConfig(next.accessories?.audio, current.accessories.audio),
      },
      max_turns:
        typeof next.max_turns === 'number' && Number.isFinite(next.max_turns)
          ? Math.max(MIN_MAX_TURNS, Math.min(MAX_MAX_TURNS, Math.round(next.max_turns)))
          : current.max_turns,
      access_mode:
        next.access_mode !== undefined
          ? normalizeAccessMode(next.access_mode)
          : current.access_mode,
      allow_unsafe_local_bash:
        typeof next.allow_unsafe_local_bash === 'boolean'
          ? next.allow_unsafe_local_bash
          : current.allow_unsafe_local_bash,
      firecrawl_api_url:
        typeof next.firecrawl_api_url === 'string'
          ? next.firecrawl_api_url.trim().replace(/\/+$/, '')
          : current.firecrawl_api_url || '',
      custom_providers: customProviders,
      planning_timeout: normalizePlanningTimeoutConfig(next.planning_timeout, current.planning_timeout),
    };
    const payload = JSON.stringify(merged, null, 2) + '\n';
    const tmpPath = `${this.llmConfigPath}.${process.pid}.${crypto.randomBytes(4).toString('hex')}.tmp`;
    fs.writeFileSync(tmpPath, payload, 'utf-8');
    fs.renameSync(tmpPath, this.llmConfigPath);
    return merged;
  }

  getCustomProviders(): CustomProviderConfig[] {
    return this.getLLMConfig().custom_providers || [];
  }

  getCustomProvider(providerId: string): CustomProviderConfig | null {
    const clean = normalizeProviderId(providerId);
    return this.getCustomProviders().find((cp) => cp.id === clean) || null;
  }

  upsertCustomProvider(input: {
    id?: string;
    name: string;
    base_url: string;
    protocol?: CustomProviderProtocol;
    enabled?: boolean;
    models?: Array<string | CustomProviderModelConfig>;
    timeout_seconds?: number | null;
  }): CustomProviderConfig {
    const current = this.getLLMConfig();
    const list = [...(current.custom_providers || [])];
    const now = new Date().toISOString();
    const cleanName = (input.name || '').trim();
    const cleanBase = (input.base_url || '').trim().replace(/\/+$/, '');
    if (!cleanName) {
      throw new Error('Custom provider name is required.');
    }
    if (!cleanBase || !/^https?:\/\//i.test(cleanBase)) {
      throw new Error('A valid http:// or https:// Base URL is required.');
    }

    const existingIdx = input.id
      ? list.findIndex((cp) => cp.id === normalizeProviderId(input.id))
      : -1;
    const id =
      existingIdx >= 0
        ? list[existingIdx].id
        : input.id && /^custom_[a-z0-9][a-z0-9_-]{0,48}$/.test(input.id.trim().toLowerCase())
          ? input.id.trim().toLowerCase()
          : slugifyCustomProviderId(cleanName, list.map((c) => c.id));

    const prev = existingIdx >= 0 ? list[existingIdx] : null;
    const rawTimeout = Number(input.timeout_seconds);
    const nextItem: CustomProviderConfig = {
      id,
      name: cleanName,
      base_url: cleanBase,
      protocol:
        input.protocol === 'anthropic_compatible' ? 'anthropic_compatible' : 'openai_compatible',
      enabled: input.enabled !== undefined ? Boolean(input.enabled) : prev?.enabled ?? true,
      models:
        input.models !== undefined
          ? normalizeCustomProviderModels(input.models)
          : prev?.models || [],
      ...(Number.isFinite(rawTimeout) && rawTimeout >= 10 && rawTimeout <= 600
        ? { timeout_seconds: Math.round(rawTimeout) }
        : input.timeout_seconds === null
          ? {}
          : prev?.timeout_seconds
            ? { timeout_seconds: prev.timeout_seconds }
            : {}),
      created_at: prev?.created_at || now,
      updated_at: now,
    };

    if (existingIdx >= 0) {
      list[existingIdx] = nextItem;
    } else {
      list.push(nextItem);
    }

    this.saveLLMConfig({ custom_providers: list });
    return nextItem;
  }

  deleteCustomProvider(providerId: string): boolean {
    const clean = normalizeProviderId(providerId);
    const current = this.getLLMConfig();
    const nextList = (current.custom_providers || []).filter((cp) => cp.id !== clean);
    if (nextList.length === (current.custom_providers || []).length) {
      return false;
    }
    this.saveLLMConfig({ custom_providers: nextList });
    return true;
  }

  // ---------------------------------------------------------------------------
  // Skills (./Skills/*.md)
  // ---------------------------------------------------------------------------
  listSkills(): SkillDocument[] {
    if (!fs.existsSync(this.skillsDir)) return [];
    return fs
      .readdirSync(this.skillsDir)
      .filter((f) => f.endsWith('.md'))
      .sort()
      .map((filename) => {
        const content = fs.readFileSync(path.join(this.skillsDir, filename), 'utf-8');
        const nameMatch = content.match(/^name:\s*(.+)$/m);
        const descMatch = content.match(/^description:\s*(.+)$/m);
        return {
          name: (nameMatch ? nameMatch[1].trim().replace(/^["']|["']$/g, '') : filename.replace(/\.md$/, '')),
          description: descMatch
            ? descMatch[1].trim().replace(/^["']|["']$/g, '')
            : 'Custom agent skill definition',
          filename,
          content,
        };
      });
  }

  saveSkill(name: string, content: string): SkillDocument {
    const safeName = name.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
    if (!safeName) throw new Error('Invalid skill name.');
    const filename = `${safeName}.md`;
    const full = path.join(this.skillsDir, filename);
    fs.writeFileSync(full, content, 'utf-8');
    const descMatch = content.match(/^description:\s*(.+)$/m);
    return {
      name: safeName,
      description: descMatch ? descMatch[1].trim() : 'Custom agent skill definition',
      filename,
      content,
    };
  }

  deleteSkill(name: string): void {
    const safeName = name.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_');
    const full = path.join(this.skillsDir, `${safeName}.md`);
    if (fs.existsSync(full)) fs.unlinkSync(full);
  }

  // ---------------------------------------------------------------------------
  // Pure-Python MCP Servers (./MCPs/servers.json)
  // ---------------------------------------------------------------------------
  listMCPServers(): MCPServerItem[] {
    if (!fs.existsSync(this.mcpConfigPath)) return [];
    try {
      const parsed = JSON.parse(fs.readFileSync(this.mcpConfigPath, 'utf-8')) as {
        servers?: MCPServerItem[];
      };
      return Array.isArray(parsed.servers) ? parsed.servers : [];
    } catch {
      return [];
    }
  }

  private static validateMcpServerUrl(rawUrl: string): string {
    let parsed: URL;
    try {
      parsed = new URL(rawUrl);
    } catch {
      throw new Error(`Invalid MCP server URL: ${rawUrl}`);
    }
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
      throw new Error(`MCP server URL must use http:// or https://: ${rawUrl}`);
    }
    if (parsed.username || parsed.password) {
      throw new Error('MCP server URL must not contain embedded credentials.');
    }
    const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');
    if (
      !host ||
      host === 'localhost' ||
      host.endsWith('.localhost') ||
      host.endsWith('.local') ||
      host.endsWith('.internal') ||
      host === 'metadata.google.internal'
    ) {
      throw new Error(`MCP server URL targets a restricted or local host: ${host}`);
    }
    if (net.isIPv4(host)) {
      const [a, b] = host.split('.').map(Number);
      if (
        a === 0 ||
        a === 10 ||
        a === 127 ||
        (a === 169 && b === 254) ||
        (a === 172 && b >= 16 && b <= 31) ||
        (a === 192 && b === 168) ||
        (a === 100 && b >= 64 && b <= 127) ||
        a >= 224
      ) {
        throw new Error(`MCP server URL targets a private, loopback, or metadata IP: ${host}`);
      }
    } else if (net.isIPv6(host)) {
      const first = parseInt(host.split(':')[0] || '0', 16);
      if (
        host === '::' ||
        host === '::1' ||
        host.startsWith('::ffff:') ||
        (first & 0xfe00) === 0xfc00 ||
        (first & 0xffc0) === 0xfe80 ||
        (first & 0xff00) === 0xff00
      ) {
        throw new Error(`MCP server URL targets a private, loopback, or metadata IP: ${host}`);
      }
    }
    return rawUrl;
  }

  saveMCPServers(servers: MCPServerItem[]): MCPServerItem[] {
    const clean = servers
      .filter((s) => s && typeof s.name === 'string' && s.name.trim() && typeof s.url === 'string' && s.url.trim())
      .map((s) => ({
        name: s.name.trim(),
        url: WorkspaceAndConfigStore.validateMcpServerUrl(s.url.trim()),
        transport: (s.transport === 'sse' ? 'sse' : 'streamable_http') as 'streamable_http' | 'sse',
        enabled: s.enabled !== false,
        timeout_seconds: Math.max(2, Math.min(180, Number(s.timeout_seconds) || 30)),
      }));
    fs.writeFileSync(this.mcpConfigPath, JSON.stringify({ servers: clean }, null, 2) + '\n', 'utf-8');
    return clean;
  }

  // ---------------------------------------------------------------------------
  // Chat Threads (./data/threads.json)
  // ---------------------------------------------------------------------------
  listThreads(): ChatThreadRecord[] {
    if (!fs.existsSync(this.threadsPath)) return [];
    try {
      const parsed = JSON.parse(fs.readFileSync(this.threadsPath, 'utf-8')) as ChatThreadRecord[];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  saveThreads(threads: ChatThreadRecord[]): void {
    fs.mkdirSync(path.dirname(this.threadsPath), { recursive: true });
    const tmpPath = `${this.threadsPath}.${process.pid}.${crypto.randomBytes(6).toString('hex')}.tmp`;
    fs.writeFileSync(tmpPath, JSON.stringify(threads, null, 2) + '\n', 'utf-8');
    fs.renameSync(tmpPath, this.threadsPath);
  }

  updateThreads<T>(mutator: (threads: ChatThreadRecord[]) => T): T {
    const fresh = this.listThreads();
    const result = mutator(fresh);
    this.saveThreads(fresh);
    return result;
  }

  buildThreadSessionConfig(
    base?: Partial<ThreadSessionConfig>,
    fallbackConfig?: PersistedLLMConfig,
  ): ThreadSessionConfig {
    const cfg = fallbackConfig || this.getLLMConfig();
    const normalizedRole = normalizeRoleConfig(base, cfg.primary, {
      allowEmptyModel: true,
      customProviders: cfg.custom_providers,
    });
    return {
      ...normalizedRole,
      access_mode:
        base?.access_mode !== undefined
          ? normalizeAccessMode(base.access_mode)
          : cfg.access_mode,
    };
  }

  updateThreadSessionConfig(
    threadId: string,
    patch: Partial<ThreadSessionConfig>,
  ): ChatThreadRecord | null {
    const cfg = this.getLLMConfig();
    return this.updateThreads((threads) => {
      const target = threads.find((t) => t.id === threadId);
      if (!target) return null;
      const existingBase: ThreadSessionConfig = target.sessionConfig
        ? { ...target.sessionConfig }
        : this.buildThreadSessionConfig(undefined, cfg);

      const providerChanged =
        typeof patch.provider === 'string' &&
        normalizeProviderId(patch.provider) !== normalizeProviderId(existingBase.provider);
      const modelChanged =
        typeof patch.model === 'string' &&
        patch.model.trim() !== existingBase.model;

      const mergedRaw: Partial<ThreadSessionConfig> = {
        ...existingBase,
        ...patch,
      };
      // When switching provider or model without specifying explicit limits, clear stale cached limits
      // so the new provider/model resolves cleanly instead of inheriting the old model's window.
      if ((providerChanged || modelChanged) && !('context_window' in patch)) {
        mergedRaw.context_window = null;
        mergedRaw.context_window_source = null;
      }
      if ((providerChanged || modelChanged) && !('manual_context_window' in patch)) {
        mergedRaw.manual_context_window = null;
      }
      if ((providerChanged || modelChanged) && !('max_output_tokens' in patch)) {
        mergedRaw.max_output_tokens = null;
      }
      if (providerChanged && !('base_url' in patch)) {
        const nextProv = normalizeProviderId(patch.provider);
        const customMatch = (cfg.custom_providers || []).find((cp) => cp.id === nextProv);
        mergedRaw.base_url = customMatch?.base_url || PROVIDER_DEFAULTS[nextProv]?.baseUrl || '';
      }

      const normalized = normalizeRoleConfig(mergedRaw, existingBase, {
        allowEmptyModel: true,
        customProviders: cfg.custom_providers,
      });
      target.sessionConfig = {
        ...normalized,
        access_mode:
          mergedRaw.access_mode !== undefined
            ? normalizeAccessMode(mergedRaw.access_mode)
            : existingBase.access_mode,
      };
      target.updatedAt = new Date().toISOString();
      return target;
    });
  }
}
