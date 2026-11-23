import type {
  AccessoriesConfig,
  AccessoryLLMConfig,
  CustomProviderConfig,
  MediaModality,
  PersistedLLMConfig,
  ProviderAvailability,
  ProviderModelOption,
  RoleLLMConfig,
} from '../types/agent.ts';
import { getThinkingProfile, normalizeThinkingLevel } from './thinkingModes.ts';

export interface ProviderPreset {
  id: string;
  label: string;
  baseUrl: string;
  defaultModels: string[];
  envVars: string[];
  requiresApiKey: boolean;
}

export const LLM_PROVIDER_PRESETS: ProviderPreset[] = [
  {
    id: 'river_ai',
    label: 'River AI',
    baseUrl: 'https://api.river.ai/v1',
    defaultModels: [
      'deepseek-ai/DeepSeek-V4.1-Flash',
      'GLM-5.3',
      'moonshotai/Kimi-K3.1-Pro',
      'Qwen/Qwen3.8-27B-FP8',
    ],
    envVars: ['RIVER_AI_API'],
    requiresApiKey: true,
  },
  {
    id: 'nvidia',
    label: 'NVIDIA NIM',
    baseUrl: 'https://integrate.api.nvidia.com/v1',
    defaultModels: [
      'nvidia/nemotron-3-ultra-550b-a55b',
      'nvidia/nemotron-3.5-super-vl-120b-a12b',
      'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
      'moonshotai/kimi-k3.1-pro',
      'mistralai/mistral-large-4-1t-a52b',
    ],
    envVars: ['NVIDIA_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'openai',
    label: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    defaultModels: ['gpt-5.2', 'o3-mini', 'gpt-4o', 'gpt-4.1'],
    envVars: ['OPENAI_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'anthropic',
    label: 'Anthropic (OpenAI-Compat)',
    baseUrl: 'https://api.anthropic.com/v1',
    defaultModels: ['claude-opus-4-6', 'claude-sonnet-4-6', 'claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022'],
    envVars: ['ANTHROPIC_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'gemini',
    label: 'Google Gemini',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    defaultModels: ['gemini-2.5-pro', 'gemini-2.5-flash'],
    envVars: ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    defaultModels: [
      'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
      'anthropic/claude-opus-4.6',
      'openai/gpt-5.2',
      'deepseek/deepseek-r1',
      'nvidia/nemotron-3-ultra-550b-a55b:free',
      'x-ai/grok-3-mini',
    ],
    envVars: ['OPENROUTER_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'deepseek',
    label: 'DeepSeek Direct',
    baseUrl: 'https://api.deepseek.com/v1',
    defaultModels: ['deepseek-chat', 'deepseek-reasoner'],
    envVars: ['DEEPSEEK_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'groq',
    label: 'Groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    defaultModels: ['llama-3.3-70b-versatile', 'qwen-qwq-32b'],
    envVars: ['GROQ_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'together',
    label: 'Together AI',
    baseUrl: 'https://api.together.xyz/v1',
    defaultModels: ['meta-llama/Llama-3.3-70B-Instruct-Turbo', 'Qwen/Qwen2.5-Coder-32B-Instruct'],
    envVars: ['TOGETHER_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'fireworks',
    label: 'Fireworks AI',
    baseUrl: 'https://api.fireworks.ai/inference/v1',
    defaultModels: ['accounts/fireworks/models/deepseek-v3', 'accounts/fireworks/models/llama-v3p3-70b-instruct'],
    envVars: ['FIREWORKS_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'ollama',
    label: 'Ollama (Local)',
    baseUrl: 'http://127.0.0.1:11434/v1',
    defaultModels: ['qwen2.5-coder:14b', 'llama3.3:latest', 'deepseek-r1:14b'],
    envVars: ['OLLAMA_API_KEY'],
    requiresApiKey: false,
  },
  {
    id: 'vllm',
    label: 'vLLM (Self-Hosted)',
    baseUrl: 'http://127.0.0.1:8000/v1',
    defaultModels: ['Qwen/Qwen2.5-Coder-32B-Instruct'],
    envVars: ['VLLM_API_KEY'],
    requiresApiKey: false,
  },
  {
    id: 'custom',
    label: 'Custom OpenAI-Compatible',
    baseUrl: '',
    defaultModels: [],
    envVars: ['PRIMARY_API_KEY', 'WORKER_API_KEY'],
    requiresApiKey: true,
  },
];

export const REASONING_EFFORT_OPTIONS = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const;
export const MIN_MAX_TURNS = 1;
export const MAX_MAX_TURNS = 180;

const PROVIDER_ALIASES: Record<string, string> = {
  river: 'river_ai',
  google: 'gemini',
  nvidia_nim: 'nvidia',
  nim: 'nvidia',
  local: 'ollama',
};

export function canonicalizeProviderId(providerId: string): string {
  const clean = (providerId || '').trim().toLowerCase();
  return PROVIDER_ALIASES[clean] || clean;
}

export function isCustomProviderId(providerId: string): boolean {
  const clean = canonicalizeProviderId(providerId);
  return clean.startsWith('custom_');
}

export function getProviderPreset(providerId: string): ProviderPreset | undefined {
  const canonical = canonicalizeProviderId(providerId);
  return LLM_PROVIDER_PRESETS.find((p) => p.id === canonical);
}

export function getProviderDisplayName(
  providerId: string,
  customProviders?: CustomProviderConfig[],
  providers?: Record<string, ProviderAvailability>,
): string {
  const canonical = canonicalizeProviderId(providerId);
  const custom = customProviders?.find((p) => p.id === canonical);
  if (custom?.name) return custom.name;
  const availName = providers?.[canonical]?.name;
  if (availName) return availName;
  return getProviderPreset(canonical)?.label || providerId;
}

export function applyProviderPreset(
  current: RoleLLMConfig,
  nextProviderId: string,
  options?: {
    clearModelOnProviderSwitch?: boolean;
    customProviders?: CustomProviderConfig[];
  },
): RoleLLMConfig {
  const canonicalNext = canonicalizeProviderId(nextProviderId);
  const customProvider = options?.customProviders?.find((p) => p.id === canonicalNext);
  if (customProvider) {
    const isSameProvider = canonicalizeProviderId(current.provider) === customProvider.id;
    const firstModel = customProvider.models[0];
    const nextModel = isSameProvider
      ? current.model || (options?.clearModelOnProviderSwitch ? '' : firstModel?.id || '')
      : options?.clearModelOnProviderSwitch
        ? ''
        : firstModel?.id || '';
    const matchedCustomModel = customProvider.models.find(
      (m) => m.id.toLowerCase() === nextModel.trim().toLowerCase(),
    );
    const profile = getThinkingProfile(customProvider.id, nextModel);
    return {
      ...current,
      provider: customProvider.id,
      base_url: customProvider.base_url,
      model: nextModel,
      context_window: matchedCustomModel?.context_window ?? (isSameProvider ? current.context_window : null),
      context_window_source: matchedCustomModel?.context_window ? 'custom_provider_config' : null,
      manual_context_window: isSameProvider ? current.manual_context_window : null,
      max_output_tokens: matchedCustomModel?.max_output_tokens ?? (isSameProvider ? current.max_output_tokens : null),
      reasoning_effort: normalizeThinkingLevel(profile, current.reasoning_effort),
    };
  }

  const preset = getProviderPreset(canonicalNext);
  if (!preset) {
    return {
      ...current,
      provider: canonicalNext,
      context_window: undefined,
      context_window_source: null,
      manual_context_window: null,
      max_output_tokens: undefined,
    };
  }

  const isSameProvider = canonicalizeProviderId(current.provider) === preset.id;
  const knownPresetUrls = new Set(LLM_PROVIDER_PRESETS.map((p) => p.baseUrl).filter(Boolean));
  const nextBaseUrl =
    preset.id === 'custom'
      ? knownPresetUrls.has(current.base_url.trim().replace(/\/+$/, ''))
        ? ''
        : current.base_url
      : preset.baseUrl;

  const nextModel = isSameProvider
    ? current.model || (options?.clearModelOnProviderSwitch ? '' : preset.defaultModels[0] || '')
    : options?.clearModelOnProviderSwitch
      ? ''
      : preset.defaultModels[0] || '';

  const profile = getThinkingProfile(preset.id, nextModel);
  return {
    ...current,
    provider: preset.id,
    base_url: nextBaseUrl,
    model: nextModel,
    context_window: isSameProvider ? current.context_window : undefined,
    context_window_source: isSameProvider ? current.context_window_source : null,
    manual_context_window: isSameProvider ? current.manual_context_window : null,
    max_output_tokens: isSameProvider ? current.max_output_tokens : undefined,
    reasoning_effort: normalizeThinkingLevel(profile, current.reasoning_effort),
  };
}

export function mergeProviderModels(
  liveModels: ProviderModelOption[],
  presetDefaultIds: string[],
  currentModel?: string,
): ProviderModelOption[] {
  const seen = new Set<string>();
  const out: ProviderModelOption[] = [];

  for (const item of liveModels) {
    const id = (item.id || '').trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push({ ...item, id, source: item.source || 'live' });
  }

  // If live models were returned, keep them as the primary source and only append missing presets at the end if empty
  if (out.length === 0) {
    for (const id of presetDefaultIds) {
      const clean = id.trim();
      if (!clean || seen.has(clean)) continue;
      seen.add(clean);
      out.push({
        id: clean,
        is_free: clean.toLowerCase().endsWith(':free'),
        source: 'preset',
      });
    }
  }

  const cur = (currentModel || '').trim();
  if (cur && !seen.has(cur)) {
    out.unshift({
      id: cur,
      is_free: cur.toLowerCase().endsWith(':free'),
      source: 'custom',
    });
  }

  return out;
}

export function filterProviderModels(
  models: ProviderModelOption[],
  query: string,
): ProviderModelOption[] {
  const clean = (query || '').trim().toLowerCase();
  if (!clean) return models;
  const tokens = clean.split(/\s+/).filter(Boolean);
  return models.filter((m) => {
    const haystack = `${m.id} ${m.name || ''} ${m.owned_by || ''} ${m.is_free ? 'free' : ''}`.toLowerCase();
    return tokens.every((tok) => haystack.includes(tok));
  });
}

export function formatContextLength(tokens?: number | null): string | null {
  if (!tokens || !Number.isFinite(tokens) || tokens <= 0) return null;
  if (tokens >= 1_000_000) {
    const m = tokens / 1_000_000;
    return `${Number.isInteger(m) ? m : m.toFixed(1)}M ctx`;
  }
  if (tokens >= 1000) {
    return `${Math.round(tokens / 1000)}k ctx`;
  }
  return `${tokens} ctx`;
}

export function formatOutputLength(tokens?: number | null): string | null {
  if (!tokens || !Number.isFinite(tokens) || tokens <= 0) return null;
  if (tokens >= 1_000_000) {
    const m = tokens / 1_000_000;
    return `${Number.isInteger(m) ? m : m.toFixed(1)}M out`;
  }
  if (tokens >= 1000) {
    return `${Math.round(tokens / 1000)}k out`;
  }
  return `${tokens} out`;
}

export const DEFAULT_ACCESSORY_PROVIDER = 'openrouter';
export const DEFAULT_ACCESSORY_MODEL = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free';
export const DEFAULT_ACCESSORY_BASE_URL = 'https://openrouter.ai/api/v1';

export function getDefaultAccessoryEntry(): AccessoryLLMConfig {
  return {
    enabled: true,
    provider: DEFAULT_ACCESSORY_PROVIDER,
    model: DEFAULT_ACCESSORY_MODEL,
    base_url: DEFAULT_ACCESSORY_BASE_URL,
    temperature: 0.2,
    max_tokens: 4096,
    reasoning_effort: 'low',
  };
}

export function getDefaultAccessoriesConfig(): AccessoriesConfig {
  return {
    image: getDefaultAccessoryEntry(),
    video: getDefaultAccessoryEntry(),
    audio: getDefaultAccessoryEntry(),
  };
}

export function normalizeAccessoriesConfig(
  raw?: Partial<Record<MediaModality, Partial<AccessoryLLMConfig>>> | null,
): AccessoriesConfig {
  const defaults = getDefaultAccessoriesConfig();
  const out: AccessoriesConfig = { ...defaults };
  for (const mod of ['image', 'video', 'audio'] as const) {
    const incoming = raw?.[mod];
    if (incoming && typeof incoming === 'object') {
      out[mod] = {
        ...defaults[mod],
        ...incoming,
        enabled: typeof incoming.enabled === 'boolean' ? incoming.enabled : defaults[mod].enabled,
        provider: (incoming.provider || defaults[mod].provider).trim(),
        model: (incoming.model || defaults[mod].model).trim(),
        base_url: (incoming.base_url || defaults[mod].base_url).trim(),
        temperature:
          typeof incoming.temperature === 'number' && Number.isFinite(incoming.temperature)
            ? incoming.temperature
            : defaults[mod].temperature,
        max_tokens:
          typeof incoming.max_tokens === 'number' && Number.isFinite(incoming.max_tokens)
            ? Math.round(incoming.max_tokens)
            : defaults[mod].max_tokens,
        reasoning_effort: (incoming.reasoning_effort || defaults[mod].reasoning_effort).trim(),
      };
    }
  }
  return out;
}

const OMNI_ALL_PATTERNS = [
  'omni',
  'gemini-1.5',
  'gemini-2',
  'gemini-2.5',
  'gemini-3',
  'gemini-4',
  'gemini-flash',
  'gemini-pro',
  'qwen2.5-omni',
  'qwen3-omni',
  'minicpm-o',
];

const TEXT_ONLY_OVERRIDES = [
  'deepseek-r1',
  'deepseek-v3',
  'deepseek-chat',
  'deepseek-reasoner',
  'deepseek-coder',
  'qwen2.5-coder',
  'qwen3-coder',
  'o1-mini',
  'o3-mini',
  'llama-3.1',
  'llama-3.3',
  'gpt-3.5',
  'text-embedding',
  'embed',
  'rerank',
];

const VISION_PATTERNS = [
  'deepseek-v4',
  'gpt-4o',
  'gpt-4.1',
  'gpt-4-vision',
  'gpt-4-turbo',
  'gpt-5',
  'claude-3',
  'claude-4',
  'claude-sonnet',
  'claude-opus',
  'claude-haiku',
  'gemma-3',
  'gemma-4',
  'llama-4',
  'llama-3.2-11b-vision',
  'llama-3.2-90b-vision',
  'pixtral',
  'mistral-small-3',
  'mistral-large-2',
  'grok-2-vision',
  'grok-3',
  'grok-4',
  'kimi-k2.5',
  'kimi-k3',
  'glm-4v',
  'glm-4.5v',
  'glm-5',
  'qwen2-vl',
  'qwen2.5-vl',
  'qwen3-vl',
  '-vl',
  '_vl',
  'vision',
  'multimodal',
  'internvl',
  'llava',
  'minicpm-v',
  'molmo',
  'cogvlm',
  'phi-3.5-vision',
  'phi-4-multimodal',
];

const AUDIO_PATTERNS = [
  'gpt-4o-audio',
  'gpt-audio',
  'voxtral',
  'phi-4-multimodal',
  'ultravox',
  'whisper',
];

/**
 * Mirrors Agent-Core's detect_model_input_modalities so the UI accurately displays
 * which modalities are handled natively by the Primary LLM vs delegated to Accessory sub-agents.
 */
export function detectModelInputModalities(
  _provider?: string | null,
  model?: string | null,
  explicitModalities?: string[] | null,
): Set<'text' | MediaModality> {
  if (Array.isArray(explicitModalities) && explicitModalities.length > 0) {
    const out = new Set<'text' | MediaModality>(['text']);
    for (const item of explicitModalities) {
      const clean = String(item || '').trim().toLowerCase();
      if (clean === 'image' || clean === 'video' || clean === 'audio') {
        out.add(clean);
      }
    }
    return out;
  }

  const modalities = new Set<'text' | MediaModality>(['text']);
  const m = (model || '').trim().toLowerCase();
  if (!m) return modalities;

  if (OMNI_ALL_PATTERNS.some((pat) => m.includes(pat))) {
    if (!m.includes('embedding') && !m.includes('imagen') && !m.includes('tts')) {
      modalities.add('image');
      modalities.add('audio');
      modalities.add('video');
      return modalities;
    }
  }

  if (
    TEXT_ONLY_OVERRIDES.some((pat) => m.includes(pat)) &&
    !['vl', 'vision', 'omni', 'audio', 'multimodal'].some((tag) => m.includes(tag))
  ) {
    return modalities;
  }

  if (/(?:^|\/)(?:o1|o3|o4)(?:$|[-_:]|\b)/.test(m) && !m.includes('mini')) {
    modalities.add('image');
  }

  if (VISION_PATTERNS.some((pat) => m.includes(pat))) {
    modalities.add('image');
  }

  if (AUDIO_PATTERNS.some((pat) => m.includes(pat))) {
    modalities.add('audio');
  }

  if (m.includes('video') && !m.includes('embed')) {
    modalities.add('video');
  }

  return modalities;
}

export function classifyFileModality(mimeType: string, filename: string): MediaModality | null {
  const mime = (mimeType || '').toLowerCase().split(';')[0].trim();
  const ext = (filename || '').toLowerCase().split('.').pop() || '';
  if (mime === 'image/svg+xml' || ext === 'svg') return null;
  if (mime.startsWith('image/') || ['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(ext)) {
    return 'image';
  }
  if (
    mime.startsWith('audio/') ||
    ['wav', 'mp3', 'ogg', 'opus', 'flac', 'm4a', 'aac'].includes(ext)
  ) {
    return 'audio';
  }
  if (mime.startsWith('video/') || ['mp4', 'webm', 'mov', 'mkv', 'avi', 'm4v'].includes(ext)) {
    return 'video';
  }
  return null;
}

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function validateLLMConfig(
  config: PersistedLLMConfig,
  scope: 'all' | 'llm' | 'accessories' = 'all',
): { valid: boolean; error?: string } {
  const roles: Array<{ label: string; cfg: RoleLLMConfig }> = [];

  if (scope === 'all' || scope === 'llm') {
    roles.push(
      { label: 'Main Agent', cfg: config.primary },
      { label: 'Sub-Agent', cfg: config.worker },
    );
  }

  if ((scope === 'all' || scope === 'accessories') && config.accessories) {
    const modLabels: Array<{ key: MediaModality; label: string }> = [
      { key: 'image', label: 'Image Accessory' },
      { key: 'video', label: 'Video Accessory' },
      { key: 'audio', label: 'Audio Accessory' },
    ];
    for (const { key, label } of modLabels) {
      const entry = config.accessories[key];
      if (entry && entry.enabled) {
        roles.push({ label, cfg: entry });
      }
    }
  }

  for (const { label, cfg } of roles) {
    if (!cfg.provider.trim()) {
      return { valid: false, error: `${label}: model provider is required.` };
    }
    if (!cfg.model.trim()) {
      return { valid: false, error: `${label}: model identifier is required.` };
    }
    if (!cfg.base_url.trim() || !/^https?:\/\/.+/.test(cfg.base_url.trim())) {
      return { valid: false, error: `${label}: base URL must start with http:// or https://.` };
    }
    if (Number.isNaN(cfg.temperature) || cfg.temperature < 0 || cfg.temperature > 2) {
      return { valid: false, error: `${label}: temperature must be between 0 and 2.` };
    }
    if (Number.isNaN(cfg.max_tokens) || cfg.max_tokens < 128 || cfg.max_tokens > 20_000_000) {
      return { valid: false, error: `${label}: max output tokens must be between 128 and 20,000,000.` };
    }
  }

  if (
    (scope === 'all' || scope === 'llm') &&
    (!Number.isFinite(config.max_turns) ||
      !Number.isInteger(config.max_turns) ||
      config.max_turns < MIN_MAX_TURNS ||
      config.max_turns > MAX_MAX_TURNS)
  ) {
    return {
      valid: false,
      error: `Max execution turns must be a whole number between ${MIN_MAX_TURNS} and ${MAX_MAX_TURNS}.`,
    };
  }

  return { valid: true };
}
