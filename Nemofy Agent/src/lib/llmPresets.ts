import type { PersistedLLMConfig, ProviderModelOption, RoleLLMConfig } from '../types/agent.ts';
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

export function getProviderPreset(providerId: string): ProviderPreset | undefined {
  const canonical = canonicalizeProviderId(providerId);
  return LLM_PROVIDER_PRESETS.find((p) => p.id === canonical);
}

export function applyProviderPreset(current: RoleLLMConfig, nextProviderId: string): RoleLLMConfig {
  const canonicalNext = canonicalizeProviderId(nextProviderId);
  const preset = getProviderPreset(canonicalNext);
  if (!preset) {
    return { ...current, provider: canonicalNext };
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
    ? current.model || preset.defaultModels[0] || ''
    : preset.defaultModels[0] || '';

  const profile = getThinkingProfile(preset.id, nextModel);
  return {
    ...current,
    provider: preset.id,
    base_url: nextBaseUrl,
    model: nextModel,
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

export function validateLLMConfig(config: PersistedLLMConfig): { valid: boolean; error?: string } {
  const roles: Array<{ label: string; cfg: RoleLLMConfig }> = [
    { label: 'Main Agent', cfg: config.primary },
    { label: 'Sub-Agent', cfg: config.worker },
  ];

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
    if (Number.isNaN(cfg.max_tokens) || cfg.max_tokens < 128 || cfg.max_tokens > 128000) {
      return { valid: false, error: `${label}: max tokens must be between 128 and 128,000.` };
    }
  }

  if (Number.isNaN(config.max_turns) || config.max_turns < 1 || config.max_turns > 100) {
    return { valid: false, error: 'Max execution turns must be between 1 and 100.' };
  }

  return { valid: true };
}
