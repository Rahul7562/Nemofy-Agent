import type { PersistedLLMConfig, RoleLLMConfig } from '../types/agent.ts';

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
      'deepseek-ai/DeepSeek-V3.2',
      'openai/gpt-4o',
    ],
    envVars: ['RIVER_AI_API'],
    requiresApiKey: true,
  },
  {
    id: 'openai',
    label: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    defaultModels: ['gpt-4o', 'gpt-4o-mini', 'o3-mini', 'gpt-4.1'],
    envVars: ['OPENAI_API_KEY'],
    requiresApiKey: true,
  },
  {
    id: 'anthropic',
    label: 'Anthropic (OpenAI-Compat)',
    baseUrl: 'https://api.anthropic.com/v1',
    defaultModels: ['claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022'],
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
      'anthropic/claude-3.7-sonnet',
      'openai/gpt-4o',
      'deepseek/deepseek-chat-v3',
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

export const REASONING_EFFORT_OPTIONS = ['off', 'low', 'medium', 'high'] as const;

export function getProviderPreset(providerId: string): ProviderPreset | undefined {
  return LLM_PROVIDER_PRESETS.find((p) => p.id === providerId.toLowerCase());
}

export function applyProviderPreset(current: RoleLLMConfig, nextProviderId: string): RoleLLMConfig {
  const preset = getProviderPreset(nextProviderId);
  if (!preset) {
    return { ...current, provider: nextProviderId };
  }
  return {
    ...current,
    provider: preset.id,
    base_url: preset.baseUrl || current.base_url,
    model: preset.defaultModels[0] || current.model,
  };
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
