export type ThinkingLevel = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export interface ThinkingOption {
  value: ThinkingLevel;
  label: string;
  shortLabel: string;
  description: string;
}

export interface ModelThinkingProfile {
  supported: boolean;
  familyLabel: string;
  options: ThinkingOption[];
  defaultLevel: ThinkingLevel;
  canDisable: boolean;
}

const OPTION_META: Record<ThinkingLevel, ThinkingOption> = {
  off: {
    value: 'off',
    label: 'Off',
    shortLabel: 'Off',
    description: 'Direct response without extra reasoning tokens',
  },
  minimal: {
    value: 'minimal',
    label: 'Minimal',
    shortLabel: 'Minimal',
    description: 'Fastest light check before answering',
  },
  low: {
    value: 'low',
    label: 'Low',
    shortLabel: 'Low',
    description: 'Brief reasoning for straightforward tasks',
  },
  medium: {
    value: 'medium',
    label: 'Medium',
    shortLabel: 'Medium',
    description: 'Balanced reasoning depth and response speed',
  },
  high: {
    value: 'high',
    label: 'High',
    shortLabel: 'High',
    description: 'Deep multi-step reasoning for harder problems',
  },
  xhigh: {
    value: 'xhigh',
    label: 'XHigh',
    shortLabel: 'XHigh',
    description: 'Extra-high reasoning budget for toughest tasks',
  },
  max: {
    value: 'max',
    label: 'Max',
    shortLabel: 'Max',
    description: 'Maximum thinking depth supported by this model',
  },
};

function makeOptions(levels: ThinkingLevel[]): ThinkingOption[] {
  return levels.map((lvl) => OPTION_META[lvl]);
}

/**
 * Strip provider/gateway prefixes (e.g. "openai/gpt-5.2", "accounts/fireworks/models/deepseek-r1")
 * to normalize the underlying model identifier for capability matching.
 */
export function normalizeModelSlug(model: string): string {
  const clean = (model || '').trim().toLowerCase();
  if (!clean) return '';
  const parts = clean.split('/');
  return parts[parts.length - 1] || clean;
}

/**
 * Resolve the model-specific thinking / reasoning effort profile.
 * Different models expose different levels (e.g., max vs xhigh, low/high/max,
 * low/medium/high, low/high, or no thinking support at all).
 */
export function getThinkingProfile(provider: string, model: string): ModelThinkingProfile {
  const slug = normalizeModelSlug(model);
  const prov = (provider || '').trim().toLowerCase();

  // 1. OpenAI GPT-5.2 / 5.3 / 5.4 / o3-pro / codex-max -> supports xhigh (no max)
  if (
    /^gpt-5\.[2-9]/.test(slug) ||
    slug.includes('gpt-5-pro') ||
    slug.includes('o3-pro') ||
    slug.includes('codex-max') ||
    slug.includes('codex-5')
  ) {
    return {
      supported: true,
      familyLabel: 'OpenAI Extended (xhigh)',
      options: makeOptions(['off', 'low', 'medium', 'high', 'xhigh']),
      defaultLevel: 'medium',
      canDisable: true,
    };
  }

  // 2. OpenAI GPT-5 / GPT-5-mini / GPT-5-nano -> minimal, low, medium, high
  if (/^gpt-5(\b|-)/.test(slug)) {
    return {
      supported: true,
      familyLabel: 'OpenAI GPT-5',
      options: makeOptions(['minimal', 'low', 'medium', 'high']),
      defaultLevel: 'medium',
      canDisable: false,
    };
  }

  // 3. OpenAI o1 / o3 / o4 reasoning series -> low, medium, high (always-on reasoning)
  if (/^o[134]($|[-.])/.test(slug)) {
    return {
      supported: true,
      familyLabel: 'OpenAI o-series',
      options: makeOptions(['low', 'medium', 'high']),
      defaultLevel: 'medium',
      canDisable: false,
    };
  }

  // 4. Anthropic Claude 4 / 4.5 / 4.6 / 4.7 / 3.7 Sonnet -> off, low, medium, high, max (no xhigh)
  if (
    slug.includes('claude-opus-4') ||
    slug.includes('claude-sonnet-4') ||
    slug.includes('claude-4') ||
    slug.includes('claude-3-7-sonnet') ||
    slug.includes('claude-3.7-sonnet')
  ) {
    return {
      supported: true,
      familyLabel: 'Claude Adaptive (max)',
      options: makeOptions(['off', 'low', 'medium', 'high', 'max']),
      defaultLevel: 'medium',
      canDisable: true,
    };
  }

  // 5. Google Gemini Pro / Gemini 3 / Gemini 4 -> low, medium, high, max
  if (
    slug.includes('gemini-2.5-pro') ||
    slug.includes('gemini-3-pro') ||
    slug.includes('gemini-3.1-pro') ||
    slug.includes('gemini-4')
  ) {
    return {
      supported: true,
      familyLabel: 'Gemini Pro (max)',
      options: makeOptions(['low', 'medium', 'high', 'max']),
      defaultLevel: 'medium',
      canDisable: false,
    };
  }

  // 6. Google Gemini Flash thinking models -> off, minimal, low, medium, high
  if (slug.includes('gemini-2.5-flash') || slug.includes('gemini-3-flash')) {
    return {
      supported: true,
      familyLabel: 'Gemini Flash',
      options: makeOptions(['off', 'minimal', 'low', 'medium', 'high']),
      defaultLevel: 'off',
      canDisable: true,
    };
  }

  // 7. River AI & Hybrid Low/High/Max models: DeepSeek V4.1, GLM-5.3, Kimi-K3 / K3.1
  // Verified live on River AI: accepts none (off), low, high, max (rejects medium & xhigh)
  if (
    slug.includes('deepseek-v4') ||
    slug.includes('glm-5') ||
    slug.includes('kimi-k3')
  ) {
    return {
      supported: true,
      familyLabel: 'Hybrid Thinking (max)',
      options: makeOptions(['off', 'low', 'high', 'max']),
      defaultLevel: 'off',
      canDisable: true,
    };
  }

  // 8. DeepSeek R1 / deepseek-reasoner -> always-on reasoning: low, medium, high, max
  if (slug.includes('deepseek-r1') || slug.includes('deepseek-reasoner')) {
    return {
      supported: true,
      familyLabel: 'DeepSeek Reasoner (max)',
      options: makeOptions(['low', 'medium', 'high', 'max']),
      defaultLevel: 'high',
      canDisable: false,
    };
  }

  // 9. DeepSeek V3.1 / V3.2 / deepseek-chat -> off, low, medium, high
  if (slug.includes('deepseek-v3') || slug.includes('deepseek-chat')) {
    return {
      supported: true,
      familyLabel: 'DeepSeek Hybrid',
      options: makeOptions(['off', 'low', 'medium', 'high']),
      defaultLevel: 'off',
      canDisable: true,
    };
  }

  // 10. xAI Grok 3 Mini -> only low and high
  if (slug.includes('grok-3-mini')) {
    return {
      supported: true,
      familyLabel: 'Grok 3 Mini',
      options: makeOptions(['low', 'high']),
      defaultLevel: 'low',
      canDisable: false,
    };
  }

  // 11. xAI Grok 4 -> low, medium, high, xhigh
  if (slug.includes('grok-4') || slug.includes('grok-3-reason')) {
    return {
      supported: true,
      familyLabel: 'Grok Reasoning (xhigh)',
      options: makeOptions(['low', 'medium', 'high', 'xhigh']),
      defaultLevel: 'medium',
      canDisable: false,
    };
  }

  // 12. Qwen3 / QwQ / NVIDIA Nemotron reasoning models -> off, low, medium, high
  if (
    slug.includes('qwen3') ||
    slug.includes('qwen-3') ||
    slug.includes('qwq') ||
    slug.includes('nemotron-3') ||
    slug.includes('nemotron-super') ||
    slug.includes('nemotron-ultra') ||
    slug.includes('nemotron-nano')
  ) {
    return {
      supported: true,
      familyLabel: slug.includes('nemotron') ? 'NVIDIA Nemotron Thinking' : 'Hybrid Thinking',
      options: makeOptions(['off', 'low', 'medium', 'high']),
      defaultLevel: 'off',
      canDisable: true,
    };
  }

  // 13. Explicitly non-thinking standard models
  if (
    /^gpt-4($|o|-|.)/.test(slug) ||
    slug.includes('gpt-3.5') ||
    slug.includes('claude-3-5-') ||
    slug.includes('claude-3.5-') ||
    slug.includes('claude-3-haiku') ||
    slug.includes('claude-3-opus') ||
    slug.includes('llama-3') ||
    slug.includes('llama3') ||
    slug.includes('qwen2.5') ||
    slug.includes('qwen-2.5') ||
    slug.includes('gemini-2.0') ||
    slug.includes('gemini-1.5')
  ) {
    return {
      supported: false,
      familyLabel: 'Standard (No Thinking)',
      options: [
        {
          value: 'off',
          label: 'Standard',
          shortLabel: 'Standard',
          description: 'This model does not expose configurable thinking levels',
        },
      ],
      defaultLevel: 'off',
      canDisable: true,
    };
  }

  // 14. Provider-aware adaptive fallback for custom / newly released models
  if (prov === 'anthropic') {
    return {
      supported: true,
      familyLabel: 'Anthropic Custom',
      options: makeOptions(['off', 'low', 'medium', 'high', 'max']),
      defaultLevel: 'off',
      canDisable: true,
    };
  }

  return {
    supported: true,
    familyLabel: 'Adaptive Thinking',
    options: makeOptions(['off', 'low', 'medium', 'high']),
    defaultLevel: 'off',
    canDisable: true,
  };
}

const LEVEL_RANK: Record<ThinkingLevel, number> = {
  off: 0,
  minimal: 1,
  low: 2,
  medium: 3,
  high: 4,
  xhigh: 5,
  max: 6,
};

/**
 * Adapt any previously selected thinking level to the closest valid option
 * supported by the newly active model profile.
 *
 * Examples:
 * - "max" on Claude -> "xhigh" on GPT-5.2 -> "high" on o3-mini
 * - "xhigh" on GPT-5.2 -> "max" on Claude / DeepSeek-V4.1-Flash -> "high" on o3-mini
 * - "medium" on GPT-5 -> "high" on DeepSeek-V4.1-Flash (which has low/high/max)
 * - "off" -> "low" on always-on reasoning models like o3-mini
 * - any level -> "off" on non-thinking models like gpt-4o
 */
export function normalizeThinkingLevel(
  profile: ModelThinkingProfile,
  requested?: string | null,
): ThinkingLevel {
  if (!profile.supported || profile.options.length === 0) {
    return 'off';
  }

  const allowed = profile.options.map((o) => o.value);
  const raw = (requested || '').trim().toLowerCase();
  const normalized = (raw === 'none' || raw === '' ? 'off' : raw) as ThinkingLevel;

  if (allowed.includes(normalized)) {
    return normalized;
  }

  // Exact semantic equivalents across vendor terminology
  if (normalized === 'max' && allowed.includes('xhigh')) return 'xhigh';
  if (normalized === 'xhigh' && allowed.includes('max')) return 'max';

  // If user asked for off/none on an always-on thinking model, pick lowest non-off or default
  if (normalized === 'off') {
    if (allowed.includes('minimal')) return 'minimal';
    if (allowed.includes('low')) return 'low';
    return profile.defaultLevel;
  }

  // Otherwise map to the closest available rank (preferring upward when tied, e.g. medium -> high)
  const targetRank = LEVEL_RANK[normalized];
  if (targetRank === undefined) {
    return profile.defaultLevel;
  }

  let best = profile.defaultLevel;
  let bestDist = Infinity;
  for (const candidate of allowed) {
    if (candidate === 'off' && targetRank > 0) continue;
    const dist = Math.abs(LEVEL_RANK[candidate] - targetRank);
    if (dist < bestDist || (dist === bestDist && LEVEL_RANK[candidate] > LEVEL_RANK[best])) {
      best = candidate;
      bestDist = dist;
    }
  }
  return best;
}

export function formatThinkingBadge(level?: string | null): string | null {
  const clean = (level || '').trim().toLowerCase() as ThinkingLevel;
  if (!clean || clean === 'off' || clean === ('none' as string)) return null;
  return OPTION_META[clean]?.shortLabel || clean;
}
