import type {
  ChatThread,
  ContextCategoryBreakdown,
  ContextCategoryKey,
  ContextClassification,
  ContextUsageSnapshot,
  LiveAgentActivity,
  PersistedLLMConfig,
} from '../types/agent.ts';

export const AUTO_COMPACT_THRESHOLD_PERCENT = 80;
export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const MIN_CONTEXT_WINDOW = 512;
export const DEFAULT_MAX_OUTPUT_TOKENS = 16_384;
export const MIN_MAX_OUTPUT_TOKENS = 256;

// Mirrors Agent-Core/agent_core/context_manager.py _MODEL_CONTEXT_WINDOWS
const MODEL_CONTEXT_WINDOWS: ReadonlyArray<readonly [string, number]> = [
  // Google Gemini
  ['gemini-1.5-pro', 2_000_000],
  ['gemini-1.5-flash', 1_000_000],
  ['gemini-2.5-pro', 1_048_576],
  ['gemini-2.5-flash', 1_048_576],
  ['gemini-2.0', 1_048_576],
  ['gemini-3', 1_048_576],
  ['gemini-4', 1_048_576],
  ['gemini', 1_048_576],
  // Meta Llama 4 & 3
  ['llama-4-scout', 10_000_000],
  ['llama-4-maverick', 1_000_000],
  ['llama-4', 1_000_000],
  ['llama-3.3', 131_072],
  ['llama-3.2', 131_072],
  ['llama-3.1', 131_072],
  ['llama3.3', 131_072],
  ['llama3.1', 131_072],
  // OpenAI GPT-5 / GPT-4.1 / GPT-4o / o-series
  ['gpt-5.2', 400_000],
  ['gpt-5', 400_000],
  ['gpt-4.1', 1_047_576],
  ['o4-mini', 200_000],
  ['o3-mini', 200_000],
  ['o3', 200_000],
  ['o1', 200_000],
  ['gpt-4o', 128_000],
  ['gpt-4-turbo', 128_000],
  ['gpt-3.5', 16_385],
  // Anthropic Claude
  ['claude-opus-4', 200_000],
  ['claude-sonnet-4', 200_000],
  ['claude-haiku-4', 200_000],
  ['claude-3-7', 200_000],
  ['claude-3.7', 200_000],
  ['claude-3-5', 200_000],
  ['claude-3.5', 200_000],
  ['claude-3', 200_000],
  ['claude', 200_000],
  // Moonshot Kimi
  ['kimi-k3', 256_000],
  ['kimi-k2.5', 256_000],
  ['kimi-k2', 131_072],
  ['kimi', 131_072],
  // NVIDIA Nemotron
  ['nemotron-3.5', 131_072],
  ['nemotron-3-ultra', 131_072],
  ['nemotron-3-super', 131_072],
  ['nemotron-3-nano', 131_072],
  ['nemotron', 131_072],
  // DeepSeek
  ['deepseek-v4', 128_000],
  ['deepseek-v3', 128_000],
  ['deepseek-r1', 128_000],
  ['deepseek-chat', 128_000],
  ['deepseek-reasoner', 128_000],
  ['deepseek', 128_000],
  // Zhipu GLM
  ['glm-5', 128_000],
  ['glm-4.5', 128_000],
  ['glm-4', 128_000],
  // Qwen
  ['qwen3', 131_072],
  ['qwen2.5', 131_072],
  ['qwq', 131_072],
  ['qwen', 131_072],
  // xAI Grok
  ['grok-4', 256_000],
  ['grok-3', 131_072],
  ['grok-2', 131_072],
  ['grok', 131_072],
  // Mistral
  ['mistral-large-4', 256_000],
  ['mistral-large', 128_000],
  ['mistral-small-3', 128_000],
  ['pixtral', 128_000],
  ['codestral', 256_000],
  ['mistral', 128_000],
  // Baidu ERNIE
  ['ernie-5', 128_000],
  ['ernie-4', 128_000],
  ['ernie', 128_000],
  // Gemma / Phi
  ['gemma-4', 128_000],
  ['gemma-3', 128_000],
  ['gemma', 32_768],
  ['phi-4', 128_000],
  ['phi-3.5', 128_000],
];

export const CONTEXT_CATEGORY_META: ReadonlyArray<{
  key: ContextCategoryKey;
  label: string;
  description: string;
  color: string;
}> = [
  {
    key: 'system_prompt',
    label: 'System Prompt',
    description: 'Core agent instructions, guardrails, and active plan',
    color: '#6B6660',
  },
  {
    key: 'tools',
    label: 'Tools & MCPs',
    description: 'Bound sandbox, web, sub-agent, and MCP tool definitions',
    color: '#2E7D6B',
  },
  {
    key: 'skills',
    label: 'Skills',
    description: 'Skill catalog and active domain skill guidelines',
    color: '#8B5CF6',
  },
  {
    key: 'user_messages',
    label: 'User Messages',
    description: 'Conversation prompts and attached media overhead',
    color: '#D95A2B',
  },
  {
    key: 'assistant_messages',
    label: 'Assistant & Reasoning',
    description: 'Agent responses, tool calls, and retained reasoning',
    color: '#2563EB',
  },
  {
    key: 'tool_outputs',
    label: 'Tool Outputs & Sub-Agents',
    description: 'File reads, bash output, web results, and sub-agent reports',
    color: '#C78A28',
  },
  {
    key: 'compacted_summary',
    label: 'Compacted Memory',
    description: 'Rolling memory summary preserved from 80% auto-compaction',
    color: '#B83B2E',
  },
];

// Mirrors Agent-Core/agent_core/context_manager.py _MODEL_MAX_OUTPUT_TOKENS
const MODEL_MAX_OUTPUT_TOKENS: ReadonlyArray<readonly [string, number]> = [
  // Google Gemini
  ['gemini-2.5-pro', 65_536],
  ['gemini-2.5-flash', 65_536],
  ['gemini-3', 65_536],
  ['gemini-4', 65_536],
  ['gemini-2.0', 8_192],
  ['gemini-1.5', 8_192],
  ['gemini', 65_536],
  // OpenAI GPT-5 / o-series / GPT-4.1 / GPT-4o
  ['gpt-5.2', 128_000],
  ['gpt-5', 128_000],
  ['o4-mini', 100_000],
  ['o3-mini', 100_000],
  ['o3', 100_000],
  ['o1', 100_000],
  ['gpt-4.1', 32_768],
  ['gpt-4o', 16_384],
  ['gpt-4-turbo', 4_096],
  ['gpt-3.5', 4_096],
  // Anthropic Claude
  ['claude-opus-4-6', 128_000],
  ['claude-opus-4.6', 128_000],
  ['claude-sonnet-4-6', 64_000],
  ['claude-sonnet-4.6', 64_000],
  ['claude-opus-4', 32_000],
  ['claude-sonnet-4', 64_000],
  ['claude-haiku-4', 64_000],
  ['claude-3-7', 64_000],
  ['claude-3.7', 64_000],
  ['claude-3-5', 8_192],
  ['claude-3.5', 8_192],
  ['claude-3', 4_096],
  ['claude', 64_000],
  // DeepSeek
  ['deepseek-v4', 32_768],
  ['deepseek-r1', 32_768],
  ['deepseek-reasoner', 32_768],
  ['deepseek-v3', 8_192],
  ['deepseek-chat', 8_192],
  ['deepseek', 16_384],
  // Moonshot Kimi
  ['kimi-k3', 32_768],
  ['kimi-k2.5', 32_768],
  ['kimi-k2', 16_384],
  ['kimi', 16_384],
  // NVIDIA Nemotron
  ['nemotron-3.5', 32_768],
  ['nemotron-3-ultra', 32_768],
  ['nemotron-3-super', 32_768],
  ['nemotron-3-nano', 32_768],
  ['nemotron', 16_384],
  // Qwen
  ['qwen3', 32_768],
  ['qwq', 32_768],
  ['qwen2.5', 8_192],
  ['qwen', 16_384],
  // xAI Grok
  ['grok-4', 32_768],
  ['grok-3', 32_768],
  ['grok-2', 16_384],
  ['grok', 16_384],
  // Meta Llama 4 & 3
  ['llama-4', 16_384],
  ['llama-3.3', 8_192],
  ['llama-3.2', 8_192],
  ['llama-3.1', 8_192],
  ['llama3.3', 8_192],
  ['llama3.1', 8_192],
  // Mistral
  ['mistral-large-4', 32_768],
  ['codestral', 32_768],
  ['mistral-large', 16_384],
  ['mistral-small-3', 16_384],
  ['pixtral', 16_384],
  ['mistral', 16_384],
  // Zhipu GLM
  ['glm-5', 16_384],
  ['glm-4.5', 16_384],
  ['glm-4', 8_192],
  // Baidu ERNIE
  ['ernie-5', 16_384],
  ['ernie-4', 8_192],
  ['ernie', 8_192],
  // Gemma / Phi
  ['gemma-4', 8_192],
  ['gemma-3', 8_192],
  ['gemma', 8_192],
  ['phi-4', 16_384],
  ['phi-3.5', 8_192],
];

export function resolveModelContextWindow(
  provider?: string | null,
  model?: string | null,
  explicitContextWindow?: number | null,
): { contextWindow: number; source: string } {
  if (
    typeof explicitContextWindow === 'number' &&
    Number.isFinite(explicitContextWindow) &&
    explicitContextWindow >= MIN_CONTEXT_WINDOW
  ) {
    return { contextWindow: Math.round(explicitContextWindow), source: 'provider_catalog' };
  }

  const cleanModel = (model || '').trim().toLowerCase();
  if (cleanModel) {
    const bare = cleanModel.split('/').pop()?.split(':')[0] || cleanModel;
    for (const target of [cleanModel, bare]) {
      for (const [pattern, limit] of MODEL_CONTEXT_WINDOWS) {
        if (target.includes(pattern)) {
          return { contextWindow: limit, source: 'model_registry' };
        }
      }
    }
  }

  const cleanProvider = (provider || '').trim().toLowerCase();
  if (cleanProvider === 'ollama' || cleanProvider === 'local' || cleanProvider === 'vllm') {
    return { contextWindow: 32_768, source: 'provider_default' };
  }

  return { contextWindow: DEFAULT_CONTEXT_WINDOW, source: 'default' };
}

export function resolveModelMaxOutputTokens(
  provider?: string | null,
  model?: string | null,
  explicitMaxOutputTokens?: number | null,
): { maxOutputTokens: number; source: string } {
  if (
    typeof explicitMaxOutputTokens === 'number' &&
    Number.isFinite(explicitMaxOutputTokens) &&
    explicitMaxOutputTokens >= MIN_MAX_OUTPUT_TOKENS
  ) {
    return { maxOutputTokens: Math.round(explicitMaxOutputTokens), source: 'provider_catalog' };
  }

  const cleanModel = (model || '').trim().toLowerCase();
  if (cleanModel) {
    const bare = cleanModel.split('/').pop()?.split(':')[0] || cleanModel;
    for (const target of [cleanModel, bare]) {
      for (const [pattern, limit] of MODEL_MAX_OUTPUT_TOKENS) {
        if (target.includes(pattern)) {
          return { maxOutputTokens: limit, source: 'model_registry' };
        }
      }
    }
  }

  const cleanProvider = (provider || '').trim().toLowerCase();
  if (cleanProvider === 'ollama' || cleanProvider === 'local' || cleanProvider === 'vllm') {
    return { maxOutputTokens: 8_192, source: 'provider_default' };
  }

  return { maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS, source: 'default' };
}

export function resolveModelLimits(
  provider?: string | null,
  model?: string | null,
  explicitContextWindow?: number | null,
  explicitMaxOutputTokens?: number | null,
): {
  contextWindow: number;
  contextWindowSource: string;
  maxOutputTokens: number;
  maxOutputTokensSource: string;
} {
  const ctx = resolveModelContextWindow(provider, model, explicitContextWindow);
  const out = resolveModelMaxOutputTokens(provider, model, explicitMaxOutputTokens);
  // Output limit can never exceed total context window
  const clampedOut = Math.min(out.maxOutputTokens, ctx.contextWindow);
  return {
    contextWindow: ctx.contextWindow,
    contextWindowSource: ctx.source,
    maxOutputTokens: clampedOut,
    maxOutputTokensSource: out.source,
  };
}

export function classifyContextUsage(usagePercent: number): ContextClassification {
  const pct = Math.max(0, Number.isFinite(usagePercent) ? usagePercent : 0);
  if (pct >= AUTO_COMPACT_THRESHOLD_PERCENT) {
    return {
      level: 'critical',
      label: 'Auto-Compacting',
      description:
        'Reached the 80% threshold — older turns are automatically compacted to preserve memory.',
    };
  }
  if (pct >= 70) {
    return {
      level: 'high',
      label: 'High',
      description: 'Approaching the 80% auto-compaction threshold.',
    };
  }
  if (pct >= 50) {
    return {
      level: 'moderate',
      label: 'Moderate',
      description: 'Healthy active context with headroom before 80% auto-compaction.',
    };
  }
  return {
    level: 'optimal',
    label: 'Optimal',
    description: 'Light context footprint with full recall clarity.',
  };
}

export function formatTokenCount(tokens: number): string {
  const safe = Math.max(0, Math.round(Number.isFinite(tokens) ? tokens : 0));
  if (safe >= 1_000_000) {
    const millions = safe / 1_000_000;
    const rounded = millions >= 10 || Number.isInteger(millions) ? millions.toFixed(0) : millions.toFixed(1);
    return `${rounded.replace(/\.0$/, '')}M`;
  }
  if (safe >= 10_000) {
    return `${Math.round(safe / 1000)}k`;
  }
  if (safe >= 1_000) {
    return `${(safe / 1000).toFixed(1).replace(/\.0$/, '')}k`;
  }
  return String(safe);
}

export function estimateTextTokens(text?: string | null): number {
  if (!text) return 0;
  const s = String(text);
  if (!s) return 0;
  return Math.max(1, Math.ceil(s.length / 3.7));
}

function roundOneDecimal(val: number): number {
  return Math.round(val * 10) / 10;
}

export function rescaleSnapshotForModel(
  snapshot: ContextUsageSnapshot,
  provider?: string | null,
  model?: string | null,
  explicitContextWindow?: number | null,
  explicitMaxOutputTokens?: number | null,
  configuredMaxTokens?: number | null,
): ContextUsageSnapshot {
  const activeModel = (model || '').trim();
  const activeProvider = (provider || '').trim();
  const sameModel =
    activeModel.length > 0 &&
    snapshot.model.trim().toLowerCase() === activeModel.toLowerCase();

  let targetWindow = snapshot.context_window;
  let targetSource = snapshot.context_window_source;

  if (
    typeof explicitContextWindow === 'number' &&
    Number.isFinite(explicitContextWindow) &&
    explicitContextWindow >= MIN_CONTEXT_WINDOW
  ) {
    targetWindow = Math.round(explicitContextWindow);
    targetSource = 'provider_catalog';
  } else if (!sameModel && activeModel) {
    const resolved = resolveModelContextWindow(activeProvider, activeModel, explicitContextWindow);
    targetWindow = resolved.contextWindow;
    targetSource = resolved.source;
  } else if (!targetWindow || targetWindow < MIN_CONTEXT_WINDOW) {
    const resolved = resolveModelContextWindow(activeProvider, activeModel, explicitContextWindow);
    targetWindow = resolved.contextWindow;
    targetSource = resolved.source;
  }

  let targetMaxOutput = snapshot.max_output_tokens;
  let targetMaxOutputSource = snapshot.max_output_tokens_source;
  if (
    typeof explicitMaxOutputTokens === 'number' &&
    Number.isFinite(explicitMaxOutputTokens) &&
    explicitMaxOutputTokens >= MIN_MAX_OUTPUT_TOKENS
  ) {
    targetMaxOutput = Math.min(Math.round(explicitMaxOutputTokens), targetWindow);
    targetMaxOutputSource = 'provider_catalog';
  } else if (!sameModel || !targetMaxOutput || targetMaxOutput < MIN_MAX_OUTPUT_TOKENS) {
    const resolvedOut = resolveModelMaxOutputTokens(
      activeProvider,
      activeModel || snapshot.model,
      explicitMaxOutputTokens,
    );
    targetMaxOutput = Math.min(resolvedOut.maxOutputTokens, targetWindow);
    targetMaxOutputSource = resolvedOut.source;
  } else {
    targetMaxOutput = Math.min(targetMaxOutput, targetWindow);
  }

  const effectiveConfiguredMaxTokens =
    typeof configuredMaxTokens === 'number' && Number.isFinite(configuredMaxTokens) && configuredMaxTokens > 0
      ? Math.min(Math.round(configuredMaxTokens), targetMaxOutput)
      : snapshot.configured_max_tokens
        ? Math.min(Math.round(snapshot.configured_max_tokens), targetMaxOutput)
        : targetMaxOutput;

  const usedTokens = Math.max(0, Math.round(snapshot.used_tokens || 0));
  const freeTokens = Math.max(0, targetWindow - usedTokens);
  const usagePercent = roundOneDecimal(Math.min(100, (usedTokens / Math.max(1, targetWindow)) * 100));
  const classification = classifyContextUsage(usagePercent);

  const byKey = new Map<string, ContextCategoryBreakdown>();
  for (const cat of snapshot.categories || []) {
    byKey.set(cat.key, cat);
  }

  const categories: ContextCategoryBreakdown[] = CONTEXT_CATEGORY_META.map((meta) => {
    const existing = byKey.get(meta.key);
    const toks = Math.max(0, Math.round(existing?.tokens || 0));
    return {
      key: meta.key,
      label: existing?.label || meta.label,
      description: existing?.description || meta.description,
      tokens: toks,
      percent: roundOneDecimal((toks / Math.max(1, targetWindow)) * 100),
      share_of_used: usedTokens > 0 ? roundOneDecimal((toks / usedTokens) * 100) : 0,
    };
  });

  return {
    ...snapshot,
    model: activeModel || snapshot.model,
    provider: activeProvider || snapshot.provider,
    context_window: targetWindow,
    context_window_source: targetSource,
    max_output_tokens: targetMaxOutput,
    max_output_tokens_source: targetMaxOutputSource || 'default',
    configured_max_tokens: effectiveConfiguredMaxTokens,
    used_tokens: usedTokens,
    free_tokens: freeTokens,
    usage_percent: usagePercent,
    auto_compact_threshold_percent: AUTO_COMPACT_THRESHOLD_PERCENT,
    classification,
    categories,
  };
}

export function computeModelContextUsage(params: {
  thread: ChatThread | null;
  liveActivity?: LiveAgentActivity | null;
  llmConfig: PersistedLLMConfig | null;
}): ContextUsageSnapshot {
  const { thread, liveActivity, llmConfig } = params;
  const provider = llmConfig?.primary.provider || '';
  const model = llmConfig?.primary.model || '';
  const explicitWindow = llmConfig?.primary.context_window ?? null;
  const explicitMaxOutput = llmConfig?.primary.max_output_tokens ?? null;
  const configuredMaxTokens = llmConfig?.primary.max_tokens ?? null;

  if (liveActivity?.contextUsage) {
    return rescaleSnapshotForModel(
      liveActivity.contextUsage,
      provider,
      model,
      explicitWindow,
      explicitMaxOutput,
      configuredMaxTokens,
    );
  }
  if (thread?.contextUsage) {
    return rescaleSnapshotForModel(
      thread.contextUsage,
      provider,
      model,
      explicitWindow,
      explicitMaxOutput,
      configuredMaxTokens,
    );
  }

  const limits = resolveModelLimits(provider, model, explicitWindow, explicitMaxOutput);
  const contextWindow = limits.contextWindow;
  const source = limits.contextWindowSource;
  const effectiveConfiguredMaxTokens =
    typeof configuredMaxTokens === 'number' && Number.isFinite(configuredMaxTokens) && configuredMaxTokens > 0
      ? Math.min(Math.round(configuredMaxTokens), limits.maxOutputTokens)
      : limits.maxOutputTokens;

  // Baseline overhead for Nemofy Agent system prompt, built-in sandbox+web tools, and skills catalog
  const baseSystemTokens = 720;
  const baseToolsTokens = 1_180;
  const baseSkillsTokens = 210;

  let userTokens = 0;
  let assistantTokens = 0;
  let toolOutputTokens = 0;
  let activeSkillBonus = 0;

  for (const msg of thread?.messages || []) {
    if (msg.role === 'user') {
      userTokens += estimateTextTokens(msg.content) + 6;
      for (const att of msg.attachments || []) {
        if (att.kind === 'image') userTokens += 1_024;
        else if (att.kind === 'audio') userTokens += 1_536;
        else if (att.kind === 'video') userTokens += 4_096;
      }
    } else if (msg.role === 'assistant') {
      assistantTokens += estimateTextTokens(msg.content) + 6;
      if (msg.thinking) {
        assistantTokens += estimateTextTokens(msg.thinking);
      }
      for (const ev of msg.toolEvents || []) {
        if (ev.args_summary) assistantTokens += estimateTextTokens(ev.args_summary) + 12;
        if (ev.output_summary) toolOutputTokens += estimateTextTokens(ev.output_summary) + 16;
      }
      for (const rep of msg.subagentReports || []) {
        toolOutputTokens += estimateTextTokens(JSON.stringify(rep)) + 12;
      }
      if (msg.activeSkills && msg.activeSkills.length > 0) {
        activeSkillBonus = Math.max(activeSkillBonus, msg.activeSkills.length * 320);
      }
    }
  }

  const rawCounts: Record<ContextCategoryKey, number> = {
    system_prompt: baseSystemTokens,
    tools: baseToolsTokens,
    skills: baseSkillsTokens + activeSkillBonus,
    user_messages: userTokens,
    assistant_messages: assistantTokens,
    tool_outputs: toolOutputTokens,
    compacted_summary: 0,
  };

  const usedTokens = Object.values(rawCounts).reduce((acc, n) => acc + n, 0);
  const freeTokens = Math.max(0, contextWindow - usedTokens);
  const usagePercent = roundOneDecimal(Math.min(100, (usedTokens / Math.max(1, contextWindow)) * 100));
  const classification = classifyContextUsage(usagePercent);

  const categories: ContextCategoryBreakdown[] = CONTEXT_CATEGORY_META.map((meta) => {
    const toks = rawCounts[meta.key];
    return {
      key: meta.key,
      label: meta.label,
      description: meta.description,
      tokens: toks,
      percent: roundOneDecimal((toks / Math.max(1, contextWindow)) * 100),
      share_of_used: usedTokens > 0 ? roundOneDecimal((toks / usedTokens) * 100) : 0,
    };
  });

  return {
    model: model.trim(),
    provider: provider.trim(),
    context_window: contextWindow,
    context_window_source: source,
    max_output_tokens: limits.maxOutputTokens,
    max_output_tokens_source: limits.maxOutputTokensSource,
    configured_max_tokens: effectiveConfiguredMaxTokens,
    used_tokens: usedTokens,
    free_tokens: freeTokens,
    usage_percent: usagePercent,
    auto_compact_threshold_percent: AUTO_COMPACT_THRESHOLD_PERCENT,
    classification,
    categories,
    compacted_count: 0,
    last_compacted_at: null,
    tokens_saved_last_compaction: 0,
    calibrated_from_provider: false,
  };
}
