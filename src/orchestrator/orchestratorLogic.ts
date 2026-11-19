export type OrchestratorSource = 'chat' | 'webhook';
export type ComplexityRoute = 'fast' | 'deep';
export type LlmProvider = 'river' | 'openrouter';
export type LlmOverrideSource = 'default' | 'request' | 'inline';
export type OrchestratorDomain =
  | 'engineering'
  | 'math_logic'
  | 'research'
  | 'strategy'
  | 'creative'
  | 'general';

export interface LlmToggleDefaults {
  defaultProvider?: string;
  defaultModel?: string;
  defaultReasoningEffort?: string;
}

export interface ParsedInlineLlmDirective {
  prompt: string;
  providerOverride: string;
  modelOverride: string;
  reasoningEffortOverride: string;
  directiveUsed: boolean;
}

export interface ResolvedLlmConfig {
  provider: LlmProvider;
  model: string;
  reasoningEffort: string;
  extraBodyJson: string;
  overrideSource: LlmOverrideSource;
}

export interface NormalizedOrchestratorInput {
  userPrompt: string;
  sessionId: string;
  source: OrchestratorSource;
  forceDeepMode: boolean;
  startedAt: number;
  requestProviderOverride: string;
  requestModelOverride: string;
  requestReasoningEffortOverride: string;
  inlineProviderOverride: string;
  inlineModelOverride: string;
  inlineReasoningEffortOverride: string;
  inlineDirectiveUsed: boolean;
}

export interface OrchestrationPlan {
  complexity: ComplexityRoute;
  domain: OrchestratorDomain;
  intentSummary: string;
  researchBrief: string;
  architectBrief: string;
  strategistBrief: string;
  acceptanceCriteria: string[];
  routerFallbackUsed: boolean;
}

export interface CriticEvaluation {
  verdict: 'PASS' | 'REVISE';
  qualityScore: number;
  verifiedStrengths: string[];
  criticalFlaws: string[];
  refinementDirectives: string[];
  needsRefinement: boolean;
  criticFallbackUsed: boolean;
}

export interface OrchestratorResponseMetadata {
  model: string;
  provider: LlmProvider;
  reasoningEffort: string;
  overrideSource: LlmOverrideSource;
  route: ComplexityRoute;
  domain: OrchestratorDomain;
  qualityScore: number | null;
  verdict: 'PASS' | 'REVISE' | 'DIRECT';
  refined: boolean;
  specialistCount: number;
  source: OrchestratorSource;
  sessionId: string;
  latencyMs: number;
  acceptanceCriteriaCount: number;
  criticalFlawsFound: number;
}

export interface FormattedOrchestratorResponse {
  output: string;
  text: string;
  metadata: OrchestratorResponseMetadata;
}

export const DEFAULT_LLM_PROVIDER: LlmProvider = 'river';
export const ORCHESTRATOR_BASE_MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash';
export const ORCHESTRATOR_PROVIDER_BASE_URL = 'https://api.river.ai/v1';
export const ORCHESTRATOR_REASONING_EFFORT = 'max';
export const OPENROUTER_DEFAULT_MODEL = 'baidu/ernie-5.5-flash-preview:free';
export const QUALITY_PASS_THRESHOLD = 85;

export const PROVIDER_DEFAULTS: Record<
  LlmProvider,
  { model: string; reasoningEffort: string }
> = {
  river: {
    model: ORCHESTRATOR_BASE_MODEL,
    reasoningEffort: ORCHESTRATOR_REASONING_EFFORT,
  },
  openrouter: {
    model: OPENROUTER_DEFAULT_MODEL,
    reasoningEffort: 'off',
  },
};

const ALLOWED_DOMAINS: readonly OrchestratorDomain[] = [
  'engineering',
  'math_logic',
  'research',
  'strategy',
  'creative',
  'general',
];

function asNonEmptyString(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim();
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value);
  }
  return '';
}

function asStringArray(value: unknown, fallback: string[]): string[] {
  if (!Array.isArray(value)) {
    return fallback;
  }
  const cleaned = value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter((item) => item.length > 0);
  return cleaned.length > 0 ? cleaned : fallback;
}

export function normalizeProvider(
  raw: string | undefined,
  fallback: LlmProvider = DEFAULT_LLM_PROVIDER,
): LlmProvider {
  const cleaned = asNonEmptyString(raw).toLowerCase();
  if (!cleaned) {
    return fallback;
  }
  if (
    cleaned === 'river' ||
    cleaned === 'river.ai' ||
    cleaned === 'riverai' ||
    cleaned === 'river_ai'
  ) {
    return 'river';
  }
  if (
    cleaned === 'openrouter' ||
    cleaned === 'openrouter.ai' ||
    cleaned === 'open_router' ||
    cleaned === 'or'
  ) {
    return 'openrouter';
  }
  return fallback;
}

export function isReasoningEffortToken(token: string): boolean {
  const lower = token.trim().toLowerCase();
  if (['low', 'high', 'max', 'off', 'none'].includes(lower)) {
    return true;
  }
  if (/^\d+$/.test(lower)) {
    const n = Number(lower);
    return n >= 1 && n <= 100;
  }
  return false;
}

export function normalizeReasoningEffort(
  raw: string | undefined,
  provider: LlmProvider,
): string {
  const cleaned = asNonEmptyString(raw).toLowerCase();
  if (!cleaned) {
    return PROVIDER_DEFAULTS[provider].reasoningEffort;
  }
  if (cleaned === 'none' || cleaned === 'off' || cleaned === 'disabled') {
    return 'off';
  }
  if (cleaned === 'low' || cleaned === 'high' || cleaned === 'max') {
    return cleaned;
  }
  if (/^\d+$/.test(cleaned)) {
    const n = Number(cleaned);
    if (n >= 1 && n <= 100) {
      return String(n);
    }
  }
  return PROVIDER_DEFAULTS[provider].reasoningEffort;
}

export function buildExtraBodyJson(
  provider: LlmProvider,
  reasoningEffort: string,
): string {
  if (provider !== 'river' || !reasoningEffort || reasoningEffort === 'off') {
    return '{}';
  }
  if (/^\d+$/.test(reasoningEffort)) {
    const n = Number(reasoningEffort);
    if (n >= 1 && n <= 100) {
      return JSON.stringify({ reasoning_effort: n });
    }
  }
  return JSON.stringify({ reasoning_effort: reasoningEffort });
}

/**
 * Supports optional inline chat/API toggle directives at the start of a prompt:
 *   /llm openrouter | What is 19 * 27?
 *   /llm openrouter baidu/ernie-5.5-flash-preview:free | Compare Raft and Paxos
 *   /llm river deepseek-ai/DeepSeek-V4.1-Flash max | Design a rate limiter
 *   /llm river high\nExplain CAP theorem
 */
export function parseInlineLlmDirective(
  rawPrompt: string,
): ParsedInlineLlmDirective {
  const trimmed = rawPrompt.trim();
  const match = trimmed.match(/^\/llm\s+([^\n|]+?)(?:\s*\|\s*|\r?\n)([\s\S]+)$/i);
  if (!match) {
    return {
      prompt: trimmed,
      providerOverride: '',
      modelOverride: '',
      reasoningEffortOverride: '',
      directiveUsed: false,
    };
  }

  const headerTokens = match[1].trim().split(/\s+/).filter(Boolean);
  const cleanedPrompt = match[2].trim();
  const providerOverride = headerTokens[0] ?? '';
  let modelOverride = '';
  let reasoningEffortOverride = '';

  if (headerTokens.length === 2) {
    if (isReasoningEffortToken(headerTokens[1])) {
      reasoningEffortOverride = headerTokens[1];
    } else {
      modelOverride = headerTokens[1];
    }
  } else if (headerTokens.length >= 3) {
    modelOverride = headerTokens[1];
    reasoningEffortOverride = headerTokens[2];
  }

  return {
    prompt: cleanedPrompt,
    providerOverride,
    modelOverride,
    reasoningEffortOverride,
    directiveUsed: true,
  };
}

export function resolveLlmToggleConfig(
  input: Pick<
    NormalizedOrchestratorInput,
    | 'requestProviderOverride'
    | 'requestModelOverride'
    | 'requestReasoningEffortOverride'
    | 'inlineProviderOverride'
    | 'inlineModelOverride'
    | 'inlineReasoningEffortOverride'
    | 'inlineDirectiveUsed'
  >,
  defaults: LlmToggleDefaults = {},
): ResolvedLlmConfig {
  const defaultProvider = normalizeProvider(
    defaults.defaultProvider,
    DEFAULT_LLM_PROVIDER,
  );
  const defaultModel =
    asNonEmptyString(defaults.defaultModel) ||
    PROVIDER_DEFAULTS[defaultProvider].model;
  const defaultEffort = normalizeReasoningEffort(
    defaults.defaultReasoningEffort,
    defaultProvider,
  );

  const hasInlineOverride = Boolean(
    input.inlineDirectiveUsed &&
      (input.inlineProviderOverride ||
        input.inlineModelOverride ||
        input.inlineReasoningEffortOverride),
  );
  const hasRequestOverride = Boolean(
    input.requestProviderOverride ||
      input.requestModelOverride ||
      input.requestReasoningEffortOverride,
  );

  const overrideSource: LlmOverrideSource = hasInlineOverride
    ? 'inline'
    : hasRequestOverride
      ? 'request'
      : 'default';

  const rawProviderOverride =
    asNonEmptyString(input.inlineProviderOverride) ||
    asNonEmptyString(input.requestProviderOverride);

  const provider = rawProviderOverride
    ? normalizeProvider(rawProviderOverride, defaultProvider)
    : defaultProvider;

  const providerSwitched = Boolean(
    rawProviderOverride && provider !== defaultProvider,
  );

  const explicitModel =
    asNonEmptyString(input.inlineModelOverride) ||
    asNonEmptyString(input.requestModelOverride);

  const model =
    explicitModel ||
    (providerSwitched ? PROVIDER_DEFAULTS[provider].model : defaultModel);

  const explicitEffort =
    asNonEmptyString(input.inlineReasoningEffortOverride) ||
    asNonEmptyString(input.requestReasoningEffortOverride);

  const reasoningEffort = explicitEffort
    ? normalizeReasoningEffort(explicitEffort, provider)
    : providerSwitched
      ? PROVIDER_DEFAULTS[provider].reasoningEffort
      : provider === 'openrouter'
        ? 'off'
        : defaultEffort;

  return {
    provider,
    model,
    reasoningEffort,
    extraBodyJson: buildExtraBodyJson(provider, reasoningEffort),
    overrideSource,
  };
}

export function extractJsonObject(rawText: string): Record<string, unknown> | null {
  const trimmed = rawText.trim();
  if (!trimmed) {
    return null;
  }

  // 1. Direct parse
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Fall through to fence/brace extraction
  }

  // 2. Markdown code fence extraction
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch?.[1]) {
    try {
      const parsed = JSON.parse(fenceMatch[1].trim());
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // Fall through to outermost brace scan
    }
  }

  // 3. Outermost balanced JSON object scan
  const firstBrace = trimmed.indexOf('{');
  const lastBrace = trimmed.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    try {
      const parsed = JSON.parse(trimmed.slice(firstBrace, lastBrace + 1));
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return null;
    }
  }

  return null;
}

export function normalizeOrchestratorInput(
  rawItem: Record<string, unknown>,
  nowMs: number = Date.now(),
): NormalizedOrchestratorInput {
  const isWebhook =
    rawItem.body !== undefined ||
    rawItem.headers !== undefined ||
    rawItem.webhookUrl !== undefined;

  const payload =
    isWebhook && rawItem.body && typeof rawItem.body === 'object'
      ? (rawItem.body as Record<string, unknown>)
      : rawItem;

  const rawPrompt =
    asNonEmptyString(payload.chatInput) ||
    asNonEmptyString(payload.prompt) ||
    asNonEmptyString(payload.message) ||
    asNonEmptyString(payload.query) ||
    asNonEmptyString(payload.input) ||
    asNonEmptyString(rawItem.chatInput) ||
    '';

  const inline = parseInlineLlmDirective(rawPrompt);

  const rawSession =
    asNonEmptyString(payload.sessionId) ||
    asNonEmptyString(rawItem.sessionId);

  const source: OrchestratorSource = isWebhook ? 'webhook' : 'chat';
  const sessionId = rawSession || `${source}-${nowMs}`;

  const forceDeepMode =
    payload.forceDeepMode === true ||
    payload.mode === 'deep' ||
    rawItem.forceDeepMode === true;

  const llmBlock =
    payload.llm && typeof payload.llm === 'object'
      ? (payload.llm as Record<string, unknown>)
      : {};

  const requestProviderOverride =
    asNonEmptyString(payload.provider) ||
    asNonEmptyString(payload.llmProvider) ||
    asNonEmptyString(llmBlock.provider);

  const requestModelOverride =
    asNonEmptyString(payload.model) ||
    asNonEmptyString(payload.llmModel) ||
    asNonEmptyString(llmBlock.model);

  const requestReasoningEffortOverride =
    asNonEmptyString(payload.reasoningEffort) ||
    asNonEmptyString(payload.reasoning_effort) ||
    asNonEmptyString(llmBlock.reasoningEffort) ||
    asNonEmptyString(llmBlock.reasoning_effort);

  return {
    userPrompt: inline.prompt,
    sessionId,
    source,
    forceDeepMode,
    startedAt: nowMs,
    requestProviderOverride,
    requestModelOverride,
    requestReasoningEffortOverride,
    inlineProviderOverride: inline.providerOverride,
    inlineModelOverride: inline.modelOverride,
    inlineReasoningEffortOverride: inline.reasoningEffortOverride,
    inlineDirectiveUsed: inline.directiveUsed,
  };
}

export function parseRouterPlan(
  rawOutput: string,
  userPrompt: string,
  forceDeepMode: boolean,
): OrchestrationPlan {
  const parsed = extractJsonObject(rawOutput);
  const defaultCriteria = [
    'Directly and completely answers every part of the user request',
    'Calculations, code, and factual claims are internally consistent and verified',
    'Key edge cases, constraints, and trade-offs are addressed concisely',
  ];

  if (!parsed) {
    return {
      complexity: 'deep',
      domain: 'general',
      intentSummary: userPrompt,
      researchBrief: `Gather verified facts, definitions, standards, and constraints relevant to: ${userPrompt}`,
      architectBrief: `Solve this rigorously from first principles with step-by-step verification: ${userPrompt}`,
      strategistBrief: `Challenge default assumptions, surface edge cases, trade-offs, and superior alternatives for: ${userPrompt}`,
      acceptanceCriteria: defaultCriteria,
      routerFallbackUsed: true,
    };
  }

  const rawComplexity = asNonEmptyString(parsed.complexity).toLowerCase();
  const complexity: ComplexityRoute = forceDeepMode
    ? 'deep'
    : rawComplexity === 'fast'
      ? 'fast'
      : 'deep';

  const rawDomain = asNonEmptyString(parsed.domain).toLowerCase() as OrchestratorDomain;
  const domain: OrchestratorDomain = ALLOWED_DOMAINS.includes(rawDomain)
    ? rawDomain
    : 'general';

  return {
    complexity,
    domain,
    intentSummary: asNonEmptyString(parsed.intent_summary) || userPrompt,
    researchBrief:
      asNonEmptyString(parsed.research_brief) ||
      `Gather verified background facts, domain standards, and factual anchors for: ${userPrompt}`,
    architectBrief:
      asNonEmptyString(parsed.architect_brief) ||
      `Build a rigorous first-principles solution and verify calculations/logic for: ${userPrompt}`,
    strategistBrief:
      asNonEmptyString(parsed.strategist_brief) ||
      `Explore non-obvious alternatives, trade-offs, failure modes, and high-leverage improvements for: ${userPrompt}`,
    acceptanceCriteria: asStringArray(parsed.acceptance_criteria, defaultCriteria),
    routerFallbackUsed: false,
  };
}

export function parseCriticEvaluation(rawOutput: string): CriticEvaluation {
  const parsed = extractJsonObject(rawOutput);
  if (!parsed) {
    return {
      verdict: 'REVISE',
      qualityScore: 72,
      verifiedStrengths: ['Multiple specialist perspectives were gathered'],
      criticalFlaws: [
        'Critic returned unstructured audit; perform a verification pass before final synthesis',
      ],
      refinementDirectives: [
        'Re-verify all quantitative claims, edge cases, and logical steps before final answer assembly',
      ],
      needsRefinement: true,
      criticFallbackUsed: true,
    };
  }

  const rawScore = Number(parsed.quality_score);
  const clampedScore = Number.isFinite(rawScore)
    ? Math.max(0, Math.min(100, Math.round(rawScore)))
    : 75;

  const rawVerdict = asNonEmptyString(parsed.verdict).toUpperCase();
  const verdict: 'PASS' | 'REVISE' =
    rawVerdict === 'PASS' && clampedScore >= QUALITY_PASS_THRESHOLD
      ? 'PASS'
      : 'REVISE';

  const criticalFlaws = asStringArray(parsed.critical_flaws, []);
  const needsRefinement =
    verdict === 'REVISE' || clampedScore < QUALITY_PASS_THRESHOLD;

  return {
    verdict,
    qualityScore: clampedScore,
    verifiedStrengths: asStringArray(parsed.verified_strengths, [
      'Cross-checked specialist outputs',
    ]),
    criticalFlaws,
    refinementDirectives: asStringArray(
      parsed.refinement_directives,
      needsRefinement
        ? ['Resolve remaining inconsistencies and tighten technical precision']
        : [],
    ),
    needsRefinement,
    criticFallbackUsed: false,
  };
}

export function formatFinalOrchestratorResponse(params: {
  finalText: string;
  route: ComplexityRoute;
  domain: OrchestratorDomain;
  qualityScore: number | null;
  verdict: 'PASS' | 'REVISE' | 'DIRECT';
  refined: boolean;
  source: OrchestratorSource;
  sessionId: string;
  startedAt: number;
  nowMs?: number;
  acceptanceCriteriaCount: number;
  criticalFlawsFound: number;
  provider?: LlmProvider;
  model?: string;
  reasoningEffort?: string;
  overrideSource?: LlmOverrideSource;
}): FormattedOrchestratorResponse {
  const cleanedOutput = params.finalText.trim();
  const nowMs = params.nowMs ?? Date.now();
  const latencyMs = Math.max(0, nowMs - params.startedAt);
  const provider = params.provider ?? DEFAULT_LLM_PROVIDER;

  return {
    output: cleanedOutput,
    text: cleanedOutput,
    metadata: {
      model: params.model ?? PROVIDER_DEFAULTS[provider].model,
      provider,
      reasoningEffort:
        params.reasoningEffort ?? PROVIDER_DEFAULTS[provider].reasoningEffort,
      overrideSource: params.overrideSource ?? 'default',
      route: params.route,
      domain: params.domain,
      qualityScore: params.qualityScore,
      verdict: params.verdict,
      refined: params.refined,
      specialistCount: params.route === 'fast' ? 1 : params.refined ? 7 : 6,
      source: params.source,
      sessionId: params.sessionId,
      latencyMs,
      acceptanceCriteriaCount: params.acceptanceCriteriaCount,
      criticalFlawsFound: params.criticalFlawsFound,
    },
  };
}
