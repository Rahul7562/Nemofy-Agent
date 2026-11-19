import {
  DEFAULT_LLM_PROVIDER,
  OPENROUTER_DEFAULT_MODEL,
  ORCHESTRATOR_BASE_MODEL,
  ORCHESTRATOR_PROVIDER_BASE_URL,
  ORCHESTRATOR_REASONING_EFFORT,
  QUALITY_PASS_THRESHOLD,
} from './orchestratorLogic.ts';

export const ORCHESTRATOR_WORKFLOW_ID = 'dsV41MultiAgent01';
export const ORCHESTRATOR_WORKFLOW_NAME =
  'Multi-Agent Cognitive Orchestrator (Universal LLM Toggle)';
export const ORCHESTRATOR_CHAT_WEBHOOK_ID = '7b3e4f92-41f0-4a1d-9c52-8841f0c0a001';
export const ORCHESTRATOR_API_WEBHOOK_ID = '7b3e4f92-41f0-4a1d-9c52-8841f0c0a002';
export const ORCHESTRATOR_API_WEBHOOK_PATH = 'orchestrator/run';

export interface ProviderCredentialRef {
  id: string;
  name: string;
}

export interface OrchestratorCredentials {
  riverAi: ProviderCredentialRef;
  openRouter: ProviderCredentialRef;
}

export interface N8nWorkflowNode {
  id: string;
  name: string;
  type: string;
  typeVersion: number;
  position: [number, number];
  parameters: Record<string, unknown>;
  webhookId?: string;
  credentials?: Record<string, { id: string; name: string }>;
  retryOnFail?: boolean;
  maxTries?: number;
  waitBetweenTries?: number;
}

export interface N8nConnectionTarget {
  node: string;
  type: string;
  index: number;
}

export type N8nConnections = Record<
  string,
  Record<string, N8nConnectionTarget[][]>
>;

export interface N8nWorkflowDefinition {
  id: string;
  name: string;
  active: boolean;
  nodes: N8nWorkflowNode[];
  connections: N8nConnections;
  settings: Record<string, unknown>;
  meta: Record<string, unknown>;
  pinData: Record<string, unknown>;
  versionId: string;
}

const DEFAULT_CREDENTIALS: OrchestratorCredentials = {
  riverAi: {
    id: 'riverAiCredV4101',
    name: 'River AI',
  },
  openRouter: {
    id: 'o3dspeUuaShSW6fK',
    name: 'OpenRouter account',
  },
};

function makeSandboxCodeTool(params: {
  id: string;
  name: string;
  position: [number, number];
  toolName: string;
  description: string;
}): N8nWorkflowNode {
  return {
    id: params.id,
    name: params.name,
    type: '@n8n/n8n-nodes-langchain.toolCode',
    typeVersion: 1.3,
    position: params.position,
    parameters: {
      name: params.toolName,
      description: params.description,
      language: 'javaScript',
      jsCode: `// Deterministic JavaScript verification sandbox for orchestrator sub-agents.
// Accepts single expressions (with or without trailing ';'), multi-statement scripts, or explicit return bodies.
let raw = String(query ?? '').trim();
const fence = raw.match(/\`\`\`(?:js|javascript)?\\s*([\\s\\S]*?)\`\`\`/i);
if (fence && fence[1]) {
  raw = fence[1].trim();
}
if (!raw) {
  return JSON.stringify({ ok: false, error: 'Empty code query' });
}
const forbidden = /\\b(process|require|import|globalThis|child_process|fs|fetch|eval|Function)\\b/;
if (forbidden.test(raw)) {
  return JSON.stringify({ ok: false, error: 'Blocked unsafe identifier in verification snippet' });
}
const logs = [];
const sandboxConsole = {
  log: (...args) => logs.push(args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ')),
};
function runBody(body) {
  const fn = new Function('Math', 'JSON', 'Date', 'Array', 'Object', 'Number', 'String', 'Map', 'Set', 'console', '"use strict";\\n' + body);
  return fn(Math, JSON, Date, Array, Object, Number, String, Map, Set, sandboxConsole);
}
try {
  let result;
  if (/\\breturn\\b/.test(raw)) {
    result = runBody(raw);
  } else {
    const noTrailingSemi = raw.replace(/;+\\s*$/, '');
    try {
      result = runBody('return (' + noTrailingSemi + ');');
    } catch {
      const lines = noTrailingSemi.split(/\\r?\\n/).map((l) => l.trim()).filter(Boolean);
      if (lines.length > 1) {
        const last = lines[lines.length - 1].replace(/;+\\s*$/, '');
        const prefix = lines.slice(0, -1).join('\\n');
        try {
          result = runBody(prefix + '\\nreturn (' + last + ');');
        } catch {
          result = runBody(raw);
        }
      } else {
        result = runBody(raw);
      }
    }
  }
  return JSON.stringify({ ok: true, result: result !== undefined ? result : null, logs });
} catch (err) {
  return JSON.stringify({ ok: false, error: err instanceof Error ? err.message : String(err), logs });
}`,
    },
  };
}

const NORMALIZE_INPUT_CODE = `const item = $input.first()?.json ?? {};
const isWebhook =
  item.body !== undefined ||
  item.headers !== undefined ||
  item.webhookUrl !== undefined;

const payload =
  isWebhook && item.body && typeof item.body === 'object' ? item.body : item;

const pickString = (v) => {
  if (typeof v === 'string') return v.trim();
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  return '';
};

const rawPrompt =
  pickString(payload.chatInput) ||
  pickString(payload.prompt) ||
  pickString(payload.message) ||
  pickString(payload.query) ||
  pickString(payload.input) ||
  pickString(item.chatInput) ||
  '';

function isEffortToken(tok) {
  const lower = String(tok || '').trim().toLowerCase();
  if (['low', 'high', 'max', 'off', 'none'].includes(lower)) return true;
  if (/^\\d+$/.test(lower)) {
    const n = Number(lower);
    return n >= 1 && n <= 100;
  }
  return false;
}

let userPrompt = rawPrompt;
let inlineProviderOverride = '';
let inlineModelOverride = '';
let inlineReasoningEffortOverride = '';
let inlineDirectiveUsed = false;

const directiveMatch = rawPrompt.match(/^\\/llm\\s+([^\\n|]+?)(?:\\s*\\|\\s*|\\r?\\n)([\\s\\S]+)$/i);
if (directiveMatch) {
  const tokens = directiveMatch[1].trim().split(/\\s+/).filter(Boolean);
  userPrompt = directiveMatch[2].trim();
  inlineProviderOverride = tokens[0] || '';
  if (tokens.length === 2) {
    if (isEffortToken(tokens[1])) {
      inlineReasoningEffortOverride = tokens[1];
    } else {
      inlineModelOverride = tokens[1];
    }
  } else if (tokens.length >= 3) {
    inlineModelOverride = tokens[1];
    inlineReasoningEffortOverride = tokens[2];
  }
  inlineDirectiveUsed = true;
}

const source = isWebhook ? 'webhook' : 'chat';
const now = Date.now();
const sessionId =
  pickString(payload.sessionId) ||
  pickString(item.sessionId) ||
  source + '-' + now;

const forceDeepMode =
  payload.forceDeepMode === true ||
  payload.mode === 'deep' ||
  item.forceDeepMode === true;

const llmBlock = payload.llm && typeof payload.llm === 'object' ? payload.llm : {};

const requestProviderOverride =
  pickString(payload.provider) ||
  pickString(payload.llmProvider) ||
  pickString(llmBlock.provider);

const requestModelOverride =
  pickString(payload.model) ||
  pickString(payload.llmModel) ||
  pickString(llmBlock.model);

const requestReasoningEffortOverride =
  pickString(payload.reasoningEffort) ||
  pickString(payload.reasoning_effort) ||
  pickString(llmBlock.reasoningEffort) ||
  pickString(llmBlock.reasoning_effort);

return [
  {
    json: {
      userPrompt,
      sessionId,
      source,
      forceDeepMode,
      startedAt: now,
      requestProviderOverride,
      requestModelOverride,
      requestReasoningEffortOverride,
      inlineProviderOverride,
      inlineModelOverride,
      inlineReasoningEffortOverride,
      inlineDirectiveUsed,
    },
  },
];`;

const RESOLVE_LLM_CONFIG_CODE = `const item = $input.first()?.json ?? {};

const PROVIDER_DEFAULTS = {
  river: {
    model: '${ORCHESTRATOR_BASE_MODEL}',
    reasoningEffort: '${ORCHESTRATOR_REASONING_EFFORT}',
  },
  openrouter: {
    model: '${OPENROUTER_DEFAULT_MODEL}',
    reasoningEffort: 'off',
  },
};

const pickString = (v) => {
  if (typeof v === 'string') return v.trim();
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  return '';
};

function normalizeProvider(raw, fallback) {
  const c = pickString(raw).toLowerCase();
  if (!c) return fallback;
  if (c === 'river' || c === 'river.ai' || c === 'riverai' || c === 'river_ai') return 'river';
  if (c === 'openrouter' || c === 'openrouter.ai' || c === 'open_router' || c === 'or') return 'openrouter';
  return fallback;
}

function normalizeEffort(raw, provider) {
  const c = pickString(raw).toLowerCase();
  if (!c) return PROVIDER_DEFAULTS[provider].reasoningEffort;
  if (c === 'none' || c === 'off' || c === 'disabled') return 'off';
  if (c === 'low' || c === 'high' || c === 'max') return c;
  if (/^\\d+$/.test(c)) {
    const n = Number(c);
    if (n >= 1 && n <= 100) return String(n);
  }
  return PROVIDER_DEFAULTS[provider].reasoningEffort;
}

function buildExtraBodyJson(provider, effort) {
  if (provider !== 'river' || !effort || effort === 'off') return '{}';
  if (/^\\d+$/.test(effort)) {
    const n = Number(effort);
    if (n >= 1 && n <= 100) return JSON.stringify({ reasoning_effort: n });
  }
  return JSON.stringify({ reasoning_effort: effort });
}

const defaultProvider = normalizeProvider(item.defaultProvider, '${DEFAULT_LLM_PROVIDER}');
const defaultModel = pickString(item.defaultModel) || PROVIDER_DEFAULTS[defaultProvider].model;
const defaultEffort = normalizeEffort(item.defaultReasoningEffort, defaultProvider);

const hasInlineOverride = Boolean(
  item.inlineDirectiveUsed &&
    (item.inlineProviderOverride || item.inlineModelOverride || item.inlineReasoningEffortOverride),
);
const hasRequestOverride = Boolean(
  item.requestProviderOverride || item.requestModelOverride || item.requestReasoningEffortOverride,
);

const overrideSource = hasInlineOverride ? 'inline' : hasRequestOverride ? 'request' : 'default';

const rawProviderOverride =
  pickString(item.inlineProviderOverride) || pickString(item.requestProviderOverride);

const provider = rawProviderOverride
  ? normalizeProvider(rawProviderOverride, defaultProvider)
  : defaultProvider;

const providerSwitched = Boolean(rawProviderOverride && provider !== defaultProvider);

const explicitModel =
  pickString(item.inlineModelOverride) || pickString(item.requestModelOverride);

const model =
  explicitModel ||
  (providerSwitched ? PROVIDER_DEFAULTS[provider].model : defaultModel);

const explicitEffort =
  pickString(item.inlineReasoningEffortOverride) ||
  pickString(item.requestReasoningEffortOverride);

const reasoningEffort = explicitEffort
  ? normalizeEffort(explicitEffort, provider)
  : providerSwitched
    ? PROVIDER_DEFAULTS[provider].reasoningEffort
    : provider === 'openrouter'
      ? 'off'
      : defaultEffort;

const extraBodyJson = buildExtraBodyJson(provider, reasoningEffort);

return [
  {
    json: {
      ...item,
      provider,
      model,
      reasoningEffort,
      extraBodyJson,
      overrideSource,
    },
  },
];`;

const PARSE_ROUTER_PLAN_CODE = `const norm = $('Resolve Active LLM Config').first().json;
const rawOutput = String($input.first()?.json?.output ?? '');

function extractJson(text) {
  const trimmed = text.trim();
  if (!trimmed) return null;
  try {
    const direct = JSON.parse(trimmed);
    if (direct && typeof direct === 'object' && !Array.isArray(direct)) return direct;
  } catch {}
  const fence = trimmed.match(/\`\`\`(?:json)?\\s*([\\s\\S]*?)\`\`\`/i);
  if (fence && fence[1]) {
    try {
      const fenced = JSON.parse(fence[1].trim());
      if (fenced && typeof fenced === 'object' && !Array.isArray(fenced)) return fenced;
    } catch {}
  }
  const first = trimmed.indexOf('{');
  const last = trimmed.lastIndexOf('}');
  if (first !== -1 && last > first) {
    try {
      const sliced = JSON.parse(trimmed.slice(first, last + 1));
      if (sliced && typeof sliced === 'object' && !Array.isArray(sliced)) return sliced;
    } catch {}
  }
  return null;
}

const allowedDomains = ['engineering', 'math_logic', 'research', 'strategy', 'creative', 'general'];
const defaultCriteria = [
  'Directly and completely answers every part of the user request',
  'Calculations, code, and factual claims are internally consistent and verified',
  'Key edge cases, constraints, and trade-offs are addressed concisely',
];

const parsed = extractJson(rawOutput);
const pick = (v, fb) => (typeof v === 'string' && v.trim() ? v.trim() : fb);
const pickArr = (v, fb) =>
  Array.isArray(v) && v.filter((x) => typeof x === 'string' && x.trim()).length > 0
    ? v.filter((x) => typeof x === 'string' && x.trim()).map((x) => x.trim())
    : fb;

const rawComplexity = parsed ? pick(parsed.complexity, 'deep').toLowerCase() : 'deep';
const complexity = norm.forceDeepMode ? 'deep' : rawComplexity === 'fast' ? 'fast' : 'deep';
const rawDomain = parsed ? pick(parsed.domain, 'general').toLowerCase() : 'general';
const domain = allowedDomains.includes(rawDomain) ? rawDomain : 'general';

return [
  {
    json: {
      ...norm,
      complexity,
      domain,
      intentSummary: parsed ? pick(parsed.intent_summary, norm.userPrompt) : norm.userPrompt,
      researchBrief: parsed
        ? pick(parsed.research_brief, 'Gather verified facts, constraints, and standards for: ' + norm.userPrompt)
        : 'Gather verified facts, constraints, and standards for: ' + norm.userPrompt,
      architectBrief: parsed
        ? pick(parsed.architect_brief, 'Solve rigorously from first principles with tool verification: ' + norm.userPrompt)
        : 'Solve rigorously from first principles with tool verification: ' + norm.userPrompt,
      strategistBrief: parsed
        ? pick(parsed.strategist_brief, 'Explore alternative architectures, edge cases, and failure modes for: ' + norm.userPrompt)
        : 'Explore alternative architectures, edge cases, and failure modes for: ' + norm.userPrompt,
      acceptanceCriteria: parsed ? pickArr(parsed.acceptance_criteria, defaultCriteria) : defaultCriteria,
      routerFallbackUsed: !parsed,
    },
  },
];`;

const ASSEMBLE_DOSSIER_CODE = `const plan = $('Parse Orchestration Plan').first().json;
const merged = $input.all();

function readAgentOutput(nodeName, fallbackIndex) {
  try {
    const val = $(nodeName).first()?.json?.output;
    if (typeof val === 'string' && val.trim()) return val.trim();
  } catch {}
  const fb = merged[fallbackIndex]?.json?.output;
  return typeof fb === 'string' ? fb.trim() : '';
}

const researchOutput = readAgentOutput('Research & Verification Specialist', 0);
const solverOutput = readAgentOutput('First-Principles Solver & Code Specialist', 1);
const strategistOutput = readAgentOutput('Lateral Strategist & Contrarian Specialist', 2);

return [
  {
    json: {
      ...plan,
      researchOutput,
      solverOutput,
      strategistOutput,
    },
  },
];`;

const EVALUATE_CRITIQUE_CODE = `const dossier = $('Assemble Dossier').first().json;
const rawOutput = String($input.first()?.json?.output ?? '');

function extractJson(text) {
  const trimmed = text.trim();
  if (!trimmed) return null;
  try {
    const direct = JSON.parse(trimmed);
    if (direct && typeof direct === 'object' && !Array.isArray(direct)) return direct;
  } catch {}
  const fence = trimmed.match(/\`\`\`(?:json)?\\s*([\\s\\S]*?)\`\`\`/i);
  if (fence && fence[1]) {
    try {
      const fenced = JSON.parse(fence[1].trim());
      if (fenced && typeof fenced === 'object' && !Array.isArray(fenced)) return fenced;
    } catch {}
  }
  const first = trimmed.indexOf('{');
  const last = trimmed.lastIndexOf('}');
  if (first !== -1 && last > first) {
    try {
      const sliced = JSON.parse(trimmed.slice(first, last + 1));
      if (sliced && typeof sliced === 'object' && !Array.isArray(sliced)) return sliced;
    } catch {}
  }
  return null;
}

const parsed = extractJson(rawOutput);
const pickArr = (v, fb) =>
  Array.isArray(v)
    ? v.filter((x) => typeof x === 'string' && x.trim()).map((x) => x.trim())
    : fb;

let qualityScore = 72;
let verdict = 'REVISE';
let verifiedStrengths = ['Multiple specialist perspectives were gathered'];
let criticalFlaws = ['Critic output was unstructured; perform a targeted verification pass'];
let refinementDirectives = ['Re-verify calculations, edge cases, and factual precision'];
let criticFallbackUsed = true;

if (parsed) {
  criticFallbackUsed = false;
  const num = Number(parsed.quality_score);
  qualityScore = Number.isFinite(num) ? Math.max(0, Math.min(100, Math.round(num))) : 75;
  const rawVerdict = typeof parsed.verdict === 'string' ? parsed.verdict.trim().toUpperCase() : 'REVISE';
  verdict = rawVerdict === 'PASS' && qualityScore >= ${QUALITY_PASS_THRESHOLD} ? 'PASS' : 'REVISE';
  verifiedStrengths = pickArr(parsed.verified_strengths, ['Cross-checked specialist outputs']);
  criticalFlaws = pickArr(parsed.critical_flaws, []);
  refinementDirectives = pickArr(
    parsed.refinement_directives,
    verdict === 'REVISE' ? ['Resolve identified gaps and tighten precision'] : [],
  );
}

const needsRefinement = verdict === 'REVISE' || qualityScore < ${QUALITY_PASS_THRESHOLD};

return [
  {
    json: {
      ...dossier,
      verdict,
      qualityScore,
      verifiedStrengths,
      criticalFlaws,
      refinementDirectives,
      needsRefinement,
      gateDecision: needsRefinement ? 'revise' : 'pass',
      criticFallbackUsed,
    },
  },
];`;

const PREPARE_SYNTHESIS_CODE = `const critique = $('Evaluate Critique Score').first().json;
let repairedDraft = '';

if (critique.needsRefinement) {
  try {
    const repairVal = $('Deep Refinement & Repair Agent').first()?.json?.output;
    if (typeof repairVal === 'string') {
      repairedDraft = repairVal.trim();
    }
  } catch {
    const incoming = $input.first()?.json?.output;
    if (typeof incoming === 'string') {
      repairedDraft = incoming.trim();
    }
  }
}

const criteriaList = (critique.acceptanceCriteria || []).map((c, i) => (i + 1) + '. ' + c).join('\\n');
const strengthsList = (critique.verifiedStrengths || []).map((s) => '- ' + s).join('\\n') || '- None listed';
const flawsList = (critique.criticalFlaws || []).map((f) => '- ' + f).join('\\n') || '- None found';
const directivesList = (critique.refinementDirectives || []).map((d) => '- ' + d).join('\\n') || '- None';

const synthesisBrief = [
  '=== ORIGINAL USER REQUEST ===',
  critique.userPrompt,
  '',
  '=== INTERPRETED GOAL & ACCEPTANCE CRITERIA ===',
  critique.intentSummary,
  criteriaList,
  '',
  '=== SPECIALIST 1: RESEARCH & VERIFICATION ===',
  critique.researchOutput,
  '',
  '=== SPECIALIST 2: FIRST-PRINCIPLES SOLVER & CODE ===',
  critique.solverOutput,
  '',
  '=== SPECIALIST 3: LATERAL STRATEGIST & CONTRARIAN ===',
  critique.strategistOutput,
  '',
  '=== ADVERSARIAL CRITIC AUDIT (Score: ' + critique.qualityScore + '/100 | Verdict: ' + critique.verdict + ') ===',
  'Verified Strengths:',
  strengthsList,
  'Critical Flaws Caught:',
  flawsList,
  'Refinement Directives:',
  directivesList,
  repairedDraft ? '\\n=== DEEP REFINEMENT & REPAIR PATCH (AUTHORITATIVE FIXES) ===\\n' + repairedDraft : '',
].join('\\n');

return [
  {
    json: {
      ...critique,
      repairedDraft,
      synthesisBrief,
    },
  },
];`;

const FORMAT_FINAL_RESPONSE_CODE = `const plan = $('Parse Orchestration Plan').first().json;
const llm = $('Resolve Active LLM Config').first().json;
const rawFinal = String($input.first()?.json?.output ?? '').trim();
const isFast = plan.complexity === 'fast';

let qualityScore = null;
let verdict = 'DIRECT';
let refined = false;
let criticalFlawsFound = 0;

if (!isFast) {
  try {
    const critique = $('Evaluate Critique Score').first().json;
    qualityScore = typeof critique.qualityScore === 'number' ? critique.qualityScore : null;
    verdict = critique.verdict || 'PASS';
    refined = Boolean(critique.needsRefinement);
    criticalFlawsFound = Array.isArray(critique.criticalFlaws) ? critique.criticalFlaws.length : 0;
  } catch {}
}

const latencyMs = Math.max(0, Date.now() - (Number(plan.startedAt) || Date.now()));

return [
  {
    json: {
      output: rawFinal,
      text: rawFinal,
      metadata: {
        model: llm.model || '${ORCHESTRATOR_BASE_MODEL}',
        provider: llm.provider || '${DEFAULT_LLM_PROVIDER}',
        reasoningEffort: llm.reasoningEffort || '${ORCHESTRATOR_REASONING_EFFORT}',
        overrideSource: llm.overrideSource || 'default',
        route: plan.complexity,
        domain: plan.domain,
        qualityScore,
        verdict,
        refined,
        specialistCount: isFast ? 1 : refined ? 7 : 6,
        source: plan.source,
        sessionId: plan.sessionId,
        latencyMs,
        acceptanceCriteriaCount: Array.isArray(plan.acceptanceCriteria) ? plan.acceptanceCriteria.length : 0,
        criticalFlawsFound,
      },
    },
  },
];`;

export function buildMultiAgentOrchestratorWorkflow(
  credentials: OrchestratorCredentials = DEFAULT_CREDENTIALS,
): N8nWorkflowDefinition {
  const nodes: N8nWorkflowNode[] = [
    // =========================================================================
    // CANVAS ARCHITECTURE ZONES (STICKY NOTES)
    // =========================================================================
    {
      id: 'sticky-zone-1',
      name: 'Zone 1: Entry, LLM Config & Meta-Router',
      type: 'n8n-nodes-base.stickyNote',
      typeVersion: 1,
      position: [-1520, -320],
      parameters: {
        content:
          '## Stage 1 · Dual Entry, Centralized LLM Toggle & Cognitive Meta-Router\n- **Chat UI + REST Webhook** (`POST /webhook/orchestrator/run`)\n- **Single-Point LLM Toggle**: Edit **`LLM Defaults (Toggle Here)`** to switch provider (`river` / `openrouter`), model slug, or `reasoningEffort` (`max` / `high` / `low` / `off`) for ALL 8 agents at once.\n- **Per-Request Overrides**: Send `{"provider":"openrouter","model":"..."}` to the Webhook, or prefix Chat messages with `/llm openrouter | your question`.',
        height: 780,
        width: 1720,
        color: 4,
      },
    },
    {
      id: 'sticky-zone-0',
      name: 'Zone 0: Universal LLM Toggle & Provider Hub',
      type: 'n8n-nodes-base.stickyNote',
      typeVersion: 1,
      position: [-1520, 500],
      parameters: {
        content:
          '## Universal LLM Toggle Hub (Powers All 8 AI Agents)\n- **Slot 1 (`river`)**: River AI OpenAI-compatible engine (`deepseek-ai/DeepSeek-V4.1-Flash` + dynamic `reasoning_effort` via `extraBody`).\n- **Slot 2 (`openrouter`)**: OpenRouter engine (`baidu/ernie-5.5-flash-preview:free` or any OpenRouter slug).\n- **Extensible**: Increase `Number of Inputs` on `Universal LLM Toggle (Model Selector)` to plug in Anthropic, Gemini, Ollama, or OpenAI without rewiring the swarm.',
        height: 620,
        width: 1720,
        color: 2,
      },
    },
    {
      id: 'sticky-zone-2',
      name: 'Zone 2: System 1 Fast Lane',
      type: 'n8n-nodes-base.stickyNote',
      typeVersion: 1,
      position: [240, -640],
      parameters: {
        content:
          '## Stage 2A · System 1 Fast Direct Agent\nLow-latency path for greetings, quick lookups, and single-step questions.\nTool-assisted (**Exact Calculator + Session Memory**) driven by the active toggled LLM.',
        height: 440,
        width: 880,
        color: 6,
      },
    },
    {
      id: 'sticky-zone-3',
      name: 'Zone 3: System 2 Parallel Specialist Swarm',
      type: 'n8n-nodes-base.stickyNote',
      typeVersion: 1,
      position: [240, -140],
      parameters: {
        content:
          '## Stage 2B · System 2 Parallel Specialist Swarm (3 Concurrent Sub-Agents)\nAll driven dynamically by the **Universal LLM Toggle Hub** with specialized tools:\n1. **Research & Verification** (Wikipedia + Think)\n2. **First-Principles Solver & Code** (JS Verification Sandbox + Calculator + Think)\n3. **Lateral Strategist & Contrarian** (Divergent Think)',
        height: 1260,
        width: 960,
        color: 5,
      },
    },
    {
      id: 'sticky-zone-4',
      name: 'Zone 4: Adversarial Red-Team & Self-Repair',
      type: 'n8n-nodes-base.stickyNote',
      typeVersion: 1,
      position: [1260, -140],
      parameters: {
        content:
          '## Stage 3 · Fan-In Dossier, Adversarial Critic & Conditional Self-Repair\n- **Adversarial Critic** cross-examines all 3 dossiers, detects contradictions, verifies math, and scores `0–100`.\n- **Quality Gate (<85 or REVISE)** triggers the **Deep Refinement & Repair Agent** (`JS Repair Sandbox + Calculator`) before final synthesis.',
        height: 1260,
        width: 1580,
        color: 3,
      },
    },
    {
      id: 'sticky-zone-5',
      name: 'Zone 5: Executive Lead Synthesizer & Response',
      type: 'n8n-nodes-base.stickyNote',
      typeVersion: 1,
      position: [2900, -340],
      parameters: {
        content:
          '## Stage 4 · Executive Lead Synthesizer & Unified Response\nFuses verified specialist insights, critic audit, and repair patch into one authoritative answer, then emits `{ output, text, metadata }` (with active `provider`, `model`, `reasoningEffort`) to **Chat UI** or **Webhook API**.',
        height: 940,
        width: 880,
        color: 7,
      },
    },

    // =========================================================================
    // ZONE 0: UNIVERSAL LLM TOGGLE & PROVIDER HUB (SHARED BY ALL 8 AGENTS)
    // =========================================================================
    {
      id: 'node-llm-selector',
      name: 'Universal LLM Toggle (Model Selector)',
      type: '@n8n/n8n-nodes-langchain.modelSelector',
      typeVersion: 1,
      position: [-680, 680],
      parameters: {
        numberInputs: 2,
        rules: {
          rule: [
            {
              modelIndex: 1,
              conditions: {
                options: {
                  caseSensitive: true,
                  leftValue: '',
                  typeValidation: 'strict',
                  version: 2,
                },
                combinator: 'and',
                conditions: [
                  {
                    id: 'cond-provider-river',
                    leftValue: "={{ $('Resolve Active LLM Config').first().json.provider }}",
                    rightValue: 'river',
                    operator: {
                      type: 'string',
                      operation: 'equals',
                    },
                  },
                ],
              },
            },
            {
              modelIndex: 2,
              conditions: {
                options: {
                  caseSensitive: true,
                  leftValue: '',
                  typeValidation: 'strict',
                  version: 2,
                },
                combinator: 'and',
                conditions: [
                  {
                    id: 'cond-provider-openrouter',
                    leftValue: "={{ $('Resolve Active LLM Config').first().json.provider }}",
                    rightValue: 'openrouter',
                    operator: {
                      type: 'string',
                      operation: 'equals',
                    },
                  },
                ],
              },
            },
            {
              modelIndex: 1,
              conditions: {
                options: {
                  caseSensitive: true,
                  leftValue: '',
                  typeValidation: 'strict',
                  version: 2,
                },
                combinator: 'and',
                conditions: [
                  {
                    id: 'cond-provider-fallback',
                    leftValue: "={{ $('Resolve Active LLM Config').first().json.provider }}",
                    rightValue: '',
                    operator: {
                      type: 'string',
                      operation: 'notEmpty',
                      singleValue: true,
                    },
                  },
                ],
              },
            },
          ],
        },
      },
    },
    {
      id: 'node-engine-river',
      name: 'River AI — Model Engine',
      type: '@n8n/n8n-nodes-langchain.lmChatOpenAi',
      typeVersion: 1.3,
      position: [-940, 920],
      parameters: {
        model: {
          __rl: true,
          mode: 'id',
          value: "={{ $('Resolve Active LLM Config').first().json.model }}",
        },
        responsesApiEnabled: false,
        options: {
          baseURL: ORCHESTRATOR_PROVIDER_BASE_URL,
          temperature: 0.25,
          maxTokens: 8192,
          maxRetries: 4,
          timeout: 180000,
          extraBody: "={{ $('Resolve Active LLM Config').first().json.extraBodyJson }}",
        },
      },
      credentials: {
        openAiApi: {
          id: credentials.riverAi.id,
          name: credentials.riverAi.name,
        },
      },
    },
    {
      id: 'node-engine-openrouter',
      name: 'OpenRouter — Model Engine',
      type: '@n8n/n8n-nodes-langchain.lmChatOpenRouter',
      typeVersion: 1,
      position: [-420, 920],
      parameters: {
        model: "={{ $('Resolve Active LLM Config').first().json.model }}",
        options: {
          temperature: 0.25,
          maxTokens: 8192,
          maxRetries: 5,
          timeout: 180000,
        },
      },
      credentials: {
        openRouterApi: {
          id: credentials.openRouter.id,
          name: credentials.openRouter.name,
        },
      },
    },

    // =========================================================================
    // STAGE 1: DUAL TRIGGERS + NORMALIZER + LLM TOGGLE + META-ROUTER + GATE
    // =========================================================================
    {
      id: 'node-chat-trigger',
      name: 'Chat Trigger (Orchestrator UI)',
      type: '@n8n/n8n-nodes-langchain.chatTrigger',
      typeVersion: 1.4,
      position: [-1420, -80],
      webhookId: ORCHESTRATOR_CHAT_WEBHOOK_ID,
      parameters: {
        public: true,
        mode: 'hostedChat',
        authentication: 'none',
        availableInChat: true,
        agentName: 'Cognitive Multi-Agent Orchestrator',
        agentDescription:
          'Generalized multi-agent n8n orchestrator with a Universal LLM Toggle (defaults to River AI deepseek-ai/DeepSeek-V4.1-Flash with reasoning_effort=max; toggleable to OpenRouter or any connected provider).',
        suggestedPrompts: {
          prompts: [
            {
              icon: { type: 'icon', value: 'code' },
              text: 'Design a multi-region rate limiter for 50k req/s comparing Token Bucket, Sliding Window, and GCRA with exact Redis memory math and race-condition fixes.',
            },
            {
              icon: { type: 'icon', value: 'calculator' },
              text: 'Verify how many unique 5-card poker hands beat a straight, show the exact combinatorics, and double-check every step in the JS sandbox.',
            },
            {
              icon: { type: 'icon', value: 'lightbulb' },
              text: '/llm openrouter | Critique a microservices migration plan for a 12-engineer SaaS team and propose a lower-risk alternative architecture.',
            },
          ],
        },
        initialMessages:
          'Cognitive Multi-Agent Orchestrator online (Default: River AI · deepseek-ai/DeepSeek-V4.1-Flash · reasoning_effort=max).\nSwitch models anytime via the "LLM Defaults (Toggle Here)" node, Webhook JSON (`provider`, `model`, `reasoningEffort`), or inline chat command: `/llm openrouter | your question`.',
        options: {
          responseMode: 'lastNode',
          title: 'Cognitive Multi-Agent Orchestrator',
          subtitle:
            'Universal LLM Toggle (River AI / OpenRouter) · Adaptive Router · 3 Parallel Specialists · Adversarial Critic · Self-Repair · Lead Synthesizer',
        },
      },
    },
    {
      id: 'node-webhook-trigger',
      name: 'Webhook Trigger (API)',
      type: 'n8n-nodes-base.webhook',
      typeVersion: 2.1,
      position: [-1420, 140],
      webhookId: ORCHESTRATOR_API_WEBHOOK_ID,
      parameters: {
        httpMethod: 'POST',
        path: ORCHESTRATOR_API_WEBHOOK_PATH,
        responseMode: 'lastNode',
        options: {},
      },
    },
    {
      id: 'node-normalize-input',
      name: 'Normalize Input & Session',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [-1180, 30],
      parameters: {
        mode: 'runOnceForAllItems',
        jsCode: NORMALIZE_INPUT_CODE,
      },
    },
    {
      id: 'node-llm-defaults',
      name: 'LLM Defaults (Toggle Here)',
      type: 'n8n-nodes-base.set',
      typeVersion: 3.4,
      position: [-940, 30],
      parameters: {
        mode: 'manual',
        duplicateItem: false,
        assignments: {
          assignments: [
            {
              id: 'cfg-default-provider',
              name: 'defaultProvider',
              value: DEFAULT_LLM_PROVIDER,
              type: 'string',
            },
            {
              id: 'cfg-default-model',
              name: 'defaultModel',
              value: ORCHESTRATOR_BASE_MODEL,
              type: 'string',
            },
            {
              id: 'cfg-default-reasoning',
              name: 'defaultReasoningEffort',
              value: ORCHESTRATOR_REASONING_EFFORT,
              type: 'string',
            },
          ],
        },
        includeOtherFields: true,
        include: 'all',
        options: {},
      },
    },
    {
      id: 'node-resolve-llm',
      name: 'Resolve Active LLM Config',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [-700, 30],
      parameters: {
        mode: 'runOnceForAllItems',
        jsCode: RESOLVE_LLM_CONFIG_CODE,
      },
    },
    {
      id: 'node-meta-router',
      name: 'Meta-Router & Planner Agent',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [-440, 30],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: "={{ 'User Request:\\n' + $json.userPrompt + '\\n\\nForce Deep Mode: ' + String($json.forceDeepMode) }}",
        options: {
          maxIterations: 4,
          systemMessage: `You are the Cognitive Meta-Router & Task Decomposer inside a high-intelligence Multi-Agent Orchestrator.

Your job is NOT to answer the user directly. Your job is to classify complexity and decompose the request so specialized downstream agents produce an answer far superior to a single LLM pass.

Routing policy:
- Choose "fast" ONLY for trivial greetings, simple single-fact questions, or one-line arithmetic with zero ambiguity.
- Choose "deep" for ANY multi-step reasoning, coding, system architecture, math/logic puzzles, research comparisons, business/product strategy, debugging, or multi-constraint prompts — OR whenever Force Deep Mode is true.

Return STRICT JSON ONLY (no markdown fences, no commentary outside JSON) with this exact schema:
{
  "complexity": "fast" | "deep",
  "domain": "engineering" | "math_logic" | "research" | "strategy" | "creative" | "general",
  "intent_summary": "Crisp 1-2 sentence statement of the user's true goal and hidden constraints",
  "research_brief": "Specific factual anchors, definitions, standards, or prior art the Research Specialist must verify",
  "architect_brief": "Exact analytical/coding/mathematical derivation and verification checks the First-Principles Solver must perform",
  "strategist_brief": "Counter-examples, alternative architectures, hidden failure modes, and high-leverage optimizations the Lateral Strategist must investigate",
  "acceptance_criteria": [
    "Concrete verifiable criterion 1",
    "Concrete verifiable criterion 2",
    "Concrete verifiable criterion 3"
  ]
}`,
        },
      },
    },
    {
      id: 'node-memory-router',
      name: 'Session Memory (Router)',
      type: '@n8n/n8n-nodes-langchain.memoryBufferWindow',
      typeVersion: 1.3,
      position: [-440, 270],
      parameters: {
        sessionIdType: 'customKey',
        sessionKey: "={{ $('Normalize Input & Session').first().json.sessionId + ':router' }}",
        contextWindowLength: 8,
      },
    },
    {
      id: 'node-parse-plan',
      name: 'Parse Orchestration Plan',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [-180, 30],
      parameters: {
        mode: 'runOnceForAllItems',
        jsCode: PARSE_ROUTER_PLAN_CODE,
      },
    },
    {
      id: 'node-complexity-gate',
      name: 'Complexity Gate (Fast vs Deep)',
      type: 'n8n-nodes-base.if',
      typeVersion: 2.2,
      position: [40, 30],
      parameters: {
        conditions: {
          options: {
            caseSensitive: true,
            leftValue: '',
            typeValidation: 'strict',
            version: 2,
          },
          combinator: 'and',
          conditions: [
            {
              id: 'cond-is-fast',
              leftValue: '={{ $json.complexity }}',
              rightValue: 'fast',
              operator: {
                type: 'string',
                operation: 'equals',
              },
            },
          ],
        },
        options: {},
      },
    },

    // =========================================================================
    // STAGE 2A: SYSTEM 1 FAST PATH
    // =========================================================================
    {
      id: 'node-fast-agent',
      name: 'Fast Direct Agent',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [580, -520],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: '={{ $json.userPrompt }}',
        options: {
          maxIterations: 5,
          systemMessage:
            'You are the System-1 Fast Lane of a Cognitive Multi-Agent Orchestrator. Give a crisp, accurate, direct answer. Use the calculator tool for any arithmetic rather than doing mental math.',
        },
      },
    },
    {
      id: 'node-memory-fast',
      name: 'Session Memory (Fast)',
      type: '@n8n/n8n-nodes-langchain.memoryBufferWindow',
      typeVersion: 1.3,
      position: [460, -320],
      parameters: {
        sessionIdType: 'customKey',
        sessionKey: "={{ $('Normalize Input & Session').first().json.sessionId + ':synth' }}",
        contextWindowLength: 10,
      },
    },
    {
      id: 'node-calc-fast',
      name: 'Calculator (Fast)',
      type: '@n8n/n8n-nodes-langchain.toolCalculator',
      typeVersion: 1,
      position: [700, -320],
      parameters: {},
    },

    // =========================================================================
    // STAGE 2B: SYSTEM 2 PARALLEL SPECIALIST SWARM (3 CONCURRENT SUB-AGENTS)
    // =========================================================================
    {
      id: 'node-agent-researcher',
      name: 'Research & Verification Specialist',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [580, 40],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: "={{ 'User Request: ' + $json.userPrompt + '\\n\\nGoal: ' + $json.intentSummary + '\\n\\nYour Research Brief: ' + $json.researchBrief + '\\n\\nAcceptance Criteria:\\n- ' + ($json.acceptanceCriteria || []).join('\\n- ') }}",
        options: {
          maxIterations: 10,
          systemMessage: `You are Specialist 1: Research & Epistemic Verification Agent inside a Cognitive Multi-Agent Orchestrator.

Your mission is to ground the swarm in verified facts, canonical definitions, industry standards, and empirical constraints.
- Only call Wikipedia for short 1-3 word encyclopedic topics when factual lookup is genuinely needed (skip Wikipedia for pure math/code/Redis engineering prompts); max 1 tool call total.
- Explicitly tag key claims with [VERIFIED], [INFERRED], or [OPEN QUESTION].
- Be ultra-concise and high-signal (strictly under 400 words). Output a tight bulleted Research Dossier.`,
        },
      },
    },
    {
      id: 'node-tool-wiki',
      name: 'Wikipedia Research Tool',
      type: '@n8n/n8n-nodes-langchain.toolWikipedia',
      typeVersion: 1,
      position: [460, 230],
      parameters: {},
    },
    {
      id: 'node-tool-think-researcher',
      name: 'Think Tool (Researcher)',
      type: '@n8n/n8n-nodes-langchain.toolThink',
      typeVersion: 1.1,
      position: [700, 230],
      parameters: {},
    },

    {
      id: 'node-agent-solver',
      name: 'First-Principles Solver & Code Specialist',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [580, 420],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: "={{ 'User Request: ' + $json.userPrompt + '\\n\\nGoal: ' + $json.intentSummary + '\\n\\nYour Solver Brief: ' + $json.architectBrief + '\\n\\nAcceptance Criteria:\\n- ' + ($json.acceptanceCriteria || []).join('\\n- ') }}",
        options: {
          maxIterations: 12,
          systemMessage: `You are Specialist 2: First-Principles Solver, Mathematician & Software Architect inside a Cognitive Multi-Agent Orchestrator.

Your mission is to derive the most rigorous, logically airtight primary solution.
- Verify key calculations or edge cases in at most 1 batched call to js_verification_sandbox (pass a single JS script returning an object with all computed numbers) or Calculator, then immediately output your final dossier.
- Do not repeat redundant tool calls once a value is verified.
- Keep your Technical Solution Dossier crisp and high-density (under 550 words, excluding essential code blocks), including all tool-verified numbers.`,
        },
      },
    },
    {
      id: 'node-tool-calc-solver',
      name: 'Calculator (Solver)',
      type: '@n8n/n8n-nodes-langchain.toolCalculator',
      typeVersion: 1,
      position: [380, 610],
      parameters: {},
    },
    makeSandboxCodeTool({
      id: 'node-tool-sandbox-solver',
      name: 'JS Verification Sandbox',
      position: [580, 610],
      toolName: 'js_verification_sandbox',
      description:
        'Executes a pure JavaScript expression or function body to deterministically verify math, combinatorics, data transforms, algorithmic edge cases, or logic checks. Pass valid JS code returning a value.',
    }),
    {
      id: 'node-tool-think-solver',
      name: 'Think Tool (Solver)',
      type: '@n8n/n8n-nodes-langchain.toolThink',
      typeVersion: 1.1,
      position: [780, 610],
      parameters: {},
    },

    {
      id: 'node-agent-strategist',
      name: 'Lateral Strategist & Contrarian Specialist',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [580, 810],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: "={{ 'User Request: ' + $json.userPrompt + '\\n\\nGoal: ' + $json.intentSummary + '\\n\\nYour Strategy Brief: ' + $json.strategistBrief + '\\n\\nAcceptance Criteria:\\n- ' + ($json.acceptanceCriteria || []).join('\\n- ') }}",
        options: {
          maxIterations: 6,
          systemMessage: `You are Specialist 3: Lateral Strategist, Systems Thinker & Contrarian inside a Cognitive Multi-Agent Orchestrator.

Your mission is to break tunnel vision and discover what a single linear LLM pass misses (keep your response strictly under 350 words, bullet-dense, zero fluff):
1. Challenge the obvious framing: is there a cleaner, faster, cheaper, or more resilient paradigm?
2. Surface non-obvious failure modes, race conditions, security holes, second-order effects, and edge cases.
3. Highlight 2-3 high-leverage upgrades that elevate the answer from "good" to "extraordinary".`,
        },
      },
    },
    {
      id: 'node-tool-think-strategist',
      name: 'Think Tool (Strategist)',
      type: '@n8n/n8n-nodes-langchain.toolThink',
      typeVersion: 1.1,
      position: [580, 1000],
      parameters: {},
    },

    // =========================================================================
    // STAGE 3: FAN-IN AGGREGATION + ADVERSARIAL CRITIC + SELF-REPAIR LOOP
    // =========================================================================
    {
      id: 'node-merge-specialists',
      name: 'Merge Specialist Outputs',
      type: 'n8n-nodes-base.merge',
      typeVersion: 3.2,
      position: [1320, 420],
      parameters: {
        mode: 'append',
        numberInputs: 3,
      },
    },
    {
      id: 'node-assemble-dossier',
      name: 'Assemble Dossier',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [1540, 420],
      parameters: {
        mode: 'runOnceForAllItems',
        jsCode: ASSEMBLE_DOSSIER_CODE,
      },
    },
    {
      id: 'node-agent-critic',
      name: 'Adversarial Critic & Fact-Checker',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [1800, 420],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: "={{ 'ORIGINAL USER REQUEST:\\n' + $json.userPrompt + '\\n\\nACCEPTANCE CRITERIA:\\n- ' + ($json.acceptanceCriteria || []).join('\\n- ') + '\\n\\n--- SPECIALIST 1 (RESEARCH) ---\\n' + $json.researchOutput + '\\n\\n--- SPECIALIST 2 (SOLVER) ---\\n' + $json.solverOutput + '\\n\\n--- SPECIALIST 3 (STRATEGIST) ---\\n' + $json.strategistOutput }}",
        options: {
          maxIterations: 10,
          systemMessage: `You are the Adversarial Red-Team Critic & Fact-Checker inside a Cognitive Multi-Agent Orchestrator.

Your job is to ruthlessly audit the 3 specialist dossiers BEFORE any answer reaches the user:
1. Cross-examine claims across Research, Solver, and Strategist for contradictions or silent assumptions.
2. Only call Calculator if a number looks suspicious or unverified (max 1 tool call); otherwise immediately emit your JSON verdict.
3. Check every Acceptance Criterion and hunt for missing edge cases, bugs, security issues, or hand-waving.
4. Score the combined material honestly from 0 to 100:
   - Score >= ${QUALITY_PASS_THRESHOLD} and verdict "PASS" ONLY if there are no material errors, gaps, or unresolved contradictions.
   - Score < ${QUALITY_PASS_THRESHOLD} and verdict "REVISE" if anything needs surgical repair or deeper verification.

Return STRICT, COMPACT JSON ONLY (under 250 words, no markdown fences, no prose outside JSON):
{
  "verdict": "PASS" | "REVISE",
  "quality_score": 0,
  "verified_strengths": ["Best verified insights to keep"],
  "critical_flaws": ["Concrete errors, contradictions, or gaps found (empty array if none)"],
  "refinement_directives": ["Exact surgical fixes required before synthesis"]
}`,
        },
      },
    },
    {
      id: 'node-tool-calc-critic',
      name: 'Calculator (Critic)',
      type: '@n8n/n8n-nodes-langchain.toolCalculator',
      typeVersion: 1,
      position: [1680, 660],
      parameters: {},
    },
    {
      id: 'node-tool-think-critic',
      name: 'Think Tool (Critic)',
      type: '@n8n/n8n-nodes-langchain.toolThink',
      typeVersion: 1.1,
      position: [1920, 660],
      parameters: {},
    },
    {
      id: 'node-evaluate-critique',
      name: 'Evaluate Critique Score',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [2080, 420],
      parameters: {
        mode: 'runOnceForAllItems',
        jsCode: EVALUATE_CRITIQUE_CODE,
      },
    },
    {
      id: 'node-quality-gate',
      name: 'Quality Gate (Score >= 85?)',
      type: 'n8n-nodes-base.if',
      typeVersion: 2.2,
      position: [2300, 420],
      parameters: {
        conditions: {
          options: {
            caseSensitive: true,
            leftValue: '',
            typeValidation: 'strict',
            version: 2,
          },
          combinator: 'and',
          conditions: [
            {
              id: 'cond-quality-pass',
              leftValue: '={{ $json.gateDecision }}',
              rightValue: 'pass',
              operator: {
                type: 'string',
                operation: 'equals',
              },
            },
          ],
        },
        options: {},
      },
    },
    {
      id: 'node-agent-refiner',
      name: 'Deep Refinement & Repair Agent',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [2560, 640],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: "={{ 'ORIGINAL USER REQUEST:\\n' + $json.userPrompt + '\\n\\nCRITICAL FLAWS TO FIX:\\n- ' + ($json.criticalFlaws || []).join('\\n- ') + '\\n\\nREFINEMENT DIRECTIVES:\\n- ' + ($json.refinementDirectives || []).join('\\n- ') + '\\n\\nSOLVER DRAFT:\\n' + $json.solverOutput + '\\n\\nRESEARCH CONTEXT:\\n' + $json.researchOutput + '\\n\\nSTRATEGIST NOTES:\\n' + $json.strategistOutput }}",
        options: {
          maxIterations: 10,
          systemMessage: `You are the Deep Refinement & Repair Specialist inside a Cognitive Multi-Agent Orchestrator.

The Adversarial Critic flagged specific flaws or gaps (quality score < ${QUALITY_PASS_THRESHOLD}).
Your job is to fix EVERY single flaw with mathematical and engineering rigor:
- Make at most 1 focused tool call (js_repair_sandbox or Calculator) if needed to verify corrected computations, then immediately output your Repair Patch.
- Resolve all contradictions between the earlier specialists explicitly.
- Keep the Repair Patch concise and surgical (under 400 words) so the Executive Lead Synthesizer can merge it cleanly.`,
        },
      },
    },
    makeSandboxCodeTool({
      id: 'node-tool-sandbox-refiner',
      name: 'JS Repair Sandbox',
      position: [2440, 880],
      toolName: 'js_repair_sandbox',
      description:
        'Executes pure JavaScript checks to verify repaired calculations, edge cases, and algorithmic invariants during the self-repair loop.',
    }),
    {
      id: 'node-tool-calc-refiner',
      name: 'Calculator (Refiner)',
      type: '@n8n/n8n-nodes-langchain.toolCalculator',
      typeVersion: 1,
      position: [2680, 880],
      parameters: {},
    },

    // =========================================================================
    // STAGE 4: EXECUTIVE LEAD SYNTHESIZER + UNIFIED RESPONSE
    // =========================================================================
    {
      id: 'node-prepare-synthesis',
      name: 'Prepare Synthesis Context',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [2960, 200],
      parameters: {
        mode: 'runOnceForAllItems',
        jsCode: PREPARE_SYNTHESIS_CODE,
      },
    },
    {
      id: 'node-agent-synthesizer',
      name: 'Executive Lead Synthesizer',
      type: '@n8n/n8n-nodes-langchain.agent',
      typeVersion: 3.1,
      position: [3220, 200],
      retryOnFail: true,
      maxTries: 3,
      waitBetweenTries: 5000,
      parameters: {
        promptType: 'define',
        text: '={{ $json.synthesisBrief }}',
        options: {
          maxIterations: 4,
          systemMessage: `You are the Executive Lead Synthesizer — the final voice of the Cognitive Multi-Agent Orchestrator.

You have received:
1. Verified factual grounding (Research Specialist)
2. Tool-verified primary solution (First-Principles Solver)
3. Alternative framings, failure modes, and upgrades (Lateral Strategist)
4. Adversarial red-team audit (Critic) and any Repair Patch

Craft ONE extraordinary, cohesive, direct, publication-grade final response for the user:
- Lead immediately with the clearest, most actionable answer.
- Seamlessly integrate verified facts, exact calculations/code, and key trade-offs/edge cases caught by the swarm.
- If a Repair Patch is present, treat its verified corrections as authoritative.
- Do NOT expose internal JSON, meta-commentary about prompts, or verbose "Agent 1 said / Agent 2 said" play-by-play — present unified mastery, with crisp structure (headings, code blocks, or tables only where they genuinely increase clarity).`,
        },
      },
    },
    {
      id: 'node-memory-synthesizer',
      name: 'Session Memory (Synthesizer)',
      type: '@n8n/n8n-nodes-langchain.memoryBufferWindow',
      typeVersion: 1.3,
      position: [3220, 440],
      parameters: {
        sessionIdType: 'customKey',
        sessionKey: "={{ $('Normalize Input & Session').first().json.sessionId + ':synth' }}",
        contextWindowLength: 10,
      },
    },
    {
      id: 'node-format-response',
      name: 'Format Final Response',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [3520, -120],
      parameters: {
        mode: 'runOnceForAllItems',
        jsCode: FORMAT_FINAL_RESPONSE_CODE,
      },
    },
  ];

  const connections: N8nConnections = {
    // Universal LLM Toggle Provider Hub -> All 8 AI Agents
    'River AI — Model Engine': {
      ai_languageModel: [
        [
          {
            node: 'Universal LLM Toggle (Model Selector)',
            type: 'ai_languageModel',
            index: 0,
          },
        ],
      ],
    },
    'OpenRouter — Model Engine': {
      ai_languageModel: [
        [
          {
            node: 'Universal LLM Toggle (Model Selector)',
            type: 'ai_languageModel',
            index: 1,
          },
        ],
      ],
    },
    'Universal LLM Toggle (Model Selector)': {
      ai_languageModel: [
        [
          { node: 'Meta-Router & Planner Agent', type: 'ai_languageModel', index: 0 },
          { node: 'Fast Direct Agent', type: 'ai_languageModel', index: 0 },
          { node: 'Research & Verification Specialist', type: 'ai_languageModel', index: 0 },
          { node: 'First-Principles Solver & Code Specialist', type: 'ai_languageModel', index: 0 },
          { node: 'Lateral Strategist & Contrarian Specialist', type: 'ai_languageModel', index: 0 },
          { node: 'Adversarial Critic & Fact-Checker', type: 'ai_languageModel', index: 0 },
          { node: 'Deep Refinement & Repair Agent', type: 'ai_languageModel', index: 0 },
          { node: 'Executive Lead Synthesizer', type: 'ai_languageModel', index: 0 },
        ],
      ],
    },

    // Entry, Centralized LLM Config & Triage
    'Chat Trigger (Orchestrator UI)': {
      main: [[{ node: 'Normalize Input & Session', type: 'main', index: 0 }]],
    },
    'Webhook Trigger (API)': {
      main: [[{ node: 'Normalize Input & Session', type: 'main', index: 0 }]],
    },
    'Normalize Input & Session': {
      main: [[{ node: 'LLM Defaults (Toggle Here)', type: 'main', index: 0 }]],
    },
    'LLM Defaults (Toggle Here)': {
      main: [[{ node: 'Resolve Active LLM Config', type: 'main', index: 0 }]],
    },
    'Resolve Active LLM Config': {
      main: [[{ node: 'Meta-Router & Planner Agent', type: 'main', index: 0 }]],
    },
    'Session Memory (Router)': {
      ai_memory: [
        [{ node: 'Meta-Router & Planner Agent', type: 'ai_memory', index: 0 }],
      ],
    },
    'Meta-Router & Planner Agent': {
      main: [[{ node: 'Parse Orchestration Plan', type: 'main', index: 0 }]],
    },
    'Parse Orchestration Plan': {
      main: [[{ node: 'Complexity Gate (Fast vs Deep)', type: 'main', index: 0 }]],
    },

    // System 1 vs System 2 Split
    'Complexity Gate (Fast vs Deep)': {
      main: [
        // true => Fast lane
        [{ node: 'Fast Direct Agent', type: 'main', index: 0 }],
        // false => Deep parallel specialist swarm (3 concurrent branches)
        [
          { node: 'Research & Verification Specialist', type: 'main', index: 0 },
          { node: 'First-Principles Solver & Code Specialist', type: 'main', index: 0 },
          { node: 'Lateral Strategist & Contrarian Specialist', type: 'main', index: 0 },
        ],
      ],
    },

    // System 1 Fast Direct Agent sub-nodes & output
    'Session Memory (Fast)': {
      ai_memory: [
        [{ node: 'Fast Direct Agent', type: 'ai_memory', index: 0 }],
      ],
    },
    'Calculator (Fast)': {
      ai_tool: [[{ node: 'Fast Direct Agent', type: 'ai_tool', index: 0 }]],
    },
    'Fast Direct Agent': {
      main: [[{ node: 'Format Final Response', type: 'main', index: 0 }]],
    },

    // Specialist 1: Research & Verification
    'Wikipedia Research Tool': {
      ai_tool: [
        [{ node: 'Research & Verification Specialist', type: 'ai_tool', index: 0 }],
      ],
    },
    'Think Tool (Researcher)': {
      ai_tool: [
        [{ node: 'Research & Verification Specialist', type: 'ai_tool', index: 0 }],
      ],
    },
    'Research & Verification Specialist': {
      main: [[{ node: 'Merge Specialist Outputs', type: 'main', index: 0 }]],
    },

    // Specialist 2: First-Principles Solver & Code
    'Calculator (Solver)': {
      ai_tool: [
        [{ node: 'First-Principles Solver & Code Specialist', type: 'ai_tool', index: 0 }],
      ],
    },
    'JS Verification Sandbox': {
      ai_tool: [
        [{ node: 'First-Principles Solver & Code Specialist', type: 'ai_tool', index: 0 }],
      ],
    },
    'Think Tool (Solver)': {
      ai_tool: [
        [{ node: 'First-Principles Solver & Code Specialist', type: 'ai_tool', index: 0 }],
      ],
    },
    'First-Principles Solver & Code Specialist': {
      main: [[{ node: 'Merge Specialist Outputs', type: 'main', index: 1 }]],
    },

    // Specialist 3: Lateral Strategist & Contrarian
    'Think Tool (Strategist)': {
      ai_tool: [
        [{ node: 'Lateral Strategist & Contrarian Specialist', type: 'ai_tool', index: 0 }],
      ],
    },
    'Lateral Strategist & Contrarian Specialist': {
      main: [[{ node: 'Merge Specialist Outputs', type: 'main', index: 2 }]],
    },

    // Stage 3: Fan-In, Adversarial Critic & Conditional Self-Repair
    'Merge Specialist Outputs': {
      main: [[{ node: 'Assemble Dossier', type: 'main', index: 0 }]],
    },
    'Assemble Dossier': {
      main: [[{ node: 'Adversarial Critic & Fact-Checker', type: 'main', index: 0 }]],
    },
    'Calculator (Critic)': {
      ai_tool: [
        [{ node: 'Adversarial Critic & Fact-Checker', type: 'ai_tool', index: 0 }],
      ],
    },
    'Think Tool (Critic)': {
      ai_tool: [
        [{ node: 'Adversarial Critic & Fact-Checker', type: 'ai_tool', index: 0 }],
      ],
    },
    'Adversarial Critic & Fact-Checker': {
      main: [[{ node: 'Evaluate Critique Score', type: 'main', index: 0 }]],
    },
    'Evaluate Critique Score': {
      main: [[{ node: 'Quality Gate (Score >= 85?)', type: 'main', index: 0 }]],
    },
    'Quality Gate (Score >= 85?)': {
      main: [
        // true => PASS directly to synthesis
        [{ node: 'Prepare Synthesis Context', type: 'main', index: 0 }],
        // false => REVISE via Deep Refinement & Repair Agent
        [{ node: 'Deep Refinement & Repair Agent', type: 'main', index: 0 }],
      ],
    },
    'JS Repair Sandbox': {
      ai_tool: [
        [{ node: 'Deep Refinement & Repair Agent', type: 'ai_tool', index: 0 }],
      ],
    },
    'Calculator (Refiner)': {
      ai_tool: [
        [{ node: 'Deep Refinement & Repair Agent', type: 'ai_tool', index: 0 }],
      ],
    },
    'Deep Refinement & Repair Agent': {
      main: [[{ node: 'Prepare Synthesis Context', type: 'main', index: 0 }]],
    },

    // Stage 4: Executive Synthesis & Unified Response
    'Prepare Synthesis Context': {
      main: [[{ node: 'Executive Lead Synthesizer', type: 'main', index: 0 }]],
    },
    'Session Memory (Synthesizer)': {
      ai_memory: [
        [{ node: 'Executive Lead Synthesizer', type: 'ai_memory', index: 0 }],
      ],
    },
    'Executive Lead Synthesizer': {
      main: [[{ node: 'Format Final Response', type: 'main', index: 0 }]],
    },
  };

  return {
    id: ORCHESTRATOR_WORKFLOW_ID,
    name: ORCHESTRATOR_WORKFLOW_NAME,
    active: true,
    nodes,
    connections,
    settings: {
      executionOrder: 'v1',
      binaryMode: 'separate',
      availableInMCP: false,
    },
    meta: {
      templateCredsSetupCompleted: true,
    },
    pinData: {},
    versionId: '7b3e4f92-41f0-4a1d-9c52-8841f0c0v002',
  };
}
