import test from 'node:test';
import assert from 'node:assert/strict';
import {
  OPENROUTER_DEFAULT_MODEL,
  ORCHESTRATOR_BASE_MODEL,
  buildExtraBodyJson,
  extractJsonObject,
  formatFinalOrchestratorResponse,
  normalizeOrchestratorInput,
  parseCriticEvaluation,
  parseInlineLlmDirective,
  parseRouterPlan,
  resolveLlmToggleConfig,
} from './orchestratorLogic';

test('extractJsonObject parses plain, fenced, and prose-wrapped JSON objects', () => {
  assert.deepEqual(extractJsonObject('{"complexity":"fast"}'), {
    complexity: 'fast',
  });
  assert.deepEqual(
    extractJsonObject('Here is the plan:\n```json\n{"complexity":"deep","domain":"engineering"}\n```'),
    { complexity: 'deep', domain: 'engineering' },
  );
  assert.deepEqual(
    extractJsonObject('Result: {"quality_score":92,"verdict":"PASS"} end.'),
    { quality_score: 92, verdict: 'PASS' },
  );
  assert.equal(extractJsonObject('not valid json at all'), null);
});

test('parseInlineLlmDirective supports provider-only, provider+model, provider+effort, and newline syntax', () => {
  const plain = parseInlineLlmDirective('Explain Raft consensus');
  assert.equal(plain.directiveUsed, false);
  assert.equal(plain.prompt, 'Explain Raft consensus');

  const orOnly = parseInlineLlmDirective('/llm openrouter | What is 19 * 27?');
  assert.equal(orOnly.directiveUsed, true);
  assert.equal(orOnly.providerOverride, 'openrouter');
  assert.equal(orOnly.modelOverride, '');
  assert.equal(orOnly.prompt, 'What is 19 * 27?');

  const riverEffort = parseInlineLlmDirective('/llm river high\nVerify combinatorics');
  assert.equal(riverEffort.directiveUsed, true);
  assert.equal(riverEffort.providerOverride, 'river');
  assert.equal(riverEffort.modelOverride, '');
  assert.equal(riverEffort.reasoningEffortOverride, 'high');
  assert.equal(riverEffort.prompt, 'Verify combinatorics');

  const full = parseInlineLlmDirective(
    '/llm river moonshotai/Kimi-K3.1-Pro 75 | Design a lock-free ring buffer',
  );
  assert.equal(full.directiveUsed, true);
  assert.equal(full.providerOverride, 'river');
  assert.equal(full.modelOverride, 'moonshotai/Kimi-K3.1-Pro');
  assert.equal(full.reasoningEffortOverride, '75');
  assert.equal(full.prompt, 'Design a lock-free ring buffer');
});

test('resolveLlmToggleConfig applies precedence (inline > request > defaults) and provider-aware fallbacks', () => {
  const defaults = {
    defaultProvider: 'river',
    defaultModel: ORCHESTRATOR_BASE_MODEL,
    defaultReasoningEffort: 'max',
  };

  // 1. Default River AI DeepSeek-V4.1-Flash Max Reasoning
  const baseInput = normalizeOrchestratorInput({ chatInput: 'Solve this' }, 1000);
  const baseCfg = resolveLlmToggleConfig(baseInput, defaults);
  assert.equal(baseCfg.provider, 'river');
  assert.equal(baseCfg.model, 'deepseek-ai/DeepSeek-V4.1-Flash');
  assert.equal(baseCfg.reasoningEffort, 'max');
  assert.equal(baseCfg.extraBodyJson, '{"reasoning_effort":"max"}');
  assert.equal(baseCfg.overrideSource, 'default');

  // 2. Webhook switches provider to openrouter without explicit model => smart fallback to OpenRouter default model
  const orReq = normalizeOrchestratorInput(
    {
      headers: { 'content-type': 'application/json' },
      body: { prompt: 'What is 19 * 27?', provider: 'openrouter' },
    },
    1000,
  );
  const orCfg = resolveLlmToggleConfig(orReq, defaults);
  assert.equal(orCfg.provider, 'openrouter');
  assert.equal(orCfg.model, OPENROUTER_DEFAULT_MODEL);
  assert.equal(orCfg.reasoningEffort, 'off');
  assert.equal(orCfg.extraBodyJson, '{}');
  assert.equal(orCfg.overrideSource, 'request');

  // 3. Inline /llm directive beats Webhook fields
  const both = normalizeOrchestratorInput(
    {
      headers: {},
      body: {
        prompt: '/llm river deepseek-ai/DeepSeek-V4.1-Flash 60 | Compute 2^10',
        provider: 'openrouter',
        model: 'baidu/ernie-5.5-flash-preview:free',
      },
    },
    1000,
  );
  const inlineCfg = resolveLlmToggleConfig(both, defaults);
  assert.equal(both.userPrompt, 'Compute 2^10');
  assert.equal(inlineCfg.provider, 'river');
  assert.equal(inlineCfg.model, 'deepseek-ai/DeepSeek-V4.1-Flash');
  assert.equal(inlineCfg.reasoningEffort, '60');
  assert.equal(inlineCfg.extraBodyJson, '{"reasoning_effort":60}');
  assert.equal(inlineCfg.overrideSource, 'inline');

  // 4. Reasoning effort off produces empty JSON object
  assert.equal(buildExtraBodyJson('river', 'off'), '{}');
  assert.equal(buildExtraBodyJson('openrouter', 'max'), '{}');
});

test('normalizeOrchestratorInput handles both Chat Trigger and Webhook payloads', () => {
  const chat = normalizeOrchestratorInput(
    { chatInput: 'Design a consensus algorithm', sessionId: 'chat-42' },
    1000,
  );
  assert.equal(chat.source, 'chat');
  assert.equal(chat.userPrompt, 'Design a consensus algorithm');
  assert.equal(chat.sessionId, 'chat-42');
  assert.equal(chat.forceDeepMode, false);

  const webhook = normalizeOrchestratorInput(
    {
      headers: { 'content-type': 'application/json' },
      body: {
        prompt: 'Audit this SQL schema',
        mode: 'deep',
        sessionId: 'api-9',
        provider: 'openrouter',
        model: 'baidu/ernie-5.5-flash-preview:free',
      },
    },
    2000,
  );
  assert.equal(webhook.source, 'webhook');
  assert.equal(webhook.userPrompt, 'Audit this SQL schema');
  assert.equal(webhook.sessionId, 'api-9');
  assert.equal(webhook.forceDeepMode, true);
  assert.equal(webhook.requestProviderOverride, 'openrouter');
  assert.equal(webhook.requestModelOverride, 'baidu/ernie-5.5-flash-preview:free');
});

test('parseRouterPlan respects fast/deep routes, forceDeepMode, and malformed fallbacks', () => {
  const fast = parseRouterPlan(
    JSON.stringify({
      complexity: 'fast',
      domain: 'math_logic',
      intent_summary: 'Multiply two numbers',
      acceptance_criteria: ['Exact integer output'],
    }),
    'What is 17 * 23?',
    false,
  );
  assert.equal(fast.complexity, 'fast');
  assert.equal(fast.domain, 'math_logic');
  assert.equal(fast.routerFallbackUsed, false);

  const forcedDeep = parseRouterPlan(
    JSON.stringify({ complexity: 'fast', domain: 'general' }),
    'What is 17 * 23?',
    true,
  );
  assert.equal(forcedDeep.complexity, 'deep');

  const fallback = parseRouterPlan('unparseable output', 'Build an LRU cache', false);
  assert.equal(fallback.complexity, 'deep');
  assert.equal(fallback.routerFallbackUsed, true);
  assert.ok(fallback.acceptanceCriteria.length >= 3);
});

test('parseCriticEvaluation enforces score >= 85 pass threshold and triggers self-repair otherwise', () => {
  const pass = parseCriticEvaluation(
    JSON.stringify({
      verdict: 'PASS',
      quality_score: 91,
      verified_strengths: ['Math verified in sandbox'],
      critical_flaws: [],
      refinement_directives: [],
    }),
  );
  assert.equal(pass.verdict, 'PASS');
  assert.equal(pass.qualityScore, 91);
  assert.equal(pass.needsRefinement, false);

  const lowScorePass = parseCriticEvaluation(
    JSON.stringify({
      verdict: 'PASS',
      quality_score: 84,
      critical_flaws: ['Edge case for n=0 omitted'],
    }),
  );
  assert.equal(lowScorePass.verdict, 'REVISE');
  assert.equal(lowScorePass.needsRefinement, true);

  const highScoreRevise = parseCriticEvaluation(
    JSON.stringify({
      verdict: 'REVISE',
      quality_score: 88,
      critical_flaws: ['Security warning missing'],
    }),
  );
  assert.equal(highScoreRevise.verdict, 'REVISE');
  assert.equal(highScoreRevise.needsRefinement, true);

  const malformed = parseCriticEvaluation('looks mostly fine');
  assert.equal(malformed.criticFallbackUsed, true);
  assert.equal(malformed.needsRefinement, true);
});

test('formatFinalOrchestratorResponse emits dynamic active LLM provider/model metadata', () => {
  const res = formatFinalOrchestratorResponse({
    finalText: '  Final verified answer  ',
    route: 'deep',
    domain: 'engineering',
    qualityScore: 94,
    verdict: 'PASS',
    refined: false,
    source: 'webhook',
    sessionId: 'sess-1',
    startedAt: 1000,
    nowMs: 4250,
    acceptanceCriteriaCount: 4,
    criticalFlawsFound: 0,
    provider: 'openrouter',
    model: OPENROUTER_DEFAULT_MODEL,
    reasoningEffort: 'off',
    overrideSource: 'request',
  });

  assert.equal(res.output, 'Final verified answer');
  assert.equal(res.text, 'Final verified answer');
  assert.equal(res.metadata.provider, 'openrouter');
  assert.equal(res.metadata.model, OPENROUTER_DEFAULT_MODEL);
  assert.equal(res.metadata.reasoningEffort, 'off');
  assert.equal(res.metadata.overrideSource, 'request');
  assert.equal(res.metadata.latencyMs, 3250);
  assert.equal(res.metadata.specialistCount, 6);
});
