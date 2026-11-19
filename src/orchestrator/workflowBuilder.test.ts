import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_LLM_PROVIDER,
  ORCHESTRATOR_BASE_MODEL,
  ORCHESTRATOR_PROVIDER_BASE_URL,
  ORCHESTRATOR_REASONING_EFFORT,
} from './orchestratorLogic';
import {
  buildMultiAgentOrchestratorWorkflow,
  ORCHESTRATOR_API_WEBHOOK_PATH,
  ORCHESTRATOR_WORKFLOW_ID,
} from './workflowBuilder';

test('Universal LLM Toggle hub wires River AI (slot 0) and OpenRouter (slot 1) to all 8 AI agents with dynamic expressions', () => {
  const wf = buildMultiAgentOrchestratorWorkflow({
    riverAi: { id: 'riverAiCredV4101', name: 'River AI' },
    openRouter: { id: 'o3dspeUuaShSW6fK', name: 'OpenRouter account' },
  });

  assert.equal(wf.id, ORCHESTRATOR_WORKFLOW_ID);
  assert.equal(wf.active, true);

  const nodeByName = new Map(wf.nodes.map((n) => [n.name, n]));

  // Centralized GUI Toggle node defaults to River AI DeepSeek-V4.1-Flash Max Reasoning
  const defaultsNode = nodeByName.get('LLM Defaults (Toggle Here)');
  assert.ok(defaultsNode);
  assert.equal(defaultsNode.type, 'n8n-nodes-base.set');
  assert.equal(defaultsNode.typeVersion, 3.4);
  const assignments = (
    defaultsNode.parameters.assignments as {
      assignments: Array<{ name: string; value: string }>;
    }
  ).assignments;
  const defaultsMap = Object.fromEntries(assignments.map((a) => [a.name, a.value]));
  assert.equal(defaultsMap.defaultProvider, DEFAULT_LLM_PROVIDER);
  assert.equal(defaultsMap.defaultModel, ORCHESTRATOR_BASE_MODEL);
  assert.equal(defaultsMap.defaultReasoningEffort, ORCHESTRATOR_REASONING_EFFORT);

  // Model Selector hub
  const selector = nodeByName.get('Universal LLM Toggle (Model Selector)');
  assert.ok(selector);
  assert.equal(selector.type, '@n8n/n8n-nodes-langchain.modelSelector');
  assert.equal(selector.parameters.numberInputs, 2);
  const rules = (selector.parameters.rules as { rule: Array<{ modelIndex: number }> }).rule;
  assert.equal(rules.length, 3);
  assert.equal(rules[0].modelIndex, 1);
  assert.equal(rules[1].modelIndex, 2);
  assert.equal(rules[2].modelIndex, 1);

  // River AI Model Engine uses dynamic model + dynamic extraBody
  const river = nodeByName.get('River AI — Model Engine');
  assert.ok(river);
  assert.equal(river.type, '@n8n/n8n-nodes-langchain.lmChatOpenAi');
  assert.equal(river.typeVersion, 1.3);
  assert.equal(river.parameters.responsesApiEnabled, false);
  const riverModel = river.parameters.model as { value: string };
  assert.match(riverModel.value, /Resolve Active LLM Config/);
  const riverOpts = river.parameters.options as { baseURL: string; extraBody: string };
  assert.equal(riverOpts.baseURL, ORCHESTRATOR_PROVIDER_BASE_URL);
  assert.match(riverOpts.extraBody, /extraBodyJson/);
  assert.deepEqual(river.credentials?.openAiApi, {
    id: 'riverAiCredV4101',
    name: 'River AI',
  });

  // OpenRouter Model Engine uses dynamic model
  const openRouter = nodeByName.get('OpenRouter — Model Engine');
  assert.ok(openRouter);
  assert.equal(openRouter.type, '@n8n/n8n-nodes-langchain.lmChatOpenRouter');
  assert.match(String(openRouter.parameters.model), /Resolve Active LLM Config/);
  assert.deepEqual(openRouter.credentials?.openRouterApi, {
    id: 'o3dspeUuaShSW6fK',
    name: 'OpenRouter account',
  });

  // Verify input indices into Model Selector
  assert.equal(
    wf.connections['River AI — Model Engine'].ai_languageModel[0][0].index,
    0,
  );
  assert.equal(
    wf.connections['OpenRouter — Model Engine'].ai_languageModel[0][0].index,
    1,
  );

  // Verify ALL 8 AI Agents receive ai_languageModel from the Universal LLM Toggle
  const agentNodes = wf.nodes.filter(
    (n) => n.type === '@n8n/n8n-nodes-langchain.agent',
  );
  assert.equal(agentNodes.length, 8);
  const selectorTargets = wf.connections['Universal LLM Toggle (Model Selector)']
    .ai_languageModel[0]
    .map((e) => e.node)
    .sort();
  assert.deepEqual(
    selectorTargets,
    agentNodes.map((n) => n.name).sort(),
  );
});

test('workflow wires dual triggers (Chat + Webhook), LLM toggle pipeline, 3 parallel specialists, critic, refiner, and synthesizer cleanly', () => {
  const wf = buildMultiAgentOrchestratorWorkflow();
  const nodeByName = new Map(wf.nodes.map((n) => [n.name, n]));

  // Verify every connection source and destination exists
  for (const [sourceName, channelMap] of Object.entries(wf.connections)) {
    assert.ok(nodeByName.has(sourceName), `Missing source node: ${sourceName}`);
    for (const branches of Object.values(channelMap)) {
      for (const branch of branches) {
        for (const edge of branch) {
          assert.ok(
            nodeByName.has(edge.node),
            `Missing target node: ${edge.node} from ${sourceName}`,
          );
        }
      }
    }
  }

  // Verify Normalize -> LLM Defaults -> Resolve Active LLM Config -> Meta-Router
  assert.equal(
    wf.connections['Normalize Input & Session'].main[0][0].node,
    'LLM Defaults (Toggle Here)',
  );
  assert.equal(
    wf.connections['LLM Defaults (Toggle Here)'].main[0][0].node,
    'Resolve Active LLM Config',
  );
  assert.equal(
    wf.connections['Resolve Active LLM Config'].main[0][0].node,
    'Meta-Router & Planner Agent',
  );

  // Verify Chat Trigger uses responseMode: lastNode
  const chatTrigger = nodeByName.get('Chat Trigger (Orchestrator UI)');
  assert.ok(chatTrigger);
  const chatOptions = chatTrigger.parameters.options as Record<string, unknown>;
  assert.equal(chatOptions.responseMode, 'lastNode');

  // Verify Webhook Trigger configuration
  const webhook = nodeByName.get('Webhook Trigger (API)');
  assert.ok(webhook);
  assert.equal(webhook.parameters.httpMethod, 'POST');
  assert.equal(webhook.parameters.path, ORCHESTRATOR_API_WEBHOOK_PATH);
  assert.equal(webhook.parameters.responseMode, 'lastNode');

  // Verify parallel fan-out of 3 distinct specialists from Complexity Gate false branch
  const gateBranches = wf.connections['Complexity Gate (Fast vs Deep)'].main;
  assert.equal(gateBranches[0].length, 1);
  assert.equal(gateBranches[0][0].node, 'Fast Direct Agent');
  assert.equal(gateBranches[1].length, 3);
  assert.deepEqual(
    gateBranches[1].map((e) => e.node),
    [
      'Research & Verification Specialist',
      'First-Principles Solver & Code Specialist',
      'Lateral Strategist & Contrarian Specialist',
    ],
  );
});

test('no two functional nodes overlap and all nodes sit within architectural sticky-note zones', () => {
  const wf = buildMultiAgentOrchestratorWorkflow();
  const functionalNodes = wf.nodes.filter(
    (n) => n.type !== 'n8n-nodes-base.stickyNote',
  );
  const stickyZones = wf.nodes.filter(
    (n) => n.type === 'n8n-nodes-base.stickyNote',
  );

  for (let i = 0; i < functionalNodes.length; i++) {
    for (let j = i + 1; j < functionalNodes.length; j++) {
      const a = functionalNodes[i];
      const b = functionalNodes[j];
      const dx = Math.abs(a.position[0] - b.position[0]);
      const dy = Math.abs(a.position[1] - b.position[1]);
      const overlaps = dx < 120 && dy < 120;
      assert.equal(
        overlaps,
        false,
        `Nodes "${a.name}" (${a.position}) and "${b.name}" (${b.position}) overlap on canvas`,
      );
    }
  }

  for (const node of functionalNodes) {
    const insideAnyZone = stickyZones.some((zone) => {
      const [zx, zy] = zone.position;
      const zw = Number(zone.parameters.width ?? 0);
      const zh = Number(zone.parameters.height ?? 0);
      return (
        node.position[0] >= zx &&
        node.position[0] <= zx + zw &&
        node.position[1] >= zy &&
        node.position[1] <= zy + zh
      );
    });
    assert.equal(
      insideAnyZone,
      true,
      `Node "${node.name}" at [${node.position}] falls outside all sticky-note zones`,
    );
  }
});

test('JS Verification Sandbox handles semicolons, multi-line scripts, fences, and blocks unsafe identifiers', () => {
  const wf = buildMultiAgentOrchestratorWorkflow();
  const sandboxNode = wf.nodes.find((n) => n.name === 'JS Verification Sandbox');
  assert.ok(sandboxNode);
  const jsCode = String(sandboxNode.parameters.jsCode);
  const runSandbox = (query: string) => {
    const fn = new Function('query', jsCode);
    return JSON.parse(String(fn(query))) as {
      ok: boolean;
      result?: unknown;
      error?: string;
      logs?: string[];
    };
  };

  const semi = runSandbox('100000 * 96;');
  assert.equal(semi.ok, true);
  assert.equal(semi.result, 9600000);

  const multi = runSandbox('const bytes = 100000 * 96;\nconsole.log("bytes:", bytes);\nbytes / (1024 * 1024);');
  assert.equal(multi.ok, true);
  assert.ok(Math.abs(Number(multi.result) - 9.1552734375) < 1e-6);
  assert.deepEqual(multi.logs, ['bytes: 9600000']);

  const fenced = runSandbox('```js\nconst a = 847 * 923;\nreturn { product: a };\n```');
  assert.equal(fenced.ok, true);
  assert.deepEqual(fenced.result, { product: 781781 });

  const blocked = runSandbox('process.exit(1)');
  assert.equal(blocked.ok, false);
});
