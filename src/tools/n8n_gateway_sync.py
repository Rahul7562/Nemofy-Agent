"""Phase 9+: n8n Autonomous Agent Gateway Workflow Builder & Sync Utility.

Creates and imports the ``LangGraph Autonomous AI Agent Gateway`` workflow into the
existing n8n installation WITHOUT modifying or replacing existing workflows, credentials,
or the 5678 -> 3000 web mapping.

Canvas Plug-and-Play Architecture (strict 3-system separation — zero AI Agents in n8n):
1. **Brain & Sub-Agent Model Slots**:
   - ``Brain LLM (OpenAI Chat Model)`` -> ``Read Brain LLM``
   - ``Sub-Agent LLM (OpenAI Chat Model)`` -> ``Read Sub-Agent LLM``
2. **Skills Slot (Unlimited Plug-and-Play)**:
   - Wire any number of native ``Default Data Loader`` nodes (containing Markdown skills)
     into ``Read Canvas Skills`` (``ai_document``, ``maxConnections: -1``).
3. **MCP Servers & Tools Slot (Unlimited Plug-and-Play)**:
   - Wire any number of native ``MCP Client Tool`` nodes (or any n8n AI Tool node) into
     ``Agent Heartbeat & MCP Bridge`` (``ai_tool``, ``maxConnections: -1``).
4. **Long-Running Heartbeat & Stall Watchdog (Zero 120s Hard Timeout)**:
   - ``Start Python Agent Job`` stages an async job in <50ms via ``POST /agent/jobs/create``.
   - ``Agent Heartbeat & MCP Bridge`` registers all plugged-in MCP/AI tools, starts the job,
     long-polls ``GET /agent/jobs/{job_id}?wait_ms=1500``, executes any canvas MCP tool calls
     live using n8n's authenticated MCP sessions, and ONLY fails if the Python watchdog
     detects a genuine stall (zero forward progress) or the backend goes unreachable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

GATEWAY_WORKFLOW_ID = "lgAgentGateway01"
GATEWAY_WORKFLOW_NAME = "LangGraph Autonomous AI Agent Gateway"
GATEWAY_WEBHOOK_PATH = "langgraph-agent/run"
GATEWAY_CHAT_WEBHOOK_ID = "9c4e1a20-7b12-4d9f-a101-8841f0c0b001"
GATEWAY_API_WEBHOOK_ID = "9c4e1a20-7b12-4d9f-a101-8841f0c0b002"
GATEWAY_CREDENTIAL_ID = "lgAgentApiAuth01"
GATEWAY_CREDENTIAL_NAME = "LangGraph Agent API Bearer Auth"
RIVER_OPENAI_CREDENTIAL_ID = "riverAiOpenAi01"
RIVER_OPENAI_CREDENTIAL_NAME = "River AI (OpenAI Compatible)"
N8N_MCP_BEARER_CREDENTIAL_ID = "n8nMcpBearer01"
N8N_MCP_BEARER_CREDENTIAL_NAME = "n8n Built-in MCP Bearer Auth"


def _make_read_llm_node_js(target_key: str) -> str:
    """Build LangChain Code JS that reads model/baseURL/apiKey from a connected n8n Chat Model node."""
    return r"""
const items = this.getInputData();
const first = items[0]?.json ? { ...items[0].json } : {};

const inferProvider = (baseUrl, modelName) => {
  const u = String(baseUrl || '').toLowerCase();
  if (u.includes('river.ai')) return 'river_ai';
  if (u.includes('openrouter.ai')) return 'openrouter';
  if (u.includes('generativelanguage.googleapis.com')) return 'gemini';
  if (u.includes('api.groq.com')) return 'groq';
  if (u.includes('api.deepseek.com')) return 'deepseek';
  if (u.includes('together.xyz')) return 'together';
  if (u.includes('fireworks.ai')) return 'fireworks';
  if (u.includes('api.anthropic.com')) return 'anthropic';
  if (u.includes('11434') || u.includes('ollama')) return 'ollama';
  if (u.includes('localhost') || u.includes('127.0.0.1')) return 'local';
  if (u.includes('api.openai.com')) return 'openai';
  if (u) return 'custom';
  const m = String(modelName || '').toLowerCase();
  if (m.startsWith('gpt-') || m.startsWith('o1') || m.startsWith('o3') || m.startsWith('o4')) return 'openai';
  if (m.startsWith('claude')) return 'anthropic';
  if (m.startsWith('gemini')) return 'gemini';
  return 'openai';
};

let extracted = null;
try {
  const conn = await this.getInputConnectionData('ai_languageModel', 0);
  const llm = Array.isArray(conn) ? conn[0] : conn;
  if (llm && typeof llm === 'object') {
    const kwargs = llm.lc_kwargs || {};
    const cfg = llm.clientConfig || kwargs.configuration || llm.configuration || {};
    const model = String(
      llm.model || llm.modelName || kwargs.model || kwargs.modelName || ''
    ).trim();
    const base_url = String(
      cfg.baseURL || cfg.baseUrl || llm.openAIApiBase || kwargs.baseURL || ''
    ).trim();
    const api_key = String(
      llm.openAIApiKey || cfg.apiKey || kwargs.apiKey || kwargs.openAIApiKey || ''
    ).trim();
    const temperature =
      typeof llm.temperature === 'number' && Number.isFinite(llm.temperature)
        ? llm.temperature
        : undefined;
    const max_tokens =
      typeof llm.maxTokens === 'number' && Number.isFinite(llm.maxTokens) && llm.maxTokens > 0
        ? Math.round(llm.maxTokens)
        : undefined;
    const reasoning_effort = String(
      llm.modelKwargs?.reasoning_effort ||
      kwargs.modelKwargs?.reasoning_effort ||
      llm.reasoning?.effort ||
      ''
    ).trim();

    if (model || base_url || api_key) {
      extracted = {
        provider: inferProvider(base_url, model),
        ...(model ? { model } : {}),
        ...(base_url ? { base_url } : {}),
        ...(api_key ? { api_key } : {}),
        ...(temperature !== undefined ? { temperature } : {}),
        ...(max_tokens !== undefined ? { max_tokens } : {}),
        ...(reasoning_effort ? { reasoning_effort } : {}),
      };
    }
  }
} catch (_err) {
  extracted = null;
}

return [
  {
    json: {
      ...first,
      __TARGET_KEY__: extracted,
    },
  },
];
""".strip().replace("__TARGET_KEY__", target_key)


READ_BRAIN_LLM_JS = _make_read_llm_node_js("_n8n_brain_llm")
READ_SUBAGENT_LLM_JS = _make_read_llm_node_js("_n8n_subagent_llm")


READ_CANVAS_SKILLS_JS = r"""
const items = this.getInputData();
const first = items[0]?.json ? { ...items[0].json } : {};
const discoveredSkills = [];

const normalizeSkillName = (raw, fallback) => {
  const cleaned = String(raw || '')
    .toLowerCase()
    .replace(/^skill\s*[:\-]\s*/i, '')
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^[_-]+|[_-]+$/g, '');
  if (!cleaned) return fallback;
  return /^[a-z0-9]/.test(cleaned) ? cleaned.slice(0, 64) : ('skill_' + cleaned).slice(0, 64);
};

const addSkill = (content, meta, fallbackName) => {
  const text = String(content || '').trim();
  if (!text) return;
  const m = meta && typeof meta === 'object' ? meta : {};
  const explicitName = String(m.name || m.skill_name || m.title || '').trim();
  const explicitDesc = String(m.description || '').trim();
  const nodeLabel = String(m._nodeName || fallbackName || 'canvas_skill').trim();
  discoveredSkills.push({
    name: normalizeSkillName(explicitName || nodeLabel, 'canvas_skill_' + (discoveredSkills.length + 1)),
    ...(explicitDesc ? { description: explicitDesc } : {}),
    content: text,
    source: nodeLabel ? ('n8n_canvas:' + nodeLabel) : 'n8n_canvas',
  });
};

try {
  const conn = await this.getInputConnectionData('ai_document', 0);
  const loaders = Array.isArray(conn) ? conn : (conn ? [conn] : []);

  for (let i = 0; i < loaders.length; i++) {
    const entry = loaders[i];
    if (!entry) continue;

    // 1. Native Default Data Loader (N8nJsonLoader / N8nBinaryLoader)
    if (typeof entry.processItem === 'function') {
      const nodeName = entry.context?.getNode?.()?.name || ('Skill ' + (i + 1));
      try {
        // Prevent 1000-char chunk splitting so long Markdown skills stay 100% intact
        const prevSplitter = entry.textSplitter;
        entry.textSplitter = null;
        const docs = await entry.processItem({ json: first }, 0);
        entry.textSplitter = prevSplitter;
        const docList = Array.isArray(docs) ? docs : (docs ? [docs] : []);
        if (docList.length > 0) {
          const mergedContent = docList
            .map((d) => String(d?.pageContent || '').trim())
            .filter(Boolean)
            .join('\n\n');
          const firstMeta = { ...(docList[0]?.metadata || {}), _nodeName: nodeName };
          addSkill(mergedContent, firstMeta, nodeName);
          continue;
        }
      } catch (_loaderErr) {
        // Fallback: read jsonData directly if loader had non-standard input
        try {
          const rawParam = entry.context?.getNodeParameter?.('jsonData', 0);
          if (rawParam) {
            addSkill(
              typeof rawParam === 'string' ? rawParam : JSON.stringify(rawParam, null, 2),
              { _nodeName: nodeName },
              nodeName
            );
          }
        } catch (_ignored) {}
      }
      continue;
    }

    // 2. LangChain Document or array of Documents
    if (Array.isArray(entry)) {
      const merged = entry
        .map((d) => String(d?.pageContent || d?.content || '').trim())
        .filter(Boolean)
        .join('\n\n');
      addSkill(merged, entry[0]?.metadata || entry[0], 'canvas_skill_' + (i + 1));
      continue;
    }

    if (typeof entry === 'object') {
      const content = entry.pageContent || entry.content || entry.instructions || entry.markdown;
      if (content) {
        addSkill(content, entry.metadata || entry, entry.name || ('canvas_skill_' + (i + 1)));
      }
    } else if (typeof entry === 'string') {
      addSkill(entry, {}, 'canvas_skill_' + (i + 1));
    }
  }
} catch (_err) {
  // No canvas skill nodes connected — continue cleanly with disk skills
}

return [
  {
    json: {
      ...first,
      _n8n_canvas_skills: discoveredSkills,
    },
  },
];
""".strip()


VALIDATE_INPUT_JS = r"""
const item = $input.first()?.json ?? {};
const isWebhook = Boolean(item.body && typeof item.body === 'object');
const payload = isWebhook ? item.body : item;

const pickString = (v) => (typeof v === 'string' ? v.trim() : '');
const pickNumber = (v) => {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v.trim());
    if (Number.isFinite(n)) return n;
  }
  return undefined;
};

const n8nBrainNode = item._n8n_brain_llm && typeof item._n8n_brain_llm === 'object'
  ? item._n8n_brain_llm
  : {};
const n8nSubagentNode = item._n8n_subagent_llm && typeof item._n8n_subagent_llm === 'object'
  ? item._n8n_subagent_llm
  : {};

const canvasSkillsFromNodes = Array.isArray(item._n8n_canvas_skills)
  ? item._n8n_canvas_skills
  : [];
const payloadSkills = Array.isArray(payload.canvas_skills)
  ? payload.canvas_skills
  : Array.isArray(payload.skills)
    ? payload.skills
    : [];
const mergedCanvasSkills = [...canvasSkillsFromNodes, ...payloadSkills]
  .filter((s) => s && typeof s === 'object' && typeof (s.content || s.instructions || s.markdown) === 'string')
  .map((s) => ({
    ...(pickString(s.name) ? { name: pickString(s.name) } : {}),
    ...(pickString(s.description) ? { description: pickString(s.description) } : {}),
    content: String(s.content || s.instructions || s.markdown).trim(),
    source: pickString(s.source) || 'n8n_canvas',
  }))
  .filter((s) => s.content.length > 0);

const payloadMcpServers = Array.isArray(payload.mcp_servers) ? payload.mcp_servers : [];

const KNOWN_PROVIDERS = new Set([
  'river_ai', 'river', 'openrouter', 'openai', 'gemini', 'google',
  'groq', 'deepseek', 'together', 'fireworks', 'anthropic', 'ollama', 'vllm', 'local', 'custom',
]);

const parseModelSpecToken = (token) => {
  const t = String(token || '').trim();
  if (!t) return {};
  const idx = t.indexOf(':');
  if (idx > 0) {
    const maybeProvider = t.slice(0, idx).toLowerCase();
    const rest = t.slice(idx + 1).trim();
    if (KNOWN_PROVIDERS.has(maybeProvider)) {
      return { provider: maybeProvider, ...(rest ? { model: rest } : {}) };
    }
  }
  return { model: t };
};

let rawTask =
  pickString(payload.task) ||
  pickString(payload.chatInput) ||
  pickString(payload.prompt) ||
  pickString(payload.message) ||
  pickString(payload.input);

// Optional inline chat directives: /brain provider:model, /subagent provider:model, /model provider:model
const inlineBrain = {};
const inlineSub = {};
if (rawTask) {
  rawTask = rawTask
    .replace(/(?:^|\s)\/brain\s+([^\s]+)/gi, (_, tok) => {
      Object.assign(inlineBrain, parseModelSpecToken(tok));
      return ' ';
    })
    .replace(/(?:^|\s)\/(?:subagent|worker)\s+([^\s]+)/gi, (_, tok) => {
      Object.assign(inlineSub, parseModelSpecToken(tok));
      return ' ';
    })
    .replace(/(?:^|\s)\/model\s+([^\s]+)/gi, (_, tok) => {
      const parsed = parseModelSpecToken(tok);
      Object.assign(inlineBrain, parsed);
      if (!inlineSub.model && !inlineSub.provider) {
        Object.assign(inlineSub, parsed);
      }
      return ' ';
    })
    .trim();
}

const mergeRoleModel = (rolePrefix, payloadObj, inlineObj, n8nNodeCfg) => {
  const obj = payloadObj && typeof payloadObj === 'object' ? payloadObj : {};
  const nodeProvider = pickString(n8nNodeCfg.provider).toLowerCase();

  const requestedProvider =
    pickString(inlineObj.provider) ||
    pickString(obj.provider) ||
    pickString(payload[rolePrefix + '_provider']);

  const provider = (requestedProvider || nodeProvider || '').toLowerCase();
  const switchedAwayFromNodeProvider =
    Boolean(requestedProvider) && Boolean(nodeProvider) && provider !== nodeProvider;

  const model =
    pickString(inlineObj.model) ||
    pickString(obj.model) ||
    pickString(payload[rolePrefix + '_model']) ||
    pickString(n8nNodeCfg.model);

  const explicitBaseUrl =
    pickString(obj.base_url) ||
    pickString(obj.baseUrl) ||
    pickString(payload[rolePrefix + '_base_url']);

  const base_url =
    explicitBaseUrl || (switchedAwayFromNodeProvider ? '' : pickString(n8nNodeCfg.base_url));

  const explicitApiKey =
    pickString(obj.api_key) ||
    pickString(obj.apiKey) ||
    pickString(payload[rolePrefix + '_api_key']);

  const api_key =
    explicitApiKey || (switchedAwayFromNodeProvider ? '' : pickString(n8nNodeCfg.api_key));

  const temperature =
    pickNumber(obj.temperature) ??
    pickNumber(payload[rolePrefix + '_temperature']) ??
    pickNumber(n8nNodeCfg.temperature);

  const max_tokens =
    pickNumber(obj.max_tokens ?? obj.maxTokens) ??
    pickNumber(payload[rolePrefix + '_max_tokens']) ??
    pickNumber(n8nNodeCfg.max_tokens);

  const reasoning_effort =
    pickString(obj.reasoning_effort) ||
    pickString(obj.reasoningEffort) ||
    pickString(payload[rolePrefix + '_reasoning_effort']) ||
    pickString(n8nNodeCfg.reasoning_effort);

  const out = {};
  if (provider) out.provider = provider;
  if (model) out.model = model;
  if (base_url) out.base_url = base_url;
  if (api_key) out.api_key = api_key;
  if (temperature !== undefined) out.temperature = temperature;
  if (max_tokens !== undefined) out.max_tokens = Math.round(max_tokens);
  if (reasoning_effort) out.reasoning_effort = reasoning_effort;
  return Object.keys(out).length > 0 ? out : undefined;
};

const primaryModelOverride = mergeRoleModel(
  'brain',
  payload.primary_model || payload.brain_model_config,
  inlineBrain,
  n8nBrainNode
);
const workerModelOverride = mergeRoleModel(
  'subagent',
  payload.worker_model || payload.subagent_model_config,
  inlineSub,
  n8nSubagentNode
);

const decision = pickString(payload.decision).toLowerCase();
const isResume = ['approve', 'reject', 'edit'].includes(decision);

const threadId =
  pickString(payload.thread_id) ||
  pickString(payload.threadId) ||
  pickString(payload.sessionId) ||
  pickString(item.sessionId) ||
  ('n8n-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8));

const maxTurnsRaw = Number(payload.max_turns ?? payload.maxTurns);
const maxTurns =
  Number.isInteger(maxTurnsRaw) && maxTurnsRaw >= 1 && maxTurnsRaw <= 100
    ? maxTurnsRaw
    : undefined;

const stallTimeoutRaw = pickNumber(payload.stall_timeout_seconds ?? payload.stallTimeoutSeconds);
const stallTimeoutSeconds =
  stallTimeoutRaw !== undefined && stallTimeoutRaw >= 0.2 && stallTimeoutRaw <= 3600
    ? stallTimeoutRaw
    : undefined;

let isValid = true;
let validationError = '';

if (isResume) {
  if (!pickString(payload.thread_id) && !pickString(payload.threadId) && !pickString(payload.sessionId)) {
    isValid = false;
    validationError = 'thread_id is required when resuming an approval decision.';
  }
} else if (!rawTask) {
  isValid = false;
  validationError = 'Task is required. Provide a non-empty task, prompt, message, or chatInput.';
} else if (rawTask.length > 32000) {
  isValid = false;
  validationError = 'Task exceeds maximum allowed length of 32,000 characters.';
}

const endpoint = isResume ? 'resume' : 'run';
const jobCreateBody = isResume
  ? {
      endpoint: 'resume',
      thread_id: threadId,
      decision,
      ...(payload.arguments && typeof payload.arguments === 'object'
        ? { arguments: payload.arguments }
        : {}),
      ...(pickString(payload.reason) ? { reason: pickString(payload.reason) } : {}),
      ...(primaryModelOverride ? { primary_model: primaryModelOverride } : {}),
      ...(workerModelOverride ? { worker_model: workerModelOverride } : {}),
      ...(mergedCanvasSkills.length > 0 ? { canvas_skills: mergedCanvasSkills } : {}),
      ...(payloadMcpServers.length > 0 ? { mcp_servers: payloadMcpServers } : {}),
      ...(stallTimeoutSeconds !== undefined ? { stall_timeout_seconds: stallTimeoutSeconds } : {}),
    }
  : {
      endpoint: 'run',
      task: rawTask,
      thread_id: threadId,
      ...(maxTurns !== undefined ? { max_turns: maxTurns } : {}),
      ...(primaryModelOverride ? { primary_model: primaryModelOverride } : {}),
      ...(workerModelOverride ? { worker_model: workerModelOverride } : {}),
      ...(mergedCanvasSkills.length > 0 ? { canvas_skills: mergedCanvasSkills } : {}),
      ...(payloadMcpServers.length > 0 ? { mcp_servers: payloadMcpServers } : {}),
      ...(stallTimeoutSeconds !== undefined ? { stall_timeout_seconds: stallTimeoutSeconds } : {}),
    };

// Never include api_key in selected_models metadata
const sanitizeSummary = (cfg) =>
  cfg
    ? {
        provider: cfg.provider || 'custom',
        model: cfg.model || '',
        base_url: cfg.base_url || '',
      }
    : null;

return [
  {
    json: {
      isValid,
      validationError,
      source: isWebhook ? 'webhook' : 'chat',
      endpoint,
      thread_id: threadId,
      task: rawTask,
      selected_models: {
        brain: sanitizeSummary(primaryModelOverride),
        subagent: sanitizeSummary(workerModelOverride),
      },
      canvas_skills_count: mergedCanvasSkills.length,
      canvas_skill_names: mergedCanvasSkills.map((s) => s.name || 'unnamed'),
      requestBody: jobCreateBody,
      startedAt: Date.now(),
    },
  },
];
""".strip()


def _make_heartbeat_bridge_js(agent_api_base_url: str) -> str:
    """Build LangChain Code JS for the Heartbeat Monitor + Plug-and-Play MCP Client Tool Bridge."""
    base_url = agent_api_base_url.rstrip("/")
    return r"""
const API_BASE = '__API_BASE__';
const items = this.getInputData();
const startResp = items[0]?.json ?? {};
const httpReq =
  (typeof helpers !== 'undefined' && helpers && typeof helpers.httpRequest === 'function'
    ? helpers.httpRequest.bind(helpers)
    : this?.helpers?.httpRequest?.bind(this.helpers));

const statusCode = Number(startResp.statusCode ?? startResp.status ?? 200);
const startBody =
  startResp.body && typeof startResp.body === 'object' ? startResp.body : startResp;
const startErr =
  startResp.error && typeof startResp.error === 'object'
    ? (startResp.error.message || JSON.stringify(startResp.error))
    : typeof startResp.error === 'string'
      ? startResp.error
      : '';

// If Start Python Agent Job failed (e.g. API offline or 401/503), pass error cleanly to formatter
if (
  (Number.isFinite(statusCode) && statusCode >= 400) ||
  !startBody.job_id ||
  !startBody.job_token
) {
  return [
    {
      json: {
        statusCode: Number.isFinite(statusCode) && statusCode >= 400 ? statusCode : 502,
        body: startBody,
        error: startErr || startBody.detail || 'Could not start Python LangGraph Agent job.',
      },
    },
  ];
}

const jobId = String(startBody.job_id);
const jobToken = String(startBody.job_token);
const jobHeaders = {
  'Content-Type': 'application/json',
  Accept: 'application/json',
  'X-Job-Token': jobToken,
};

// --- Helper: Convert Zod v3 / Zod v4 / JSON Schema into standard JSON Schema ---
const unwrapZod = (schema) => {
  let cur = schema;
  let optional = false;
  let defaultVal = undefined;
  let description = cur?.description || '';
  for (let depth = 0; depth < 8 && cur && typeof cur === 'object'; depth++) {
    const def = cur._zod?.def || cur._def || {};
    if (!description && (def.description || cur.description)) {
      description = def.description || cur.description || '';
    }
    const kind = String(def.typeName || def.type || '').toLowerCase();
    if (kind === 'zodoptional' || kind === 'optional' || kind === 'zodnullable' || kind === 'nullable') {
      optional = true;
      cur = def.innerType || def.schema || cur.unwrap?.();
      continue;
    }
    if (kind === 'zoddefault' || kind === 'default') {
      optional = true;
      defaultVal = typeof def.defaultValue === 'function' ? def.defaultValue() : def.defaultValue;
      cur = def.innerType || def.schema;
      continue;
    }
    if (kind === 'zodeffects' || kind === 'effects' || kind === 'pipe') {
      cur = def.schema || def.in || def.innerType;
      continue;
    }
    break;
  }
  return { node: cur, optional, defaultVal, description };
};

const zodToJsonSchema = (schema) => {
  if (!schema || typeof schema !== 'object') {
    return { type: 'object', properties: {} };
  }
  // Already plain JSON Schema
  if (schema.type === 'object' && schema.properties && !schema._def && !schema._zod) {
    return schema;
  }
  const { node, description } = unwrapZod(schema);
  const def = node?._zod?.def || node?._def || {};
  const kind = String(def.typeName || def.type || '').toLowerCase();

  if (kind === 'zodstring' || kind === 'string') {
    return { type: 'string', ...(description ? { description } : {}) };
  }
  if (kind === 'zodnumber' || kind === 'number' || kind === 'int') {
    return { type: 'number', ...(description ? { description } : {}) };
  }
  if (kind === 'zodboolean' || kind === 'boolean') {
    return { type: 'boolean', ...(description ? { description } : {}) };
  }
  if (kind === 'zodarray' || kind === 'array') {
    const elem = def.type || def.element;
    return {
      type: 'array',
      items: elem ? zodToJsonSchema(elem) : { type: 'string' },
      ...(description ? { description } : {}),
    };
  }
  if (kind === 'zodenum' || kind === 'enum') {
    const vals = Array.isArray(def.values)
      ? def.values
      : def.entries
        ? Object.values(def.entries)
        : undefined;
    return { type: 'string', ...(vals ? { enum: vals } : {}), ...(description ? { description } : {}) };
  }
  if (kind === 'zodobject' || kind === 'object') {
    const rawShape = typeof def.shape === 'function' ? def.shape() : (def.shape || node.shape || {});
    const properties = {};
    const required = [];
    for (const [key, fieldSchema] of Object.entries(rawShape)) {
      const unwrapped = unwrapZod(fieldSchema);
      const propJson = zodToJsonSchema(fieldSchema);
      if (unwrapped.description && !propJson.description) {
        propJson.description = unwrapped.description;
      }
      if (unwrapped.defaultVal !== undefined) {
        propJson.default = unwrapped.defaultVal;
      }
      properties[key] = propJson;
      if (!unwrapped.optional) {
        required.push(key);
      }
    }
    return {
      type: 'object',
      properties,
      ...(required.length > 0 ? { required } : {}),
      ...(description ? { description } : {}),
    };
  }
  return { type: 'object', properties: {}, ...(description ? { description } : {}) };
};

// --- Step 1: Collect all connected native MCP Client Tool / n8n AI Tool nodes ---
const liveToolsByName = {};
const bridgeToolSpecs = [];
const bridgeWarnings = [];

const registerTool = (toolObj) => {
  if (!toolObj || typeof toolObj !== 'object' || !toolObj.name) return;
  const name = String(toolObj.name).trim();
  if (!name || liveToolsByName[name]) return;
  liveToolsByName[name] = toolObj;
  let argsSchema = { type: 'object', properties: {} };
  try {
    if (toolObj.schema) {
      argsSchema = zodToJsonSchema(toolObj.schema);
    }
  } catch (_schemaErr) {
    argsSchema = { type: 'object', properties: {} };
  }
  bridgeToolSpecs.push({
    name,
    description: String(toolObj.description || '').slice(0, 4000),
    args_schema: argsSchema,
    is_mcp: true,
  });
};

try {
  const connectedTools = await this.getInputConnectionData('ai_tool', 0);
  const rawList = Array.isArray(connectedTools)
    ? connectedTools
    : connectedTools
      ? [connectedTools]
      : [];

  for (const entry of rawList) {
    if (!entry) continue;
    if (typeof entry.getTools === 'function') {
      const subTools = entry.getTools();
      if (Array.isArray(subTools)) {
        for (const t of subTools) registerTool(t);
      }
    } else if (Array.isArray(entry.tools)) {
      for (const t of entry.tools) registerTool(t);
    } else if (Array.isArray(entry)) {
      for (const t of entry) registerTool(t);
    } else if (entry.name && (typeof entry.invoke === 'function' || typeof entry.call === 'function')) {
      registerTool(entry);
    }
  }
} catch (toolConnErr) {
  bridgeWarnings.push(
    'One or more connected MCP/Tool nodes failed to initialize: ' +
      String(toolConnErr?.message || toolConnErr)
  );
}

// --- Step 2: Launch staged Python job with discovered canvas tools ---
try {
  await httpReq({
    method: 'POST',
    url: `${API_BASE}/agent/jobs/${encodeURIComponent(jobId)}/start`,
    headers: jobHeaders,
    body: {
      n8n_bridge_tools: bridgeToolSpecs,
    },
    json: true,
    timeout: 25000,
  });
} catch (launchErr) {
  return [
    {
      json: {
        statusCode: 502,
        body: {},
        error: 'Failed to launch Python LangGraph job: ' + String(launchErr?.message || launchErr),
      },
    },
  ];
}

// --- Step 3: Heartbeat long-poll + Live Canvas MCP Tool execution loop (NO 120s timeout!) ---
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let heartbeatCount = 0;
let consecutivePingErrors = 0;
let lastPoll = null;
const cancelSignal =
  typeof this.getExecutionCancelSignal === 'function' ? this.getExecutionCancelSignal() : null;

while (true) {
  if (cancelSignal?.aborted) {
    try {
      await httpReq({
        method: 'POST',
        url: `${API_BASE}/agent/jobs/${encodeURIComponent(jobId)}/cancel`,
        headers: jobHeaders,
        body: {},
        json: true,
        timeout: 5000,
      });
    } catch (_e) {}
    return [
      {
        json: {
          statusCode: 499,
          body: { status: 'error', error: 'Execution cancelled by n8n.' },
          error: 'Execution cancelled by n8n.',
        },
      },
    ];
  }

  let poll;
  try {
    poll = await httpReq({
      method: 'GET',
      url: `${API_BASE}/agent/jobs/${encodeURIComponent(jobId)}?wait_ms=1500`,
      headers: jobHeaders,
      json: true,
      timeout: 25000,
    });
    consecutivePingErrors = 0;
    heartbeatCount += 1;
    lastPoll = poll;
  } catch (pingErr) {
    consecutivePingErrors += 1;
    if (consecutivePingErrors >= 5) {
      return [
        {
          json: {
            statusCode: 502,
            body: lastPoll || {},
            error:
              'Lost heartbeat connection to Python LangGraph API after 5 retries: ' +
              String(pingErr?.message || pingErr),
          },
        },
      ];
    }
    await sleep(1000);
    continue;
  }

  // Execute any pending canvas MCP / AI Tool call right inside n8n!
  if (poll && poll.status === 'running' && poll.pending_tool_call) {
    const call = poll.pending_tool_call;
    const targetTool = liveToolsByName[call.tool_name];
    let ok = true;
    let output = '';
    let errorMsg = null;

    if (!targetTool) {
      ok = false;
      errorMsg = `Canvas tool '${call.tool_name}' is not connected in n8n.`;
    } else {
      try {
        const args = call.arguments && typeof call.arguments === 'object' ? call.arguments : {};
        const rawOut =
          typeof targetTool.invoke === 'function'
            ? await targetTool.invoke(args)
            : await targetTool.call(args);
        output = typeof rawOut === 'string' ? rawOut : JSON.stringify(rawOut);
      } catch (execErr) {
        ok = false;
        errorMsg = String(execErr?.message || execErr);
      }
    }

    try {
      await httpReq({
        method: 'POST',
        url: `${API_BASE}/agent/jobs/${encodeURIComponent(jobId)}/tool-result`,
        headers: jobHeaders,
        body: {
          call_id: call.call_id,
          ok,
          output,
          error: errorMsg,
        },
        json: true,
        timeout: 20000,
      });
    } catch (_submitErr) {
      // Next poll will inspect job status
    }
    continue;
  }

  // Keep waiting as long as agent is actively making forward progress!
  if (poll && (poll.status === 'running' || poll.status === 'staged') && !poll.is_stalled) {
    continue;
  }

  // Terminal state reached: completed, awaiting_approval, stalled, error, or cancelled
  const terminalBody =
    poll && poll.result && typeof poll.result === 'object'
      ? { ...poll.result }
      : {
          thread_id: poll?.thread_id || startBody.thread_id,
          status: poll?.status === 'stalled' ? 'stalled' : 'error',
          final_response: '',
          plan: [],
          active_skills: [],
          turn_count: Number(poll?.turn_count || 0),
          pending_approval: null,
          subagent_reports: [],
          error:
            poll?.error ||
            (poll?.is_stalled
              ? 'Agent stalled with zero forward progress.'
              : 'Agent job terminated unexpectedly.'),
        };

  return [
    {
      json: {
        statusCode: poll?.status === 'stalled' ? 504 : poll?.status === 'error' ? 500 : 200,
        body: terminalBody,
        jobMeta: {
          job_id: jobId,
          job_status: poll?.status || 'unknown',
          is_stalled: Boolean(poll?.is_stalled),
          heartbeat_count: heartbeatCount,
          progress_seq: Number(poll?.progress_seq || 0),
          elapsed_seconds: Number(poll?.elapsed_seconds || 0),
          idle_seconds: Number(poll?.idle_seconds || 0),
          current_phase: String(poll?.current_phase || ''),
          last_event: String(poll?.last_event || ''),
          canvas_mcp_tools: bridgeToolSpecs.map((t) => t.name),
          canvas_skills: Array.isArray(poll?.canvas_skills) ? poll.canvas_skills : [],
          warnings: [...bridgeWarnings, ...(Array.isArray(poll?.warnings) ? poll.warnings : [])],
        },
      },
    },
  ];
}
""".strip().replace("__API_BASE__", base_url)


FORMAT_RESPONSE_JS = r"""
const reqMeta = $('Validate Input & Session').first().json;
const raw = $input.first()?.json ?? {};

const statusCode = Number(raw.statusCode ?? raw.status ?? 200);
const body = raw.body && typeof raw.body === 'object' ? raw.body : raw;
const jobMeta = raw.jobMeta && typeof raw.jobMeta === 'object' ? raw.jobMeta : null;
const httpError = raw.error && typeof raw.error === 'object'
  ? (raw.error.message || JSON.stringify(raw.error))
  : typeof raw.error === 'string'
    ? raw.error
    : '';

const isStalled = body.status === 'stalled' || Boolean(jobMeta?.is_stalled);
const isHttpFailure =
  !isStalled && (
    (Number.isFinite(statusCode) && statusCode >= 400) ||
    Boolean(httpError && !body.thread_id && !body.final_response)
  );

const latencyMs = Math.max(0, Date.now() - (Number(reqMeta.startedAt) || Date.now()));

if (isStalled) {
  const detail =
    body.error ||
    'Agent stalled with no forward progress (phase=' + String(jobMeta?.current_phase || 'unknown') + ').';
  return [
    {
      json: {
        ok: false,
        status: 'stalled',
        http_status: 504,
        thread_id: String(body.thread_id || reqMeta.thread_id),
        output: 'Agent stalled: ' + detail,
        text: 'Agent stalled: ' + detail,
        final_response: '',
        plan: Array.isArray(body.plan) ? body.plan : [],
        active_skills: Array.isArray(body.active_skills) ? body.active_skills : [],
        turn_count: Number(body.turn_count || 0),
        pending_approval: null,
        active_models: body.active_models ?? reqMeta.selected_models ?? null,
        job_monitor: jobMeta,
        error: detail,
        latency_ms: latencyMs,
      },
    },
  ];
}

if (isHttpFailure) {
  const detail =
    (typeof body.detail === 'string' && body.detail) ||
    httpError ||
    ('Python LangGraph API returned HTTP ' + statusCode);
  return [
    {
      json: {
        ok: false,
        status: 'error',
        http_status: Number.isFinite(statusCode) ? statusCode : 502,
        thread_id: reqMeta.thread_id,
        output: 'Gateway error: ' + detail,
        text: 'Gateway error: ' + detail,
        final_response: '',
        plan: [],
        active_skills: [],
        turn_count: 0,
        pending_approval: null,
        job_monitor: jobMeta,
        error: detail,
        latency_ms: latencyMs,
      },
    },
  ];
}

const agentStatus = String(body.status || 'completed');
const finalResponse = String(body.final_response || '');
const pendingApproval = body.pending_approval ?? null;

let displayText = finalResponse;
if (agentStatus === 'awaiting_approval' && pendingApproval) {
  displayText =
    'Approval required before executing sensitive tool `' +
    String(pendingApproval.tool_name || 'unknown') +
    '`.\n' +
    'Thread ID: `' + String(body.thread_id || reqMeta.thread_id) + '`\n' +
    'Expected effect: ' + String(pendingApproval.expected_effect || '') + '\n' +
    'Arguments:\n```json\n' +
    JSON.stringify(pendingApproval.arguments || {}, null, 2) +
    '\n```\n' +
    'Send `{ "thread_id": "' +
    String(body.thread_id || reqMeta.thread_id) +
    '", "decision": "approve" | "reject" | "edit" }` to continue.';
}

return [
  {
    json: {
      ok: agentStatus !== 'error' && agentStatus !== 'stalled',
      status: agentStatus,
      http_status: Number.isFinite(statusCode) ? statusCode : 200,
      thread_id: String(body.thread_id || reqMeta.thread_id),
      output: displayText,
      text: displayText,
      final_response: finalResponse,
      plan: Array.isArray(body.plan) ? body.plan : [],
      active_skills: Array.isArray(body.active_skills) ? body.active_skills : [],
      turn_count: Number(body.turn_count || 0),
      pending_approval: pendingApproval,
      subagent_reports: Array.isArray(body.subagent_reports) ? body.subagent_reports : [],
      active_models: body.active_models ?? reqMeta.selected_models ?? null,
      canvas_mcp_tools: jobMeta?.canvas_mcp_tools ?? [],
      canvas_skills: jobMeta?.canvas_skills ?? reqMeta.canvas_skill_names ?? [],
      job_monitor: jobMeta,
      error: body.error ?? null,
      latency_ms: latencyMs,
    },
  },
];
""".strip()


FORMAT_VALIDATION_ERROR_JS = r"""
const reqMeta = $('Validate Input & Session').first().json;
const msg = String(reqMeta.validationError || 'Invalid input payload.');
return [
  {
    json: {
      ok: false,
      status: 'validation_error',
      http_status: 400,
      thread_id: reqMeta.thread_id,
      output: 'Validation error: ' + msg,
      text: 'Validation error: ' + msg,
      final_response: '',
      plan: [],
      active_skills: [],
      turn_count: 0,
      pending_approval: null,
      error: msg,
    },
  },
];
""".strip()


EXAMPLE_CANVAS_SKILL_MD = """---
name: executive_brief
description: Structure executive summaries with a TL;DR, Key Metrics, Risks, and Next Steps.
---
# Skill: Executive Brief

When this skill is loaded, format your final response cleanly using these four sections:
1. **TL;DR**: 1-2 crisp sentences stating the exact conclusion.
2. **Key Findings / Metrics**: Bullet points grounded strictly in observed data or tool output.
3. **Risks & Constraints**: Brief list of any caveats or blockers (or "None observed").
4. **Recommended Next Steps**: Concrete, numbered actions.
""".strip()


def _build_langchain_slot_node(
    *,
    node_id: str,
    name: str,
    position: list[int],
    js_code: str,
    slot_type: str,
    max_connections: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "name": name,
        "type": "@n8n/n8n-nodes-langchain.code",
        "typeVersion": 1,
        "position": position,
        "notes": notes,
        "parameters": {
            "code": {
                "execute": {
                    "code": js_code,
                }
            },
            "inputs": {
                "input": [
                    {"type": "main", "maxConnections": -1, "required": True},
                    {"type": slot_type, "maxConnections": max_connections, "required": False},
                ]
            },
            "outputs": {
                "output": [
                    {"type": "main"},
                ]
            },
        },
    }


def _build_openai_chat_model_node(
    *,
    node_id: str,
    name: str,
    position: list[int],
    default_model: str,
    temperature: float,
    max_tokens: int,
    credential_id: str,
    credential_name: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "name": name,
        "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
        "typeVersion": 1.3,
        "position": position,
        "notes": notes,
        "notesInFlow": True,
        "parameters": {
            "model": {
                "__rl": True,
                "mode": "list",
                "value": default_model,
                "cachedResultName": default_model,
            },
            "responsesApiEnabled": False,
            "options": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        },
        "credentials": {
            "openAiApi": {
                "id": credential_id,
                "name": credential_name,
            }
        },
    }


def _build_canvas_skill_node(
    *,
    node_id: str,
    name: str,
    position: list[int],
    skill_name: str,
    skill_description: str,
    markdown_content: str,
) -> dict[str, Any]:
    """Build a native n8n Default Data Loader node wired as a plug-and-play Skill."""
    return {
        "id": node_id,
        "name": name,
        "type": "@n8n/n8n-nodes-langchain.documentDefaultDataLoader",
        "typeVersion": 1.1,
        "position": position,
        "notes": (
            "Plug-and-play Skill node. Duplicate this node, edit the Markdown instructions & "
            "metadata, and wire its Document output into 'Read Canvas Skills'."
        ),
        "notesInFlow": True,
        "parameters": {
            "dataType": "json",
            "jsonMode": "expressionData",
            "jsonData": markdown_content,
            "textSplittingMode": "simple",
            "options": {
                "metadata": {
                    "metadataValues": [
                        {"name": "name", "value": skill_name},
                        {"name": "description", "value": skill_description},
                    ]
                }
            },
        },
    }


def _build_canvas_mcp_client_node(
    *,
    node_id: str,
    name: str,
    position: list[int],
    endpoint_url: str,
    credential_id: str,
    credential_name: str,
    include_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Build a native n8n MCP Client Tool node wired into the plug-and-play MCP Servers slot."""
    params: dict[str, Any] = {
        "endpointUrl": endpoint_url,
        "serverTransport": "httpStreamable",
        "authentication": "bearerAuth",
        "include": "selected" if include_tools else "all",
        "options": {"timeout": 60000},
    }
    if include_tools:
        params["includeTools"] = list(include_tools)

    return {
        "id": node_id,
        "name": name,
        "type": "@n8n/n8n-nodes-langchain.mcpClientTool",
        "typeVersion": 1.4,
        "position": position,
        "notes": (
            "Plug-and-play MCP Server node. Add or duplicate any MCP Client Tool (or any n8n AI "
            "Tool node) and wire its Tools output into 'Agent Heartbeat & MCP Bridge'."
        ),
        "notesInFlow": True,
        "parameters": params,
        "credentials": {
            "httpBearerAuth": {
                "id": credential_id,
                "name": credential_name,
            }
        },
    }


def build_gateway_workflow(
    *,
    agent_api_base_url: str = "http://127.0.0.1:8100",
    credential_id: str = GATEWAY_CREDENTIAL_ID,
    credential_name: str = GATEWAY_CREDENTIAL_NAME,
    openai_credential_id: str = RIVER_OPENAI_CREDENTIAL_ID,
    openai_credential_name: str = RIVER_OPENAI_CREDENTIAL_NAME,
    mcp_credential_id: str = N8N_MCP_BEARER_CREDENTIAL_ID,
    mcp_credential_name: str = N8N_MCP_BEARER_CREDENTIAL_NAME,
    n8n_mcp_endpoint_url: str = "http://127.0.0.1:5678/mcp-server/http",
    use_header_auth_credential: bool = True,
) -> dict[str, Any]:
    """Construct the n8n workflow JSON for ``LangGraph Autonomous AI Agent Gateway``."""
    base_url = agent_api_base_url.rstrip("/")

    start_job_http_node: dict[str, Any] = {
        "id": "node-lg-start-job",
        "name": "Start Python Agent Job",
        "type": "n8n-nodes-base.httpRequest",
        "typeVersion": 4.2,
        "position": [860, 0],
        "onError": "continueRegularOutput",
        "notes": (
            "Stages an async LangGraph job in <50ms and returns an ephemeral job_token. "
            "Eliminates the 120s gateway timeout."
        ),
        "parameters": {
            "method": "POST",
            "url": f"{base_url}/agent/jobs/create",
            "authentication": ("genericCredentialType" if use_header_auth_credential else "none"),
            "genericAuthType": "httpHeaderAuth",
            "sendHeaders": True,
            "headerParameters": {
                "parameters": [
                    {"name": "Content-Type", "value": "application/json"},
                    {"name": "Accept", "value": "application/json"},
                ]
            },
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": "={{ JSON.stringify($json.requestBody) }}",
            "options": {
                "timeout": 30000,
                "response": {
                    "response": {
                        "fullResponse": True,
                        "neverError": True,
                    }
                },
            },
        },
    }
    if use_header_auth_credential:
        start_job_http_node["credentials"] = {
            "httpHeaderAuth": {
                "id": credential_id,
                "name": credential_name,
            }
        }

    nodes: list[dict[str, Any]] = [
        {
            "id": "sticky-lg-gateway",
            "name": "Gateway Architecture Note",
            "type": "n8n-nodes-base.stickyNote",
            "typeVersion": 1,
            "position": [-660, -340],
            "parameters": {
                "content": (
                    "## LangGraph Autonomous AI Agent Gateway (Models + Skills + MCP Slots + Heartbeat Watchdog)\n"
                    "- **Model Slots**: Click **`Brain LLM (OpenAI Chat Model)`** or **`Sub-Agent LLM (OpenAI Chat Model)`** to choose any credential & model from the dropdown.\n"
                    "- **Skills Slot (Unlimited)**: Duplicate **`Skill: Executive Brief`** (or add any *Default Data Loader* node with Markdown instructions) and wire its `Document` port into **`Read Canvas Skills`** — the agent automatically discovers and loads it via `load_skill`.\n"
                    "- **MCP Servers Slot (Unlimited)**: Add any native **`MCP Client Tool`** node (or any n8n AI Tool node) and wire its `Tools` port into **`Agent Heartbeat & MCP Bridge`** — Python LangGraph discovers and executes those tools live using n8n's credentials.\n"
                    "- **No 120s Timeout**: **`Agent Heartbeat & MCP Bridge`** continuously pings Python LangGraph for forward progress and only aborts if the agent genuinely stalls or loses connectivity."
                ),
                "height": 250,
                "width": 2240,
                "color": 4,
            },
        },
        {
            "id": "sticky-lg-skills-slot",
            "name": "Skills Slot Zone",
            "type": "n8n-nodes-base.stickyNote",
            "typeVersion": 1,
            "position": [100, 250],
            "parameters": {
                "content": (
                    "### Plug-and-Play Skills Slot\n"
                    "Duplicate **`Skill: Executive Brief`** or add *Default Data Loader* nodes and wire `Document` -> **`Read Canvas Skills`**."
                ),
                "height": 270,
                "width": 320,
                "color": 5,
            },
        },
        {
            "id": "sticky-lg-mcp-slot",
            "name": "MCP Servers Slot Zone",
            "type": "n8n-nodes-base.stickyNote",
            "typeVersion": 1,
            "position": [1020, 250],
            "parameters": {
                "content": (
                    "### Plug-and-Play MCP Servers Slot\n"
                    "Add any **`MCP Client Tool`** (or n8n AI Tool) node and wire `Tools` -> **`Agent Heartbeat & MCP Bridge`**."
                ),
                "height": 270,
                "width": 360,
                "color": 6,
            },
        },
        {
            "id": "node-lg-chat-trigger",
            "name": "Chat Trigger (Agent Gateway UI)",
            "type": "@n8n/n8n-nodes-langchain.chatTrigger",
            "typeVersion": 1.4,
            "position": [-600, -40],
            "webhookId": GATEWAY_CHAT_WEBHOOK_ID,
            "parameters": {
                "public": True,
                "mode": "hostedChat",
                "authentication": "none",
                "availableInChat": True,
                "agentName": GATEWAY_WORKFLOW_NAME,
                "agentDescription": (
                    "Gateway forwarding tasks to the Python LangGraph Autonomous AI Agent "
                    "with plug-and-play Model, Skill, and MCP Server slots plus heartbeat stall detection."
                ),
                "initialMessages": (
                    "Connected to the Python LangGraph Autonomous AI Agent Gateway. "
                    "Plug OpenAI Chat Models, Skill nodes, and MCP Client Tool nodes directly on the n8n canvas "
                    "(or override models inline with `/brain provider:model` and `/subagent provider:model`)."
                ),
                "options": {
                    "responseMode": "lastNode",
                    "title": GATEWAY_WORKFLOW_NAME,
                    "subtitle": "n8n Models + Skills + MCP Slots -> Python LangGraph",
                },
            },
        },
        {
            "id": "node-lg-webhook-trigger",
            "name": "Webhook Trigger (Agent Gateway API)",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2.1,
            "position": [-600, 180],
            "webhookId": GATEWAY_API_WEBHOOK_ID,
            "parameters": {
                "httpMethod": "POST",
                "path": GATEWAY_WEBHOOK_PATH,
                "responseMode": "lastNode",
                "options": {},
            },
        },
        _build_langchain_slot_node(
            node_id="node-lg-read-brain-llm",
            name="Read Brain LLM",
            position=[-340, 70],
            js_code=READ_BRAIN_LLM_JS,
            slot_type="ai_languageModel",
            max_connections=1,
            notes="Reads selected model, Base URL, credential API key & temperature from the connected Brain LLM OpenAI node.",
        ),
        _build_openai_chat_model_node(
            node_id="node-lg-brain-openai",
            name="Brain LLM (OpenAI Chat Model)",
            position=[-340, 340],
            default_model="deepseek-ai/DeepSeek-V4.1-Flash",
            temperature=0.2,
            max_tokens=4096,
            credential_id=openai_credential_id,
            credential_name=openai_credential_name,
            notes="Select Brain (Supervisor) provider credential & model from dropdown.",
        ),
        _build_langchain_slot_node(
            node_id="node-lg-read-subagent-llm",
            name="Read Sub-Agent LLM",
            position=[-80, 70],
            js_code=READ_SUBAGENT_LLM_JS,
            slot_type="ai_languageModel",
            max_connections=1,
            notes="Reads selected model, Base URL, credential API key & temperature from the connected Sub-Agent LLM OpenAI node.",
        ),
        _build_openai_chat_model_node(
            node_id="node-lg-subagent-openai",
            name="Sub-Agent LLM (OpenAI Chat Model)",
            position=[-80, 340],
            default_model="deepseek-ai/DeepSeek-V4.1-Flash",
            temperature=0.1,
            max_tokens=2048,
            credential_id=openai_credential_id,
            credential_name=openai_credential_name,
            notes="Select Sub-Agent (Worker) provider credential & model from dropdown.",
        ),
        _build_langchain_slot_node(
            node_id="node-lg-read-skills",
            name="Read Canvas Skills",
            position=[180, 70],
            js_code=READ_CANVAS_SKILLS_JS,
            slot_type="ai_document",
            max_connections=-1,
            notes="Skills Slot Hub: accepts unlimited Default Data Loader / Document skill nodes and passes them to Python LangGraph.",
        ),
        _build_canvas_skill_node(
            node_id="node-lg-skill-executive-brief",
            name="Skill: Executive Brief",
            position=[180, 360],
            skill_name="executive_brief",
            skill_description="Structure executive summaries with a TL;DR, Key Metrics, Risks, and Next Steps.",
            markdown_content=EXAMPLE_CANVAS_SKILL_MD,
        ),
        {
            "id": "node-lg-validate-input",
            "name": "Validate Input & Session",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [420, 70],
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": VALIDATE_INPUT_JS,
            },
        },
        {
            "id": "node-lg-valid-if",
            "name": "Input Valid?",
            "type": "n8n-nodes-base.if",
            "typeVersion": 2.2,
            "position": [640, 70],
            "parameters": {
                "conditions": {
                    "options": {
                        "caseSensitive": True,
                        "leftValue": "",
                        "typeValidation": "strict",
                        "version": 2,
                    },
                    "combinator": "and",
                    "conditions": [
                        {
                            "id": "cond-input-valid",
                            "leftValue": "={{ $json.isValid }}",
                            "rightValue": True,
                            "operator": {
                                "type": "boolean",
                                "operation": "true",
                                "singleValue": True,
                            },
                        }
                    ],
                },
                "options": {},
            },
        },
        start_job_http_node,
        _build_langchain_slot_node(
            node_id="node-lg-heartbeat-mcp-bridge",
            name="Agent Heartbeat & MCP Bridge",
            position=[1120, 0],
            js_code=_make_heartbeat_bridge_js(base_url),
            slot_type="ai_tool",
            max_connections=-1,
            notes=(
                "MCP Servers Slot + Long-Running Heartbeat Monitor: discovers all plugged-in MCP Client "
                "Tool / AI Tool nodes, executes their tool calls live when requested by Python LangGraph, "
                "and polls progress continuously without hitting a 120s timeout."
            ),
        ),
        _build_canvas_mcp_client_node(
            node_id="node-lg-mcp-n8n-canvas",
            name="MCP Server: n8n Canvas",
            position=[1120, 360],
            endpoint_url=n8n_mcp_endpoint_url,
            credential_id=mcp_credential_id,
            credential_name=mcp_credential_name,
            include_tools=["search_workflows", "list_credentials"],
        ),
        {
            "id": "node-lg-format-response",
            "name": "Format Agent Response",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [1380, 0],
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": FORMAT_RESPONSE_JS,
            },
        },
        {
            "id": "node-lg-format-validation-error",
            "name": "Format Validation Error",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [860, 220],
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": FORMAT_VALIDATION_ERROR_JS,
            },
        },
    ]

    connections: dict[str, Any] = {
        "Chat Trigger (Agent Gateway UI)": {
            "main": [[{"node": "Read Brain LLM", "type": "main", "index": 0}]]
        },
        "Webhook Trigger (Agent Gateway API)": {
            "main": [[{"node": "Read Brain LLM", "type": "main", "index": 0}]]
        },
        "Brain LLM (OpenAI Chat Model)": {
            "ai_languageModel": [
                [{"node": "Read Brain LLM", "type": "ai_languageModel", "index": 0}]
            ]
        },
        "Read Brain LLM": {"main": [[{"node": "Read Sub-Agent LLM", "type": "main", "index": 0}]]},
        "Sub-Agent LLM (OpenAI Chat Model)": {
            "ai_languageModel": [
                [{"node": "Read Sub-Agent LLM", "type": "ai_languageModel", "index": 0}]
            ]
        },
        "Read Sub-Agent LLM": {
            "main": [[{"node": "Read Canvas Skills", "type": "main", "index": 0}]]
        },
        "Skill: Executive Brief": {
            "ai_document": [[{"node": "Read Canvas Skills", "type": "ai_document", "index": 0}]]
        },
        "Read Canvas Skills": {
            "main": [[{"node": "Validate Input & Session", "type": "main", "index": 0}]]
        },
        "Validate Input & Session": {
            "main": [[{"node": "Input Valid?", "type": "main", "index": 0}]]
        },
        "Input Valid?": {
            "main": [
                [{"node": "Start Python Agent Job", "type": "main", "index": 0}],
                [{"node": "Format Validation Error", "type": "main", "index": 0}],
            ]
        },
        "Start Python Agent Job": {
            "main": [[{"node": "Agent Heartbeat & MCP Bridge", "type": "main", "index": 0}]]
        },
        "MCP Server: n8n Canvas": {
            "ai_tool": [[{"node": "Agent Heartbeat & MCP Bridge", "type": "ai_tool", "index": 0}]]
        },
        "Agent Heartbeat & MCP Bridge": {
            "main": [[{"node": "Format Agent Response", "type": "main", "index": 0}]]
        },
    }

    return {
        "id": GATEWAY_WORKFLOW_ID,
        "name": GATEWAY_WORKFLOW_NAME,
        "active": True,
        "nodes": nodes,
        "connections": connections,
        "settings": {
            "executionOrder": "v1",
            "saveDataErrorExecution": "all",
            "saveDataSuccessExecution": "all",
            "saveManualExecutions": True,
        },
        "meta": {
            "templateCredsSetupCompleted": True,
        },
        "pinData": {},
        "versionId": str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"langgraph-gateway-slots-heartbeat-{base_url}")
        ),
    }


def find_n8n_binary(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / ".n8n-runtime/node_modules/.bin/n8n",
        repo_root / "node_modules/.bin/n8n",
        Path("/usr/local/bin/n8n"),
        Path("/usr/bin/n8n"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ensure_workflow_ownership(sqlite_path: Path, workflow_id: str) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        projects = [row[0] for row in cur.execute("SELECT id FROM project WHERE type = 'personal'")]
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.000")
        for pid in projects:
            exists = cur.execute(
                "SELECT workflowId FROM shared_workflow WHERE workflowId = ? AND projectId = ?",
                (workflow_id, pid),
            ).fetchone()
            if not exists:
                cur.execute(
                    "INSERT INTO shared_workflow (workflowId, projectId, role, createdAt, updatedAt) "
                    "VALUES (?, ?, 'workflow:owner', ?, ?)",
                    (workflow_id, pid, now, now),
                )
        conn.commit()
    finally:
        conn.close()


def _ensure_credential_ownership(sqlite_path: Path, credential_id: str) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        projects = [row[0] for row in cur.execute("SELECT id FROM project WHERE type = 'personal'")]
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.000")
        for pid in projects:
            exists = cur.execute(
                "SELECT credentialsId FROM shared_credentials WHERE credentialsId = ? AND projectId = ?",
                (credential_id, pid),
            ).fetchone()
            if not exists:
                cur.execute(
                    "INSERT INTO shared_credentials (credentialsId, projectId, role, createdAt, updatedAt) "
                    "VALUES (?, ?, 'credential:owner', ?, ?)",
                    (credential_id, pid, now, now),
                )
        conn.commit()
    finally:
        conn.close()


def _import_n8n_credential(
    *,
    repo_root: Path,
    user_folder: Path,
    credential_id: str,
    credential_name: str,
    credential_type: str,
    credential_data: dict[str, Any],
) -> bool:
    sqlite_path = user_folder / ".n8n" / "database.sqlite"
    if not sqlite_path.exists():
        return False

    n8n_bin = find_n8n_binary(repo_root)
    if n8n_bin is None:
        return False

    payload = [
        {
            "id": credential_id,
            "name": credential_name,
            "type": credential_type,
            "data": credential_data,
        }
    ]

    env = dict(os.environ)
    env["N8N_USER_FOLDER"] = str(user_folder)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as tmp:
        json.dump(payload, tmp)
        tmp_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            [str(n8n_bin), "import:credentials", f"--input={tmp_path}"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"n8n import:credentials failed ({proc.returncode}): {proc.stderr or proc.stdout}"
            )
        _ensure_credential_ownership(sqlite_path, credential_id)
        return True
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_river_ai_openai_credential(
    *,
    repo_root: Path,
    user_folder: Path,
    api_key: str | None = None,
    base_url: str = "https://api.river.ai/v1",
) -> bool:
    """Import/update encrypted River AI openAiApi credential in n8n for the OpenAI Chat Model nodes."""
    key = (api_key or os.getenv("RIVER_AI_API") or "").strip()
    if not key:
        return False
    return _import_n8n_credential(
        repo_root=repo_root,
        user_folder=user_folder,
        credential_id=RIVER_OPENAI_CREDENTIAL_ID,
        credential_name=RIVER_OPENAI_CREDENTIAL_NAME,
        credential_type="openAiApi",
        credential_data={"apiKey": key, "url": base_url.rstrip("/")},
    )


def ensure_n8n_mcp_bearer_credential(
    *,
    repo_root: Path,
    user_folder: Path,
    mcp_token: str | None = None,
) -> bool:
    """Import/update encrypted httpBearerAuth credential in n8n for the default MCP Client Tool node."""
    token = (mcp_token or os.getenv("N8N_MCP_TOKEN") or "").strip()
    if not token:
        return False
    return _import_n8n_credential(
        repo_root=repo_root,
        user_folder=user_folder,
        credential_id=N8N_MCP_BEARER_CREDENTIAL_ID,
        credential_name=N8N_MCP_BEARER_CREDENTIAL_NAME,
        credential_type="httpBearerAuth",
        credential_data={"token": token},
    )


def sync_gateway_workflow(
    *,
    repo_root: Path,
    user_folder: Path,
    agent_api_base_url: str = "http://127.0.0.1:8100",
    import_to_n8n: bool = True,
) -> dict[str, Any]:
    """Export workflow JSON and optionally import + publish into existing n8n DB."""
    load_dotenv(repo_root / ".env")
    workflow = build_gateway_workflow(agent_api_base_url=agent_api_base_url)
    workflows_dir = repo_root / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    exported_path = workflows_dir / "langgraph-autonomous-agent-gateway.json"
    exported_path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")

    result: dict[str, Any] = {
        "workflow_id": GATEWAY_WORKFLOW_ID,
        "workflow_name": GATEWAY_WORKFLOW_NAME,
        "exported_path": str(exported_path),
        "agent_api_base_url": agent_api_base_url,
        "openai_credential_synced": False,
        "mcp_bearer_credential_synced": False,
        "imported": False,
        "published": False,
    }

    sqlite_path = user_folder / ".n8n" / "database.sqlite"
    if not import_to_n8n or not sqlite_path.exists():
        return result

    n8n_bin = find_n8n_binary(repo_root)
    if n8n_bin is None:
        raise FileNotFoundError("n8n binary not found; cannot import gateway workflow.")

    result["openai_credential_synced"] = ensure_river_ai_openai_credential(
        repo_root=repo_root,
        user_folder=user_folder,
    )
    result["mcp_bearer_credential_synced"] = ensure_n8n_mcp_bearer_credential(
        repo_root=repo_root,
        user_folder=user_folder,
    )

    env = dict(os.environ)
    env["N8N_USER_FOLDER"] = str(user_folder)

    imp = subprocess.run(
        [str(n8n_bin), "import:workflow", f"--input={exported_path}"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if imp.returncode != 0:
        raise RuntimeError(
            f"n8n import:workflow failed ({imp.returncode}): {imp.stderr or imp.stdout}"
        )

    _ensure_workflow_ownership(sqlite_path, GATEWAY_WORKFLOW_ID)
    result["imported"] = True

    pub = subprocess.run(
        [str(n8n_bin), "publish:workflow", f"--id={GATEWAY_WORKFLOW_ID}"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    result["published"] = pub.returncode == 0
    if not result["published"]:
        result["publish_note"] = (pub.stderr or pub.stdout).strip()

    return result


if __name__ == "__main__":
    root = Path.cwd()
    folder = Path(os.getenv("N8N_USER_FOLDER", "./.n8n-data")).resolve()
    api_url = os.getenv("AGENT_API_PUBLIC_URL", "http://127.0.0.1:8100")
    out = sync_gateway_workflow(
        repo_root=root,
        user_folder=folder,
        agent_api_base_url=api_url,
        import_to_n8n=True,
    )
    print(json.dumps(out, indent=2))
