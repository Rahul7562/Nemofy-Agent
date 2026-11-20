"""Phase 9: n8n Autonomous Agent Gateway Workflow Builder & Sync Utility.

Creates and imports the ``LangGraph Autonomous AI Agent Gateway`` workflow
into the existing n8n installation WITHOUT modifying or replacing existing
workflows, credentials, or the 5678 -> 3000 web mapping.

Model selection directly inside n8n:
- Uses TWO native n8n OpenAI Chat Model nodes (``@n8n/n8n-nodes-langchain.lmChatOpenAi``):
  1. ``Brain LLM (OpenAI Chat Model)`` — picks provider credential & Brain model from dropdown
  2. ``Sub-Agent LLM (OpenAI Chat Model)`` — picks provider credential & Sub-Agent model from dropdown
- Lightweight LangChain Code bridge nodes read the selected model, baseURL, encrypted credential
  API key, temperature, and maxTokens from those two OpenAI nodes and forward them to the Python
  LangGraph FastAPI service (``POST /agent/run`` & ``POST /agent/resume``).
- Note: n8n never runs an AI agent itself — all reasoning, planning, tools, sub-agents,
  checkpoints, and HITL guardrails run strictly in Python LangGraph.
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


def _make_read_llm_node_js(target_key: str) -> str:
    """Build LangChain Code JS that reads model/baseURL/apiKey from a connected n8n Chat Model node."""
    return r"""
const inputItems = this.getInputData();
const firstJson = inputItems[0]?.json ? { ...inputItems[0].json } : {};

let llm = null;
try {
  const connected = await this.getInputConnectionData('ai_languageModel', 0);
  llm = Array.isArray(connected) ? connected[0] : connected;
} catch (_err) {
  llm = null;
}

const inferProvider = (baseUrl) => {
  const u = String(baseUrl || '').toLowerCase();
  if (!u || u.includes('api.openai.com')) return 'openai';
  if (u.includes('river.ai')) return 'river_ai';
  if (u.includes('openrouter.ai')) return 'openrouter';
  if (u.includes('generativelanguage.googleapis.com')) return 'gemini';
  if (u.includes('groq.com')) return 'groq';
  if (u.includes('deepseek.com')) return 'deepseek';
  if (u.includes('together.xyz')) return 'together';
  if (u.includes('fireworks.ai')) return 'fireworks';
  if (u.includes('anthropic.com')) return 'anthropic';
  if (u.includes('11434') || u.includes('ollama')) return 'ollama';
  return 'custom';
};

const pickStr = (...vals) => {
  for (const v of vals) {
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return '';
};

let extracted = null;
if (llm && typeof llm === 'object') {
  const cfg = llm.clientConfig || llm.configuration || llm.lc_kwargs?.configuration || {};
  const model = pickStr(
    llm.model,
    llm.modelName,
    llm.lc_kwargs?.model,
    llm.lc_kwargs?.modelName
  );
  const baseUrl = pickStr(
    cfg.baseURL,
    cfg.baseUrl,
    llm.baseURL,
    llm.baseUrl,
    llm.lc_kwargs?.baseURL,
    llm.lc_kwargs?.baseUrl,
    'https://api.openai.com/v1'
  );
  const apiKey = pickStr(
    cfg.apiKey,
    llm.apiKey,
    llm.openAIApiKey,
    llm.lc_kwargs?.apiKey,
    llm.lc_kwargs?.openAIApiKey
  );
  const temperature =
    typeof llm.temperature === 'number' && Number.isFinite(llm.temperature)
      ? llm.temperature
      : undefined;
  const maxTokens =
    typeof llm.maxTokens === 'number' && Number.isFinite(llm.maxTokens) && llm.maxTokens >= 64
      ? Math.round(llm.maxTokens)
      : undefined;
  const reasoningEffort = pickStr(
    llm.modelKwargs?.reasoning_effort,
    llm.lc_kwargs?.modelKwargs?.reasoning_effort
  );

  extracted = {
    provider: inferProvider(baseUrl),
    ...(model ? { model } : {}),
    ...(baseUrl ? { base_url: baseUrl } : {}),
    ...(apiKey ? { api_key: apiKey } : {}),
    ...(temperature !== undefined ? { temperature } : {}),
    ...(maxTokens !== undefined ? { max_tokens: maxTokens } : {}),
    ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
  };
}

firstJson.__TARGET_KEY__ = extracted;
return [{ json: firstJson }];
""".replace("__TARGET_KEY__", target_key).strip()


READ_BRAIN_LLM_JS = _make_read_llm_node_js("_n8n_brain_llm")
READ_SUBAGENT_LLM_JS = _make_read_llm_node_js("_n8n_subagent_llm")


VALIDATE_INPUT_JS = r"""
const item = $input.first()?.json ?? {};
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

const pickNumber = (v) => {
  if (v === null || v === undefined || v === '') return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
};

// Model configurations extracted from the two connected native n8n OpenAI Chat Model nodes
const n8nBrainNode =
  item._n8n_brain_llm && typeof item._n8n_brain_llm === 'object'
    ? item._n8n_brain_llm
    : {};
const n8nSubagentNode =
  item._n8n_subagent_llm && typeof item._n8n_subagent_llm === 'object'
    ? item._n8n_subagent_llm
    : {};

let rawTask =
  pickString(payload.task) ||
  pickString(payload.chatInput) ||
  pickString(payload.prompt) ||
  pickString(payload.message) ||
  pickString(payload.query) ||
  pickString(payload.input) ||
  pickString(item.chatInput) ||
  '';

// Optional inline chat directives (highest precedence over n8n OpenAI nodes):
//   /brain openrouter:anthropic/claude-3.7-sonnet
//   /subagent openai:gpt-4o-mini
//   /model openai:gpt-4o   (sets both brain & subagent)
const inlineBrain = {};
const inlineSub = {};
const parseModelSpecToken = (specToken) => {
  const cleaned = pickString(specToken);
  if (!cleaned) return {};
  const colonIdx = cleaned.indexOf(':');
  if (colonIdx > 0) {
    const prov = cleaned.slice(0, colonIdx).trim().toLowerCase();
    const mod = cleaned.slice(colonIdx + 1).trim();
    return { ...(prov ? { provider: prov } : {}), ...(mod ? { model: mod } : {}) };
  }
  return { model: cleaned };
};

if (rawTask) {
  rawTask = rawTask
    .replace(/(?:^|\s)\/(?:brain|primary)\s+([^\s]+)/gi, (_, tok) => {
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
const requestBody = isResume
  ? {
      thread_id: threadId,
      decision,
      ...(payload.arguments && typeof payload.arguments === 'object'
        ? { arguments: payload.arguments }
        : {}),
      ...(pickString(payload.reason) ? { reason: pickString(payload.reason) } : {}),
      ...(primaryModelOverride ? { primary_model: primaryModelOverride } : {}),
      ...(workerModelOverride ? { worker_model: workerModelOverride } : {}),
    }
  : {
      task: rawTask,
      thread_id: threadId,
      ...(maxTurns !== undefined ? { max_turns: maxTurns } : {}),
      ...(primaryModelOverride ? { primary_model: primaryModelOverride } : {}),
      ...(workerModelOverride ? { worker_model: workerModelOverride } : {}),
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
      requestBody,
      startedAt: Date.now(),
    },
  },
];
""".strip()


FORMAT_RESPONSE_JS = r"""
const reqMeta = $('Validate Input & Session').first().json;
const raw = $input.first()?.json ?? {};

const statusCode = Number(raw.statusCode ?? raw.status ?? 200);
const body = raw.body && typeof raw.body === 'object' ? raw.body : raw;
const httpError = raw.error && typeof raw.error === 'object'
  ? (raw.error.message || JSON.stringify(raw.error))
  : typeof raw.error === 'string'
    ? raw.error
    : '';

const isHttpFailure =
  (Number.isFinite(statusCode) && statusCode >= 400) ||
  Boolean(httpError && !body.thread_id && !body.final_response);

const latencyMs = Math.max(0, Date.now() - (Number(reqMeta.startedAt) || Date.now()));

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
      ok: agentStatus !== 'error',
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


def _build_langchain_reader_node(
    *,
    node_id: str,
    name: str,
    position: list[int],
    js_code: str,
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
                    {"type": "ai_languageModel", "maxConnections": 1, "required": False},
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


def build_gateway_workflow(
    *,
    agent_api_base_url: str = "http://127.0.0.1:8100",
    credential_id: str = GATEWAY_CREDENTIAL_ID,
    credential_name: str = GATEWAY_CREDENTIAL_NAME,
    openai_credential_id: str = RIVER_OPENAI_CREDENTIAL_ID,
    openai_credential_name: str = RIVER_OPENAI_CREDENTIAL_NAME,
    use_header_auth_credential: bool = True,
) -> dict[str, Any]:
    """Construct the n8n workflow JSON for ``LangGraph Autonomous AI Agent Gateway``."""
    base_url = agent_api_base_url.rstrip("/")

    http_node: dict[str, Any] = {
        "id": "node-lg-http-request",
        "name": "Call Python LangGraph API",
        "type": "n8n-nodes-base.httpRequest",
        "typeVersion": 4.2,
        "position": [680, 0],
        "onError": "continueRegularOutput",
        "parameters": {
            "method": "POST",
            "url": f"={base_url}/agent/{{{{ $json.endpoint }}}}",
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
                "timeout": 120000,
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
        http_node["credentials"] = {
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
                    "## LangGraph Autonomous AI Agent Gateway (Native OpenAI Chat Model Selector)\n"
                    "- **Pick Brain Model**: Click **`Brain LLM (OpenAI Chat Model)`** -> choose any n8n OpenAI-compatible credential (River AI, OpenAI, OpenRouter, Groq, Gemini, Ollama) -> pick any model from the live dropdown.\n"
                    "- **Pick Sub-Agent Model**: Click **`Sub-Agent LLM (OpenAI Chat Model)`** -> choose any credential & model independently for isolated LangGraph sub-agents.\n"
                    "- **Swap Provider Nodes**: You can even swap either node for any other n8n Chat Model node.\n"
                    "- **Strict Separation**: n8n only reads the selected model & credential settings; all reasoning, tools, checkpoints, and HITL approvals run in Python LangGraph."
                ),
                "height": 240,
                "width": 1680,
                "color": 4,
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
                    "using models selected in the native n8n OpenAI Chat Model nodes."
                ),
                "initialMessages": (
                    "Connected to the Python LangGraph Autonomous AI Agent Gateway. "
                    "Select Brain & Sub-Agent models using the two OpenAI Chat Model nodes in n8n "
                    "(or override per message with `/brain provider:model` and `/subagent provider:model`)."
                ),
                "options": {
                    "responseMode": "lastNode",
                    "title": GATEWAY_WORKFLOW_NAME,
                    "subtitle": "n8n OpenAI Model Nodes -> Python LangGraph Brain & Sub-Agents",
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
        _build_langchain_reader_node(
            node_id="node-lg-read-brain-llm",
            name="Read Brain LLM",
            position=[-340, 70],
            js_code=READ_BRAIN_LLM_JS,
            notes="Reads selected model, Base URL, credential API key & temperature from the connected Brain LLM OpenAI node.",
        ),
        _build_openai_chat_model_node(
            node_id="node-lg-brain-openai",
            name="Brain LLM (OpenAI Chat Model)",
            position=[-340, 320],
            default_model="deepseek-ai/DeepSeek-V4.1-Flash",
            temperature=0.2,
            max_tokens=4096,
            credential_id=openai_credential_id,
            credential_name=openai_credential_name,
            notes="Select Brain (Supervisor) provider credential & model from dropdown.",
        ),
        _build_langchain_reader_node(
            node_id="node-lg-read-subagent-llm",
            name="Read Sub-Agent LLM",
            position=[-80, 70],
            js_code=READ_SUBAGENT_LLM_JS,
            notes="Reads selected model, Base URL, credential API key & temperature from the connected Sub-Agent LLM OpenAI node.",
        ),
        _build_openai_chat_model_node(
            node_id="node-lg-subagent-openai",
            name="Sub-Agent LLM (OpenAI Chat Model)",
            position=[-80, 320],
            default_model="deepseek-ai/DeepSeek-V4.1-Flash",
            temperature=0.1,
            max_tokens=2048,
            credential_id=openai_credential_id,
            credential_name=openai_credential_name,
            notes="Select Sub-Agent (Worker) provider credential & model from dropdown.",
        ),
        {
            "id": "node-lg-validate-input",
            "name": "Validate Input & Session",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [180, 70],
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
            "position": [420, 70],
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
        http_node,
        {
            "id": "node-lg-format-response",
            "name": "Format Agent Response",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [940, 0],
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
            "position": [680, 220],
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
            "main": [[{"node": "Validate Input & Session", "type": "main", "index": 0}]]
        },
        "Validate Input & Session": {
            "main": [[{"node": "Input Valid?", "type": "main", "index": 0}]]
        },
        "Input Valid?": {
            "main": [
                [{"node": "Call Python LangGraph API", "type": "main", "index": 0}],
                [{"node": "Format Validation Error", "type": "main", "index": 0}],
            ]
        },
        "Call Python LangGraph API": {
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
        "versionId": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"langgraph-gateway-openai-{base_url}")),
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


def ensure_river_ai_openai_credential(
    *,
    repo_root: Path,
    user_folder: Path,
    api_key: str | None = None,
    base_url: str = "https://api.river.ai/v1",
) -> bool:
    """Import/update encrypted River AI openAiApi credential in n8n for the OpenAI Chat Model nodes."""
    key = (api_key or os.getenv("RIVER_AI_API") or "").strip()
    sqlite_path = user_folder / ".n8n" / "database.sqlite"
    if not key or not sqlite_path.exists():
        return False

    n8n_bin = find_n8n_binary(repo_root)
    if n8n_bin is None:
        return False

    payload = [
        {
            "id": RIVER_OPENAI_CREDENTIAL_ID,
            "name": RIVER_OPENAI_CREDENTIAL_NAME,
            "type": "openAiApi",
            "data": {
                "apiKey": key,
                "url": base_url.rstrip("/"),
            },
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
        _ensure_credential_ownership(sqlite_path, RIVER_OPENAI_CREDENTIAL_ID)
        return True
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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
