# Production-Grade Autonomous AI Agent

**LangGraph (Brain) + Docker Sandbox (Local Hands) + n8n MCP (SaaS Arm)**

A modular, fault-tolerant, multipurpose autonomous AI agent upgraded directly on top of the existing n8n workflow automation repository—strictly preserving existing n8n workflows, persistent SQLite data, credentials, and the `5678 -> 3000` web interface mapping.

---

## 1. Architecture

Three strictly separated subsystems:

```text
User / CLI (main.py) / n8n Gateway Workflow ("LangGraph Autonomous AI Agent Gateway")
                                |
                                v
               Authenticated FastAPI Service (src/api.py :8100)
                                |
                                v
              +-----------------------------------+
              |      BRAIN — LangGraph StateGraph |
              |   PLAN -> ACT -> OBSERVE -> REFLECT
              |  AsyncSqliteSaver (checkpoints.db)|
              |  Central Guardrails (HITL interrupt)
              |  Context Compaction (>4,000 chars)|
              +-----------------+-----------------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
   LOCAL HANDS             EXTENSIBILITY           SAAS ARM
 Docker Sandbox         Skills & Sub-Agents        n8n MCP
 (python:3.12-slim)     ./skills/*.md (lazy)     Streamable HTTP
 /workspace only        Isolated Worker Graphs   MultiServerMCPClient
 bash_exec, read/write  WORKER_MODEL default     Gmail, Sheets, Calendar,
 edit_file, list_dir    Inherited HITL safety    Slack, Webhooks, Workflows
 web_fetch (Markdown)
```

- **Brain (`src/graph.py`, `src/state.py`, `src/planner.py`)**: Cyclic `langgraph.graph.StateGraph` (`PLAN -> ACT -> OBSERVE -> REFLECT -> (CONTINUE -> PLAN | COMPLETE -> RESPONSE)`) with durable `AsyncSqliteSaver` checkpointing (`./data/checkpoints.db`), `MAX_TURNS` loop protection (default `30`), bounded tool retries, and automatic 4,000-character context compaction into `/workspace/.logs/`.
- **Local Hands (`src/sandbox.py`, `src/tools/local.py`, `Dockerfile.sandbox`)**: Isolated `python:3.12-slim` Docker containers (CPU/memory/PID limits, `cap_drop=ALL`, no Docker socket mount, workspace confined to `/workspace`) plus a hardened `local_isolated` subprocess fallback when Docker daemon is absent on the host.
- **SaaS Arm (`src/tools/mcp_n8n.py`)**: Connects to the existing n8n MCP server (`http://localhost:5678/mcp-server/http`) over Streamable HTTP using `langchain-mcp-adapters` (`MultiServerMCPClient`), Bearer auth, startup schema caching, optional `ALLOWED_N8N_TOOLS` allowlisting, reconnection, and graceful offline degradation.
- **Safety Gate (`src/guardrails.py`)**: Central SAFE vs. SENSITIVE classifier using LangGraph `interrupt()` so all sensitive operations—whether initiated by the supervisor or a sub-agent—pause with full JSON arguments and expected effect for human `approve`, `reject`, or `edit` (with validation and mandatory re-approval).

---

## 2. Installation Instructions

### Prerequisites
- Python **3.11+** (`uv` recommended, or `pip`)
- Node.js **22+** (for the preserved `server.ts` n8n proxy & existing n8n 2.35.7 runtime)
- Docker Engine (recommended for container-level sandbox isolation; `local_isolated` mode is available if Docker is unavailable)

### Setup Python Environment

```bash
# Using uv (recommended)
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]"

# Or standard venv + pip
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env and set RIVER_AI_API, AGENT_API_TOKEN, and N8N_MCP_TOKEN
```

---

## 3. River AI Configuration

The agent uses **DeepSeek V4.1 Flash through River AI** via its OpenAI-compatible endpoint (`https://api.river.ai/v1`), verified against both the live endpoint and existing repository configuration.

```env
RIVER_AI_API=<your-river-ai-api-key>
PRIMARY_PROVIDER=river_ai
PRIMARY_BASE_URL=https://api.river.ai/v1
PRIMARY_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
```

Verify connectivity without running inference:

```bash
.venv/bin/python main.py --check
```

---

## 4. Primary and Worker Model Setup

Both `PRIMARY_MODEL` and `WORKER_MODEL` are provider-agnostic and switchable purely via environment variables:

```env
# Supervisor reasoning model
PRIMARY_PROVIDER=river_ai
PRIMARY_BASE_URL=https://api.river.ai/v1
PRIMARY_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
PRIMARY_TEMPERATURE=0.2
PRIMARY_MAX_TOKENS=4096

# Economical isolated sub-agent worker model
WORKER_PROVIDER=river_ai
WORKER_BASE_URL=https://api.river.ai/v1
WORKER_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
WORKER_TEMPERATURE=0.1
WORKER_MAX_TOKENS=2048

# Timeouts & retries
MODEL_TIMEOUT_SECONDS=90
MODEL_MAX_RETRIES=2
```

To switch to OpenRouter or OpenAI without changing code, update `PRIMARY_PROVIDER`, `PRIMARY_BASE_URL`, `PRIMARY_MODEL`, and set `OPENROUTER_API_KEY` or `OPENAI_API_KEY`.

---

## 5. Docker Sandbox Configuration

Build the sandbox image from `Dockerfile.sandbox` (`python:3.12-slim` + Python data/web packages + headless Playwright Chromium, non-root `agent` user, `/workspace` mountpoint, no Docker socket):

```bash
docker build -f Dockerfile.sandbox -t autonomous-agent-sandbox:latest .
```

Configure limits in `.env`:

```env
SANDBOX_BACKEND=auto               # auto | docker | local_isolated
SANDBOX_IMAGE=autonomous-agent-sandbox:latest
SANDBOX_MEMORY_LIMIT=512m
SANDBOX_CPU_QUOTA=100000
SANDBOX_CPU_PERIOD=100000
SANDBOX_PIDS_LIMIT=128
SANDBOX_TIMEOUT_SECONDS=60
SANDBOX_NETWORK_MODE=bridge
WORKSPACE_DIR=./workspace
```

---

## 6. n8n MCP Endpoint and Authentication Setup

The existing n8n 2.35.7 instance exposes its built-in Streamable HTTP MCP Server at `/mcp-server/http`.

```env
N8N_MCP_URL=http://127.0.0.1:5678/mcp-server/http
N8N_MCP_TOKEN=<your-n8n-mcp-server-jwt-token>
ALLOWED_N8N_TOOLS=                 # Optional comma-separated allowlist
N8N_MCP_TIMEOUT_SECONDS=30
```

In n8n Settings -> **Instance-level MCP**, enable MCP access and generate an MCP Server Access Token (`audience=mcp-server-api`). Any workflow marked **Available in MCP** (with supported triggers such as Webhook, Chat, Form, or Schedule) plus n8n's built-in MCP tools are discovered automatically and cached at startup.

---

## 7. Network Configuration

- **Preserved n8n Web Mapping**: `127.0.0.1:5678 -> 0.0.0.0:3000` (managed by `server.ts` — untouched).
- **Same-host deployment**:
  - Python Agent API listens on `0.0.0.0:8100` (`AGENT_API_PUBLIC_URL=http://127.0.0.1:8100`).
  - Python Agent reaches n8n MCP at `N8N_MCP_URL=http://127.0.0.1:5678/mcp-server/http`.
  - n8n Gateway Workflow HTTP Request node calls `http://127.0.0.1:8100/agent/run` (and `/agent/resume`).
- **Docker Compose / multi-container network**:
  - Do **not** use `localhost` across separate containers. Use Docker service DNS names on a shared bridge network:
    - `N8N_MCP_URL=http://n8n:5678/mcp-server/http`
    - `AGENT_API_PUBLIC_URL=http://agent-api:8100`
  - Sync the Gateway workflow with the reachable URL:
    ```bash
    AGENT_API_PUBLIC_URL=http://agent-api:8100 .venv/bin/python src/tools/n8n_gateway_sync.py
    ```

---

## 8. Starting and Stopping Services

### Preserved n8n + Web Proxy (`5678 -> 3000`)

```bash
npm run dev
# Access n8n UI at http://localhost:3000
```

### Python LangGraph FastAPI Service (`:8100`)

```bash
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8100
```

Stop cleanly with `Ctrl+C` (`SIGINT`/`SIGTERM`), which flushes SQLite WAL checkpoints and stops managed sandbox containers.

---

## 9. CLI Usage

Start a new interactive session:

```bash
.venv/bin/python main.py
```

Resume an existing session (including sessions paused at a HITL approval prompt):

```bash
.venv/bin/python main.py --thread-id <existing_thread_id>
```

Run a single task non-interactively or inspect dependency health:

```bash
.venv/bin/python main.py --thread-id demo-01 --task "List files in /workspace and create a summary."
.venv/bin/python main.py --check
```

Interactive commands inside the CLI: `/plan`, `/status`, `/quit`.

---

## 10. API Usage

All agent execution endpoints require Bearer authentication (`Authorization: Bearer <AGENT_API_TOKEN>`).

### Health Check

```bash
curl -s http://127.0.0.1:8100/health | jq .
```

### Execute or Continue a Task

```bash
curl -s -X POST http://127.0.0.1:8100/agent/run \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "Create /workspace/hello.py and run it", "thread_id": "session-101"}'
```

### Resume a Paused HITL Approval

```bash
curl -s -X POST http://127.0.0.1:8100/agent/resume \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-101", "decision": "approve"}'
```

### Trigger via n8n Gateway Workflow Webhook & Native OpenAI Model Selection

You can select or switch the **Brain (supervisor)** and **Sub-Agent (worker)** LLMs directly from n8n across any OpenAI-compatible provider (`river_ai`, `openai`, `openrouter`, `gemini`, `groq`, `deepseek`, `together`, `fireworks`, `ollama`, `vllm`, or any custom `base_url`) in three ways:

1. **Native n8n OpenAI Chat Model Nodes (Visual Dropdown)**: Open `"LangGraph Autonomous AI Agent Gateway"` in n8n and click:
   - **`Brain LLM (OpenAI Chat Model)`** -> pick any n8n OpenAI-compatible credential (e.g., `River AI (OpenAI Compatible)`, OpenAI, OpenRouter, Groq, Ollama) and choose your Brain model from the live dropdown.
   - **`Sub-Agent LLM (OpenAI Chat Model)`** -> pick any credential & model independently for isolated LangGraph sub-agents.
2. **Inline Chat Directives**: Prefix any message in the n8n Chat Trigger with `/brain provider:model` and/or `/subagent provider:model` (or `/model provider:model`).
3. **Webhook / API JSON Payload**:
```bash
curl -s -X POST http://127.0.0.1:5678/webhook/langgraph-agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Inspect /workspace and summarize active n8n workflows",
    "thread_id": "n8n-session-01",
    "primary_model": { "provider": "openrouter", "model": "anthropic/claude-3.7-sonnet" },
    "worker_model": { "provider": "openai", "model": "gpt-4o-mini" }
  }'
```

Inspect available provider presets at `GET http://127.0.0.1:8100/models/presets`.

---

## 11. Adding New Skills

Drop any Markdown file into `./skills/` with optional YAML frontmatter:

```markdown
---
name: security_audit
description: Static security review checklist for Python and TypeScript repositories.
---

# Security Audit Skill
...
```

At startup, only `name` and the one-line `description` are injected into the supervisor prompt. Full instructions load lazily when the agent calls `load_skill("security_audit")`. No orchestration code changes are required.

---

## 12. Configuring MCP Tool Allowlists

Set `ALLOWED_N8N_TOOLS` in `.env` to restrict which discovered n8n MCP tools are exposed to the LangGraph supervisor:

```env
ALLOWED_N8N_TOOLS=search_workflows,get_workflow_details,execute_workflow
```

Leave `ALLOWED_N8N_TOOLS` empty to expose all tools discovered from the connected n8n MCP server. All mutating or external MCP tools remain gated behind central HITL approval.

---

## 13. Creating Sub-Agents

The supervisor delegates independent workstreams via `spawn_subagent(task, role, allowed_tools, context, max_turns)`:

- Built-in default roles: `deep_research`, `repo_inspection`, `code_review`, `data_analysis`, `documentation`, `general`.
- Each worker runs an independent LangGraph `StateGraph` with its own message history and uses `WORKER_MODEL` by default.
- Workers can only invoke tools explicitly listed in `allowed_tools` (never `spawn_subagent`).
- All worker tool calls pass through `src/guardrails.py`—sub-agents cannot bypass HITL approval.

---

## 14. HITL Approval and Resume Procedures

When the supervisor or any sub-agent requests a **SENSITIVE** action (email, Slack, Calendar, Sheets mutation, deletion, destructive bash, external webhook/workflow execution):

1. LangGraph pauses execution via `interrupt()` before the tool runs.
2. The CLI / API / n8n Gateway displays:
   - Exact `tool_name`
   - Complete formatted JSON `arguments`
   - `expected_effect` and `risk_reason`
   - Initiating `source` (`supervisor` or `subagent:<role>`)
3. Operator responds with:
   - `approve` -> executes the exact displayed arguments once and resumes.
   - `reject` -> never executes the tool; records rejection and lets supervisor respond safely.
   - `edit` -> validates new JSON arguments against the tool schema, re-displays the modified arguments, and requires explicit `approve` before execution.

---

## 15. Session Checkpointing and Recovery

- State is persisted after every graph step to `./data/checkpoints.db` (`AsyncSqliteSaver` with SQLite WAL mode).
- If the CLI or API restarts while a session is mid-task or awaiting approval, re-opening the same `thread_id` restores the exact plan, completed steps (never re-executed), conversation context, and pending approval prompt.
- Tool outputs exceeding `CONTEXT_COMPACTION_THRESHOLD` (`4000` chars) are archived automatically to `/workspace/.logs/<timestamp>_<tool>_<id>.log`.

---

## 16. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Missing API key ... Set RIVER_AI_API` | `RIVER_AI_API` not configured in `.env` | Add valid `RIVER_AI_API` key to `.env` and restart |
| `401 Unauthorized` on `/agent/run` | Missing/mismatched `AGENT_API_TOKEN` | Pass `Authorization: Bearer <AGENT_API_TOKEN>` matching `.env` |
| `n8n_mcp.connected: false` | n8n offline, MCP disabled, or invalid `N8N_MCP_TOKEN` | Verify n8n `/healthz`, enable MCP access in n8n, check `N8N_MCP_URL` & `N8N_MCP_TOKEN` |
| `sandbox.backend: "local_isolated"` | Docker daemon not reachable on host | Start Docker daemon & build `Dockerfile.sandbox`, or continue using workspace-confined `local_isolated` |
| Gateway webhook returns `404` | Workflow not published or n8n not reloaded | Re-run `.venv/bin/python src/tools/n8n_gateway_sync.py` and reload n8n |
| Verification test suite | Run full verification suite | `.venv/bin/pytest -v` and `.venv/bin/python tests/self_test.py` |
