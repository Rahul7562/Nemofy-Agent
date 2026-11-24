# Nemofy Agent & Generalized `Agent-Core` LangGraph Platform

> **Production-ready, pure-Python LangGraph autonomous AI agent engine paired with a protected Express gateway, AES-256-GCM encrypted API key vault, sandboxed execution runtime, Firecrawl Local web research, and a Vercel-ready React + Tailwind CSS workspace.**

```text
+======================================================================================+
|  VERIFICATION STATUS:  100/100 Python Tests (0 Warnings)  |  92/92 TypeScript Tests  |
|  STATIC ANALYSIS:      0 Ruff Errors  |  0 tsc Errors  |  21/21 Python Modules OK    |
|  LIVE BENCHMARKS:      6/6 Runtime Suites PASS  |  Live River AI & OpenRouter PASS   |
+======================================================================================+
```

---

## Table of Contents

1. [What Is Nemofy Agent?](#1-what-is-nemofy-agent)
2. [Full-Stack System Architecture (ASCII Diagram)](#2-full-stack-system-architecture)
3. [Inside `Agent-Core`: Cyclic LangGraph Supervisor Engine](#3-inside-agent-core-cyclic-langgraph-supervisor-engine)
4. [Feature Deep Dive](#4-feature-deep-dive)
   - [4.1 Generalized Standalone `agent_core` Python Library](#41-generalized-standalone-agent_core-python-library)
   - [4.2 Dual-Model, Multi-Provider & Multimodal Orchestration](#42-dual-model-multi-provider--multimodal-orchestration)
   - [4.3 Security Model, Sandbox Isolation & HITL Guardrails](#43-security-model-sandbox-isolation--hitl-guardrails)
   - [4.4 Context Window Budgeting & Strict 80% Auto-Compaction](#44-context-window-budgeting--strict-80-auto-compaction)
   - [4.5 Dynamic Skills, Direct MCP Servers & Firecrawl Local](#45-dynamic-skills-direct-mcp-servers--firecrawl-local)
   - [4.6 Protected Web Workspace (`Nemofy Agent/`)](#46-protected-web-workspace-nemofy-agent)
5. [Repository Structure & Module Map](#5-repository-structure--module-map)
6. [Live Benchmarks, Latency Analysis & Acceptance Evidence](#6-live-benchmarks-latency-analysis--acceptance-evidence)
7. [Quickstart: How to Use After Downloading](#7-quickstart-how-to-use-after-downloading)
   - [Mode A: Full-Stack Local Development](#mode-a-full-stack-local-development-recommended)
   - [Mode B: Embed `Agent-Core` as a Standalone Python Library](#mode-b-embed-agent-core-as-a-standalone-python-library)
   - [Mode C: Pure-Python Interactive CLI](#mode-c-pure-python-interactive-cli)
   - [Mode D: Production Docker Compose Stack](#mode-d-production-docker-compose-stack)
   - [Mode E: Deploy `Nemofy Agent/` Frontend on Vercel](#mode-e-deploy-nemofy-agent-frontend-on-vercel)
8. [Configuration & Environment Variables](#8-configuration--environment-variables)
9. [HTTP API Reference](#9-http-api-reference)
10. [Testing, Self-Check & Benchmarking](#10-testing-self-check--benchmarking)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)

---

## 1. What Is Nemofy Agent?

**Nemofy Agent** is an end-to-end autonomous AI agent platform built on two cleanly decoupled layers:

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **`Agent-Core/`** | Python 3.11+, LangGraph, FastAPI, SQLite WAL, Docker SDK | Generalized cyclic LangGraph reasoning/execution runtime (`Plan → Act → Observe → Reflect → Response`), isolated sandbox, HITL guardrails, context compaction, skills, MCP client, sub-agents, and async job manager. Can be imported **standalone** in future Python projects. |
| **`server.ts` & `server/`** | Node.js 22+, Express 4, TypeScript, AES-256-GCM | Unified API gateway on port `:3000`: server-enforced session auth, per-user encrypted API key vault, thread/filesystem persistence, provider model catalog, background job orchestration, and automatic supervision of Python `Agent-Core` (`:8100`) and `Firecrawl Local` (`:3002`). |
| **`Nemofy Agent/`** | React 19, Vite 8, Tailwind CSS 4, TypeScript | Protected, Vercel-ready SPA with Light & Dark mode, multi-thread chat, live thinking/tool/file-diff activity stream, interactive HITL approval cards, context usage ring, `/workspace` file explorer, and unified Settings modal. |

---

## 2. Full-Stack System Architecture

```text
+======================================================================================+
|                           BROWSER CLIENT (Light / Dark UI)                           |
|  Nemofy Agent SPA (React 19 + Vite 8 + Tailwind v4)                                  |
|  [LoginGate] -> [Sidebar Threads] [ChatView + ActivityPanel + HITL] [WorkspaceDrawer]|
|                 [ContextUsageRing] [ModelSelectorCombobox] [SettingsModal]           |
+=========================================+============================================+
                                          | HTTPS / Same-Origin Cookie (HttpOnly)
                                          v
+======================================================================================+
|                    EXPRESS API GATEWAY  (server.ts  :3000)                           |
|                                                                                      |
|  +-----------------------+  +------------------------+  +-------------------------+  |
|  | AuthStore             |  | ProfileSecretVaultStore|  | WorkspaceAndConfigStore |  |
|  | scrypt + timingSafeEq |  | AES-256-GCM + scrypt   |  | Atomic JSON persistence |  |
|  | Session cookies       |  | data/vaults/<user>.json|  | /workspace, Skills, MCP |  |
|  +-----------------------+  +-----------+------------+  +-------------------------+  |
|                                         | Syncs decrypted keys in-memory only        |
|  +-----------------------+  +-----------v------------+  +-------------------------+  |
|  | ChatJobOrchestrator   |  | ProviderCatalog        |  | Process Managers        |  |
|  | Background polling    |  | Live /models + context |  | AgentCoreProcessManager |  |
|  | Thread race-locking   |  | windows + SSRF guard   |  | FirecrawlProcessManager |  |
|  +-----------+-----------+  +------------------------+  +-----------+-------------+  |
+==============|======================================================|================+
               | Bearer AGENT_API_TOKEN (127.0.0.1:8100)              | 127.0.0.1:3002
               v                                                      v
+===========================================================+  +-----------------------+
|          AGENT-CORE ENGINE (FastAPI + LangGraph :8100)    |  | FIRECRAWL LOCAL :3002 |
|                                                           |  | Native Rust binary    |
|  [JobManager + Heartbeat Watchdog]                        |  | - HTML -> Markdown    |
|         |                                                 |  | - JS via Chromium     |
|         v                                                 |  | - Native PDF/DOCX/    |
|  [LangGraph Supervisor: Plan -> Act -> Observe -> Reflect]|  |   XLSX/PPTX parser    |
|    |-- ContextManager (80% auto-compaction + budgets)     |  +-----------------------+
|    |-- Guardrails & HITL Interrupts (SAFE vs SENSITIVE)   |
|    |-- SandboxManager (Docker OR local_isolated)          |
|    |-- SkillRegistry (/Skills/*.md dynamic loading)       |
|    |-- Sub-Agent Runner (isolated worker LangGraph)       |
|    |-- Multimodal Accessories (Image / Video / Audio)     |
|    |-- MCPRegistryManager (/MCPs/servers.json)            |
|    +-- Checkpointer (SQLite WAL | :memory: | MemorySaver) |
+============================+==============================+
                             | OpenAI-Compatible / Native HTTPS
                             v
+======================================================================================+
|  12+ LLM PROVIDERS: River AI, OpenRouter, OpenAI, Anthropic, Google Gemini, NVIDIA,  |
|                     DeepSeek, Groq, Together, Fireworks, Ollama, vLLM + Custom APIs  |
+======================================================================================+
```

---

## 3. Inside `Agent-Core`: Cyclic LangGraph Supervisor Engine

The heart of `Agent-Core` (`Agent-Core/agent_core/graph.py`) is a deterministic, checkpointed cyclic LangGraph state machine. Every node transition persists state to the active checkpointer so runs survive crashes, restarts, and Human-in-the-Loop pauses without ever re-executing completed side effects.

```text
                                 User Task / Follow-Up
                                           |
                                           v
                       +---------------------------------------+
                       |             CONTEXT CHECK             |
                       |  Compute token usage vs context_window|
                       |  If usage >= 80% -> Auto-Compact old  |
                       |  turns (preserving tool-call pairs)   |
                       +-------------------+-------------------+
                                           |
                                           v
                  +--------------------------------------------------+
      +---------> |                    PLAN NODE                     |
      |           |  - Enforce max_turns & max_wall_clock_seconds    |
      |           |  - Build persona prompt + active skills catalog  |
      |           |  - Stream live <think>/reasoning tokens          |
      |           |  - Resilient retry + timeout fallback model      |
      |           |  - Empty/truncated output auto-recovery          |
      |           +------------------------+-------------------------+
      |                                    |
      |                     +--------------+--------------+
      |                     |                             |
      |             Has tool_calls?               No tool_calls
      |                     |                             |
      |                     v                             v
      |           +-------------------+       +-----------------------+
      |           |     ACT NODE      |       |     RESPONSE NODE     |
      |           | (1 tool per step) |       | Extract final answer, |
      |           +---------+---------+       | thinking, file diffs, |
      |                     |                 | & context snapshot    |
      |                     v                 +-----------+-----------+
      |           +-------------------+                   |
      |           |  GUARDRAIL GATE   |                   v
      |           |  SAFE | SENSITIVE |              [COMPLETED]
      |           +----+---------+----+
      |                |         |
      |             SAFE     SENSITIVE (in default mode)
      |                |         |
      |                |         v
      |                |   [LangGraph interrupt()] ---> PAUSE (awaiting_approval)
      |                |         |                      Survives full restart!
      |                |   Human Decision:
      |                |   approve | edit (validated) | reject
      |                |         |
      |                +----+----+
      |                     v
      |           +-------------------+
      |           |  EXECUTE TOOL     |
      |           |  (Sandbox / Web / |
      |           |   Skills / MCP /  |
      |           |   Sub-Agent)      |
      |           +---------+---------+
      |                     |
      |                     v
      |           +-------------------+
      |           |   OBSERVE NODE    |
      |           |  >4,000 chars?    |
      |           |  Archive raw log  |
      |           |  /workspace/.logs |
      |           |  (read_file exempt)|
      |           +---------+---------+
      |                     |
      |                     v
      |           +-------------------+
      |           |   REFLECT NODE    |
      |           |  Track retries,   |
      |           |  workspace diffs, |
      |           |  update plan step |
      +-----------+-------------------+
```

### Why 1 Tool Call Per Checkpoint Step Matters
When a model emits multiple tool calls in a single turn (for example, `write_file` followed by a sensitive `gmail_send_email`), `act_node` executes **one pending tool call per LangGraph checkpoint step**. If tool #2 raises `GraphInterrupt` for human approval, tool #1 is already committed to the SQLite checkpoint and is **never replayed** when the user resumes—even across a full server restart.

---

## 4. Feature Deep Dive

### 4.1 Generalized Standalone `agent_core` Python Library

`Agent-Core/agent_core` is decoupled from Nemofy-specific branding and web infrastructure so you can reuse the exact same battle-tested LangGraph engine in any future Python project:

- **`AgentConfig`** (`agent_core.config`): Single immutable configuration object controlling persona, execution limits, guardrail policy, lifecycle hooks, default access mode, feature toggles (`include_default_tools`, `enable_web_tools`, `enable_skills`, `enable_subagents`, `enable_workspace_diff`, `enable_auto_compaction`), FastAPI metadata, and custom checkpointer injection.
- **`AgentPersona`**: Swap agent identity, organization, instructions, full `custom_system_prompt`, or a dynamic `system_prompt_builder` callable while keeping Nemofy's production prompt intact by default.
- **`ExecutionLimits`**: Validated safety caps for `max_turns` (`1..180`), `max_tool_retries`, `tool_output_compaction_threshold`, `auto_compact_threshold` (`0.10..0.99`), and `max_wall_clock_seconds` (terminates runaway loops cleanly).
- **`GuardrailPolicy`**: Immutable policy with `.extend(add_safe_tools=..., add_sensitive_tools=..., add_safe_bash_commands=..., custom_classifier=...)` so custom domain tools integrate with the central HITL interrupt system cleanly.
- **`AgentHooks`**: Sync or async lifecycle callbacks (`on_turn_start`, `on_tool_start`, `on_tool_end`, `on_compaction`, `on_run_complete`) wrapped in defensive error isolation so a failing hook never crashes an agent run.
- **Pluggable Persistence & Optional MCP**: Supports SQLite WAL (`checkpoints.db`), pure in-memory SQLite (`":memory:"`, creating zero files on disk), or any LangGraph `BaseCheckpointSaver` (e.g., `MemorySaver()`, PostgreSQL, Redis). Gracefully falls back to a no-op MCP manager when the sibling `MCPs/` folder is absent.
- **Public Facade**: Import everything cleanly from `agent_core` (`from agent_core import AgentConfig, AgentPersona, GuardrailPolicy, ExecutionLimits, AgentHooks, create_agent_runtime, create_app`).

### 4.2 Dual-Model, Multi-Provider & Multimodal Orchestration

- **Main Agent (Brain)** + **Sub-Agent (Worker)** split: Assign a frontier reasoning model to planning/synthesis and a fast, economical model to isolated sub-agent tasks (`spawn_subagent`).
- **12+ Built-in Providers + Unlimited Custom Providers**: River AI, OpenRouter, OpenAI, Anthropic, Google Gemini, NVIDIA NIM, DeepSeek, Groq, Together AI, Fireworks AI, Ollama, vLLM, plus user-defined custom OpenAI-compatible or Anthropic-compatible endpoints with per-provider encrypted API keys.
- **Per-Session Isolation**: Every chat thread stores its own `ThreadSessionConfig` (provider, model, reasoning effort, context window, access mode). Switching providers/models in Thread A never mutates Thread B.
- **Thinking & Reasoning Controls**: Supports adaptive reasoning effort (`off`, `low`, `medium`, `high`), streams `<think>` tags and provider-native `reasoning_content` / `reasoning_details` live to the UI, and preserves Gemini thought signatures across tool-call roundtrips.
- **Multimodal Accessories**: Automatically detects whether the primary model natively supports `image`, `video`, or `audio`. Unsupported modalities are transparently routed to dedicated multimodal accessory models (`image`, `video`, `audio`), with an automatic runtime fallback if a model unexpectedly rejects a native media payload.
- **Resilient Retry & Timeout Fallback**: Classifies transient 429 rate limits, read timeouts, and connection errors; applies jittered exponential backoff while keeping job heartbeats alive; optionally falls back to a configured secondary provider/model on timeout.

### 4.3 Security Model, Sandbox Isolation & HITL Guardrails

```text
+------------------------------------------------------------------------------+
|                         DEFENSE-IN-DEPTH SECURITY LAYERS                     |
+------------------------------------------------------------------------------+
| 1. Auth & Sessions     | scrypt password hashing + constant-time verify      |
|                        | HttpOnly SameSite cookies; Bearer token on :8100    |
+------------------------+-----------------------------------------------------+
| 2. Encrypted Vault     | Per-user AES-256-GCM + scrypt KDF + AAD binding     |
|                        | Plaintext API keys never written to llm_config.json |
+------------------------+-----------------------------------------------------+
| 3. Outbound URL Guard  | Blocks loopback, link-local, metadata (169.254/16)  |
|                        | and private RFC1918 ranges on web/MCP/provider URLs |
+------------------------+-----------------------------------------------------+
| 4. Sandbox Confinement | Docker (non-root, no socket, 512m RAM, 128 PIDs) or |
|                        | local_isolated (path + symlink escape rejection,    |
|                        | secret env scrubbing, fail-closed bash_exec)        |
+------------------------+-----------------------------------------------------+
| 5. HITL Guardrails     | Read-only bash AST allowlist -> SAFE                |
|                        | Destructive/mutating/external tools -> SENSITIVE    |
|                        | Pauses via LangGraph interrupt() until approved     |
+------------------------------------------------------------------------------+
```

- **Fail-Closed `bash_exec` Classifier**: Only strictly read-only shell pipelines (`ls`, `cat`, `grep`, `rg`, `git status/log/diff`, etc. without redirection `>`, subshells `$(...)`, or mutating flags like `sed -i` / `find -delete`) are auto-approved in `default` mode. Everything else requires explicit human approval unless the user explicitly switches the session to `full_access`.
- **Interactive HITL Decisions**:
  - **Approve**: Executes the paused tool call **exactly once** and continues the graph.
  - **Edit**: Validates modified JSON arguments against the tool's Pydantic schema and prompts for re-approval before execution.
  - **Reject**: Skips execution (`count == 0`), feeds the rejection reason back to the planner, and completes cleanly.

### 4.4 Context Window Budgeting & Strict 80% Auto-Compaction

```text
Token Budget Breakdown (e.g. 128,000-token context window)
[==== System + Tools ====|=== Active History ===|== Tool Outputs ==|     Free Headroom     ]
0%                      12%                    45%                68%                     100%
                                                                       ^
                                                                       | 80.0% Trigger
When usage_percent >= 80.0%:
  1. Find safe split boundary (never splits an AIMessage tool_call from its ToolMessage)
  2. Summarize older turns via resilient LLM compaction call -> compact summary block
  3. Preserve recent turns + active system/skill context verbatim
  4. Emit `auto_compact_context` event & update ContextUsageRing in real time
```

Additionally, any non-`read_file` tool output exceeding **4,000 characters** is automatically archived in full to `/workspace/.logs/<tool>_<timestamp>.log` and replaced in active context with a compact head/tail summary pointing to the archived path. `read_file` is intentionally exempt (with line pagination support) so source code reads stay exact.

### 4.5 Dynamic Skills, Direct MCP Servers & Firecrawl Local

- **Dynamic Skills (`/Skills/*.md`)**: Compact skill catalog is injected into the system prompt; full playbooks (such as `code_and_data.md` and `deep_research.md`, plus custom user-created Markdown skills) are loaded on demand via `load_skill` to keep baseline token overhead minimal.
- **Direct Pure-Python MCP (`/MCPs/servers.json`)**: Connects directly to external Streamable HTTP / SSE Model Context Protocol servers via `langchain-mcp-adapters`, validates URLs against private-network targets, and degrades gracefully if an external MCP server is offline.
- **Firecrawl Local Web Research (`agent_core/firecrawl.py`)**: Integrates the official Rust `firecrawl-local` v2 server (`:3002`) with 5 built-in tools (`web_search`, `web_fetch`, `web_crawl`, `web_map`, `web_extract`), native PDF/DOCX/XLSX/PPTX document extraction, and optional Chromium JS rendering—zero Redis, Postgres, or RabbitMQ sidecars required.

### 4.6 Protected Web Workspace (`Nemofy Agent/`)

- **Login Gate**: Administrator setup on first launch; instant password rotation with session invalidation.
- **Autonomous Chat View**: Grouped thread history, live streaming thinking drawer, real-time tool execution badges, workspace file/folder diff tracker (`created` / `modified` / `deleted`), multimodal file uploads, inline HITL approval/edit/reject controls, and interactive `ContextUsageRing`.
- **Workspace Filesystem Drawer**: Full `/workspace` tree browser, instant search, syntax-friendly viewer/editor, file & folder creation, and safe deletion.
- **Unified Settings Modal**: Configure Main Agent, Sub-Agent, Accessories (Image/Video/Audio), Custom Providers, Encrypted API Key Vault, Thinking/Timeout policies, Execution Access Mode, Skills, MCP Servers, Password, and Light/Dark theme.

---

## 5. Repository Structure & Module Map

```text
nemofy-agent/
├── Agent-Core/                          # Generalized pure-Python LangGraph engine
│   ├── README.md                        # Standalone library usage & API guide
│   ├── main.py                          # Interactive Rich CLI (`--check`, `--thread-id`)
│   ├── Dockerfile.sandbox               # Hardened non-root Python + Playwright sandbox image
│   ├── agent_core/
│   │   ├── __init__.py                  # Clean public library facade (`__all__`)
│   │   ├── config.py                    # AgentConfig, AgentPersona, ExecutionLimits, AgentHooks
│   │   ├── graph.py                     # Cyclic LangGraph supervisor & AgentRuntime
│   │   ├── state.py                     # Typed AgentState, PlanStep, 4k log compaction
│   │   ├── planner.py                   # System prompt builder & structured plan parser
│   │   ├── guardrails.py                # GuardrailPolicy, bash AST classifier, HITL gate
│   │   ├── sandbox.py                   # Docker & local_isolated sandbox + path confinement
│   │   ├── context_manager.py           # Exact token accounting & 80% auto-compaction
│   │   ├── models.py                    # 12+ provider specs, retry engine, timeout fallback
│   │   ├── multimodal.py                # Media capability detection & accessory sub-agents
│   │   ├── firecrawl.py                 # Firecrawl Local v2 client & document extraction
│   │   ├── skills.py                    # /Skills/*.md registry & on-demand loader
│   │   ├── subagents.py                 # Isolated worker sub-agent graph runner
│   │   ├── jobs.py                      # Async JobManager, heartbeats & stall watchdog
│   │   ├── api.py                       # FastAPI service (`create_app`, Bearer auth)
│   │   ├── message_utils.py             # Pure <think>/reasoning parser & history sanitizer
│   │   ├── workspace_diff.py            # Workspace file/directory snapshot diffing
│   │   └── tools/local.py               # Built-in file, bash, and Firecrawl tools
│   ├── benchmarks/
│   │   └── run_benchmarks.py            # Live 6-suite benchmark harness with acceptance caps
│   └── tests/                           # 100+ pytest tests + self_test.py runner
├── Nemofy Agent/                        # Standalone Vercel-deployable React + Vite SPA
│   ├── package.json                     # Self-contained frontend dependencies (Vite 8 + React 19)
│   ├── vercel.json                      # SPA rewrites & security headers
│   └── src/
│       ├── App.tsx                      # Top-level workspace & session state coordinator
│       ├── components/                  # ChatView, AgentActivityPanel, ContextUsageRing,
│       │                                # FilesystemView, SettingsModal, LoginGate, etc.
│       ├── hooks/                       # useAgentExecution, useChatThreads
│       └── lib/                         # apiClient, llmPresets, contextUtils, markdownRenderer
│                                        # + 10 TypeScript unit test suites
├── server.ts                            # Unified Express gateway (:3000) + Vite middleware
├── server/                              # Backend gateway modules + 7 unit test suites
│   ├── authStore.ts                     # scrypt password & session management
│   ├── profileVault.ts                  # Per-user AES-256-GCM encrypted API key vault
│   ├── configStore.ts                   # Atomic JSON stores for LLM config, threads, workspace
│   ├── providerCatalog.ts               # Live model discovery, context metadata & SSRF guard
│   ├── jobOrchestrator.ts               # Background job poller & thread concurrency locking
│   ├── agentProcessManager.ts           # Auto-bootstraps .venv & supervises Agent-Core :8100
│   ├── firecrawlProcessManager.ts       # Supervises local Firecrawl Rust binary :3002
│   └── serverConfig.ts                  # CLI/env host & port resolution
├── Skills/                              # Dynamic Markdown playbooks (code_and_data, deep_research)
├── MCPs/                                # Pure-Python MCPRegistryManager & servers.json
├── Web-design/                          # Nemofy brand design tokens & SVG logo
├── scripts/                             # Runtime build, Firecrawl Local & Chromium installers
├── Dockerfile                           # Multi-stage production container (Rust + Node + Python)
├── docker-compose.yml                   # Single-container production stack with named volumes
├── pyproject.toml                       # Python dependencies, pytest, ruff & mypy config
├── package.json                         # Root workspace scripts & Express/Vite dependencies
└── .env.example                         # Documented environment variable template
```

---

## 6. Live Benchmarks, Latency Analysis & Acceptance Evidence

All benchmarks below were executed live on this repository via `Agent-Core/benchmarks/run_benchmarks.py` and live end-to-end API runs against configured encrypted-vault providers.

### 6.1 Runtime Benchmark Suite Scorecard (`6/6 PASS`)

| # | Benchmark Suite | Measured Result | Acceptance Criterion | Verdict |
| :-: | :--- | :--- | :--- | :-: |
| 1 | **Supervisor Graph Latency & Persistence** (SQLite WAL, `:memory:`, `MemorySaver`) | Direct p50 `39.2 ms` / p95 `41.9 ms`<br>Multi-step p50 `102.3 ms` / p95 `106.2 ms`<br>`:memory:` disk leak: `False` | Direct p95 `< 350 ms`<br>Multi-step p95 `< 600 ms`<br>Zero `:memory:` file created | **PASS** |
| 2 | **Sandbox Safety & Perf (`local_isolated`)** (10 Python runs + timeout + escapes) | Exec p50 `102.5 ms` / p95 `113.3 ms`<br>1s timeout kill: `1054.0 ms`<br>Traversal, symlink, env scrub: `100%` | Exec p95 `< 500 ms`<br>Timeout kill `< 2500 ms`<br>100% security blocks | **PASS** |
| 3 | **Guardrails, HITL & Wall-Clock Safety Cap** | Approve side-effects: `1`, Reject: `0`<br>Hooks: `10` events (error isolated)<br>Wall-clock halt: `305.7 ms` (`2` turns) | Approve `== 1`, Reject `== 0`<br>Wall-clock halt `< 1000 ms`<br>Runaway turns `< 5` | **PASS** |
| 4 | **Context Accounting & 80% Auto-Compaction** | Turn 1: `59.9%` (`0` compactions)<br>Turn 2: `63.0%` (`0` compactions)<br>Turn 3: `>= 80%` -> `1` compaction | Turns 1–2 uncompacted<br>Turn 3 compacted `>= 1`<br>Summary tokens `> 0` | **PASS** |
| 5 | **Concurrent SQLite WAL Isolation** (12 parallel multi-step threads) | `12/12` completed in `961.4 ms` wall<br>Thread p95 `864.1 ms`<br>Cross-thread bleed / lock errors: `0` | `12/12` pass, `0` bleed<br>Total wall `< 10,000 ms` | **PASS** |
| 6 | **FastAPI Auth, Async Jobs & Cancellation** | `/health` clean (`skills_dir` absent)<br>Unauth `401` enforced; `8/8` jobs p95 `461.6 ms`<br>In-flight job cancel: `4.8 ms` | `401` unauth, `8/8` jobs<br>Cancel `< 1000 ms`<br>Zero host-path leaks | **PASS** |

### 6.2 Measured Latency Distribution (ASCII Bar Chart)

```text
Agent-Core Local Engine Latency (lower is better, p95 vs acceptance budget)
----------------------------------------------------------------------------------------
In-Flight Job Cancel          | #                                  4.8 ms  (cap: 1000ms)
Direct LangGraph Turn (p95)   | ####                              41.9 ms  (cap:  350ms)
Multi-Step Tool Loop (p95)    | ##########                       106.2 ms  (cap:  600ms)
Isolated Sandbox Exec (p95)   | ###########                      113.3 ms  (cap:  500ms)
Wall-Clock Runaway Halt       | ##############################   305.7 ms  (cap: 1000ms)
8x Concurrent Async Jobs (p95)| ############################################## 461.6 ms
12x Parallel WAL Threads (p95)| ######################################################## 864.1 ms
----------------------------------------------------------------------------------------
                              0ms       200ms      400ms      600ms      800ms     1000ms
```

### 6.3 Live Cloud LLM End-to-End Verification (Configured Encrypted Vault Providers)

Executed live through the running Express/FastAPI stack (`127.0.0.1:8100`) using AES-256-GCM vault keys synced in memory:

| Live Scenario | Active Provider & Model | Latency | Turns | Verified Outcome |
| :--- | :--- | :---: | :---: | :--- |
| **Multi-Step Tool Loop** (`write_file` → `read_file` → synthesis) | `river_ai` / `deepseek-ai/DeepSeek-V4.1-Flash` | `4,578 ms` | 2 | **PASS** — Both tools `ok`, file diff `created` tracked, `64` chars reasoning, `3,829 / 1,048,576` tokens (`0.4%`) |
| **Dynamic Provider Switch** (per-request override) | `openrouter` / `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | `1,574 ms` | 1 | **PASS** — Exact target response returned, `238` chars reasoning extracted |
| **Sub-Agent Delegation + Fail-Closed Sandbox** (`spawn_subagent`) | `river_ai` / `deepseek-ai/DeepSeek-V4.1-Flash` (Primary + Worker) | `14,348 ms` | 3 | **PASS** — Sub-agent report returned (`17 × 23 = 391`); blocked `bash_exec` gracefully reported without crash |

### 6.4 Automated Test & Quality Gate Summary

```text
Check                                           Result                 Notes
------------------------------------------------------------------------------------------------
Python py_compile (Agent-Core + MCPs)           21 / 21 PASS           All modules clean
Ruff Linter (Agent-Core + tests + benchmarks)   0 errors PASS          Includes ASYNC & S rules
Pytest Full Suite (Agent-Core/tests)            100 PASS / 0 FAIL      0 warnings (1 optional skip*)
TypeScript Tests (server + Nemofy Agent/src)    92 PASS / 0 FAIL       Auth, vault, catalog, UI logic
TypeScript Typecheck (root & Nemofy Agent)      0 errors PASS          Strict mode enabled
Vite Production Build (`npm run build`)         PASS                   Frontend bundle built cleanly
Headless Chromium Verification (:3000)          PASS                   Clean desktop render, 0 app errors
```

*\*1 skipped test (`test_firecrawl_pdf_and_office_documents_remote_and_local` step 3) skips cleanly only when the optional Rust `firecrawl-local` binary is not yet installed on the host; steps 1 & 2 always run, and Docker builds include `firecrawl-local` + Chromium automatically.*

---

## 7. Quickstart: How to Use After Downloading

### Prerequisites

- **Node.js 22+** and **npm**
- **Python 3.11+**
- *(Optional)* **Docker** — for container-isolated sandbox execution (`Agent-Core/Dockerfile.sandbox`) or one-command `docker compose` deployment
- *(Optional)* **Rust / Cargo** — only if you want to compile `firecrawl-local` locally outside Docker (`npm run firecrawl:install`)

---

### Mode A: Full-Stack Local Development (Recommended)

1. **Install Node dependencies**:
   ```bash
   npm install
   ```

2. **Install Python `Agent-Core` dependencies** *(optional — `npm run dev` also auto-bootstraps `.venv` automatically on cold start)*:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install -e ".[dev]"
   ```

3. **Optional environment configuration**:
   ```bash
   cp .env.example .env
   # You do NOT need to put LLM API keys in .env — you can enter them securely
   # inside the web UI Settings modal where they are stored encrypted (AES-256-GCM).
   ```

4. **Launch the unified server** (starts Express + Vite on `:3000`, Python `Agent-Core` on `:8100`, and `Firecrawl Local` on `:3002` if installed):
   ```bash
   npm run dev
   ```

5. **Open `http://localhost:3000`**:
   - Create your administrator username & password on the **Protected Access Gate** (first run).
   - Open **Settings → LLM Config**, add your provider API key (e.g., OpenAI, OpenRouter, River AI, Anthropic, Gemini, NVIDIA, DeepSeek, or local Ollama), pick your **Main Agent** & **Sub-Agent** models, and click **Test Connection**.
   - Start chatting, watch live thinking/tools/file changes, inspect `/workspace` in the Filesystem drawer, and approve/edit/reject sensitive actions via HITL prompts.

---

### Mode B: Embed `Agent-Core` as a Standalone Python Library

Use `agent_core` directly in your own LangGraph projects without needing Express, React, or `MCPs/`:

```python
import asyncio
from pathlib import Path
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agent_core import (
    AgentConfig,
    AgentHooks,
    AgentPersona,
    ExecutionLimits,
    GuardrailPolicy,
    SandboxConfig,
    SandboxManager,
    create_agent_runtime,
)

@tool("lookup_order")
def lookup_order(order_id: str) -> str:
    """Look up an order's shipping status."""
    return f"Order {order_id}: out for delivery"

async def main() -> None:
    sandbox = SandboxManager(
        SandboxConfig(workspace_dir=Path("./workspace"), backend="local_isolated")
    )
    config = AgentConfig(
        persona=AgentPersona(
            name="Support Agent",
            organization="Acme Corp",
            role_description="a concise order support specialist",
            extra_instructions="Always confirm the order ID in your reply.",
        ),
        limits=ExecutionLimits(max_turns=12, max_wall_clock_seconds=45.0),
        guardrail_policy=GuardrailPolicy().extend(add_safe_tools=("lookup_order",)),
        hooks=AgentHooks(
            on_tool_end=lambda ev: print(f"[hook] {ev['tool']} -> {ev['status']}"),
            on_run_complete=lambda ev: print(f"[hook] run finished: {ev['status']}"),
        ),
        include_default_tools=False,   # Expose ONLY your domain tools
        enable_skills=False,
        enable_subagents=False,
        custom_checkpointer=MemorySaver(),  # Or omit for durable SQLite WAL
    )

    async with create_agent_runtime(
        sandbox=sandbox,
        extra_tools=[lookup_order],
        connect_mcp=False,
        config=config,
    ) as runtime:
        result = await runtime.run_task("Where is order ORD-1042?", thread_id="ticket-42")
        print(result["final_response"])

asyncio.run(main())
```

Or mount `Agent-Core` as a standalone FastAPI microservice:

```python
from agent_core import AgentConfig, create_app

app = create_app(
    checkpoint_db_path=":memory:",   # Pure in-memory SQLite (zero disk artifacts)
    connect_mcp=False,
    agent_config=AgentConfig(app_title="Custom Domain Agent API"),
)
```

See [`Agent-Core/README.md`](Agent-Core/README.md) for the full standalone library guide.

---

### Mode C: Pure-Python Interactive CLI

```bash
# Verify configuration, sandbox, skills, and model reachability
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/main.py --check

# Launch interactive terminal agent session with HITL prompts
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/main.py --thread-id my-session
```

---

### Mode D: Production Docker Compose Stack

The multi-stage `Dockerfile` compiles the official Rust `firecrawl-local` binary, installs Chromium, Node 22, and Python 3.11 `.venv`, builds the React frontend, and runs everything inside a single self-contained image with persistent volumes (`nemofy_data`, `nemofy_workspace`):

```bash
# Build and start in detached mode on http://localhost:3000
npm run docker:up

# View logs / stop
npm run docker:logs
npm run docker:down
```

*(Optional)* Build the hardened Docker sandbox image for container-isolated code execution:
```bash
docker build -f Agent-Core/Dockerfile.sandbox -t nemofy-agent-sandbox:latest Agent-Core
```

---

### Mode E: Deploy `Nemofy Agent/` Frontend on Vercel

1. Import this repository in **Vercel** and set **Root Directory** to `Nemofy Agent`.
2. Vercel auto-detects **Vite** (`npm run build` → `dist`, SPA rewrites & security headers pre-configured in `Nemofy Agent/vercel.json`).
3. Set environment variable `VITE_AGENT_API_URL` to your hosted Express gateway URL (or set it inside **Settings → Appearance & Runtime**).

---

## 8. Configuration & Environment Variables

All supported variables are listed in `.env.example`. Most settings can also be managed live through the web UI without editing `.env`.

| Category | Key Variables | Default / Behavior |
| :--- | :--- | :--- |
| **Ports & Auth** | `PORT`, `AGENT_API_PORT`, `AGENT_API_TOKEN`, `NEMOFY_ADMIN_PASSWORD`, `VAULT_MASTER_KEY` | Web `:3000`, Agent-Core `:8100`; tokens/vault keys auto-generated securely if omitted |
| **Primary / Worker Models** | `PRIMARY_PROVIDER`, `PRIMARY_MODEL`, `PRIMARY_BASE_URL`, `PRIMARY_TEMPERATURE`, `PRIMARY_MAX_TOKENS`, `PRIMARY_REASONING_EFFORT` (and `WORKER_*`) | Defaults configurable via UI (`data/llm_config.json`) |
| **Provider API Keys** | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `RIVER_AI_API`, `NVIDIA_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY` | Recommended: store via UI in encrypted `data/vaults/<user>.vault.json` (AES-256-GCM) |
| **Resilience & Limits** | `MAX_TURNS`, `MODEL_CONNECT_TIMEOUT_SECONDS`, `MODEL_READ_TIMEOUT_SECONDS`, `MODEL_MAX_RETRIES` | `30` turns (`1..180`), `15s` connect, `180s` read, `2` retries |
| **Sandbox** | `SANDBOX_BACKEND` (`auto` \| `docker` \| `local_isolated`), `ALLOW_UNSAFE_LOCAL_BASH`, `SANDBOX_TIMEOUT_SECONDS`, `WORKSPACE_DIR` | Fail-closed (`ALLOW_UNSAFE_LOCAL_BASH=false`), `300s` timeout |
| **Firecrawl Local** | `FIRECRAWL_API_URL`, `FIRECRAWL_PORT`, `FIRECRAWL_LOCAL_AUTOSTART`, `FIRECRAWL_CHROME_PATH`, `FIRECRAWL_API_KEY` | `http://127.0.0.1:3002`, auto-starts if binary installed |
| **Storage & Paths** | `CHECKPOINT_DB_PATH`, `LLM_CONFIG_PATH`, `SKILLS_DIR`, `MCP_CONFIG_PATH` | `data/checkpoints.db`, `data/llm_config.json`, `Skills/`, `MCPs/servers.json` |

---

## 9. HTTP API Reference

### Express Gateway (`http://localhost:3000`)

| Endpoint | Auth | Purpose |
| :--- | :---: | :--- |
| `GET /api/health` | Public | Gateway & Agent-Core readiness probe |
| `POST /api/auth/setup` / `login` / `logout` / `change-password` | Public / Cookie | Administrator authentication & session rotation |
| `GET/POST/DELETE /api/threads` | Cookie | Multi-thread chat history & per-session model config |
| `POST /api/chat/send` / `resume` / `cancel` | Cookie | Launch/resume/cancel background agent jobs |
| `GET /api/chat/jobs/:jobId` | Cookie | Live polling for thinking, tools, file diffs & context usage |
| `GET/PUT/POST/DELETE /api/workspace/*` | Cookie | Confined `/workspace` file & directory CRUD |
| `GET/PUT /api/settings/llm` & `POST /api/settings/probe` | Cookie | Dual-model, accessory, custom provider & vault management |
| `GET/PUT/DELETE /api/skills` & `GET/PUT /api/mcps` | Cookie | Manage dynamic Markdown skills and MCP servers |

### Agent-Core FastAPI Service (`http://127.0.0.1:8100`)

| Endpoint | Auth | Purpose |
| :--- | :---: | :--- |
| `GET /health` | Public | Runtime, models, checkpoint, sandbox, Firecrawl, skills & MCP health (zero host-path leaks) |
| `POST /agent/run` & `POST /agent/resume` | Bearer | Synchronous LangGraph execution & HITL approval resume |
| `POST /agent/jobs/run` & `POST /agent/jobs/resume` | Bearer | Non-blocking async jobs (HTTP `202 Accepted`) |
| `GET /agent/jobs/{job_id}?wait_ms=...` | Bearer | Long-poll progress heartbeats, thinking stream & result |
| `POST /agent/jobs/{job_id}/cancel` | Bearer | Immediate cooperative cancellation of in-flight jobs |
| `POST /config/secrets` / `DELETE /config/secrets` | Bearer | In-memory sync of decrypted vault secrets from gateway |

---

## 10. Testing, Self-Check & Benchmarking

Run the complete verification suite anytime from the repository root:

```bash
# 1. Python syntax compile + Ruff lint + Full Pytest verification suite
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/tests/self_test.py

# 2. TypeScript server + frontend library unit tests (92 tests)
npx --yes tsx --test server/*.test.ts "Nemofy Agent/src/lib/"*.test.ts

# 3. TypeScript strict typecheck (root + standalone Nemofy Agent)
npm run lint
npx tsc -p "Nemofy Agent/tsconfig.json" --noEmit

# 4. Live 6-suite Agent-Core benchmark harness (latency, sandbox, HITL, compaction, WAL concurrency, HTTP jobs)
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/benchmarks/run_benchmarks.py

# 5. Production build
npm run build
```

---

## 11. Troubleshooting & FAQ

- **`bash_exec` returns an error saying Docker is unavailable and unisolated local fallback is disabled**:
  This is intentional fail-closed security (`ALLOW_UNSAFE_LOCAL_BASH=false`). Either run Docker with `Agent-Core/Dockerfile.sandbox`, or explicitly enable **Allow local bash fallback** in **Settings → Appearance & Runtime** (or set `ALLOW_UNSAFE_LOCAL_BASH=true`) on trusted personal machines.
- **Firecrawl web tools report `127.0.0.1:3002` unavailable locally**:
  Run `npm run firecrawl:install` (requires Rust/Cargo + C compiler) or run via `npm run docker:up`, which pre-compiles `firecrawl-local` and bundles Chromium automatically. Alternatively, add a `FIRECRAWL_API_KEY` in Settings to use Firecrawl Cloud.
- **How do I reset my administrator password or encrypted vault?**
  Stop the server and remove `data/auth.json` (and `data/vaults/` if rotating the vault master key), then restart `npm run dev` to run the clean first-time setup flow.

---

## License

Released under the terms in [`LICENSE.md`](LICENSE.md).
