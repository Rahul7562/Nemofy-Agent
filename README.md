# Nemofy Agent

Pure-Python autonomous LangGraph AI agent platform paired with a protected, Vercel-ready **Nemofy Agent** web workspace synced with **https://nemofy.tech/** (Light & Dark mode).

## Directory Layout

- **`Agent-Core/`** — Generalized, reusable pure-Python autonomous AI agent engine (LangGraph stateful execution loop, `AgentConfig` / `AgentPersona` / `GuardrailPolicy` / `ExecutionLimits` / `AgentHooks`, Main Agent & Sub-Agent delegation, SQLite WAL / `:memory:` / custom checkpointers, isolated Python/Bash sandbox, HITL guardrails, FastAPI service, CLI, `pytest` suite, and live benchmarks — see [`Agent-Core/README.md`](Agent-Core/README.md)).
- **`Nemofy Agent/`** — Standalone, Vercel-deployable React + Vite + Tailwind CSS web application with protected login gate, **Chat**, **Filesystem** (`/workspace`), and **Settings** (**LLM Config** for Main Agent & Sub-Agent, **Change Password**, **Skills & MCPs**, **Appearance & Runtime**).
- **`Web-design/`** — Shared Nemofy brand design tokens (`design-tokens.css`), exact SVG brand mark (`nemofy-logo.svg`), and design documentation.
- **`Skills/`** — Dynamic Markdown agent skills (`code_and_data.md`, `deep_research.md`, plus custom skills created from Settings).
- **`MCPs/`** — Pure-Python Model Context Protocol (MCP) registry and client loader (`manager.py`, `servers.json`).
- **`server/` & `server.ts`** — Unified Express API gateway with server-side session authentication, `/workspace` filesystem routes, settings persistence, and Python `Agent-Core` process supervision.

## Running Locally

```bash
# Start unified web server (:3000) + pure-Python Agent-Core FastAPI (:8100)
npm run dev
```

Or run the Python CLI, test suite, or live benchmarks directly:

```bash
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/main.py --check
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/tests/self_test.py
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/benchmarks/run_benchmarks.py
```

## Deploying `Nemofy Agent/` on Vercel

1. Connect this repository in Vercel and set **Root Directory** to `Nemofy Agent`.
2. Vercel automatically detects Vite (`vercel.json` is pre-configured with SPA rewrites).
3. Optionally configure `VITE_AGENT_API_URL` (or set the backend URL inside **Settings → Appearance & Runtime**) to connect your Vercel deployment to a hosted `Agent-Core` server.
