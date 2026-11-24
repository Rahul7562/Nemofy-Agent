# Agent-Core (Generalized Pure-Python LangGraph Engine)

`agent_core` is a modular, production-hardened LangGraph autonomous agent runtime designed to work both inside **Nemofy Agent** and as a standalone reusable Python library in future applications.

## Architecture

`agent_core` implements a cyclic **Plan → Act → Observe → Reflect → Response** supervisor graph with:

- **Configurable Runtime (`agent_core.config`)**:
  - `AgentConfig` — single typed object controlling persona, safety limits, guardrail policy, lifecycle hooks, feature toggles (`include_default_tools`, `enable_web_tools`, `enable_skills`, `enable_subagents`, `enable_workspace_diff`, `enable_auto_compaction`), and custom LangGraph checkpointer injection.
  - `AgentPersona` — swap identity, organization, instructions, full custom system prompt, or a dynamic `system_prompt_builder` callable without touching graph internals.
  - `ExecutionLimits` — validated bounds for `max_turns`, `max_tool_retries`, `tool_output_compaction_threshold`, `auto_compact_threshold`, and `max_wall_clock_seconds`.
  - `AgentHooks` — defensive sync or async lifecycle hooks (`on_turn_start`, `on_tool_start`, `on_tool_end`, `on_compaction`, `on_run_complete`); hook exceptions are logged and isolated from execution.
- **Pluggable Persistence**:
  - SQLite WAL checkpointer (`AsyncSqliteSaver`) for durable pause/resume across restarts.
  - True in-memory SQLite (`checkpoint_db_path=":memory:"`, creates zero disk artifacts).
  - Direct custom LangGraph checkpointer injection via `AgentConfig(custom_checkpointer=MemorySaver())` (or PostgreSQL/Redis savers).
- **Composable Guardrails & HITL (`agent_core.guardrails`)**:
  - Immutable `GuardrailPolicy` with `.extend(safe_tools=..., sensitive_tools=...)` to classify domain-specific tools cleanly.
  - Deterministic pause/resume via LangGraph `interrupt(...)` with `approve`, `reject`, and `edit` decisions and zero duplicate side effects.
- **Sandboxed Execution (`agent_core.sandbox`)**:
  - `docker` (hardened ephemeral container) or `local_isolated` (path-confined workspace, secret environment scrubbing, bounded timeouts).
- **Context & Token Management (`agent_core.context_manager`)**:
  - Exact model window resolution, per-category token accounting, 4,000-char tool log archiving to `/workspace/.logs/`, and strict threshold-based auto-compaction preserving tool-call/result pairs.
- **Modular Utilities**:
  - `agent_core.message_utils` — pure message sanitization, `<think>` / reasoning extraction, finish-reason diagnostics, and empty-output recovery.
  - `agent_core.workspace_diff` — deterministic workspace file change tracking.
  - Optional sibling `MCPs` integration — `agent_core` gracefully falls back to a no-op registry when `MCPs/` is not installed.

## Standalone Library Quickstart

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
    """Look up order status by ID."""
    return f"Order {order_id}: shipped"

async def main() -> None:
    sandbox = SandboxManager(SandboxConfig(workspace_dir=Path("./workspace"), backend="local_isolated"))
    config = AgentConfig(
        persona=AgentPersona(
            name="Support Agent",
            organization="Acme Corp",
            role_description="a concise order support specialist",
            extra_instructions="Always confirm order IDs before responding.",
        ),
        limits=ExecutionLimits(max_turns=12, max_wall_clock_seconds=45.0),
        guardrail_policy=GuardrailPolicy().extend(add_safe_tools=("lookup_order",)),
        hooks=AgentHooks(on_run_complete=lambda ev: print("Done:", ev["status"])),
        include_default_tools=False,
        enable_skills=False,
        enable_subagents=False,
        custom_checkpointer=MemorySaver(),
    )

    async with create_agent_runtime(
        sandbox=sandbox,
        extra_tools=[lookup_order],
        connect_mcp=False,
        config=config,
    ) as runtime:
        result = await runtime.run_task("Check order ORD-1042", thread_id="ticket-42")
        print(result["final_response"])

asyncio.run(main())
```

## Embedding the FastAPI Service

```python
from agent_core import AgentConfig, create_app

app = create_app(
    checkpoint_db_path=":memory:",
    connect_mcp=False,
    agent_config=AgentConfig(app_title="My Custom Agent API"),
)
```

## Verification & Live Benchmarks

Run static checks and the full test suite from the repository root:

```bash
PYTHONPATH=Agent-Core:. .venv/bin/ruff check Agent-Core MCPs
PYTHONPATH=Agent-Core:. .venv/bin/pytest Agent-Core/tests -v
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/tests/self_test.py
```

Run the live end-to-end benchmark suite with safety limits and pass/fail acceptance criteria:

```bash
PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/benchmarks/run_benchmarks.py
```
