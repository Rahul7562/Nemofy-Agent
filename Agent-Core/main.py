"""Interactive Async Rich CLI for Nemofy Agent-Core (Pure Python).

Usage:
    python Agent-Core/main.py
    python Agent-Core/main.py --thread-id <existing_thread_id>
    python Agent-Core/main.py --thread-id <id> --task "Inspect /workspace and summarize"
    python Agent-Core/main.py --check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure Agent-Core and repo root (for MCPs/) are on sys.path when invoked directly
_AGENT_CORE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _AGENT_CORE_DIR.parent
for _p in (str(_AGENT_CORE_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from agent_core.graph import AgentRuntime, create_agent_runtime
from agent_core.models import ModelConfigurationError, load_model_spec, probe_model_connectivity
from agent_core.sandbox import SandboxManager
from agent_core.skills import SkillRegistry
from agent_core.state import PlanStep, make_initial_state
from agent_core.tools.local import current_session_id
from MCPs.manager import MCPRegistryManager

load_dotenv()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemofy-agent",
        description="Interactive CLI for the Pure-Python LangGraph + Sandbox + MCP Autonomous Agent.",
    )
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--no-mcp", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def render_banner(console: Console, *, thread_id: str, runtime: AgentRuntime | None = None) -> None:
    primary = load_model_spec("primary", include_persisted=True)
    worker = load_model_spec("worker", include_persisted=True)
    mcp_status = (
        f"connected ({len(runtime.mcp_tool_names)} tools)"
        if runtime and runtime.mcp_manager.health_status()["connected"]
        else "standby (local hands active)"
    )
    sandbox_backend = runtime.sandbox.active_backend if runtime else "uninitialized"
    skills_count = len(runtime.skills.list_skills()) if runtime else 0

    body = (
        f"[bold]Thread ID:[/bold]     {thread_id}\n"
        f"[bold]Main Agent:[/bold]    {primary.provider} / {primary.model}\n"
        f"[bold]Sub-Agent:[/bold]     {worker.provider} / {worker.model}\n"
        f"[bold]Sandbox:[/bold]       {sandbox_backend}\n"
        f"[bold]Skills:[/bold]        {skills_count} discovered in ./Skills/\n"
        f"[bold]MCP Servers:[/bold]   {mcp_status}"
    )
    console.print(Panel(body, title="Nemofy Agent-Core", border_style="blue"))


def render_plan(console: Console, plan: list[PlanStep]) -> None:
    if not plan:
        return
    table = Table(title="Execution Plan Checklist", show_lines=False)
    table.add_column("Status", width=12)
    table.add_column("Step", width=10)
    table.add_column("Action", width=22)
    table.add_column("Description")

    icon_map = {
        "completed": "[green][DONE][/green]",
        "in_progress": "[yellow][RUN][/yellow]",
        "pending": "[dim][WAIT][/dim]",
        "failed": "[red][FAIL][/red]",
        "skipped": "[magenta][SKIP][/magenta]",
    }
    for step in plan:
        table.add_row(
            icon_map.get(step["status"], step["status"]),
            step["step_id"],
            step["tool_or_action"],
            step["description"],
        )
    console.print(table)


def prompt_human_approval(console: Console, approval_payload: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(approval_payload.get("tool_name", "unknown"))
    args_json = str(
        approval_payload.get("arguments_json")
        or json.dumps(approval_payload.get("arguments", {}), indent=2, default=str)
    )
    effect = str(approval_payload.get("expected_effect", ""))
    risk = str(approval_payload.get("risk_reason", ""))
    source = str(approval_payload.get("source", "supervisor"))
    edited = bool(approval_payload.get("edited", False))
    val_err = approval_payload.get("validation_error")

    lines = [
        "[bold red]SENSITIVE ACTION PAUSED FOR APPROVAL[/bold red]",
        f"[bold]Tool:[/bold]            {tool_name}",
        f"[bold]Source:[/bold]          {source}",
        f"[bold]Edited Args:[/bold]     {'YES (please review & re-approve)' if edited else 'No'}",
        f"[bold]Expected Effect:[/bold] {effect}",
        f"[bold]Policy Reason:[/bold]   {risk}",
        f"\n[bold]Complete JSON Arguments:[/bold]\n{args_json}",
    ]
    if val_err:
        lines.append(f"\n[bold red]Validation Error:[/bold red] {val_err}")

    console.print(Panel("\n".join(lines), title="Human-in-the-Loop Guardrail", border_style="red"))

    choice = Prompt.ask(
        "Decision", choices=["approve", "reject", "edit"], default="reject", console=console
    )
    if choice == "approve":
        return {"decision": "approve"}
    if choice == "reject":
        reason = Prompt.ask(
            "Rejection reason (optional)", default="Rejected in CLI", console=console
        )
        return {"decision": "reject", "reason": reason}

    raw_json = Prompt.ask(
        "Enter replacement arguments as JSON object",
        default=json.dumps(approval_payload.get("arguments", {})),
        console=console,
    )
    try:
        parsed_args = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON:[/red] {exc}")
        return {"decision": "edit", "arguments": {"__invalid_json__": raw_json}}
    return {"decision": "edit", "arguments": parsed_args}


async def stream_turn_with_hitl(
    runtime: AgentRuntime,
    console: Console,
    *,
    thread_id: str,
    task: str | None = None,
) -> dict[str, Any]:
    token = current_session_id.set(thread_id)
    try:
        config = runtime.make_config(thread_id)
        current = await runtime.inspect_thread(thread_id)
        if current["status"] == "awaiting_approval" and task is None:
            console.print(
                f"[yellow]Resuming previously interrupted thread '{thread_id}'...[/yellow]"
            )
        elif task is not None:
            existing = await runtime.graph.aget_state(config)
            has_history = bool(existing and existing.values and existing.values.get("messages"))
            if has_history:
                payload: dict[str, Any] = {
                    "messages": [HumanMessage(content=task.strip())],
                    "turn_count": 0,
                    "max_turns": runtime.max_turns,
                    "next_action": "continue",
                    "final_response": "",
                    "error": None,
                    "thread_id": thread_id,
                }
            else:
                payload = dict(
                    make_initial_state(
                        [HumanMessage(content=task.strip())],
                        thread_id=thread_id,
                        max_turns=runtime.max_turns,
                    )
                )
            async for chunk in runtime.graph.astream(payload, config=config, stream_mode="updates"):
                _render_stream_chunk(console, chunk)

        while True:
            state_summary = await runtime.inspect_thread(thread_id)
            if state_summary["status"] != "awaiting_approval":
                if state_summary.get("plan"):
                    render_plan(console, state_summary["plan"])
                if state_summary.get("error"):
                    console.print(
                        Panel(str(state_summary["error"]), title="Error", border_style="red")
                    )
                elif state_summary.get("final_response"):
                    console.print(
                        Panel(
                            str(state_summary["final_response"]),
                            title="Agent Response",
                            border_style="green",
                        )
                    )
                return state_summary

            approval_payload = state_summary["pending_approval"] or {}
            decision = prompt_human_approval(console, approval_payload)
            async for chunk in runtime.graph.astream(
                Command(resume=decision),
                config=config,
                stream_mode="updates",
            ):
                _render_stream_chunk(console, chunk)
    finally:
        current_session_id.reset(token)


def _render_stream_chunk(console: Console, chunk: dict[str, Any]) -> None:
    for node_name, update in chunk.items():
        if node_name == "__interrupt__":
            console.print("[bold yellow]Graph paused at guardrail interrupt.[/bold yellow]")
            continue
        if not isinstance(update, dict):
            continue
        if node_name == "plan":
            turn = update.get("turn_count")
            console.print(f"[cyan]• PLAN[/cyan] (turn {turn})")
            if update.get("plan"):
                render_plan(console, update["plan"])
        elif node_name == "act":
            for obs in update.get("last_observations", []):
                console.print(
                    f"[blue]• ACT[/blue] {obs.get('tool', 'tool')} -> {obs.get('status', 'ok')}"
                )
        elif node_name == "observe":
            console.print("[dim]• OBSERVE[/dim] plan state updated")
        elif node_name == "reflect":
            console.print(
                f"[dim]• REFLECT[/dim] next={update.get('next_action')} — "
                f"{str(update.get('reflection', ''))[:140]}"
            )


async def run_readiness_check(console: Console, *, connect_mcp: bool = True) -> int:
    probe = await probe_model_connectivity("primary", run_inference=False)
    sandbox = SandboxManager()
    skills = SkillRegistry()
    mcp = MCPRegistryManager()
    if connect_mcp:
        await mcp.connect()

    report = {
        "primary_model_probe": probe,
        "worker_model": load_model_spec("worker", include_persisted=True).safe_summary(),
        "sandbox": sandbox.health_status(),
        "skills": [s.name for s in skills.list_skills()],
        "mcps": mcp.health_status(),
    }
    console.print_json(json.dumps(report, indent=2, default=str))
    sandbox.cleanup_all()
    return 0


async def interactive_loop(args: argparse.Namespace) -> int:
    console = Console()
    if args.check:
        return await run_readiness_check(console, connect_mcp=not args.no_mcp)

    thread_id = (args.thread_id or f"cli-{uuid.uuid4().hex[:10]}").strip()
    try:
        async with create_agent_runtime(connect_mcp=not args.no_mcp) as runtime:
            render_banner(console, thread_id=thread_id, runtime=runtime)
            existing_state = await runtime.inspect_thread(thread_id)
            if existing_state["status"] == "awaiting_approval":
                await stream_turn_with_hitl(runtime, console, thread_id=thread_id, task=None)

            if args.task:
                await stream_turn_with_hitl(runtime, console, thread_id=thread_id, task=args.task)
                return 0

            console.print("[dim]Type a task and press Enter. Commands: /plan, /status, /quit[/dim]")
            while True:
                try:
                    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]", console=console).strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Session saved to checkpoint store. Goodbye.[/dim]")
                    return 0

                if not user_input:
                    continue
                if user_input.lower() in {"/quit", "/exit", "exit", "quit"}:
                    return 0
                if user_input.lower() == "/plan":
                    st = await runtime.inspect_thread(thread_id)
                    render_plan(console, st.get("plan", []))
                    continue
                if user_input.lower() == "/status":
                    st = await runtime.inspect_thread(thread_id)
                    console.print_json(json.dumps(st, indent=2, default=str))
                    continue

                await stream_turn_with_hitl(runtime, console, thread_id=thread_id, task=user_input)
    except ModelConfigurationError as exc:
        console.print(Panel(str(exc), title="Model Configuration Error", border_style="red"))
        return 1
    except Exception as exc:  # noqa: BLE001
        console.print(
            Panel(f"{type(exc).__name__}: {exc}", title="Fatal Error", border_style="red")
        )
        return 1


def cli_entrypoint() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(interactive_loop(args)))


if __name__ == "__main__":
    cli_entrypoint()
