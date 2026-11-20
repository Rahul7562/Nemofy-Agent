"""Isolated LangGraph Sub-Agent System (spawn_subagent)."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agent_core.guardrails import execute_tool_with_guardrails
from agent_core.jobs import emit_progress
from agent_core.state import compact_tool_output

logger = logging.getLogger(__name__)

DEFAULT_SUBAGENT_MAX_TURNS = 8

ROLE_DEFAULT_TOOLS: dict[str, list[str]] = {
    "deep_research": ["web_fetch", "read_file", "write_file", "list_dir", "load_skill"],
    "repo_inspection": ["list_dir", "read_file", "bash_exec", "load_skill"],
    "code_review": ["list_dir", "read_file", "bash_exec", "load_skill"],
    "data_analysis": ["list_dir", "read_file", "write_file", "bash_exec", "load_skill"],
    "documentation": ["list_dir", "read_file", "write_file", "edit_file", "load_skill"],
    "general": ["list_dir", "read_file", "web_fetch", "load_skill"],
}


class SubagentReport(BaseModel):
    role: str
    task: str
    status: Literal["completed", "failed", "max_turns_reached"]
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    turns_used: int = 0
    error: str | None = None


class WorkerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    max_turns: int
    tools_used: list[str]
    final_output: str


def _parse_worker_report(
    *,
    role: str,
    task: str,
    raw_output: str,
    tools_used: list[str],
    turns_used: int,
    max_turns: int,
    error: str | None = None,
) -> SubagentReport:
    if error:
        return SubagentReport(
            role=role,
            task=task,
            status="failed",
            summary=f"Worker failed: {error}",
            key_findings=[],
            artifacts=[],
            tools_used=tools_used,
            turns_used=turns_used,
            error=error,
        )

    status: Literal["completed", "failed", "max_turns_reached"] = (
        "max_turns_reached" if turns_used >= max_turns and not raw_output.strip() else "completed"
    )

    text = raw_output.strip()
    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            try:
                candidate = json.loads(text[first : last + 1])
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                parsed = None

    if parsed:
        summary = str(parsed.get("summary") or text[:800]).strip()
        findings = [
            str(item).strip() for item in (parsed.get("key_findings") or []) if str(item).strip()
        ]
        artifacts = [
            str(item).strip() for item in (parsed.get("artifacts") or []) if str(item).strip()
        ]
        return SubagentReport(
            role=role,
            task=task,
            status=status,
            summary=summary,
            key_findings=findings,
            artifacts=artifacts,
            tools_used=tools_used,
            turns_used=turns_used,
        )

    lines = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]
    summary = lines[0][:500] if lines else "Worker completed without textual output."
    key_findings = [ln[:240] for ln in lines[1:6]] if len(lines) > 1 else []
    return SubagentReport(
        role=role,
        task=task,
        status=status,
        summary=summary if len(text) <= 1200 else text[:1200].rstrip() + "...",
        key_findings=key_findings,
        artifacts=[],
        tools_used=tools_used,
        turns_used=turns_used,
    )


async def run_isolated_subagent(
    *,
    task: str,
    role: str,
    worker_model: BaseChatModel,
    available_tools: dict[str, BaseTool],
    allowed_tool_names: list[str] | None = None,
    mcp_tool_names: set[str] | None = None,
    context: str = "",
    max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
    workspace_dir: str | None = None,
) -> SubagentReport:
    normalized_role = role.strip().lower() or "general"
    requested_names = (
        allowed_tool_names
        if allowed_tool_names is not None
        else ROLE_DEFAULT_TOOLS.get(normalized_role, ROLE_DEFAULT_TOOLS["general"])
    )

    permitted_tools: dict[str, BaseTool] = {
        name: available_tools[name]
        for name in requested_names
        if name in available_tools and name != "spawn_subagent"
    }
    mcp_names = mcp_tool_names or set()

    bound_model: Any = (
        worker_model.bind_tools(list(permitted_tools.values()))
        if permitted_tools and hasattr(worker_model, "bind_tools")
        else worker_model
    )

    async def worker_reason(state: WorkerState) -> dict[str, Any]:
        next_turn = state["turn_count"] + 1
        emit_progress(
            "subagent_reason",
            detail=f"Sub-agent '{normalized_role}' reasoning (worker turn {next_turn})",
        )
        response = await bound_model.ainvoke(state["messages"])
        content_text = str(getattr(response, "content", "") or "")
        emit_progress(
            "subagent_reason_done",
            detail=f"Sub-agent '{normalized_role}' turn {next_turn} complete",
        )
        return {
            "messages": [response],
            "turn_count": next_turn,
            "final_output": content_text,
        }

    async def worker_act(state: WorkerState) -> dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        tool_messages: list[ToolMessage] = []
        used = list(state["tools_used"])

        for tc in tool_calls:
            tc_name = str(tc.get("name", ""))
            tc_args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
            tc_id = str(tc.get("id", f"worker-tc-{len(tool_messages)}"))

            if tc_name not in permitted_tools:
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"PERMISSION DENIED: Sub-agent '{normalized_role}' is not authorized "
                            f"to call tool '{tc_name}'. Assigned tools: "
                            f"{sorted(permitted_tools.keys())}"
                        ),
                        tool_call_id=tc_id,
                        name=tc_name,
                    )
                )
                continue

            tool_obj = permitted_tools[tc_name]
            emit_progress(
                "subagent_tool_start",
                tool=tc_name,
                detail=f"Sub-agent '{normalized_role}' running tool '{tc_name}'",
            )
            raw_result, _audit = await execute_tool_with_guardrails(
                tool_obj,
                tc_args,
                is_mcp_tool=(tc_name in mcp_names),
                source=f"subagent:{normalized_role}",
            )
            emit_progress(
                "subagent_tool_done",
                tool=tc_name,
                detail=f"Sub-agent '{normalized_role}' finished tool '{tc_name}'",
            )
            compacted, _saved = compact_tool_output(
                f"subagent_{normalized_role}_{tc_name}",
                raw_result,
                workspace_dir=workspace_dir,
            )
            if tc_name not in used:
                used.append(tc_name)
            tool_messages.append(ToolMessage(content=compacted, tool_call_id=tc_id, name=tc_name))

        return {"messages": tool_messages, "tools_used": used}

    def route_worker(state: WorkerState) -> Literal["worker_act", "__end__"]:
        if state["turn_count"] >= state["max_turns"]:
            return "__end__"
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            return "worker_act"
        return "__end__"

    builder: StateGraph[WorkerState] = StateGraph(WorkerState)
    builder.add_node("worker_reason", worker_reason)
    builder.add_node("worker_act", worker_act)
    builder.add_edge(START, "worker_reason")
    builder.add_conditional_edges(
        "worker_reason",
        route_worker,
        {"worker_act": "worker_act", "__end__": END},
    )
    builder.add_edge("worker_act", "worker_reason")
    worker_graph = builder.compile()

    sys_prompt = (
        f"You are an isolated specialist sub-agent (role='{normalized_role}') executing a "
        f"delegated task for the parent LangGraph supervisor.\n"
        f"Assigned tools: {', '.join(sorted(permitted_tools.keys())) or '(none)'}.\n"
        "You have no access to tools outside your assigned list and cannot bypass safety gates.\n"
        "Accuracy rules: verify calculations or file facts with your assigned tools instead of "
        "guessing, and never fabricate results.\n"
        "When finished, return a concise JSON object with keys: "
        '{"summary": "...", "key_findings": ["..."], "artifacts": ["..."]}.'
    )
    user_prompt = f"Delegated Task:\n{task.strip()}"
    if context.strip():
        user_prompt += f"\n\nTask Context:\n{context.strip()}"

    initial_worker_state: WorkerState = {
        "messages": [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)],
        "turn_count": 0,
        "max_turns": max(1, max_turns),
        "tools_used": [],
        "final_output": "",
    }

    try:
        final_state = await worker_graph.ainvoke(initial_worker_state)
        return _parse_worker_report(
            role=normalized_role,
            task=task,
            raw_output=str(final_state.get("final_output", "")),
            tools_used=list(final_state.get("tools_used", [])),
            turns_used=int(final_state.get("turn_count", 0)),
            max_turns=max_turns,
        )
    except GraphInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Isolated sub-agent '%s' failed: %s", normalized_role, exc)
        return _parse_worker_report(
            role=normalized_role,
            task=task,
            raw_output="",
            tools_used=[],
            turns_used=0,
            max_turns=max_turns,
            error=str(exc),
        )


def make_spawn_subagent_tool(
    *,
    worker_model_factory: Any,
    available_tools: dict[str, BaseTool] | Any,
    mcp_tool_names: set[str] | Any | None = None,
    workspace_dir: str | None = None,
) -> BaseTool:
    @tool("spawn_subagent")
    async def spawn_subagent(
        task: str,
        role: str = "general",
        allowed_tools: list[str] | None = None,
        context: str = "",
        max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
    ) -> str:
        """Delegate a complex, independent task to an isolated LangGraph worker sub-agent."""
        worker_model = (
            worker_model_factory() if callable(worker_model_factory) else worker_model_factory
        )
        resolved_tools: dict[str, BaseTool] = (
            available_tools() if callable(available_tools) else available_tools
        )
        resolved_mcp_names: set[str] | None = (
            mcp_tool_names() if callable(mcp_tool_names) else mcp_tool_names
        )
        report = await run_isolated_subagent(
            task=task,
            role=role,
            worker_model=worker_model,
            available_tools=resolved_tools,
            allowed_tool_names=allowed_tools,
            mcp_tool_names=resolved_mcp_names,
            context=context,
            max_turns=max_turns,
            workspace_dir=workspace_dir,
        )
        return report.model_dump_json(indent=2)

    return spawn_subagent
