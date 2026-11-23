"""Isolated LangGraph Sub-Agent System (spawn_subagent)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent_core.guardrails import execute_tool_with_guardrails, normalize_tool_arguments
from agent_core.jobs import emit_progress, track_active_task
from agent_core.models import invoke_with_rate_limit_retry
from agent_core.state import compact_tool_output
from agent_core.tools.local import current_session_id

_WORKER_CHECKPOINTER = MemorySaver()
_WORKER_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", flags=re.IGNORECASE)
_WORKER_UNCLOSED_THINK_RE = re.compile(r"<think>[\s\S]*$", flags=re.IGNORECASE)


def _normalize_worker_finish_reason(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if "length" in s or "max_tokens" in s:
        return "length"
    if "tool_calls" in s:
        return "tool_calls"
    if "stop" in s or "end_turn" in s:
        return "stop"
    return s


def _extract_worker_text(msg: Any) -> str:
    raw = getattr(msg, "content", "")
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = str(item.get("type") or "").lower()
                if t in ("thinking", "reasoning"):
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
        text = "\n".join(parts)
    else:
        text = str(raw or "")
    text = _WORKER_THINK_RE.sub("", text)
    text = _WORKER_UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def _build_worker_recovery_messages(
    messages: list[BaseMessage],
    nudge: str,
) -> list[BaseMessage]:
    msgs = list(messages)
    if not msgs:
        return [HumanMessage(content=nudge)]
    last = msgs[-1]
    if isinstance(last, HumanMessage):
        existing = str(last.content or "").rstrip()
        msgs[-1] = HumanMessage(content=f"{existing}\n\n{nudge}" if existing else nudge)
        return msgs
    if isinstance(last, ToolMessage):
        msgs.append(AIMessage(content="I reviewed the tool outputs above."))
        msgs.append(HumanMessage(content=nudge))
        return msgs
    msgs.append(HumanMessage(content=nudge))
    return msgs


def _sanitize_worker_ai_message(msg: AIMessage) -> AIMessage:
    valid_tcs = list(getattr(msg, "tool_calls", None) or [])
    extra = dict(getattr(msg, "additional_kwargs", None) or {})
    inv_tcs = list(getattr(msg, "invalid_tool_calls", None) or [])
    if "tool_calls" not in extra and "function_call" not in extra and not inv_tcs:
        return msg
    extra.pop("tool_calls", None)
    extra.pop("function_call", None)
    return AIMessage(
        content=msg.content,
        additional_kwargs=extra,
        response_metadata=dict(getattr(msg, "response_metadata", None) or {}),
        tool_calls=valid_tcs,
        invalid_tool_calls=[],
    )


logger = logging.getLogger(__name__)

DEFAULT_SUBAGENT_MAX_TURNS = 8

ROLE_DEFAULT_TOOLS: dict[str, list[str]] = {
    "deep_research": [
        "web_search",
        "web_fetch",
        "web_map",
        "web_crawl",
        "web_extract",
        "read_file",
        "write_file",
        "list_dir",
        "load_skill",
    ],
    "repo_inspection": ["list_dir", "read_file", "bash_exec", "load_skill"],
    "code_review": ["list_dir", "read_file", "bash_exec", "load_skill"],
    "data_analysis": ["list_dir", "read_file", "write_file", "bash_exec", "load_skill"],
    "documentation": ["list_dir", "read_file", "write_file", "edit_file", "load_skill"],
    "general": ["list_dir", "read_file", "web_search", "web_fetch", "web_map", "load_skill"],
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

    async def _invoke_worker_resilient(
        model_to_call: Any,
        base_obj: Any,
        prompt_msgs: list[BaseMessage],
        *,
        bind_permitted_tools: bool = True,
    ) -> AIMessage:
        worker_label = f"sub-agent ({normalized_role})"
        try:
            res = await invoke_with_rate_limit_retry(
                lambda: model_to_call.ainvoke(prompt_msgs),
                label=worker_label,
            )
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            err_lower = str(exc).lower()
            extra_body = getattr(base_obj, "extra_body", None)
            if (
                isinstance(extra_body, dict)
                and "reasoning_effort" in extra_body
                and ("reasoning" in err_lower or "400" in err_lower or "invalid_request" in err_lower)
                and hasattr(base_obj, "model_copy")
            ):
                stripped = {k: v for k, v in extra_body.items() if k != "reasoning_effort"}
                fb = base_obj.model_copy(update={"extra_body": stripped or None})
                fb_bound = (
                    fb.bind_tools(list(permitted_tools.values()))
                    if bind_permitted_tools and permitted_tools and hasattr(fb, "bind_tools")
                    else fb
                )
                res = await invoke_with_rate_limit_retry(
                    lambda: fb_bound.ainvoke(prompt_msgs),
                    label=worker_label,
                )
            else:
                raise
        return res if isinstance(res, AIMessage) else AIMessage(content=str(getattr(res, "content", "") or ""))

    def _bump_worker_model(
        cur_base: Any,
        *,
        bind_permitted_tools: bool,
        attempt: int,
    ) -> tuple[Any, Any]:
        updates: dict[str, Any] = {}
        cur_max = getattr(cur_base, "max_tokens", None)
        if isinstance(cur_max, int) and cur_max < 16384:
            updates["max_tokens"] = min(max(cur_max * 2, 4096), 16384)
        extra_body = getattr(cur_base, "extra_body", None)
        if isinstance(extra_body, dict) and "reasoning_effort" in extra_body:
            new_extra = dict(extra_body)
            if attempt == 1 and str(new_extra.get("reasoning_effort", "")).lower() not in ("low", "minimal", "none"):
                new_extra["reasoning_effort"] = "low"
            else:
                new_extra.pop("reasoning_effort", None)
            updates["extra_body"] = new_extra or None
        if not updates or not hasattr(cur_base, "model_copy"):
            target = (
                cur_base.bind_tools(list(permitted_tools.values()))
                if bind_permitted_tools and permitted_tools and hasattr(cur_base, "bind_tools")
                else cur_base
            )
            return target, cur_base
        try:
            copied = cur_base.model_copy(update=updates)
            target = (
                copied.bind_tools(list(permitted_tools.values()))
                if bind_permitted_tools and permitted_tools and hasattr(copied, "bind_tools")
                else copied
            )
            return target, copied
        except Exception:  # noqa: BLE001
            target = (
                cur_base.bind_tools(list(permitted_tools.values()))
                if bind_permitted_tools and permitted_tools and hasattr(cur_base, "bind_tools")
                else cur_base
            )
            return target, cur_base

    async def worker_reason(state: WorkerState) -> dict[str, Any]:
        next_turn = state["turn_count"] + 1
        is_final_turn = next_turn >= state["max_turns"]
        emit_progress(
            "subagent_reason",
            detail=f"Sub-agent '{normalized_role}' reasoning (worker turn {next_turn})",
        )
        prompt_msgs = list(state["messages"])
        bind_tools_now = not is_final_turn
        if is_final_turn and state["turn_count"] > 0:
            prompt_msgs = _build_worker_recovery_messages(
                prompt_msgs,
                "This is your final worker turn. Do not call any more tools; synthesize and return your final JSON report now based on all tool outputs above.",
            )
        active_model = bound_model if bind_tools_now else worker_model
        response = await _invoke_worker_resilient(
            active_model,
            worker_model,
            prompt_msgs,
            bind_permitted_tools=bind_tools_now,
        )
        content_text = _extract_worker_text(response)
        tcs = list(getattr(response, "tool_calls", None) or []) if bind_tools_now else []
        inv_tcs = list(getattr(response, "invalid_tool_calls", None) or [])
        finish_reason = _normalize_worker_finish_reason(
            (getattr(response, "response_metadata", None) or {}).get("finish_reason")
        )

        retry_base = worker_model
        for attempt in range(1, 3):
            if tcs or (content_text and not inv_tcs):
                break
            nudge = (
                "Your previous output truncated or produced invalid tool JSON. "
                "Keep reasoning concise and emit either a smaller valid tool call or your final JSON report now."
                if (finish_reason == "length" or inv_tcs)
                else "Your previous output was empty. Please call the next tool or return your final JSON report now."
            )
            if not bind_tools_now:
                nudge = (
                    "Your previous output was empty or truncated. "
                    "Keep reasoning minimal and immediately return your concise final JSON report now."
                )
            retry_model = active_model
            if finish_reason == "length":
                retry_model, retry_base = _bump_worker_model(
                    retry_base,
                    bind_permitted_tools=bind_tools_now,
                    attempt=attempt,
                )
            retry_msgs = _build_worker_recovery_messages(prompt_msgs, nudge)
            try:
                response = await _invoke_worker_resilient(
                    retry_model,
                    retry_base,
                    retry_msgs,
                    bind_permitted_tools=bind_tools_now,
                )
            except GraphInterrupt:
                raise
            except Exception:  # noqa: BLE001
                break
            content_text = _extract_worker_text(response)
            tcs = list(getattr(response, "tool_calls", None) or []) if bind_tools_now else []
            inv_tcs = list(getattr(response, "invalid_tool_calls", None) or [])
            finish_reason = _normalize_worker_finish_reason(
                (getattr(response, "response_metadata", None) or {}).get("finish_reason")
            )

        response = _sanitize_worker_ai_message(response)

        emit_progress(
            "subagent_reason_done",
            detail=f"Sub-agent '{normalized_role}' turn {next_turn} complete",
        )
        return {
            "messages": [response],
            "turn_count": next_turn,
            "final_output": content_text,
        }

    def _find_pending_worker_calls(
        msgs: list[BaseMessage],
    ) -> list[tuple[int, dict[str, Any]]]:
        last_ai_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], AIMessage):
                last_ai_idx = i
                break
        if last_ai_idx < 0:
            return []
        raw_tcs = list(getattr(msgs[last_ai_idx], "tool_calls", None) or [])
        if not raw_tcs:
            return []
        answered = {
            str(getattr(m, "tool_call_id", ""))
            for m in msgs[last_ai_idx + 1 :]
            if isinstance(m, ToolMessage)
        }
        return [
            (idx, tc)
            for idx, tc in enumerate(raw_tcs)
            if str(tc.get("id") or f"worker-tc-{idx}") not in answered
        ]

    async def worker_act(state: WorkerState) -> dict[str, Any]:
        pending = _find_pending_worker_calls(list(state["messages"]))
        if not pending:
            return {"tools_used": list(state["tools_used"])}

        tool_messages: list[ToolMessage] = []
        used = list(state["tools_used"])

        # Execute one pending tool call per checkpointed step so HITL resume never replays
        # already-completed worker tools.
        idx, tc = pending[0]
        tc_name = str(tc.get("name", "")).strip()
        tc_args = normalize_tool_arguments(tc_name, tc.get("args"))
        tc_id = str(tc.get("id") or f"worker-tc-{idx}")

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
            return {"messages": tool_messages, "tools_used": used}

        tool_obj = permitted_tools[tc_name]
        emit_progress(
            "subagent_tool_start",
            tool=tc_name,
            detail=f"Sub-agent '{normalized_role}' running tool '{tc_name}'",
        )
        try:
            raw_result, _audit = await execute_tool_with_guardrails(
                tool_obj,
                tc_args,
                is_mcp_tool=(tc_name in mcp_names),
                source=f"subagent:{normalized_role}",
            )
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            raw_result = f"ERROR: Tool '{tc_name}' raised {type(exc).__name__}: {exc}"

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

    def route_after_worker_act(state: WorkerState) -> Literal["worker_act", "worker_reason"]:
        if _find_pending_worker_calls(list(state["messages"])):
            return "worker_act"
        return "worker_reason"

    builder: StateGraph[WorkerState] = StateGraph(WorkerState)
    builder.add_node("worker_reason", worker_reason)
    builder.add_node("worker_act", worker_act)
    builder.add_edge(START, "worker_reason")
    builder.add_conditional_edges(
        "worker_reason",
        route_worker,
        {"worker_act": "worker_act", "__end__": END},
    )
    builder.add_conditional_edges(
        "worker_act",
        route_after_worker_act,
        {"worker_act": "worker_act", "worker_reason": "worker_reason"},
    )
    worker_graph = builder.compile(checkpointer=_WORKER_CHECKPOINTER)

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

    parent_session = current_session_id.get()
    key_material = json.dumps(
        {
            "session": parent_session,
            "role": normalized_role,
            "task": task.strip(),
            "context": context.strip(),
            "tools": sorted(permitted_tools.keys()),
        },
        sort_keys=True,
    )
    sub_thread_id = f"sub-{hashlib.sha256(key_material.encode('utf-8')).hexdigest()[:24]}"
    worker_config: dict[str, Any] = {"configurable": {"thread_id": sub_thread_id}}

    def _extract_pending_interrupt_value(snapshot: Any) -> tuple[bool, Any]:
        for task_info in getattr(snapshot, "tasks", ()) or ():
            interrupts = getattr(task_info, "interrupts", ()) or ()
            if interrupts:
                return True, getattr(interrupts[0], "value", interrupts[0])
        return False, None

    def _clear_worker_checkpoint() -> None:
        try:
            delete_fn = getattr(_WORKER_CHECKPOINTER, "delete_thread", None)
            if callable(delete_fn):
                delete_fn(sub_thread_id)
            else:
                storage = getattr(_WORKER_CHECKPOINTER, "storage", None)
                if isinstance(storage, dict):
                    storage.pop(sub_thread_id, None)
        except Exception:  # noqa: BLE001
            pass

    subagent_label = f"Sub-agent ({normalized_role})"
    try:
        state_snap = await worker_graph.aget_state(worker_config)
        has_pending, pending_payload = _extract_pending_interrupt_value(state_snap)
        if has_pending:
            # Parent graph is resuming after HITL approval: retrieve decision and resume
            # worker_graph from its saved checkpoint instead of restarting at turn 0.
            decision = interrupt(pending_payload)
            with track_active_task("subagent", subagent_label, task_id=sub_thread_id):
                final_state = await worker_graph.ainvoke(
                    Command(resume=decision), config=worker_config
                )
        else:
            with track_active_task("subagent", subagent_label, task_id=sub_thread_id):
                final_state = await worker_graph.ainvoke(
                    initial_worker_state, config=worker_config
                )

        while isinstance(final_state, dict) and final_state.get("__interrupt__"):
            intr_list = final_state["__interrupt__"]
            first_intr = intr_list[0] if isinstance(intr_list, (list, tuple)) and intr_list else intr_list
            payload = getattr(first_intr, "value", first_intr)
            decision = interrupt(payload)
            with track_active_task("subagent", subagent_label, task_id=sub_thread_id):
                final_state = await worker_graph.ainvoke(
                    Command(resume=decision), config=worker_config
                )

        _clear_worker_checkpoint()
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
        _clear_worker_checkpoint()
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
