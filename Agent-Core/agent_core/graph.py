"""Pure-Python LangGraph Supervisor Engine (Plan -> Act -> Observe -> Reflect)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aiosqlite
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent_core.guardrails import execute_tool_with_guardrails
from agent_core.jobs import ProgressCallback, current_progress_callback, emit_progress
from agent_core.models import (
    ModelOverride,
    ModelSpec,
    build_chat_model,
    get_primary_model,
    get_worker_model,
    load_model_spec,
    resolve_model_spec_with_override,
)
from agent_core.planner import (
    build_supervisor_system_prompt,
    get_next_pending_step,
    parse_structured_plan,
    update_plan_step,
)
from agent_core.sandbox import SandboxManager
from agent_core.skills import (
    DynamicSkill,
    SkillRegistry,
    build_loaded_skills_context,
    current_dynamic_skills,
    make_load_skill_tool,
    parse_dynamic_skills_payload,
)
from agent_core.state import (
    DEFAULT_MAX_TOOL_RETRIES,
    DEFAULT_MAX_TURNS,
    AgentState,
    PlanStep,
    compact_tool_output,
    make_initial_state,
)
from agent_core.subagents import make_spawn_subagent_tool
from agent_core.tools.local import build_local_tools, current_session_id
from MCPs.manager import MCPRegistryManager

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "./data/checkpoints.db")

current_primary_model: ContextVar[BaseChatModel | None] = ContextVar(
    "current_primary_model", default=None
)
current_worker_model: ContextVar[BaseChatModel | None] = ContextVar(
    "current_worker_model", default=None
)
current_extra_tools: ContextVar[dict[str, BaseTool] | None] = ContextVar(
    "current_extra_tools", default=None
)
current_extra_mcp_names: ContextVar[set[str] | None] = ContextVar(
    "current_extra_mcp_names", default=None
)


def _describe_model_instance(
    model_obj: BaseChatModel | None,
    spec: ModelSpec | None = None,
) -> dict[str, Any]:
    if spec is not None:
        return {
            "provider": spec.provider,
            "model": spec.model,
            "base_url": spec.base_url,
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
        }
    if model_obj is None:
        return {"provider": "unknown", "model": "unknown", "base_url": ""}
    model_name = (
        getattr(model_obj, "model_name", None)
        or getattr(model_obj, "model", None)
        or type(model_obj).__name__
    )
    base_url = getattr(model_obj, "openai_api_base", None) or getattr(model_obj, "base_url", None)
    return {
        "provider": "custom",
        "model": str(model_name),
        "base_url": str(base_url or ""),
    }


ModelOverrideResolver = Callable[[Literal["primary", "worker"], ModelSpec], BaseChatModel]


def _resolve_override_model(
    role: Literal["primary", "worker"],
    override: BaseChatModel | ModelSpec | ModelOverride | dict[str, Any] | None,
    *,
    resolver: ModelOverrideResolver | None = None,
) -> tuple[BaseChatModel | None, ModelSpec | None]:
    if override is None:
        return None, None
    if isinstance(override, BaseChatModel):
        return override, None
    factory = resolver or (lambda r, sp: build_chat_model(r, spec=sp, require_api_key=True))
    if isinstance(override, ModelSpec):
        return factory(role, override), override
    spec = resolve_model_spec_with_override(role, override=override)
    return factory(role, spec), spec


@dataclass
class AgentRuntime:
    """Container holding all wired components of the pure-Python autonomous agent."""

    graph: Any
    sandbox: SandboxManager
    skills: SkillRegistry
    mcp_manager: MCPRegistryManager
    tools_by_name: dict[str, BaseTool]
    mcp_tool_names: set[str]
    checkpoint_db_path: Path
    max_turns: int
    max_tool_retries: int
    default_primary_summary: dict[str, Any] | None = None
    default_worker_summary: dict[str, Any] | None = None
    model_override_resolver: ModelOverrideResolver | None = None

    def make_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(150, self.max_turns * 6 + 30),
        }

    async def run_task(
        self,
        task: str,
        *,
        thread_id: str,
        max_turns: int | None = None,
        primary_model_override: BaseChatModel
        | ModelSpec
        | ModelOverride
        | dict[str, Any]
        | None = None,
        worker_model_override: BaseChatModel
        | ModelSpec
        | ModelOverride
        | dict[str, Any]
        | None = None,
        extra_tools: dict[str, BaseTool] | None = None,
        extra_mcp_tool_names: set[str] | None = None,
        dynamic_skills: dict[str, DynamicSkill] | list[dict[str, Any]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not task or not task.strip():
            raise ValueError("Task must be a non-empty string.")

        prim_model, prim_spec = _resolve_override_model(
            "primary", primary_model_override, resolver=self.model_override_resolver
        )
        work_model, work_spec = _resolve_override_model(
            "worker", worker_model_override, resolver=self.model_override_resolver
        )

        active_models = {
            "primary": (
                _describe_model_instance(prim_model, prim_spec)
                if prim_model is not None
                else (self.default_primary_summary or load_model_spec("primary").safe_summary())
            ),
            "worker": (
                _describe_model_instance(work_model, work_spec)
                if work_model is not None
                else (self.default_worker_summary or load_model_spec("worker").safe_summary())
            ),
        }

        parsed_skills: dict[str, DynamicSkill] | None = None
        if isinstance(dynamic_skills, dict):
            parsed_skills = dynamic_skills
        elif isinstance(dynamic_skills, list):
            parsed_skills = parse_dynamic_skills_payload(dynamic_skills)

        token = current_session_id.set(thread_id)
        prim_tok = current_primary_model.set(prim_model)
        work_tok = current_worker_model.set(work_model)
        tools_tok = current_extra_tools.set(extra_tools or None)
        mcp_tok = current_extra_mcp_names.set(extra_mcp_tool_names or None)
        skills_tok = current_dynamic_skills.set(parsed_skills or None)
        prog_tok = current_progress_callback.set(progress_callback)
        try:
            config = self.make_config(thread_id)
            existing = await self.graph.aget_state(config)
            has_history = bool(existing and existing.values and existing.values.get("messages"))

            if has_history:
                input_payload: dict[str, Any] = {
                    "messages": [HumanMessage(content=task.strip())],
                    "turn_count": 0,
                    "max_turns": max_turns if max_turns is not None else self.max_turns,
                    "next_action": "continue",
                    "final_response": "",
                    "error": None,
                    "thread_id": thread_id,
                }
            else:
                input_payload = dict(
                    make_initial_state(
                        [HumanMessage(content=task.strip())],
                        thread_id=thread_id,
                        max_turns=max_turns if max_turns is not None else self.max_turns,
                    )
                )

            await self.graph.ainvoke(input_payload, config=config)
            result = await self.inspect_thread(thread_id)
            result["active_models"] = active_models
            return result
        finally:
            current_progress_callback.reset(prog_tok)
            current_dynamic_skills.reset(skills_tok)
            current_extra_mcp_names.reset(mcp_tok)
            current_extra_tools.reset(tools_tok)
            current_worker_model.reset(work_tok)
            current_primary_model.reset(prim_tok)
            current_session_id.reset(token)

    async def resume_approval(
        self,
        *,
        thread_id: str,
        decision: str | dict[str, Any],
        primary_model_override: BaseChatModel
        | ModelSpec
        | ModelOverride
        | dict[str, Any]
        | None = None,
        worker_model_override: BaseChatModel
        | ModelSpec
        | ModelOverride
        | dict[str, Any]
        | None = None,
        extra_tools: dict[str, BaseTool] | None = None,
        extra_mcp_tool_names: set[str] | None = None,
        dynamic_skills: dict[str, DynamicSkill] | list[dict[str, Any]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        prim_model, prim_spec = _resolve_override_model(
            "primary", primary_model_override, resolver=self.model_override_resolver
        )
        work_model, work_spec = _resolve_override_model(
            "worker", worker_model_override, resolver=self.model_override_resolver
        )

        active_models = {
            "primary": (
                _describe_model_instance(prim_model, prim_spec)
                if prim_model is not None
                else (self.default_primary_summary or load_model_spec("primary").safe_summary())
            ),
            "worker": (
                _describe_model_instance(work_model, work_spec)
                if work_model is not None
                else (self.default_worker_summary or load_model_spec("worker").safe_summary())
            ),
        }

        parsed_skills: dict[str, DynamicSkill] | None = None
        if isinstance(dynamic_skills, dict):
            parsed_skills = dynamic_skills
        elif isinstance(dynamic_skills, list):
            parsed_skills = parse_dynamic_skills_payload(dynamic_skills)

        token = current_session_id.set(thread_id)
        prim_tok = current_primary_model.set(prim_model)
        work_tok = current_worker_model.set(work_model)
        tools_tok = current_extra_tools.set(extra_tools or None)
        mcp_tok = current_extra_mcp_names.set(extra_mcp_tool_names or None)
        skills_tok = current_dynamic_skills.set(parsed_skills or None)
        prog_tok = current_progress_callback.set(progress_callback)
        try:
            config = self.make_config(thread_id)
            await self.graph.ainvoke(Command(resume=decision), config=config)
            result = await self.inspect_thread(thread_id)
            result["active_models"] = active_models
            return result
        finally:
            current_progress_callback.reset(prog_tok)
            current_dynamic_skills.reset(skills_tok)
            current_extra_mcp_names.reset(mcp_tok)
            current_extra_tools.reset(tools_tok)
            current_worker_model.reset(work_tok)
            current_primary_model.reset(prim_tok)
            current_session_id.reset(token)

    async def inspect_thread(self, thread_id: str) -> dict[str, Any]:
        config = self.make_config(thread_id)
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values if snapshot and snapshot.values else {}

        interrupts: list[dict[str, Any]] = []
        if snapshot and getattr(snapshot, "tasks", None):
            for task_obj in snapshot.tasks:
                for intr in getattr(task_obj, "interrupts", ()) or ():
                    val = getattr(intr, "value", intr)
                    if isinstance(val, dict):
                        interrupts.append(val)
                    else:
                        interrupts.append({"value": val})

        status = "awaiting_approval" if interrupts else "completed"
        if values.get("error") and not interrupts:
            status = "error"

        return {
            "thread_id": thread_id,
            "status": status,
            "final_response": values.get("final_response", ""),
            "plan": values.get("plan", []),
            "active_skills": values.get("active_skills", []),
            "turn_count": values.get("turn_count", 0),
            "pending_approval": interrupts[0] if interrupts else None,
            "interrupts": interrupts,
            "subagent_reports": values.get("subagent_reports", []),
            "error": values.get("error"),
        }


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return "Complete the user task."


def _latest_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def build_supervisor_graph(
    *,
    primary_model: BaseChatModel,
    worker_model: BaseChatModel | None = None,
    sandbox: SandboxManager | None = None,
    skills: SkillRegistry | None = None,
    mcp_tools: list[BaseTool] | None = None,
    extra_tools: list[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES,
    compaction_threshold: int | None = None,
    allow_private_web_fetch: bool = False,
) -> tuple[Any, dict[str, BaseTool], set[str]]:
    resolved_sandbox = sandbox or SandboxManager()
    resolved_skills = skills or SkillRegistry()
    resolved_worker_model = worker_model or primary_model

    local_tools = build_local_tools(
        resolved_sandbox,
        allow_private_web_fetch=allow_private_web_fetch,
    )
    load_skill_tool = make_load_skill_tool(resolved_skills)

    mcp_list = list(mcp_tools or [])
    mcp_names: set[str] = {t.name for t in mcp_list}

    base_tools: dict[str, BaseTool] = {}
    for t in [*local_tools, load_skill_tool, *mcp_list, *(extra_tools or [])]:
        base_tools[t.name] = t

    def _active_base_tools() -> dict[str, BaseTool]:
        dyn = current_extra_tools.get()
        if not dyn:
            return base_tools
        return {**base_tools, **dyn}

    def _active_mcp_names() -> set[str]:
        dyn_mcp = current_extra_mcp_names.get()
        if not dyn_mcp:
            return mcp_names
        return mcp_names | dyn_mcp

    spawn_tool = make_spawn_subagent_tool(
        worker_model_factory=lambda: current_worker_model.get() or resolved_worker_model,
        available_tools=_active_base_tools,
        mcp_tool_names=_active_mcp_names,
        workspace_dir=str(resolved_sandbox.config.workspace_dir),
    )

    tools_by_name: dict[str, BaseTool] = {**base_tools, spawn_tool.name: spawn_tool}
    all_tool_list = list(tools_by_name.values())
    default_bound_model: Any = (
        primary_model.bind_tools(all_tool_list)
        if hasattr(primary_model, "bind_tools")
        else primary_model
    )

    def _active_tools_map() -> dict[str, BaseTool]:
        dyn = current_extra_tools.get()
        if not dyn:
            return tools_by_name
        return {**tools_by_name, **dyn}

    async def plan_node(state: AgentState) -> dict[str, Any]:
        turn_count = int(state.get("turn_count", 0)) + 1
        effective_max = int(state.get("max_turns", max_turns))
        messages = list(state.get("messages", []))
        current_plan: list[PlanStep] = list(state.get("plan", []))
        active_skills: list[str] = list(state.get("active_skills", []))
        last_observations = list(state.get("last_observations", []))

        completed_count = sum(1 for s in current_plan if s["status"] == "completed")
        emit_progress(
            "plan_start",
            turn_count=turn_count,
            steps_completed=completed_count,
            detail=f"Planning turn {turn_count}/{effective_max}",
        )

        if turn_count > effective_max:
            return {
                "turn_count": turn_count,
                "next_action": "complete",
                "reflection": f"Reached MAX_TURNS ({effective_max}); stopping loop safely.",
            }

        user_goal = _latest_human_text(messages)
        needs_new_plan = not current_plan or all(
            s["status"] in ("completed", "skipped") for s in current_plan
        )
        had_failure = any(obs.get("status") in ("error", "rejected") for obs in last_observations)

        updated_plan = list(current_plan)
        if needs_new_plan:
            step_num = len(updated_plan) + 1
            step_desc = (
                user_goal[:240]
                if not current_plan or turn_count == 1
                else "Continue task based on observed tool results or synthesize final answer."
            )
            step_action = (
                "supervisor_decide"
                if not current_plan or turn_count == 1
                else "supervisor_continue_or_synthesize"
            )
            updated_plan.append(
                PlanStep(
                    step_id=f"step-{step_num}",
                    description=step_desc,
                    status="in_progress",
                    tool_or_action=step_action,
                    result="",
                )
            )
            current_step_id = f"step-{step_num}"
        else:
            if had_failure:
                step_num = len(updated_plan) + 1
                updated_plan.append(
                    PlanStep(
                        step_id=f"step-{step_num}",
                        description="Recover from previous tool failure/rejection and adjust plan.",
                        status="in_progress",
                        tool_or_action="replan_recover",
                        result="",
                    )
                )
                current_step_id = f"step-{step_num}"
            else:
                nxt = get_next_pending_step(updated_plan)
                if nxt is not None:
                    current_step_id = nxt["step_id"]
                    updated_plan = update_plan_step(
                        updated_plan, current_step_id, status="in_progress"
                    )
                else:
                    current_step_id = updated_plan[-1]["step_id"]

        active_tools = _active_tools_map()
        active_tool_list = list(active_tools.values())
        dynamic_skill_catalog = resolved_skills.format_catalog_for_prompt()
        loaded_skills_text = build_loaded_skills_context(active_skills, resolved_skills)
        sys_prompt = build_supervisor_system_prompt(
            skill_catalog_text=dynamic_skill_catalog,
            loaded_skills_text=loaded_skills_text,
            available_tool_names=list(active_tools.keys()),
            plan=updated_plan,
            turn_count=turn_count,
            max_turns=effective_max,
        )

        override_primary = current_primary_model.get()
        has_dynamic_tools = bool(current_extra_tools.get())
        if override_primary is not None:
            active_bound_model: Any = (
                override_primary.bind_tools(active_tool_list)
                if hasattr(override_primary, "bind_tools")
                else override_primary
            )
        elif has_dynamic_tools and hasattr(primary_model, "bind_tools"):
            active_bound_model = primary_model.bind_tools(active_tool_list)
        else:
            active_bound_model = default_bound_model
        try:
            ai_reply = await active_bound_model.ainvoke(
                [SystemMessage(content=sys_prompt), *messages]
            )
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            err_msg = f"Primary model error during planning: {exc}"
            logger.error(err_msg)
            return {
                "turn_count": turn_count,
                "plan": update_plan_step(
                    updated_plan, current_step_id, status="failed", result=err_msg
                ),
                "current_step_id": current_step_id,
                "next_action": "complete",
                "error": err_msg,
                "final_response": f"Agent stopped due to model error: {err_msg}",
            }

        if not isinstance(ai_reply, AIMessage):
            ai_reply = AIMessage(content=str(getattr(ai_reply, "content", "") or ""))

        tool_calls = getattr(ai_reply, "tool_calls", None) or []
        emit_progress(
            "plan_llm_done",
            turn_count=turn_count,
            detail=(
                f"Planned {len(tool_calls)} tool call(s)"
                if tool_calls
                else "Produced direct response"
            ),
        )

        raw_content = str(ai_reply.content or "")
        if '"step_id"' in raw_content and ('"plan"' in raw_content or '"steps"' in raw_content):
            parsed_steps, _ = parse_structured_plan(
                raw_content,
                existing_plan=current_plan,
                fallback_goal=user_goal,
            )
            if parsed_steps:
                updated_plan = parsed_steps
                nxt = get_next_pending_step(updated_plan)
                if nxt:
                    current_step_id = nxt["step_id"]
                    updated_plan = update_plan_step(
                        updated_plan, current_step_id, status="in_progress"
                    )

        if tool_calls:
            planned_tools = ", ".join(str(tc.get("name", "tool")) for tc in tool_calls)
            updated_plan = [
                (
                    PlanStep(
                        step_id=s["step_id"],
                        description=s["description"],
                        status=s["status"],
                        tool_or_action=planned_tools,
                        result=s["result"],
                    )
                    if s["step_id"] == current_step_id
                    else s
                )
                for s in updated_plan
            ]

        return {
            "messages": [ai_reply],
            "plan": updated_plan,
            "current_step_id": current_step_id,
            "turn_count": turn_count,
            "pending_approval": None,
            "error": None,
        }

    async def act_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        last_ai = _latest_ai_message(messages)
        tool_calls = getattr(last_ai, "tool_calls", None) or [] if last_ai else []

        if not tool_calls:
            return {"last_observations": [], "pending_approval": None}

        active_tools = _active_tools_map()
        active_mcp_names = _active_mcp_names()
        retry_counts = dict(state.get("tool_retry_counts", {}))
        active_skills = list(state.get("active_skills", []))
        subagent_reports = list(state.get("subagent_reports", []))
        thread_id = str(state.get("thread_id") or "default")
        ws_dir = resolved_sandbox.get_session_workspace(thread_id)

        tool_messages: list[ToolMessage] = []
        observations: list[dict[str, Any]] = []

        for idx, tc in enumerate(tool_calls):
            tc_name = str(tc.get("name", "")).strip()
            tc_args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
            tc_id = str(tc.get("id") or f"call-{idx}")

            retry_key = f"{tc_name}:{json.dumps(tc_args, sort_keys=True, default=str)[:120]}"
            prior_failures = retry_counts.get(retry_key, 0)
            if prior_failures >= max_tool_retries:
                err_text = (
                    f"RETRY LIMIT EXCEEDED: Tool '{tc_name}' already failed "
                    f"{prior_failures} times with the same arguments (limit={max_tool_retries}). "
                    "Choose a different approach or inform the user."
                )
                tool_messages.append(
                    ToolMessage(content=err_text, tool_call_id=tc_id, name=tc_name)
                )
                observations.append(
                    {"tool": tc_name, "status": "error", "summary": err_text, "archived_log": None}
                )
                continue

            tool_obj = active_tools.get(tc_name)
            if tool_obj is None:
                err_text = (
                    f"ERROR: Unknown tool '{tc_name}'. Available tools: "
                    f"{sorted(active_tools.keys())}"
                )
                retry_counts[retry_key] = prior_failures + 1
                tool_messages.append(
                    ToolMessage(content=err_text, tool_call_id=tc_id, name=tc_name)
                )
                observations.append(
                    {"tool": tc_name, "status": "error", "summary": err_text, "archived_log": None}
                )
                continue

            emit_progress(
                "act_tool_start",
                tool=tc_name,
                detail=f"Running tool '{tc_name}' ({idx + 1}/{len(tool_calls)})",
            )
            try:
                raw_output, audit = await execute_tool_with_guardrails(
                    tool_obj,
                    tc_args,
                    is_mcp_tool=(tc_name in active_mcp_names),
                    source="supervisor",
                )
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                err_text = f"ERROR: Tool '{tc_name}' raised {type(exc).__name__}: {exc}"
                retry_counts[retry_key] = prior_failures + 1
                tool_messages.append(
                    ToolMessage(content=err_text, tool_call_id=tc_id, name=tc_name)
                )
                observations.append(
                    {"tool": tc_name, "status": "error", "summary": err_text, "archived_log": None}
                )
                continue

            emit_progress(
                "act_tool_done",
                tool=tc_name,
                detail=f"Finished tool '{tc_name}'",
            )

            if not audit.get("approved", True):
                obs_status = "rejected"
            elif raw_output.startswith("ERROR:"):
                obs_status = "error"
                retry_counts[retry_key] = prior_failures + 1
            else:
                obs_status = "ok"
                retry_counts.pop(retry_key, None)

            if tc_name == "load_skill" and obs_status == "ok":
                skill_name = str(tc_args.get("name", "")).strip().lower()
                if skill_name and skill_name not in active_skills:
                    active_skills.append(skill_name)

            if tc_name == "spawn_subagent" and obs_status == "ok":
                try:
                    parsed_report = json.loads(raw_output)
                    if isinstance(parsed_report, dict):
                        subagent_reports.append(parsed_report)
                except json.JSONDecodeError:
                    subagent_reports.append({"raw": raw_output[:400]})

            compacted_text, archived_log = compact_tool_output(
                tc_name,
                raw_output,
                workspace_dir=ws_dir,
                threshold=compaction_threshold,
            )

            tool_messages.append(
                ToolMessage(content=compacted_text, tool_call_id=tc_id, name=tc_name)
            )
            observations.append(
                {
                    "tool": tc_name,
                    "status": obs_status,
                    "summary": compacted_text[:300],
                    "archived_log": archived_log,
                    "audit": audit,
                }
            )

        return {
            "messages": tool_messages,
            "active_skills": active_skills,
            "subagent_reports": subagent_reports,
            "tool_retry_counts": retry_counts,
            "last_observations": observations,
            "pending_approval": None,
        }

    async def observe_node(state: AgentState) -> dict[str, Any]:
        plan = list(state.get("plan", []))
        current_step_id = state.get("current_step_id")
        observations = list(state.get("last_observations", []))
        last_ai = _latest_ai_message(list(state.get("messages", [])))

        if current_step_id and plan:
            if observations:
                has_err = any(o.get("status") == "error" for o in observations)
                has_reject = any(o.get("status") == "rejected" for o in observations)
                summary = "; ".join(
                    f"{o['tool']}={o['status']}: {str(o.get('summary', ''))[:120]}"
                    for o in observations
                )
                step_status: Literal["completed", "failed", "skipped"] = (
                    "failed" if has_err else "skipped" if has_reject else "completed"
                )
                plan = update_plan_step(plan, current_step_id, status=step_status, result=summary)
            elif last_ai and not getattr(last_ai, "tool_calls", None):
                final_snippet = str(last_ai.content or "")[:160]
                plan = update_plan_step(
                    plan, current_step_id, status="completed", result=final_snippet
                )

        completed_steps = sum(1 for s in plan if s["status"] == "completed")
        emit_progress(
            "observe",
            steps_completed=completed_steps,
            detail=f"Observed {len(observations)} tool result(s); completed steps={completed_steps}",
        )
        return {"plan": plan}

    async def reflect_node(state: AgentState) -> dict[str, Any]:
        turn_count = int(state.get("turn_count", 0))
        effective_max = int(state.get("max_turns", max_turns))
        observations = list(state.get("last_observations", []))
        last_ai = _latest_ai_message(list(state.get("messages", [])))
        emit_progress(
            "reflect", turn_count=turn_count, detail=f"Reflecting after turn {turn_count}"
        )

        if state.get("error"):
            return {
                "next_action": "complete",
                "reflection": f"Terminating due to recorded error: {state['error']}",
            }

        if turn_count >= effective_max:
            return {
                "next_action": "complete",
                "reflection": f"Reached MAX_TURNS ({effective_max}); producing final summary.",
            }

        if observations:
            statuses = [o.get("status") for o in observations]
            if "error" in statuses:
                note = "Tool error observed; returning to PLAN to recover or re-plan."
            elif "rejected" in statuses:
                note = "Sensitive action rejected by human; returning to PLAN to acknowledge or adjust."
            else:
                note = "Tool results observed; returning to PLAN for next step or final synthesis."
            return {"next_action": "continue", "reflection": note}

        if last_ai and str(last_ai.content or "").strip():
            return {
                "next_action": "complete",
                "reflection": "Supervisor produced a complete final response.",
            }

        return {
            "next_action": "complete",
            "reflection": "Model returned empty content; finalizing cleanly.",
            "final_response": "No additional output was returned by the model.",
        }

    async def response_node(state: AgentState) -> dict[str, Any]:
        if state.get("final_response"):
            return {"final_response": state["final_response"]}

        messages = list(state.get("messages", []))
        last_ai = _latest_ai_message(messages)
        if last_ai and not getattr(last_ai, "tool_calls", None):
            text = str(last_ai.content or "").strip()
            if text:
                return {"final_response": text}

        turn_count = int(state.get("turn_count", 0))
        effective_max = int(state.get("max_turns", max_turns))
        plan = list(state.get("plan", []))
        completed_steps = [s for s in plan if s["status"] == "completed"]
        summary = (
            f"Stopped after reaching turn limit ({turn_count}/{effective_max}). "
            f"Completed {len(completed_steps)}/{len(plan)} planned steps."
        )
        return {
            "final_response": summary,
            "messages": [AIMessage(content=summary)],
        }

    def route_after_reflect(state: AgentState) -> Literal["plan", "response"]:
        if state.get("next_action") == "continue":
            return "plan"
        return "response"

    builder: StateGraph[AgentState] = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("act", act_node)
    builder.add_node("observe", observe_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("response", response_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "act")
    builder.add_edge("act", "observe")
    builder.add_edge("observe", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"plan": "plan", "response": "response"},
    )
    builder.add_edge("response", END)

    compiled = builder.compile(checkpointer=checkpointer)
    return compiled, tools_by_name, mcp_names


@asynccontextmanager
async def create_agent_runtime(
    *,
    primary_model: BaseChatModel | None = None,
    worker_model: BaseChatModel | None = None,
    checkpoint_db_path: Path | str | None = None,
    sandbox: SandboxManager | None = None,
    skills: SkillRegistry | None = None,
    mcp_manager: MCPRegistryManager | None = None,
    connect_mcp: bool = True,
    extra_tools: list[BaseTool] | None = None,
    max_turns: int | None = None,
    max_tool_retries: int | None = None,
    compaction_threshold: int | None = None,
    allow_private_web_fetch: bool = False,
    model_override_resolver: ModelOverrideResolver | None = None,
) -> AsyncIterator[AgentRuntime]:
    def _prepare_db_path() -> Path:
        resolved = Path(
            checkpoint_db_path if checkpoint_db_path is not None else DEFAULT_CHECKPOINT_DB_PATH
        ).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    db_path = await asyncio.to_thread(_prepare_db_path)

    resolved_sandbox = sandbox or SandboxManager()
    resolved_skills = skills or SkillRegistry()
    resolved_mcp = mcp_manager or MCPRegistryManager()

    mcp_tools: list[BaseTool] = []
    if connect_mcp:
        mcp_tools = await resolved_mcp.connect()

    resolved_primary = primary_model or get_primary_model()
    resolved_worker = worker_model or get_worker_model()

    effective_max_turns = DEFAULT_MAX_TURNS if max_turns is None else max_turns
    effective_max_retries = (
        DEFAULT_MAX_TOOL_RETRIES if max_tool_retries is None else max_tool_retries
    )

    conn = await aiosqlite.connect(str(db_path))
    try:
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        await conn.commit()

        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

        compiled, tools_by_name, mcp_names = build_supervisor_graph(
            primary_model=resolved_primary,
            worker_model=resolved_worker,
            sandbox=resolved_sandbox,
            skills=resolved_skills,
            mcp_tools=mcp_tools,
            extra_tools=extra_tools,
            checkpointer=checkpointer,
            max_turns=effective_max_turns,
            max_tool_retries=effective_max_retries,
            compaction_threshold=compaction_threshold,
            allow_private_web_fetch=allow_private_web_fetch,
        )

        default_prim_meta = (
            load_model_spec("primary", include_persisted=True).safe_summary()
            if primary_model is None
            else _describe_model_instance(resolved_primary)
        )
        default_work_meta = (
            load_model_spec("worker", include_persisted=True).safe_summary()
            if worker_model is None
            else _describe_model_instance(resolved_worker)
        )

        yield AgentRuntime(
            graph=compiled,
            sandbox=resolved_sandbox,
            skills=resolved_skills,
            mcp_manager=resolved_mcp,
            tools_by_name=tools_by_name,
            mcp_tool_names=mcp_names,
            checkpoint_db_path=db_path,
            max_turns=effective_max_turns,
            max_tool_retries=effective_max_retries,
            default_primary_summary=default_prim_meta,
            default_worker_summary=default_work_meta,
            model_override_resolver=model_override_resolver,
        )
    finally:
        await conn.close()
