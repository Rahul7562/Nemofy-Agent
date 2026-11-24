"""Pure-Python LangGraph Supervisor Engine (Plan -> Act -> Observe -> Reflect)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import aiosqlite
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent_core.config import AgentConfig, invoke_agent_hook
from agent_core.context_manager import (
    AUTO_COMPACT_THRESHOLD_PERCENT,
    build_compacted_memory_block,
    compact_conversation_history,
    compute_context_usage,
    extract_provider_input_tokens,
    should_auto_compact,
    utc_now_iso,
)
from agent_core.guardrails import (
    bind_run_access_mode,
    current_access_mode,
    current_guardrail_policy,
    execute_tool_with_guardrails,
    normalize_tool_arguments,
)
from agent_core.jobs import (
    ProgressCallback,
    current_progress_callback,
    emit_progress,
    track_active_task,
)
from agent_core.message_utils import (
    BASH_EXIT_CODE_RE as _BASH_EXIT_CODE_RE,  # noqa: F401
)
from agent_core.message_utils import (
    KNOWN_FINISH_REASONS as _KNOWN_FINISH_REASONS,  # noqa: F401
)
from agent_core.message_utils import (
    THINK_TAG_RE as _THINK_TAG_RE,  # noqa: F401
)
from agent_core.message_utils import (
    UNCLOSED_THINK_TAG_RE as _UNCLOSED_THINK_TAG_RE,  # noqa: F401
)
from agent_core.message_utils import (
    build_empty_output_diagnostic as _build_empty_output_diagnostic,
)
from agent_core.message_utils import (
    build_recovery_messages as _build_recovery_messages,
)
from agent_core.message_utils import (
    extract_chunk_thinking_delta as _extract_chunk_thinking_delta,
)
from agent_core.message_utils import (
    extract_reasoning_from_extra as _extract_reasoning_from_extra,  # noqa: F401
)
from agent_core.message_utils import (
    extract_text_and_thinking as _extract_text_and_thinking,
)
from agent_core.message_utils import (
    get_finish_reason as _get_finish_reason,
)
from agent_core.message_utils import (
    is_tool_output_error as _is_tool_output_error,
)
from agent_core.message_utils import (
    latest_ai_message as _latest_ai_message,
)
from agent_core.message_utils import (
    latest_human_text as _latest_human_text,
)
from agent_core.message_utils import (
    normalize_finish_reason as _normalize_finish_reason,
)
from agent_core.message_utils import (
    sanitize_ai_message_for_history as _sanitize_ai_message_for_history,
)
from agent_core.message_utils import (
    sanitize_messages_for_llm as _sanitize_messages_for_llm,
)
from agent_core.message_utils import (
    summarize_tool_args as _summarize_tool_args,
)
from agent_core.models import (
    ModelConfigurationError,
    ModelOverride,
    ModelSpec,
    build_chat_model,
    classify_model_error_category,
    get_primary_model,
    get_worker_model,
    invoke_with_resilient_retry,
    is_timeout_model_error,
    load_model_spec,
    resolve_accessory_model_spec,
    resolve_model_spec_with_override,
)
from agent_core.multimodal import (
    MediaAttachment,
    MediaModality,
    build_multimodal_content_blocks,
    detect_model_input_modalities,
    format_accessory_reports_for_primary,
    is_unsupported_modality_error,
    partition_attachments_by_capability,
    run_accessory_media_subagent,
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
    AgentState,
    PlanStep,
    compact_tool_output,
    make_initial_state,
)
from agent_core.subagents import make_spawn_subagent_tool
from agent_core.tools.local import build_local_tools, current_session_id
from agent_core.workspace_diff import (
    IGNORED_WORKSPACE_DIRS as _IGNORED_WORKSPACE_DIRS,  # noqa: F401
)
from agent_core.workspace_diff import (
    diff_workspace_snapshots as _diff_workspace_snapshots,
)
from agent_core.workspace_diff import (
    snapshot_workspace as _snapshot_workspace,
)

try:
    from MCPs.manager import MCPRegistryManager
except ImportError:

    class MCPRegistryManager:  # type: ignore[no-redef]
        """No-op fallback when the optional sibling MCPs package is not installed."""

        async def connect(self) -> list[BaseTool]:
            return []

        def health_status(self) -> dict[str, Any]:
            return {"configured": 0, "connected": 0, "servers": [], "tools": []}


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
current_run_activity: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_run_activity", default=None
)
current_native_media_attachments: ContextVar[list[MediaAttachment] | None] = ContextVar(
    "current_native_media_attachments", default=None
)
current_accessory_fallback_fn: ContextVar[Callable[..., Any] | None] = ContextVar(
    "current_accessory_fallback_fn", default=None
)
current_uploads_dir: ContextVar[Path | None] = ContextVar(
    "current_uploads_dir", default=None
)
current_primary_meta: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_primary_meta", default=None
)
current_run_deadline: ContextVar[float | None] = ContextVar(
    "current_run_deadline", default=None
)


def _new_run_activity() -> dict[str, Any]:
    return {
        "thinking_segments": [],
        "tool_events": [],
        "file_changes": [],
        "accessory_reports": [],
    }


def _record_run_thinking(thinking: str | None) -> None:
    if not thinking or not thinking.strip():
        return
    clean = thinking.strip()
    activity = current_run_activity.get()
    if activity is not None:
        segments: list[str] = activity["thinking_segments"]
        if not segments or segments[-1] != clean:
            segments.append(clean)


def _record_run_tool_event(event: dict[str, Any]) -> None:
    activity = current_run_activity.get()
    if activity is None:
        return
    events: list[dict[str, Any]] = activity["tool_events"]
    ev_id = str(event.get("id") or f"tool-{len(events) + 1}")
    existing = next((t for t in events if t.get("id") == ev_id), None)
    if existing is not None:
        existing.update({k: v for k, v in event.items() if v is not None})
    else:
        events.append({"id": ev_id, **event})


def _record_run_file_changes(changes: list[dict[str, Any]]) -> None:
    if not changes:
        return
    activity = current_run_activity.get()
    if activity is not None:
        activity["file_changes"].extend(changes)


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
            "reasoning_effort": spec.reasoning_effort,
            "context_window": spec.context_window,
            "context_window_source": spec.context_window_source,
            "manual_context_window": spec.manual_context_window,
            "max_output_tokens": spec.max_output_tokens,
            "timeout_seconds": spec.timeout_seconds,
            "connect_timeout_seconds": spec.connect_timeout_seconds,
            "max_retries": spec.max_retries,
            "fallback_on_timeout": spec.fallback_on_timeout,
            "fallback_provider": spec.fallback_provider,
            "fallback_model": spec.fallback_model,
        }
    if model_obj is None:
        return {"provider": "unknown", "model": "unknown", "base_url": ""}
    model_name = (
        getattr(model_obj, "model_name", None)
        or getattr(model_obj, "model", None)
        or type(model_obj).__name__
    )
    base_url = getattr(model_obj, "openai_api_base", None) or getattr(model_obj, "base_url", None)
    explicit_ctx = getattr(model_obj, "context_window", None)
    explicit_out = getattr(model_obj, "max_output_tokens", None)
    cfg_max_tok = getattr(model_obj, "max_tokens", None)
    return {
        "provider": "custom",
        "model": str(model_name),
        "base_url": str(base_url or ""),
        **({"context_window": int(explicit_ctx)} if isinstance(explicit_ctx, int) and explicit_ctx >= 512 else {}),
        **({"max_output_tokens": int(explicit_out)} if isinstance(explicit_out, int) and explicit_out >= 256 else {}),
        **({"max_tokens": int(cfg_max_tok)} if isinstance(cfg_max_tok, int) and cfg_max_tok >= 64 else {}),
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
    config: AgentConfig = field(default_factory=AgentConfig)

    @property
    def tools(self) -> list[BaseTool]:
        """Return all registered tools as a list."""
        return list(self.tools_by_name.values())

    def make_config(self, thread_id: str, max_turns: int | None = None) -> dict[str, Any]:
        effective_turns = max_turns if max_turns is not None else self.max_turns
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(150, int(effective_turns) * 6 + 30),
        }

    async def run_task(
        self,
        task: str,
        *,
        thread_id: str,
        max_turns: int | None = None,
        access_mode: str | None = None,
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
        attachments: list[MediaAttachment | dict[str, Any]] | None = None,
        accessory_models: dict[
            str, BaseChatModel | ModelSpec | ModelOverride | dict[str, Any]
        ]
        | None = None,
        primary_input_modalities: list[str] | set[str] | None = None,
        allowed_uploads_dir: Path | None = None,
    ) -> dict[str, Any]:
        parsed_attachments: list[MediaAttachment] = []
        for raw_att in attachments or []:
            if isinstance(raw_att, MediaAttachment):
                parsed_attachments.append(raw_att)
            elif isinstance(raw_att, dict):
                parsed_attachments.append(MediaAttachment.model_validate(raw_att))

        clean_task = (task or "").strip()
        if not clean_task and parsed_attachments:
            kinds = sorted({a.kind for a in parsed_attachments})
            clean_task = f"Analyze the attached {', '.join(kinds)} file(s)."
        if not clean_task:
            raise ValueError("Task must be a non-empty string.")

        prim_model, prim_spec = _resolve_override_model(
            "primary", primary_model_override, resolver=self.model_override_resolver
        )
        try:
            work_model, work_spec = _resolve_override_model(
                "worker", worker_model_override, resolver=self.model_override_resolver
            )
        except ModelConfigurationError as exc:
            logger.warning(
                "Worker model override unavailable (%s); falling back to primary model.",
                exc,
            )
            work_model, work_spec = prim_model, prim_spec

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

        effective_access_mode = (
            access_mode if access_mode is not None else self.config.default_access_mode
        )
        run_deadline = (
            time.monotonic() + self.config.limits.max_wall_clock_seconds
            if self.config.limits.max_wall_clock_seconds is not None
            else None
        )
        run_activity = _new_run_activity()
        token = current_session_id.set(thread_id)
        prim_tok = current_primary_model.set(prim_model)
        work_tok = current_worker_model.set(work_model)
        tools_tok = current_extra_tools.set(extra_tools or None)
        mcp_tok = current_extra_mcp_names.set(extra_mcp_tool_names or None)
        skills_tok = current_dynamic_skills.set(parsed_skills or None)
        prog_tok = current_progress_callback.set(progress_callback)
        act_tok = current_run_activity.set(run_activity)
        uploads_tok = current_uploads_dir.set(allowed_uploads_dir)
        native_media_tok = current_native_media_attachments.set(None)
        fallback_fn_tok = current_accessory_fallback_fn.set(None)
        prim_meta_tok = current_primary_meta.set(dict(active_models["primary"]))
        access_mode_tok = bind_run_access_mode(effective_access_mode)
        policy_tok = current_guardrail_policy.set(self.config.guardrail_policy)
        deadline_tok = current_run_deadline.set(run_deadline)

        async def _run_accessory_for_modality(
            modality: MediaModality,
            mod_atts: list[MediaAttachment],
            user_prompt: str,
        ) -> dict[str, Any]:
            raw_acc = (accessory_models or {}).get(modality)
            acc_model_obj: BaseChatModel | None = None
            acc_provider = "openrouter"
            acc_model_name = "accessory"
            acc_enabled = True
            resolve_err: str | None = None

            if isinstance(raw_acc, BaseChatModel):
                acc_model_obj = raw_acc
                desc = _describe_model_instance(raw_acc)
                acc_provider = str(desc.get("provider") or "accessory")
                acc_model_name = str(desc.get("model") or "accessory")
            elif isinstance(raw_acc, ModelSpec):
                acc_provider = raw_acc.provider
                acc_model_name = raw_acc.model
                try:
                    acc_model_obj = (
                        self.model_override_resolver("worker", raw_acc)
                        if self.model_override_resolver
                        else build_chat_model("worker", spec=raw_acc, require_api_key=True)
                    )
                except Exception as exc:  # noqa: BLE001
                    resolve_err = str(exc)
            else:
                try:
                    acc_spec, acc_enabled = resolve_accessory_model_spec(
                        modality,
                        override=raw_acc if isinstance(raw_acc, (dict, ModelOverride)) else None,
                    )
                    acc_provider = acc_spec.provider
                    acc_model_name = acc_spec.model
                    if acc_enabled:
                        acc_model_obj = (
                            self.model_override_resolver("worker", acc_spec)
                            if self.model_override_resolver
                            else build_chat_model("worker", spec=acc_spec, require_api_key=True)
                        )
                except Exception as exc:  # noqa: BLE001
                    resolve_err = str(exc)

            ev_id = f"acc-{modality}-{len(run_activity['tool_events']) + 1}"
            file_list = ", ".join(a.name for a in mod_atts)
            tool_label = f"accessory_{modality}_subagent"
            start_event = {
                "id": ev_id,
                "tool": tool_label,
                "args_summary": f"{acc_provider}:{acc_model_name} · {file_list}"[:160],
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
            }
            _record_run_tool_event(start_event)
            emit_progress(
                "act_tool_start",
                tool=tool_label,
                tool_event=start_event,
                detail=(
                    f"Spawning {modality} Accessory sub-agent ({acc_model_name}) "
                    f"for {len(mod_atts)} file(s)..."
                ),
            )

            if not acc_enabled:
                report = {
                    "subagent_id": ev_id,
                    "role": tool_label,
                    "modality": modality,
                    "provider": acc_provider,
                    "model": acc_model_name,
                    "attachments": [a.name for a in mod_atts],
                    "status": "error",
                    "summary": f"{modality.title()} Accessory sub-agent is disabled in Settings > Accessories.",
                    "findings": (
                        f"The primary model does not natively accept {modality} inputs, and the "
                        f"{modality} Accessory model is currently disabled in Settings > Accessories."
                    ),
                    "error": "Accessory disabled",
                }
            elif acc_model_obj is None:
                report = {
                    "subagent_id": ev_id,
                    "role": tool_label,
                    "modality": modality,
                    "provider": acc_provider,
                    "model": acc_model_name,
                    "attachments": [a.name for a in mod_atts],
                    "status": "error",
                    "summary": f"Could not initialize {modality} Accessory model ({acc_provider}: {acc_model_name}).",
                    "findings": (
                        f"The primary model does not natively accept {modality} inputs, and the "
                        f"configured {modality} Accessory model ({acc_provider}: {acc_model_name}) "
                        f"could not be initialized: {resolve_err or 'missing API key'}. "
                        "Configure its API key or model under Settings > Accessories."
                    ),
                    "error": resolve_err or "Accessory unavailable",
                }
            else:
                with track_active_task(
                    "subagent",
                    f"{modality.title()} Accessory ({acc_model_name})",
                    task_id=ev_id,
                ):
                    report = await run_accessory_media_subagent(
                        modality=modality,
                        attachments=mod_atts,
                        user_task=user_prompt,
                        model=acc_model_obj,
                        provider_label=acc_provider,
                        model_label=acc_model_name,
                        allowed_uploads_dir=allowed_uploads_dir,
                    )

            done_event = {
                "id": ev_id,
                "tool": tool_label,
                "args_summary": start_event["args_summary"],
                "status": "ok" if report.get("status") == "completed" else "error",
                "output_summary": str(report.get("summary") or "")[:180],
                "finished_at": datetime.now(UTC).isoformat(),
            }
            _record_run_tool_event(done_event)
            emit_progress(
                "act_tool_done",
                tool=tool_label,
                tool_event=done_event,
                detail=(
                    f"{modality.title()} Accessory sub-agent finished ({acc_model_name})"
                    if report.get("status") == "completed"
                    else f"{modality.title()} Accessory sub-agent reported an issue"
                ),
            )
            run_activity["accessory_reports"].append(report)
            return report

        try:
            primary_meta = active_models["primary"]
            primary_modalities = detect_model_input_modalities(
                primary_meta.get("provider"),
                primary_meta.get("model"),
                explicit_modalities=primary_input_modalities,
            )
            native_atts, delegated_by_modality = partition_attachments_by_capability(
                parsed_attachments, primary_modalities
            )

            accessory_reports: list[dict[str, Any]] = []
            if delegated_by_modality:
                accessory_reports = list(
                    await asyncio.gather(
                        *(
                            _run_accessory_for_modality(mod, mod_atts, clean_task)
                            for mod, mod_atts in delegated_by_modality.items()
                        )
                    )
                )

            message_parts: list[str] = [clean_task]
            if native_atts:
                native_desc = ", ".join(f"{a.name} ({a.kind})" for a in native_atts)
                message_parts.append(f"\n[Natively attached media: {native_desc}]")
            if accessory_reports:
                message_parts.append("\n" + format_accessory_reports_for_primary(accessory_reports))

            human_text = "\n".join(message_parts).strip()
            current_native_media_attachments.set(native_atts or None)
            current_accessory_fallback_fn.set(_run_accessory_for_modality)

            effective_run_max_turns = max_turns if max_turns is not None else self.max_turns
            config = self.make_config(thread_id, max_turns=effective_run_max_turns)
            existing = await self.graph.aget_state(config)
            has_history = bool(existing and existing.values and existing.values.get("messages"))

            if has_history:
                input_payload: dict[str, Any] = {
                    "messages": [HumanMessage(content=human_text)],
                    "turn_count": 0,
                    "max_turns": effective_run_max_turns,
                    "next_action": "continue",
                    "final_response": "",
                    "error": None,
                    "thread_id": thread_id,
                }
            else:
                input_payload = dict(
                    make_initial_state(
                        [HumanMessage(content=human_text)],
                        thread_id=thread_id,
                        max_turns=effective_run_max_turns,
                    )
                )

            await self.graph.ainvoke(input_payload, config=config)
            result = await self.inspect_thread(thread_id)
            result["active_models"] = active_models
            if run_activity["thinking_segments"]:
                result["thinking"] = "\n\n".join(run_activity["thinking_segments"]).strip()
            result["tool_events"] = list(run_activity["tool_events"])
            result["file_changes"] = list(run_activity["file_changes"])
            if run_activity["accessory_reports"]:
                existing_reports = list(result.get("subagent_reports") or [])
                result["subagent_reports"] = [*run_activity["accessory_reports"], *existing_reports]
            await invoke_agent_hook(
                self.config.hooks.on_run_complete,
                {
                    "thread_id": thread_id,
                    "status": result.get("status"),
                    "turn_count": result.get("turn_count", 0),
                    "tool_events": result.get("tool_events", []),
                },
            )
            return result
        finally:
            current_run_deadline.reset(deadline_tok)
            current_guardrail_policy.reset(policy_tok)
            current_access_mode.reset(access_mode_tok)
            current_primary_meta.reset(prim_meta_tok)
            current_accessory_fallback_fn.reset(fallback_fn_tok)
            current_native_media_attachments.reset(native_media_tok)
            current_uploads_dir.reset(uploads_tok)
            current_run_activity.reset(act_tok)
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
        access_mode: str | None = None,
    ) -> dict[str, Any]:
        prim_model, prim_spec = _resolve_override_model(
            "primary", primary_model_override, resolver=self.model_override_resolver
        )
        try:
            work_model, work_spec = _resolve_override_model(
                "worker", worker_model_override, resolver=self.model_override_resolver
            )
        except ModelConfigurationError as exc:
            logger.warning(
                "Worker model override unavailable during resume (%s); falling back to primary model.",
                exc,
            )
            work_model, work_spec = prim_model, prim_spec

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

        effective_access_mode = (
            access_mode if access_mode is not None else self.config.default_access_mode
        )
        run_deadline = (
            time.monotonic() + self.config.limits.max_wall_clock_seconds
            if self.config.limits.max_wall_clock_seconds is not None
            else None
        )
        run_activity = _new_run_activity()
        token = current_session_id.set(thread_id)
        prim_tok = current_primary_model.set(prim_model)
        work_tok = current_worker_model.set(work_model)
        tools_tok = current_extra_tools.set(extra_tools or None)
        mcp_tok = current_extra_mcp_names.set(extra_mcp_tool_names or None)
        skills_tok = current_dynamic_skills.set(parsed_skills or None)
        prog_tok = current_progress_callback.set(progress_callback)
        act_tok = current_run_activity.set(run_activity)
        prim_meta_tok = current_primary_meta.set(dict(active_models["primary"]))
        access_mode_tok = bind_run_access_mode(effective_access_mode)
        policy_tok = current_guardrail_policy.set(self.config.guardrail_policy)
        deadline_tok = current_run_deadline.set(run_deadline)
        try:
            decision_label = (
                str(decision.get("decision") or "approve")
                if isinstance(decision, dict)
                else str(decision)
            )
            emit_progress(
                "resuming_approval",
                detail=(
                    "Approval received — executing approved action..."
                    if decision_label in ("approve", "edit")
                    else "Rejection received — updating plan..."
                ),
            )
            existing_snap = await self.graph.aget_state(self.make_config(thread_id))
            saved_max_turns = (
                int(existing_snap.values.get("max_turns"))
                if existing_snap and existing_snap.values and existing_snap.values.get("max_turns")
                else self.max_turns
            )
            config = self.make_config(thread_id, max_turns=saved_max_turns)
            await self.graph.ainvoke(Command(resume=decision), config=config)
            result = await self.inspect_thread(thread_id)
            result["active_models"] = active_models
            if run_activity["thinking_segments"]:
                result["thinking"] = "\n\n".join(run_activity["thinking_segments"]).strip()
            result["tool_events"] = list(run_activity["tool_events"])
            result["file_changes"] = list(run_activity["file_changes"])
            await invoke_agent_hook(
                self.config.hooks.on_run_complete,
                {
                    "thread_id": thread_id,
                    "status": result.get("status"),
                    "turn_count": result.get("turn_count", 0),
                    "tool_events": result.get("tool_events", []),
                    "resumed": True,
                },
            )
            return result
        finally:
            current_run_deadline.reset(deadline_tok)
            current_guardrail_policy.reset(policy_tok)
            current_access_mode.reset(access_mode_tok)
            current_primary_meta.reset(prim_meta_tok)
            current_run_activity.reset(act_tok)
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

        thinking_trace: str | None = None
        last_ai = _latest_ai_message(list(values.get("messages", [])))
        if last_ai is not None:
            _, thinking_trace = _extract_text_and_thinking(last_ai)

        return {
            "thread_id": thread_id,
            "status": status,
            "final_response": values.get("final_response", ""),
            "thinking": thinking_trace,
            "plan": values.get("plan", []),
            "active_skills": values.get("active_skills", []),
            "turn_count": values.get("turn_count", 0),
            "pending_approval": interrupts[0] if interrupts else None,
            "interrupts": interrupts,
            "subagent_reports": values.get("subagent_reports", []),
            "context_usage": values.get("context_usage") or None,
            "error": values.get("error"),
        }


async def _invoke_model_with_live_thinking(
    bound_model: Any,
    input_messages: list[BaseMessage],
    *,
    turn_count: int,
) -> AIMessage:
    """Stream LLM tokens when a live job progress callback is active so thinking updates in real time."""
    if current_progress_callback.get() is None or not hasattr(bound_model, "astream"):
        raw = await bound_model.ainvoke(input_messages)
        return raw if isinstance(raw, AIMessage) else AIMessage(content=str(getattr(raw, "content", "") or ""))

    merged: AIMessageChunk | None = None
    last_finish_reason = ""
    announced_tools: set[str] = set()
    announced_responding = False
    pending_delta = ""
    last_flush = asyncio.get_running_loop().time()
    in_think_tag = False
    tag_buffer = ""

    def _flush_thinking(force: bool = False) -> None:
        nonlocal pending_delta, last_flush
        if not pending_delta:
            return
        now_t = asyncio.get_running_loop().time()
        if force or len(pending_delta) >= 24 or (now_t - last_flush) >= 0.08:
            emit_progress(
                "thinking",
                turn_count=turn_count,
                thinking_delta=pending_delta,
                detail="Thinking through next step...",
            )
            pending_delta = ""
            last_flush = now_t

    stream_iter = bound_model.astream(input_messages)
    try:
        async for chunk in stream_iter:
            chunk_meta = getattr(chunk, "response_metadata", None) or {}
            if isinstance(chunk_meta, dict):
                chunk_fr = _normalize_finish_reason(
                    chunk_meta.get("finish_reason") or chunk_meta.get("normalized_finish_reason")
                )
                if chunk_fr:
                    last_finish_reason = chunk_fr

            if isinstance(chunk, AIMessageChunk):
                merged = chunk if merged is None else merged + chunk
            elif isinstance(chunk, AIMessage):
                # Non-streaming fake/custom model yielded full AIMessage in one shot
                merged = AIMessageChunk(
                    content=chunk.content,
                    additional_kwargs=dict(chunk.additional_kwargs or {}),
                    response_metadata=dict(chunk.response_metadata or {}),
                    tool_calls=list(getattr(chunk, "tool_calls", None) or []),
                )

            # 1. Streamed reasoning_content / reasoning / thinking blocks
            r_delta = _extract_chunk_thinking_delta(chunk)
            if r_delta:
                pending_delta += r_delta
                _flush_thinking()

            # 2. Inline <think>...</think> tags emitted inside plain string content
            c_raw = getattr(chunk, "content", None)
            if isinstance(c_raw, str) and c_raw:
                if not in_think_tag and "<think>" in (tag_buffer + c_raw).lower():
                    combined = tag_buffer + c_raw
                    idx = combined.lower().find("<think>")
                    after = combined[idx + len("<think>") :]
                    in_think_tag = True
                    tag_buffer = ""
                    if "</think>" in after.lower():
                        end_idx = after.lower().find("</think>")
                        pending_delta += after[:end_idx]
                        in_think_tag = False
                        _flush_thinking(force=True)
                    else:
                        pending_delta += after
                        _flush_thinking()
                elif in_think_tag:
                    if "</think>" in c_raw.lower():
                        end_idx = c_raw.lower().find("</think>")
                        pending_delta += c_raw[:end_idx]
                        in_think_tag = False
                        _flush_thinking(force=True)
                    else:
                        pending_delta += c_raw
                        _flush_thinking()
                else:
                    tag_buffer = (tag_buffer + c_raw)[-16:]
                    if not announced_responding and c_raw.strip() and not announced_tools:
                        announced_responding = True
                        _flush_thinking(force=True)
                        emit_progress(
                            "responding",
                            turn_count=turn_count,
                            detail="Drafting response...",
                        )

            # 3. Announce tool calls the moment the model starts emitting them
            tc_chunks = getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None) or []
            for tc in tc_chunks:
                t_name = str(tc.get("name") or "").strip() if isinstance(tc, dict) else ""
                if t_name and t_name not in announced_tools:
                    announced_tools.add(t_name)
                    _flush_thinking(force=True)
                    emit_progress(
                        "planning_tool",
                        turn_count=turn_count,
                        detail=f"Preparing tool call: {t_name}",
                    )
    except (GraphInterrupt, asyncio.CancelledError):
        raise
    except (
        NotImplementedError,
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.StreamError,
        json.JSONDecodeError,
    ) as stream_exc:
        logger.warning(
            "Streaming interrupted (%s); falling back to non-streaming ainvoke on turn %d.",
            type(stream_exc).__name__,
            turn_count,
        )
        _flush_thinking(force=True)
        raw = await bound_model.ainvoke(input_messages)
        return raw if isinstance(raw, AIMessage) else AIMessage(content=str(getattr(raw, "content", "") or ""))
    finally:
        aclose = getattr(stream_iter, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception as close_exc:  # noqa: BLE001
                logger.debug("Ignored stream aclose error: %s", close_exc)

    _flush_thinking(force=True)

    if merged is None:
        # Provider returned an empty SSE stream (e.g. transient free-tier [DONE] with 0 chunks);
        # try a single non-streaming fallback before declaring empty output.
        if hasattr(bound_model, "ainvoke"):
            try:
                raw = await bound_model.ainvoke(input_messages)
                if isinstance(raw, AIMessage):
                    return raw
                return AIMessage(content=str(getattr(raw, "content", "") or ""))
            except GraphInterrupt:
                raise
            except Exception as fallback_exc:  # noqa: BLE001
                logger.warning("Empty-stream ainvoke fallback failed: %s", fallback_exc)
        return AIMessage(content="")

    merged_tool_calls = list(getattr(merged, "tool_calls", None) or [])
    merged_invalid_tcs = list(getattr(merged, "invalid_tool_calls", None) or [])
    merged_extra = dict(merged.additional_kwargs or {})

    # Fallback: if streaming truncated a tool call mid-JSON and LangChain did not populate
    # invalid_tool_calls, recover the partial tool call name/args so plan_node can self-heal.
    if not merged_tool_calls and not merged_invalid_tcs:
        raw_tc_chunks = list(getattr(merged, "tool_call_chunks", None) or [])
        raw_ak_tcs = merged_extra.get("tool_calls")
        if raw_tc_chunks:
            for idx, tc in enumerate(raw_tc_chunks):
                if isinstance(tc, dict) and (tc.get("name") or tc.get("args")):
                    merged_invalid_tcs.append(
                        {
                            "name": str(tc.get("name") or "unknown_tool"),
                            "args": str(tc.get("args") or ""),
                            "id": str(tc.get("id") or f"invalid-tc-{idx}"),
                            "error": "Truncated or unparseable streamed tool call JSON.",
                            "type": "invalid_tool_call",
                        }
                    )
        elif isinstance(raw_ak_tcs, list) and raw_ak_tcs:
            for idx, tc in enumerate(raw_ak_tcs):
                if isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    merged_invalid_tcs.append(
                        {
                            "name": str(fn.get("name") or tc.get("name") or "unknown_tool"),
                            "args": str(fn.get("arguments") or ""),
                            "id": str(tc.get("id") or f"invalid-tc-{idx}"),
                            "error": "Unparsed or truncated tool call in additional_kwargs.",
                            "type": "invalid_tool_call",
                        }
                    )

    final_meta = dict(merged.response_metadata or {})
    if last_finish_reason:
        final_meta["finish_reason"] = last_finish_reason
    elif final_meta.get("finish_reason"):
        final_meta["finish_reason"] = _normalize_finish_reason(final_meta.get("finish_reason"))

    ai_kwargs: dict[str, Any] = {
        "content": merged.content,
        "additional_kwargs": merged_extra,
        "response_metadata": final_meta,
        "tool_calls": merged_tool_calls,
        "invalid_tool_calls": merged_invalid_tcs,
    }
    if getattr(merged, "id", None):
        ai_kwargs["id"] = merged.id
    if getattr(merged, "usage_metadata", None):
        ai_kwargs["usage_metadata"] = merged.usage_metadata
    return AIMessage(**ai_kwargs)


def _build_model_with_bumped_max_tokens(
    base_model: Any,
    active_tool_list: list[BaseTool],
    *,
    lower_reasoning: bool = False,
    attempt: int = 1,
) -> tuple[Any, Any, int | None]:
    """Return a copied (bound_model, base_model, max_tokens) with a larger ceiling and reduced reasoning effort."""
    cur_max = getattr(base_model, "max_tokens", None)
    updates: dict[str, Any] = {}
    effective_max: int | None = cur_max if isinstance(cur_max, int) else None

    if isinstance(cur_max, int) and cur_max < 32768:
        bumped = min(max(cur_max * 2, 8192), 32768)
        updates["max_tokens"] = bumped
        effective_max = bumped

    if lower_reasoning:
        extra_body = getattr(base_model, "extra_body", None)
        if isinstance(extra_body, dict) and "reasoning_effort" in extra_body:
            new_extra = dict(extra_body)
            cur_eff = str(new_extra.get("reasoning_effort") or "").strip().lower()
            if attempt == 1 and cur_eff not in ("low", "minimal", "none"):
                new_extra["reasoning_effort"] = "low"
            else:
                new_extra.pop("reasoning_effort", None)
            updates["extra_body"] = new_extra or None
        direct_eff = getattr(base_model, "reasoning_effort", None)
        if isinstance(direct_eff, str) and direct_eff.strip():
            if attempt == 1 and direct_eff.strip().lower() not in ("low", "minimal", "none"):
                updates["reasoning_effort"] = "low"
            else:
                updates["reasoning_effort"] = None

    if not updates or not hasattr(base_model, "model_copy"):
        bound = (
            base_model.bind_tools(active_tool_list)
            if hasattr(base_model, "bind_tools")
            else base_model
        )
        return bound, base_model, effective_max

    try:
        copied = base_model.model_copy(update=updates)
        bound = (
            copied.bind_tools(active_tool_list)
            if hasattr(copied, "bind_tools")
            else copied
        )
        return bound, copied, effective_max
    except Exception:  # noqa: BLE001
        bound = (
            base_model.bind_tools(active_tool_list)
            if hasattr(base_model, "bind_tools")
            else base_model
        )
        return bound, base_model, cur_max if isinstance(cur_max, int) else None


def build_supervisor_graph(
    *,
    primary_model: BaseChatModel,
    worker_model: BaseChatModel | None = None,
    sandbox: SandboxManager | None = None,
    skills: SkillRegistry | None = None,
    mcp_tools: list[BaseTool] | None = None,
    extra_tools: list[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_turns: int | None = None,
    max_tool_retries: int | None = None,
    compaction_threshold: int | None = None,
    allow_private_web_fetch: bool = False,
    config: AgentConfig | None = None,
    model_override_resolver: ModelOverrideResolver | None = None,
) -> tuple[Any, dict[str, BaseTool], set[str]]:
    cfg = config or AgentConfig()
    effective_max_turns_default = max_turns if max_turns is not None else cfg.limits.max_turns
    effective_max_tool_retries = (
        max_tool_retries if max_tool_retries is not None else cfg.limits.max_tool_retries
    )
    effective_compaction_threshold = (
        compaction_threshold
        if compaction_threshold is not None
        else cfg.limits.tool_output_compaction_threshold
    )
    auto_compact_threshold = cfg.limits.auto_compact_threshold
    guardrail_policy = cfg.guardrail_policy

    resolved_sandbox = sandbox or SandboxManager()
    resolved_skills = skills or SkillRegistry()
    resolved_worker_model = worker_model or primary_model

    local_tools: list[BaseTool] = (
        build_local_tools(
            resolved_sandbox,
            allow_private_web_fetch=allow_private_web_fetch,
            extract_model_factory=lambda: current_worker_model.get() or resolved_worker_model,
            include_web_tools=cfg.enable_web_tools,
        )
        if cfg.include_default_tools
        else []
    )
    skill_tools: list[BaseTool] = [make_load_skill_tool(resolved_skills)] if cfg.enable_skills else []

    mcp_list = list(mcp_tools or [])
    mcp_names: set[str] = {t.name for t in mcp_list}

    base_tools: dict[str, BaseTool] = {}
    for t in [*local_tools, *skill_tools, *mcp_list, *(extra_tools or [])]:
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

    tools_by_name: dict[str, BaseTool] = dict(base_tools)
    if cfg.enable_subagents:
        spawn_tool = make_spawn_subagent_tool(
            worker_model_factory=lambda: current_worker_model.get() or resolved_worker_model,
            available_tools=_active_base_tools,
            mcp_tool_names=_active_mcp_names,
            workspace_dir=str(resolved_sandbox.config.workspace_dir),
            guardrail_policy=guardrail_policy,
        )
        tools_by_name[spawn_tool.name] = spawn_tool

    all_tool_list = list(tools_by_name.values())
    default_bound_model: Any = (
        primary_model.bind_tools(all_tool_list)
        if all_tool_list and hasattr(primary_model, "bind_tools")
        else primary_model
    )

    def _active_tools_map() -> dict[str, BaseTool]:
        dyn = current_extra_tools.get()
        if not dyn:
            return tools_by_name
        return {**tools_by_name, **dyn}

    async def plan_node(state: AgentState) -> dict[str, Any]:
        turn_count = int(state.get("turn_count", 0)) + 1
        effective_max = int(state.get("max_turns", effective_max_turns_default))
        messages = list(state.get("messages", []))
        current_plan: list[PlanStep] = list(state.get("plan", []))
        active_skills: list[str] = list(state.get("active_skills", []))
        last_observations = list(state.get("last_observations", []))

        completed_count = sum(1 for s in current_plan if s["status"] == "completed")
        start_detail = (
            "Reviewing your request and choosing the best approach"
            if turn_count == 1
            else f"Evaluating results and planning step {turn_count}"
        )
        emit_progress(
            "plan_start",
            turn_count=turn_count,
            steps_completed=completed_count,
            detail=start_detail,
        )
        await invoke_agent_hook(
            cfg.hooks.on_turn_start,
            {
                "thread_id": str(state.get("thread_id") or "default"),
                "turn_count": turn_count,
                "max_turns": effective_max,
                "completed_steps": completed_count,
            },
        )

        if turn_count > effective_max:
            return {
                "turn_count": turn_count,
                "next_action": "complete",
                "reflection": f"Reached MAX_TURNS ({effective_max}); stopping loop safely.",
            }

        deadline = current_run_deadline.get()
        if deadline is not None and time.monotonic() >= deadline:
            limit_s = cfg.limits.max_wall_clock_seconds
            timeout_note = (
                f"Stopped after reaching max_wall_clock_seconds ({limit_s:g}s) safety limit "
                f"at turn {turn_count - 1}/{effective_max}."
            )
            return {
                "turn_count": turn_count - 1,
                "next_action": "complete",
                "reflection": timeout_note,
                "final_response": timeout_note,
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
        dynamic_skill_catalog = (
            resolved_skills.format_catalog_for_prompt() if cfg.enable_skills else ""
        )
        loaded_skills_text = (
            build_loaded_skills_context(active_skills, resolved_skills) if cfg.enable_skills else ""
        )
        compacted_summary = str(state.get("compacted_summary") or "")
        compaction_count = int(state.get("compaction_count") or 0)
        last_compacted_at: str | None = state.get("last_compacted_at")
        last_compaction_saved_tokens = int(state.get("last_compaction_saved_tokens") or 0)

        def _build_full_system_prompt(summary_text: str) -> str:
            base_sys = build_supervisor_system_prompt(
                skill_catalog_text=dynamic_skill_catalog,
                loaded_skills_text=loaded_skills_text,
                available_tool_names=list(active_tools.keys()),
                plan=updated_plan,
                turn_count=turn_count,
                max_turns=effective_max,
                persona=cfg.persona,
                enable_skills=cfg.enable_skills,
            )
            return base_sys + build_compacted_memory_block(summary_text)

        sys_prompt = _build_full_system_prompt(compacted_summary)

        override_primary = current_primary_model.get()
        has_dynamic_tools = bool(current_extra_tools.get())
        if override_primary is not None:
            active_bound_model: Any = (
                override_primary.bind_tools(active_tool_list)
                if active_tool_list and hasattr(override_primary, "bind_tools")
                else override_primary
            )
        elif has_dynamic_tools and active_tool_list and hasattr(primary_model, "bind_tools"):
            active_bound_model = primary_model.bind_tools(active_tool_list)
        else:
            active_bound_model = default_bound_model
        base_candidate = override_primary if override_primary is not None else primary_model

        active_primary_meta = current_primary_meta.get() or _describe_model_instance(base_candidate)
        active_provider_name = str(active_primary_meta.get("provider") or "")
        active_model_name = str(active_primary_meta.get("model") or "")
        explicit_ctx_window = active_primary_meta.get("context_window")
        if not isinstance(explicit_ctx_window, int) or explicit_ctx_window < 512:
            model_attr_ctx = getattr(base_candidate, "context_window", None)
            explicit_ctx_window = (
                int(model_attr_ctx)
                if isinstance(model_attr_ctx, int) and model_attr_ctx >= 512
                else None
            )
        explicit_ctx_source = (
            str(active_primary_meta.get("context_window_source")).strip()
            if active_primary_meta.get("context_window_source")
            else None
        )
        manual_ctx_window = active_primary_meta.get("manual_context_window")
        if not isinstance(manual_ctx_window, int) or manual_ctx_window < 512:
            manual_ctx_window = None
        explicit_max_out = active_primary_meta.get("max_output_tokens")
        if not isinstance(explicit_max_out, int) or explicit_max_out < 256:
            model_attr_out = getattr(base_candidate, "max_output_tokens", None)
            explicit_max_out = (
                int(model_attr_out)
                if isinstance(model_attr_out, int) and model_attr_out >= 256
                else None
            )
        configured_max_toks = active_primary_meta.get("max_tokens")
        if not isinstance(configured_max_toks, int) or configured_max_toks < 64:
            configured_max_toks = None
        active_max_retries = active_primary_meta.get("max_retries")
        if not isinstance(active_max_retries, int) or active_max_retries < 0:
            active_max_retries = 2
        active_read_timeout = active_primary_meta.get("timeout_seconds")

        pre_context_usage = compute_context_usage(
            provider=active_provider_name,
            model=active_model_name,
            explicit_context_window=explicit_ctx_window,
            explicit_context_window_source=explicit_ctx_source,
            manual_context_window=manual_ctx_window,
            explicit_max_output_tokens=explicit_max_out,
            configured_max_tokens=configured_max_toks,
            system_prompt_text=sys_prompt,
            skill_catalog_text=dynamic_skill_catalog,
            loaded_skills_text=loaded_skills_text,
            tools=active_tool_list,
            messages=messages,
            compacted_summary=compacted_summary,
            compaction_count=compaction_count,
            last_compacted_at=last_compacted_at,
            tokens_saved_last_compaction=last_compaction_saved_tokens,
            auto_compact_threshold=auto_compact_threshold,
        )

        # Auto-compact conversation strictly at >= configured threshold (default 80%) of known context window ONLY
        removal_messages: list[RemoveMessage] = []
        known_ctx_window = pre_context_usage.get("context_window")
        usage_ratio = (
            pre_context_usage["used_tokens"] / max(1, known_ctx_window)
            if isinstance(known_ctx_window, int) and known_ctx_window >= 512
            else 0.0
        )
        if (
            cfg.enable_auto_compaction
            and isinstance(known_ctx_window, int)
            and known_ctx_window >= 512
            and should_auto_compact(usage_ratio, len(messages), threshold=auto_compact_threshold)
        ):
            compact_ev_id = f"compact-{compaction_count + 1}-{turn_count}"
            compact_start_ev = {
                "id": compact_ev_id,
                "tool": "auto_compact_context",
                "args_summary": (
                    f"{pre_context_usage['usage_percent']}% of "
                    f"{pre_context_usage['context_window']:,} tokens ({active_model_name or 'primary'})"
                )[:160],
                "status": "running",
                "started_at": utc_now_iso(),
            }
            _record_run_tool_event(compact_start_ev)
            emit_progress(
                "compacting_context",
                turn_count=turn_count,
                tool="auto_compact_context",
                tool_event=compact_start_ev,
                context_usage=pre_context_usage,
                detail=(
                    f"Context reached {pre_context_usage['usage_percent']}% "
                    f"(>= {AUTO_COMPACT_THRESHOLD_PERCENT}% threshold) — auto-compacting earlier turns..."
                ),
            )
            removals, remaining_msgs, new_summary, saved_tokens = await compact_conversation_history(
                messages,
                summarizer_model=base_candidate,
                existing_summary=compacted_summary,
                keep_recent_turns=2,
            )
            if removals or (new_summary and new_summary != compacted_summary):
                removal_messages = removals
                messages = remaining_msgs
                compacted_summary = new_summary
                compaction_count += 1
                last_compacted_at = utc_now_iso()
                last_compaction_saved_tokens = saved_tokens
                sys_prompt = _build_full_system_prompt(compacted_summary)
                pre_context_usage = compute_context_usage(
                    provider=active_provider_name,
                    model=active_model_name,
                    explicit_context_window=explicit_ctx_window,
                    explicit_context_window_source=explicit_ctx_source,
                    manual_context_window=manual_ctx_window,
                    explicit_max_output_tokens=explicit_max_out,
                    configured_max_tokens=configured_max_toks,
                    system_prompt_text=sys_prompt,
                    skill_catalog_text=dynamic_skill_catalog,
                    loaded_skills_text=loaded_skills_text,
                    tools=active_tool_list,
                    messages=messages,
                    compacted_summary=compacted_summary,
                    compaction_count=compaction_count,
                    last_compacted_at=last_compacted_at,
                    tokens_saved_last_compaction=last_compaction_saved_tokens,
                    auto_compact_threshold=auto_compact_threshold,
                )
            compact_done_ev = {
                "id": compact_ev_id,
                "tool": "auto_compact_context",
                "args_summary": compact_start_ev["args_summary"],
                "status": "ok",
                "output_summary": (
                    f"Saved ~{saved_tokens:,} tokens; now at {pre_context_usage['usage_percent']}% "
                    f"({pre_context_usage['used_tokens']:,} / {pre_context_usage['context_window']:,})"
                )[:180],
                "finished_at": utc_now_iso(),
            }
            _record_run_tool_event(compact_done_ev)
            emit_progress(
                "act_tool_done",
                turn_count=turn_count,
                tool="auto_compact_context",
                tool_event=compact_done_ev,
                context_usage=pre_context_usage,
                detail=(
                    f"Auto-compacted conversation at {pre_context_usage['auto_compact_threshold_percent']}% threshold "
                    f"(now {pre_context_usage['usage_percent']}%)"
                ),
            )
            await invoke_agent_hook(
                cfg.hooks.on_compaction,
                {
                    "thread_id": str(state.get("thread_id") or "default"),
                    "turn_count": turn_count,
                    "compaction_count": compaction_count,
                    "tokens_saved": saved_tokens,
                    "usage_percent_after": pre_context_usage.get("usage_percent"),
                },
            )

        sanitized_history = _sanitize_messages_for_llm(messages)
        native_media_atts = current_native_media_attachments.get()
        if native_media_atts:
            for rev_idx in range(len(sanitized_history) - 1, -1, -1):
                if isinstance(sanitized_history[rev_idx], HumanMessage):
                    orig_hm = sanitized_history[rev_idx]
                    orig_text = (
                        orig_hm.content
                        if isinstance(orig_hm.content, str)
                        else str(orig_hm.content or "")
                    )
                    try:
                        mm_blocks = build_multimodal_content_blocks(
                            orig_text,
                            native_media_atts,
                            allowed_uploads_dir=current_uploads_dir.get(),
                        )
                        sanitized_history[rev_idx] = HumanMessage(
                            content=mm_blocks,
                            additional_kwargs=dict(getattr(orig_hm, "additional_kwargs", None) or {}),
                        )
                    except Exception as mm_exc:  # noqa: BLE001
                        logger.warning("Failed to encode native media attachment: %s", mm_exc)
                    break

        llm_input_messages: list[BaseMessage] = [SystemMessage(content=sys_prompt), *sanitized_history]

        async def _call_model_resilient(
            bound_candidate: Any,
            base_model_obj: Any,
            prompt_msgs: list[BaseMessage],
        ) -> AIMessage:
            model_label = active_model_name or "primary model"

            async def _invoke_once(target_bound: Any) -> AIMessage:
                return await invoke_with_resilient_retry(
                    lambda: _invoke_model_with_live_thinking(
                        target_bound,
                        prompt_msgs,
                        turn_count=turn_count,
                    ),
                    label=model_label,
                    provider=active_provider_name,
                    model=active_model_name,
                    stage="planning",
                    max_transient_retries=active_max_retries,
                    turn_count=turn_count,
                )

            try:
                reply = await _invoke_once(bound_candidate)
            except (GraphInterrupt, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001
                err_lower = str(exc).lower()
                extra_body = getattr(base_model_obj, "extra_body", None)
                if (
                    isinstance(extra_body, dict)
                    and "reasoning_effort" in extra_body
                    and ("reasoning" in err_lower or "400" in err_lower or "invalid_request" in err_lower)
                    and hasattr(base_model_obj, "model_copy")
                ):
                    logger.warning(
                        "Model rejected reasoning_effort=%r; retrying once without reasoning_effort: %s",
                        extra_body.get("reasoning_effort"),
                        exc,
                    )
                    stripped_extra = {k: v for k, v in extra_body.items() if k != "reasoning_effort"}
                    fallback_model = base_model_obj.model_copy(
                        update={"extra_body": stripped_extra or None}
                    )
                    fallback_bound: Any = (
                        fallback_model.bind_tools(active_tool_list)
                        if hasattr(fallback_model, "bind_tools")
                        else fallback_model
                    )
                    reply = await _invoke_once(fallback_bound)
                elif is_timeout_model_error(exc) and bool(active_primary_meta.get("fallback_on_timeout")):
                    fb_provider = str(active_primary_meta.get("fallback_provider") or "").strip()
                    fb_model_name = str(active_primary_meta.get("fallback_model") or "").strip()
                    if fb_provider and fb_model_name:
                        logger.warning(
                            "Primary model timed out during planning [provider=%s model=%s]; "
                            "falling back to configured fallback [provider=%s model=%s].",
                            active_provider_name,
                            active_model_name,
                            fb_provider,
                            fb_model_name,
                        )
                        emit_progress(
                            "thinking",
                            turn_count=turn_count,
                            detail=f"Primary model timed out — falling back to {fb_provider}/{fb_model_name}...",
                        )
                        fb_spec = resolve_model_spec_with_override(
                            "primary",
                            override={"provider": fb_provider, "model": fb_model_name},
                        )
                        fb_llm = (
                            model_override_resolver("primary", fb_spec)
                            if model_override_resolver is not None
                            else build_chat_model("primary", spec=fb_spec, require_api_key=True)
                        )
                        fb_bound: Any = (
                            fb_llm.bind_tools(active_tool_list)
                            if active_tool_list and hasattr(fb_llm, "bind_tools")
                            else fb_llm
                        )
                        reply = await _invoke_once(fb_bound)
                    else:
                        raise
                else:
                    raise
            if not isinstance(reply, AIMessage):
                reply = AIMessage(content=str(getattr(reply, "content", "") or ""))
            return reply

        emit_progress(
            "thinking",
            turn_count=turn_count,
            context_usage=pre_context_usage,
            detail=(
                "Analyzing your request..."
                if turn_count == 1
                else f"Reasoning on next step (turn {turn_count})..."
            ),
        )
        try:
            ai_reply = await _call_model_resilient(
                active_bound_model,
                base_candidate,
                llm_input_messages,
            )
        except (GraphInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001
            # Runtime safety net: if the primary model unexpectedly rejects native multimodal blocks,
            # automatically delegate those attachments to the configured Accessory sub-agent(s) and retry!
            fallback_fn = current_accessory_fallback_fn.get()
            if native_media_atts and fallback_fn is not None and is_unsupported_modality_error(exc):
                logger.warning(
                    "Primary model rejected native media (%s); automatically delegating to Accessory sub-agent(s).",
                    exc,
                )
                grouped: dict[MediaModality, list[MediaAttachment]] = {}
                for att in native_media_atts:
                    grouped.setdefault(att.kind, []).append(att)
                fallback_reports = list(
                    await asyncio.gather(
                        *(
                            fallback_fn(mod, mod_atts, user_goal)
                            for mod, mod_atts in grouped.items()
                        )
                    )
                )
                current_native_media_attachments.set(None)
                report_block = format_accessory_reports_for_primary(fallback_reports)
                text_only_history = _sanitize_messages_for_llm(messages)
                for rev_idx in range(len(text_only_history) - 1, -1, -1):
                    if isinstance(text_only_history[rev_idx], HumanMessage):
                        orig_hm = text_only_history[rev_idx]
                        orig_text = (
                            orig_hm.content
                            if isinstance(orig_hm.content, str)
                            else str(orig_hm.content or "")
                        )
                        text_only_history[rev_idx] = HumanMessage(
                            content=f"{orig_text}\n\n{report_block}".strip()
                        )
                        break
                llm_input_messages = [SystemMessage(content=sys_prompt), *text_only_history]
                try:
                    ai_reply = await _call_model_resilient(
                        active_bound_model,
                        base_candidate,
                        llm_input_messages,
                    )
                except (GraphInterrupt, asyncio.CancelledError):
                    raise
                except Exception as retry_exc:  # noqa: BLE001
                    err_cat = classify_model_error_category(retry_exc)
                    err_msg = (
                        f"Primary model error after Accessory fallback "
                        f"[{err_cat} | provider={active_provider_name or 'unknown'} | "
                        f"model={active_model_name or 'unknown'}]: {retry_exc}"
                    )
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
            else:
                err_cat = classify_model_error_category(exc)
                timeout_hint = (
                    f" (read_timeout={active_read_timeout}s, retries={active_max_retries})"
                    if is_timeout_model_error(exc) and active_read_timeout is not None
                    else ""
                )
                err_msg = (
                    f"Primary model error during planning "
                    f"[{err_cat} | provider={active_provider_name or 'unknown'} | "
                    f"model={active_model_name or 'unknown'}]{timeout_hint}: {exc}"
                )
                logger.error(
                    "Planning failure [session=%s turn=%d category=%s provider=%s model=%s]: %s",
                    current_session_id.get(),
                    turn_count,
                    err_cat,
                    active_provider_name or "unknown",
                    active_model_name or "unknown",
                    exc,
                )
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

        clean_preamble, turn_thinking = _extract_text_and_thinking(ai_reply)
        _record_run_thinking(turn_thinking)
        tool_calls = list(getattr(ai_reply, "tool_calls", None) or [])
        invalid_tool_calls = list(getattr(ai_reply, "invalid_tool_calls", None) or [])
        finish_reason = _get_finish_reason(ai_reply)

        # Self-healing recovery: if the model truncated mid-tool-call (finish_reason=length),
        # produced malformed tool JSON, or returned empty content (e.g. thinking-only output),
        # retry automatically up to 2 times with escalating token ceiling, reduced reasoning
        # effort on length truncation, and role-safe continuation guidance.
        max_recovery_attempts = 2
        recovery_attempt = 0
        current_retry_base = base_candidate
        current_retry_bound = active_bound_model
        while not tool_calls and (invalid_tool_calls or not clean_preamble) and recovery_attempt < max_recovery_attempts:
            recovery_attempt += 1
            bad_names = ", ".join(
                str(itc.get("name") or "tool") for itc in invalid_tool_calls[:3] if isinstance(itc, dict)
            )
            bad_err = "; ".join(
                str(itc.get("error") or "invalid JSON")[:120]
                for itc in invalid_tool_calls[:2]
                if isinstance(itc, dict)
            )

            retry_bound = current_retry_bound
            retry_base = current_retry_base
            if finish_reason == "length":
                bumped_bound, bumped_base, _bumped_limit = _build_model_with_bumped_max_tokens(
                    current_retry_base,
                    active_tool_list,
                    lower_reasoning=True,
                    attempt=recovery_attempt,
                )
                retry_bound = bumped_bound
                retry_base = bumped_base
                current_retry_bound = bumped_bound
                current_retry_base = bumped_base

                if bad_names:
                    recovery_note = (
                        f"Output hit max_tokens while preparing {bad_names} — retrying with "
                        f"chunked strategy (attempt {recovery_attempt}/{max_recovery_attempts})..."
                    )
                    nudge_text = (
                        f"SYSTEM RECOVERY: Your previous output hit the max_tokens output limit "
                        f"while generating arguments for `{bad_names}` and truncated mid-JSON. "
                        "Keep your internal reasoning brief and break large file writes into smaller "
                        "chunks (e.g. write a compact base with `write_file` under ~150 lines, then "
                        "extend with `edit_file` or a short Python script via `bash_exec`). "
                        "Continue the task now."
                    )
                else:
                    recovery_note = (
                        f"Output hit max_tokens before emitting a visible reply — retrying with "
                        f"higher token budget (attempt {recovery_attempt}/{max_recovery_attempts})..."
                    )
                    nudge_text = (
                        "SYSTEM RECOVERY: Your previous output hit the max_tokens limit before "
                        "finishing a tool call or visible answer. Keep your internal reasoning very "
                        "concise and immediately emit either a compact tool call (splitting large "
                        "file writes into smaller chunks) or your complete final answer now."
                    )
            elif invalid_tool_calls:
                recovery_note = (
                    f"Tool call ({bad_names or 'tool'}) had malformed JSON — retrying "
                    f"(attempt {recovery_attempt}/{max_recovery_attempts})..."
                )
                nudge_text = (
                    f"SYSTEM RECOVERY: Your previous tool call to `{bad_names or 'tool'}` failed "
                    f"argument JSON parsing ({bad_err}). Re-issue the tool call with strictly valid, "
                    "properly escaped JSON arguments, or provide your final answer."
                )
            else:
                recovery_note = (
                    f"Model returned no visible output — prompting continuation "
                    f"(attempt {recovery_attempt}/{max_recovery_attempts})..."
                )
                nudge_text = (
                    "SYSTEM RECOVERY: Your previous response had no visible text or tool calls. "
                    "Please immediately call the next required tool or provide your complete final "
                    "answer based on the conversation and tool outputs above."
                )

            logger.warning(
                "Empty/truncated model output on turn %d (finish_reason=%r, invalid_tcs=%d); %s",
                turn_count,
                finish_reason,
                len(invalid_tool_calls),
                recovery_note,
            )
            emit_progress(
                "thinking",
                turn_count=turn_count,
                detail=recovery_note,
            )

            retry_messages = _build_recovery_messages(llm_input_messages, nudge_text)
            try:
                ai_reply = await _call_model_resilient(retry_bound, retry_base, retry_messages)
            except GraphInterrupt:
                raise
            except Exception as retry_exc:  # noqa: BLE001
                # If bumping max_tokens or modifying reasoning_effort triggered a provider 400 error,
                # fall back to the original model with the role-safe retry messages.
                if retry_bound is not active_bound_model and (
                    "max_tokens" in str(retry_exc).lower()
                    or "max_completion_tokens" in str(retry_exc).lower()
                    or "reasoning" in str(retry_exc).lower()
                    or "400" in str(retry_exc).lower()
                ):
                    try:
                        ai_reply = await _call_model_resilient(
                            active_bound_model, base_candidate, retry_messages
                        )
                    except GraphInterrupt:
                        raise
                    except Exception as fb_exc:  # noqa: BLE001
                        logger.warning("Recovery fallback retry %d failed: %s", recovery_attempt, fb_exc)
                        break
                else:
                    logger.warning("Recovery retry %d failed: %s", recovery_attempt, retry_exc)
                    break

            clean_preamble, next_thinking = _extract_text_and_thinking(ai_reply)
            if next_thinking:
                _record_run_thinking(next_thinking)
                turn_thinking = (
                    f"{turn_thinking}\n\n{next_thinking}".strip()
                    if turn_thinking
                    else next_thinking
                )
            tool_calls = list(getattr(ai_reply, "tool_calls", None) or [])
            invalid_tool_calls = list(getattr(ai_reply, "invalid_tool_calls", None) or [])
            finish_reason = _get_finish_reason(ai_reply)

        dropped_invalid_notes: list[str] = []
        if tool_calls and invalid_tool_calls:
            for itc in invalid_tool_calls[:3]:
                if isinstance(itc, dict):
                    iname = str(itc.get("name") or "tool")
                    ierr = str(itc.get("error") or "malformed JSON")[:140]
                    dropped_invalid_notes.append(f"`{iname}` ({ierr})")
            logger.warning(
                "Turn %d had %d valid and %d invalid tool calls; dropping invalid calls: %s",
                turn_count,
                len(tool_calls),
                len(invalid_tool_calls),
                "; ".join(dropped_invalid_notes),
            )

        ai_reply = _sanitize_ai_message_for_history(ai_reply)
        if dropped_invalid_notes:
            meta_copy = dict(getattr(ai_reply, "response_metadata", None) or {})
            meta_copy["_dropped_invalid_tool_calls"] = dropped_invalid_notes
            ai_reply = ai_reply.model_copy(update={"response_metadata": meta_copy})
        tool_calls = list(getattr(ai_reply, "tool_calls", None) or [])

        tool_list_label = ", ".join(str(tc.get("name", "tool")) for tc in tool_calls[:4])
        if len(tool_calls) > 4:
            tool_list_label += f" +{len(tool_calls) - 4} more"
        # If the model wrote a brief preamble explaining what it is about to do before calling tools, include it in activity_note
        preamble_note = (
            clean_preamble.splitlines()[0].strip()[:180]
            if tool_calls and clean_preamble and not clean_preamble.lstrip().startswith("{")
            else None
        )
        provider_input_toks = extract_provider_input_tokens(ai_reply)
        post_context_usage = compute_context_usage(
            provider=active_provider_name,
            model=active_model_name,
            explicit_context_window=explicit_ctx_window,
            explicit_context_window_source=explicit_ctx_source,
            manual_context_window=manual_ctx_window,
            explicit_max_output_tokens=explicit_max_out,
            configured_max_tokens=configured_max_toks,
            system_prompt_text=sys_prompt,
            skill_catalog_text=dynamic_skill_catalog,
            loaded_skills_text=loaded_skills_text,
            tools=active_tool_list,
            messages=[*messages, ai_reply],
            compacted_summary=compacted_summary,
            provider_input_tokens=provider_input_toks,
            compaction_count=compaction_count,
            last_compacted_at=last_compacted_at,
            tokens_saved_last_compaction=last_compaction_saved_tokens,
            auto_compact_threshold=auto_compact_threshold,
        )

        emit_progress(
            "plan_llm_done",
            turn_count=turn_count,
            thinking=turn_thinking,
            activity_note=preamble_note,
            context_usage=post_context_usage,
            detail=(
                f"Calling {tool_list_label}"
                if tool_calls
                else "Finalizing response"
            ),
        )

        raw_content = clean_preamble
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
            "messages": [*removal_messages, ai_reply],
            "plan": updated_plan,
            "current_step_id": current_step_id,
            "turn_count": turn_count,
            "pending_approval": None,
            "compacted_summary": compacted_summary,
            "compaction_count": compaction_count,
            "last_compacted_at": last_compacted_at,
            "last_compaction_saved_tokens": last_compaction_saved_tokens,
            "context_usage": post_context_usage,
            "error": None,
        }

    def _find_pending_tool_calls(
        messages: list[BaseMessage],
    ) -> tuple[AIMessage | None, list[tuple[int, dict[str, Any]]], set[str]]:
        last_ai_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_idx = i
                break
        if last_ai_idx < 0:
            return None, [], set()
        last_ai_msg = messages[last_ai_idx]
        assert isinstance(last_ai_msg, AIMessage)
        raw_tcs = list(getattr(last_ai_msg, "tool_calls", None) or [])
        if not raw_tcs:
            return last_ai_msg, [], set()
        answered: set[str] = set()
        for msg in messages[last_ai_idx + 1 :]:
            if isinstance(msg, ToolMessage):
                tid = getattr(msg, "tool_call_id", None)
                if tid is not None:
                    answered.add(str(tid))
        pending: list[tuple[int, dict[str, Any]]] = []
        for idx, tc in enumerate(raw_tcs):
            tc_id = str(tc.get("id") or f"call-{idx}")
            if tc_id not in answered:
                pending.append((idx, tc))
        return last_ai_msg, pending, answered

    async def act_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        last_ai, pending_calls, answered_ids = _find_pending_tool_calls(messages)
        tool_calls = getattr(last_ai, "tool_calls", None) or [] if last_ai else []

        if not tool_calls:
            return {"last_observations": [], "pending_approval": None}

        if not pending_calls:
            return {"pending_approval": None}

        active_tools = _active_tools_map()
        active_mcp_names = _active_mcp_names()
        retry_counts = dict(state.get("tool_retry_counts", {}))
        active_skills = list(state.get("active_skills", []))
        subagent_reports = list(state.get("subagent_reports", []))
        thread_id = str(state.get("thread_id") or "default")
        ws_dir = resolved_sandbox.get_session_workspace(thread_id)

        # Execute exactly one pending tool call per LangGraph checkpoint step so a subsequent
        # tool raising GraphInterrupt never replays already-completed earlier tools on resume.
        tool_messages: list[ToolMessage] = []
        observations: list[dict[str, Any]] = (
            list(state.get("last_observations", [])) if answered_ids else []
        )
        is_final_pending_call = len(pending_calls) == 1

        for idx, tc in pending_calls[:1]:
            tc_name = str(tc.get("name", "")).strip()
            tc_args = normalize_tool_arguments(tc_name, tc.get("args"))
            tc_id = str(tc.get("id") or f"call-{idx}")

            retry_key = f"{tc_name}:{json.dumps(tc_args, sort_keys=True, default=str)[:120]}"
            prior_failures = retry_counts.get(retry_key, 0)
            if prior_failures >= effective_max_tool_retries:
                err_text = (
                    f"RETRY LIMIT EXCEEDED: Tool '{tc_name}' already failed "
                    f"{prior_failures} times with the same arguments (limit={effective_max_tool_retries}). "
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

            args_summary = _summarize_tool_args(tc_args)
            started_iso = datetime.now(UTC).isoformat()
            start_tool_event = {
                "id": tc_id,
                "tool": tc_name,
                "args_summary": args_summary,
                "status": "running",
                "output_summary": "",
                "started_at": started_iso,
                "finished_at": None,
            }
            _record_run_tool_event(start_tool_event)
            emit_progress(
                "act_tool_start",
                tool=tc_name,
                tool_event=start_tool_event,
                detail=f"Running {tc_name}" + (f" ({args_summary})" if args_summary else ""),
            )
            await invoke_agent_hook(
                cfg.hooks.on_tool_start,
                {
                    "thread_id": thread_id,
                    "tool": tc_name,
                    "tool_call_id": tc_id,
                    "arguments": tc_args,
                },
            )

            before_snap = _snapshot_workspace(ws_dir) if cfg.enable_workspace_diff else {}
            tool_task_label = f"{tc_name} ({args_summary})" if args_summary else tc_name
            try:
                with track_active_task(
                    "mcp" if tc_name in active_mcp_names else "tool",
                    tool_task_label,
                    task_id=f"tool-{tc_id}",
                ):
                    raw_output, audit = await execute_tool_with_guardrails(
                        tool_obj,
                        tc_args,
                        is_mcp_tool=(tc_name in active_mcp_names),
                        source="supervisor",
                        policy=guardrail_policy,
                    )
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                err_text = f"ERROR: Tool '{tc_name}' raised {type(exc).__name__}: {exc}"
                retry_counts[retry_key] = prior_failures + 1
                err_tool_event = {
                    "id": tc_id,
                    "tool": tc_name,
                    "args_summary": args_summary,
                    "status": "error",
                    "output_summary": err_text[:200],
                    "finished_at": datetime.now(UTC).isoformat(),
                }
                _record_run_tool_event(err_tool_event)
                emit_progress(
                    "act_tool_done",
                    tool=tc_name,
                    tool_event=err_tool_event,
                    detail=f"Tool '{tc_name}' failed",
                )
                await invoke_agent_hook(
                    cfg.hooks.on_tool_end,
                    {
                        "thread_id": thread_id,
                        "tool": tc_name,
                        "tool_call_id": tc_id,
                        "status": "error",
                        "error": err_text,
                    },
                )
                tool_messages.append(
                    ToolMessage(content=err_text, tool_call_id=tc_id, name=tc_name)
                )
                observations.append(
                    {"tool": tc_name, "status": "error", "summary": err_text, "archived_log": None}
                )
                continue

            if not audit.get("approved", True):
                obs_status = "rejected"
            elif _is_tool_output_error(tc_name, raw_output):
                obs_status = "error"
                retry_counts[retry_key] = prior_failures + 1
            else:
                obs_status = "ok"
                retry_counts.pop(retry_key, None)

            if cfg.enable_workspace_diff:
                after_snap = _snapshot_workspace(ws_dir)
                step_file_changes = _diff_workspace_snapshots(
                    before_snap, after_snap, tool_name=tc_name
                )
                # Fallback for identical-content overwrites via write_file/edit_file within same fs tick
                if obs_status == "ok" and tc_name in ("write_file", "edit_file") and not step_file_changes:
                    raw_target = str(tc_args.get("path") or "").strip().lstrip("/")
                    if raw_target.startswith("workspace/"):
                        raw_target = raw_target[len("workspace/") :]
                    if raw_target and raw_target in after_snap and not after_snap[raw_target][0]:
                        step_file_changes.append(
                            {
                                "kind": "file",
                                "action": "modified",
                                "path": raw_target,
                                "tool": tc_name,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )
            else:
                step_file_changes = []

            _record_run_file_changes(step_file_changes)

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
                threshold=effective_compaction_threshold,
            )

            done_tool_event = {
                "id": tc_id,
                "tool": tc_name,
                "args_summary": args_summary,
                "status": obs_status,
                "output_summary": compacted_text.replace("\n", " ").strip()[:180],
                "finished_at": datetime.now(UTC).isoformat(),
            }
            _record_run_tool_event(done_tool_event)
            emit_progress(
                "act_tool_done",
                tool=tc_name,
                tool_event=done_tool_event,
                file_changes=step_file_changes,
                detail=f"Finished {tc_name} ({obs_status})",
            )
            await invoke_agent_hook(
                cfg.hooks.on_tool_end,
                {
                    "thread_id": thread_id,
                    "tool": tc_name,
                    "tool_call_id": tc_id,
                    "status": obs_status,
                    "audit": audit,
                    "file_changes": step_file_changes,
                },
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

        dropped_invalid = (
            (getattr(last_ai, "response_metadata", None) or {}).get("_dropped_invalid_tool_calls")
            if last_ai is not None
            else None
        )
        if is_final_pending_call and isinstance(dropped_invalid, list) and dropped_invalid and tool_messages:
            note = (
                "\n\n[SYSTEM NOTE: Additional tool call(s) in the same turn were skipped due to "
                f"malformed JSON arguments: {', '.join(str(x) for x in dropped_invalid)}. "
                "Re-issue them with valid JSON if still needed.]"
            )
            last_tm = tool_messages[-1]
            tool_messages[-1] = ToolMessage(
                content=f"{last_tm.content}{note}",
                tool_call_id=last_tm.tool_call_id,
                name=last_tm.name,
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
        if observations:
            obs_tools = ", ".join(str(o.get("tool", "tool")) for o in observations[:3])
            obs_detail = f"Reviewed output from {obs_tools}"
        else:
            obs_detail = "Checking response completeness"
        emit_progress(
            "observe",
            steps_completed=completed_steps,
            detail=obs_detail,
        )
        return {"plan": plan}

    async def reflect_node(state: AgentState) -> dict[str, Any]:
        turn_count = int(state.get("turn_count", 0))
        effective_max = int(state.get("max_turns", effective_max_turns_default))
        observations = list(state.get("last_observations", []))
        last_ai = _latest_ai_message(list(state.get("messages", [])))
        emit_progress(
            "reflect",
            turn_count=turn_count,
            detail=(
                "Deciding whether more tool steps are needed"
                if observations
                else "Wrapping up final answer"
            ),
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

        deadline = current_run_deadline.get()
        if deadline is not None and time.monotonic() >= deadline:
            limit_s = cfg.limits.max_wall_clock_seconds
            return {
                "next_action": "complete",
                "reflection": f"Reached max_wall_clock_seconds ({limit_s:g}s); producing final summary.",
                "final_response": (
                    f"Stopped after reaching max_wall_clock_seconds ({limit_s:g}s) safety limit "
                    f"at turn {turn_count}/{effective_max}."
                ),
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

        if last_ai is not None:
            clean_final_text, _ = _extract_text_and_thinking(last_ai)
            if clean_final_text:
                return {
                    "next_action": "complete",
                    "reflection": "Supervisor produced a complete final response.",
                }

        plan = list(state.get("plan", []))
        diag = _build_empty_output_diagnostic(
            last_ai=last_ai,
            plan=plan,
            turn_count=turn_count,
        )
        return {
            "next_action": "complete",
            "reflection": f"Model returned empty visible output after recovery retries (finish_reason={_get_finish_reason(last_ai) or 'unknown'}).",
            "final_response": diag,
        }

    async def response_node(state: AgentState) -> dict[str, Any]:
        if state.get("final_response"):
            return {"final_response": state["final_response"]}

        messages = list(state.get("messages", []))
        last_ai = _latest_ai_message(messages)
        if last_ai and not getattr(last_ai, "tool_calls", None):
            text, _ = _extract_text_and_thinking(last_ai)
            if text:
                return {"final_response": text}

        turn_count = int(state.get("turn_count", 0))
        effective_max = int(state.get("max_turns", effective_max_turns_default))
        plan = list(state.get("plan", []))
        completed_steps = [s for s in plan if s["status"] == "completed"]
        if turn_count >= effective_max:
            summary = (
                f"Stopped after reaching turn limit ({turn_count}/{effective_max}). "
                f"Completed {len(completed_steps)}/{len(plan)} planned steps."
            )
        else:
            summary = _build_empty_output_diagnostic(
                last_ai=last_ai,
                plan=plan,
                turn_count=turn_count,
            )
        return {
            "final_response": summary,
            "messages": [AIMessage(content=summary)],
        }

    def route_after_act(state: AgentState) -> Literal["act", "observe"]:
        _last_ai, pending_calls, _answered = _find_pending_tool_calls(
            list(state.get("messages", []))
        )
        if pending_calls:
            return "act"
        return "observe"

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
    builder.add_conditional_edges(
        "act",
        route_after_act,
        {"act": "act", "observe": "observe"},
    )
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
    config: AgentConfig | None = None,
) -> AsyncIterator[AgentRuntime]:
    cfg = config or AgentConfig()
    resolved_sandbox = sandbox or SandboxManager()
    resolved_skills = skills or SkillRegistry()
    resolved_mcp = mcp_manager or MCPRegistryManager()

    mcp_tools: list[BaseTool] = []
    if connect_mcp:
        mcp_tools = await resolved_mcp.connect()

    resolved_primary = primary_model or get_primary_model()
    resolved_worker = worker_model or get_worker_model()

    effective_max_turns = cfg.limits.max_turns if max_turns is None else max_turns
    effective_max_retries = (
        cfg.limits.max_tool_retries if max_tool_retries is None else max_tool_retries
    )
    effective_compaction_threshold = (
        cfg.limits.tool_output_compaction_threshold
        if compaction_threshold is None
        else compaction_threshold
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

    raw_db_arg = (
        checkpoint_db_path if checkpoint_db_path is not None else DEFAULT_CHECKPOINT_DB_PATH
    )
    is_in_memory = str(raw_db_arg).strip() == ":memory:"

    if cfg.custom_checkpointer is not None:
        db_path = Path(":memory:") if is_in_memory else Path(raw_db_arg)
        compiled, tools_by_name, mcp_names = build_supervisor_graph(
            primary_model=resolved_primary,
            worker_model=resolved_worker,
            sandbox=resolved_sandbox,
            skills=resolved_skills,
            mcp_tools=mcp_tools,
            extra_tools=extra_tools,
            checkpointer=cfg.custom_checkpointer,
            max_turns=effective_max_turns,
            max_tool_retries=effective_max_retries,
            compaction_threshold=effective_compaction_threshold,
            allow_private_web_fetch=allow_private_web_fetch,
            config=cfg,
            model_override_resolver=model_override_resolver,
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
            config=cfg,
        )
        return

    if is_in_memory:
        db_path = Path(":memory:")
        conn = await aiosqlite.connect(":memory:")
    else:
        def _prepare_db_path() -> Path:
            resolved = Path(raw_db_arg).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved

        db_path = await asyncio.to_thread(_prepare_db_path)
        conn = await aiosqlite.connect(str(db_path))

    try:
        if not is_in_memory:
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
            compaction_threshold=effective_compaction_threshold,
            allow_private_web_fetch=allow_private_web_fetch,
            config=cfg,
            model_override_resolver=model_override_resolver,
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
            config=cfg,
        )
    finally:
        await conn.close()
