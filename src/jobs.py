"""Async Agent Job Manager, Progress Heartbeat, Stall Watchdog, and n8n MCP Tool Bridge.

Solves two core production requirements:
1. Long-running tasks without arbitrary 120s gateway timeouts:
   - Jobs run asynchronously in Python while n8n long-polls ``GET /agent/jobs/{job_id}``.
   - Every LangGraph phase transition, tool start/finish, sub-agent step, and plan advance
     emits a progress heartbeat that resets ``last_progress_at``.
   - A dedicated watchdog cancels the job ONLY if zero forward progress occurs for
     ``stall_timeout_seconds`` (default 180s), surfacing a structured ``stalled`` status.
2. Plug-and-play n8n Canvas MCP Servers & AI Tools:
   - Tools wired to the n8n canvas ``MCP Servers`` slot (native ``MCP Client Tool`` nodes
     or any n8n AI Tool node) are registered as live LangChain ``StructuredTool``s on the
     LangGraph Brain and Sub-Agents.
   - When LangGraph invokes a canvas tool, the job emits ``pending_tool_call`` on the active
     heartbeat poll; n8n executes the tool using its live authenticated MCP session and posts
     the output back to ``POST /agent/jobs/{job_id}/tool-result``.
   - Direct external ``mcp_servers`` URLs (Streamable HTTP or SSE) are also supported via
     ``MultiServerMCPClient`` per request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)

DEFAULT_STALL_TIMEOUT_SECONDS = float(os.getenv("AGENT_STALL_TIMEOUT_SECONDS", "180"))
DEFAULT_BRIDGE_TOOL_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_TOOL_TIMEOUT_SECONDS", "90"))
DEFAULT_JOB_RETENTION_SECONDS = 3600.0

ProgressCallback = Callable[[dict[str, Any]], None]

# Async-safe per-request progress reporter used by graph nodes and sub-agents
current_progress_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "current_progress_callback", default=None
)


def emit_progress(phase: str, **fields: Any) -> None:
    """Report forward progress to the active job watchdog (no-op for non-job calls)."""
    cb = current_progress_callback.get()
    if cb is None:
        return
    try:
        cb({"phase": phase, **fields})
    except Exception:  # noqa: BLE001
        logger.debug("Progress callback raised an exception", exc_info=True)


class BridgeToolSpec(BaseModel):
    """Schema of a tool plugged into the n8n canvas MCP Servers / Tools slot."""

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=4000)
    args_schema: dict[str, Any] = Field(default_factory=dict)
    source_node: str | None = Field(default=None, max_length=128)
    is_mcp: bool = True


class CanvasSkillSpec(BaseModel):
    """Dynamic Markdown skill wired to the n8n canvas Skills slot."""

    name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    content: str = Field(..., min_length=1, max_length=64_000)
    source: str = Field(default="n8n_canvas", max_length=128)


class ExternalMCPServerConfig(BaseModel):
    """Optional direct external MCP server config connected from Python per request."""

    name: str = Field(..., min_length=1, max_length=64)
    url: str = Field(..., min_length=1, max_length=1024)
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    bearer_token: str | None = Field(default=None, max_length=1024)
    headers: dict[str, str] = Field(default_factory=dict)
    include_tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    timeout_seconds: float = Field(default=30.0, ge=2.0, le=180.0)


def _json_schema_type_to_python(prop: dict[str, Any]) -> Any:
    raw_type = prop.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        raw_type = non_null[0] if non_null else "string"
    if raw_type == "string":
        return str
    if raw_type == "integer":
        return int
    if raw_type == "number":
        return float
    if raw_type == "boolean":
        return bool
    if raw_type == "array":
        return list[Any]
    if raw_type == "object":
        return dict[str, Any]
    return Any


def build_json_schema_pydantic_model(
    tool_name: str,
    args_schema: dict[str, Any] | None,
) -> type[BaseModel]:
    """Create a Pydantic input model from a JSON Schema dict so LLMs get exact tool args."""
    safe_cls_name = re.sub(r"[^a-zA-Z0-9]", "_", tool_name).strip("_") or "BridgeTool"
    schema = args_schema if isinstance(args_schema, dict) else {}
    raw_properties = schema.get("properties")
    properties: dict[str, Any] = raw_properties if isinstance(raw_properties, dict) else {}
    raw_required = schema.get("required")
    required = set(raw_required) if isinstance(raw_required, list) else set()

    fields: dict[str, Any] = {}
    for prop_name, raw_prop in properties.items():
        if not isinstance(prop_name, str) or not prop_name or prop_name.startswith("_"):
            continue
        prop = raw_prop if isinstance(raw_prop, dict) else {}
        py_type = _json_schema_type_to_python(prop)
        desc = str(prop.get("description") or "").strip() or None
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=desc))
        else:
            default_val = prop.get("default", None)
            fields[prop_name] = (py_type | None, Field(default=default_val, description=desc))

    if not fields:
        # Permissive schema for parameterless or free-form MCP tools
        return create_model(
            f"{safe_cls_name}Input",
            __config__=ConfigDict(extra="allow"),
        )

    return create_model(
        f"{safe_cls_name}Input",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


async def load_external_mcp_tools(
    servers: list[ExternalMCPServerConfig] | None,
) -> tuple[dict[str, BaseTool], set[str], list[str]]:
    """Connect direct external MCP servers per request and return ``(tools, mcp_names, warnings)``."""
    if not servers:
        return {}, set(), []

    tools_map: dict[str, BaseTool] = {}
    mcp_names: set[str] = set()
    warnings: list[str] = []

    for srv in servers:
        headers = {"Accept": "application/json, text/event-stream", **dict(srv.headers)}
        if srv.bearer_token:
            headers.setdefault("Authorization", f"Bearer {srv.bearer_token}")

        conn_cfg: dict[str, Any] = {
            srv.name: {
                "transport": srv.transport,
                "url": srv.url,
                "headers": headers,
                "timeout": timedelta(seconds=srv.timeout_seconds),
                "sse_read_timeout": timedelta(seconds=max(srv.timeout_seconds * 2, 60.0)),
            }
        }
        try:
            client = MultiServerMCPClient(conn_cfg)
            discovered = await asyncio.wait_for(client.get_tools(), timeout=srv.timeout_seconds)
            include_set = set(srv.include_tools) if srv.include_tools else None
            exclude_set = set(srv.exclude_tools) if srv.exclude_tools else set()
            for tool_obj in discovered:
                if include_set is not None and tool_obj.name not in include_set:
                    continue
                if tool_obj.name in exclude_set:
                    continue
                tools_map[tool_obj.name] = tool_obj
                mcp_names.add(tool_obj.name)
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            if srv.bearer_token:
                err_msg = err_msg.replace(srv.bearer_token, "[REDACTED]")
            warnings.append(f"External MCP server '{srv.name}' unavailable: {err_msg}")
            logger.warning("External MCP server '%s' connection failed: %s", srv.name, err_msg)

    return tools_map, mcp_names, warnings


@dataclass
class PendingBridgeToolCall:
    """In-flight tool call waiting for the n8n canvas bridge to execute and return output."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    created_at_mono: float
    future: asyncio.Future[tuple[bool, str]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }


JobStatus = Literal[
    "staged", "running", "completed", "awaiting_approval", "error", "stalled", "cancelled"
]


@dataclass
class AgentJob:
    """Observable async execution record with progress heartbeats and stall detection."""

    job_id: str
    job_token: str
    thread_id: str
    endpoint: Literal["run", "resume"]
    request_payload: dict[str, Any]
    stall_timeout_seconds: float
    bridge_tool_timeout_seconds: float = DEFAULT_BRIDGE_TOOL_TIMEOUT_SECONDS
    status: JobStatus = "staged"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at_mono: float = field(default_factory=time.monotonic)
    last_progress_at_mono: float = field(default_factory=time.monotonic)
    last_heartbeat_at_mono: float = field(default_factory=time.monotonic)
    completed_at_mono: float | None = None
    heartbeat_seq: int = 0
    progress_seq: int = 0
    current_phase: str = "staged"
    last_event: str = "Job created; waiting to start."
    turn_count: int = 0
    steps_completed: int = 0
    is_stalled: bool = False
    error: str | None = None
    result: dict[str, Any] | None = None
    canvas_mcp_tools: list[str] = field(default_factory=list)
    canvas_skills: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _pending_calls: dict[str, PendingBridgeToolCall] = field(default_factory=dict)
    _call_queue: list[str] = field(default_factory=list)
    _dispatched_call_ids: set[str] = field(default_factory=set)
    _update_event: asyncio.Event = field(default_factory=asyncio.Event)
    _runner_task: asyncio.Task[None] | None = None
    _watchdog_task: asyncio.Task[None] | None = None

    def signal_update(self) -> None:
        self._update_event.set()

    def record_progress(self, event: dict[str, Any]) -> None:
        """Record real forward progress, resetting the stall watchdog timer."""
        now = time.monotonic()
        self.last_progress_at_mono = now
        self.last_heartbeat_at_mono = now
        self.progress_seq += 1
        self.heartbeat_seq += 1

        phase = str(event.get("phase") or self.current_phase)
        self.current_phase = phase
        if "turn_count" in event and event["turn_count"] is not None:
            try:
                self.turn_count = max(self.turn_count, int(event["turn_count"]))
            except (TypeError, ValueError):
                pass
        if "steps_completed" in event and event["steps_completed"] is not None:
            try:
                self.steps_completed = max(self.steps_completed, int(event["steps_completed"]))
            except (TypeError, ValueError):
                pass
        detail = str(event.get("detail") or event.get("tool") or phase).strip()
        self.last_event = detail[:240]
        self.signal_update()

    async def invoke_bridge_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Pause this tool call and hand execution to the connected n8n MCP/Tool bridge."""
        loop = asyncio.get_running_loop()
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        fut: asyncio.Future[tuple[bool, str]] = loop.create_future()
        pending = PendingBridgeToolCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            created_at_mono=time.monotonic(),
            future=fut,
        )
        self._pending_calls[call_id] = pending
        self._call_queue.append(call_id)
        self.record_progress(
            {
                "phase": "awaiting_n8n_tool",
                "tool": tool_name,
                "detail": f"Executing n8n canvas tool '{tool_name}'",
            }
        )
        try:
            ok, output = await asyncio.wait_for(fut, timeout=self.bridge_tool_timeout_seconds)
            self.record_progress(
                {
                    "phase": "n8n_tool_completed",
                    "tool": tool_name,
                    "detail": f"n8n canvas tool '{tool_name}' finished (ok={ok})",
                }
            )
            if not ok:
                return f"ERROR: n8n canvas tool '{tool_name}' failed: {output}"
            return output
        except TimeoutError:
            self._pending_calls.pop(call_id, None)
            self._dispatched_call_ids.discard(call_id)
            return (
                f"ERROR: n8n canvas tool '{tool_name}' timed out after "
                f"{self.bridge_tool_timeout_seconds:.0f}s waiting for n8n bridge response."
            )
        finally:
            self._pending_calls.pop(call_id, None)
            self._dispatched_call_ids.discard(call_id)
            if call_id in self._call_queue:
                self._call_queue.remove(call_id)

    def next_pending_tool_call(self) -> dict[str, Any] | None:
        """Return the next in-flight bridge tool call awaiting execution by n8n."""
        for call_id in list(self._call_queue):
            pending = self._pending_calls.get(call_id)
            if pending is None or pending.future.done():
                continue
            return pending.to_payload()
        return None

    def resolve_tool_call(self, call_id: str, *, ok: bool, output: str) -> bool:
        pending = self._pending_calls.get(call_id)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result((ok, output))
        self.record_progress(
            {
                "phase": "n8n_tool_result_received",
                "tool": pending.tool_name,
                "detail": f"Received result for '{pending.tool_name}'",
            }
        )
        return True

    def fail_pending_calls(self, reason: str) -> None:
        for pending in list(self._pending_calls.values()):
            if not pending.future.done():
                pending.future.set_result((False, reason))
        self._pending_calls.clear()
        self._call_queue.clear()
        self._dispatched_call_ids.clear()

    def build_bridge_tools(
        self, specs: list[BridgeToolSpec]
    ) -> tuple[dict[str, BaseTool], set[str]]:
        """Build real LangChain ``StructuredTool`` wrappers backed by the live n8n bridge."""
        tools: dict[str, BaseTool] = {}
        mcp_names: set[str] = set()
        for spec in specs:
            args_model = build_json_schema_pydantic_model(spec.name, spec.args_schema)
            tool_name = spec.name

            async def _coro(
                _tool_name: str = tool_name,
                **kwargs: Any,
            ) -> str:
                return await self.invoke_bridge_tool(_tool_name, kwargs)

            def _sync_not_supported(
                *_args: Any, _tool_name: str = tool_name, **_kwargs: Any
            ) -> str:
                return f"ERROR: Canvas bridge tool '{_tool_name}' requires async execution."

            desc = spec.description.strip() or f"n8n canvas tool '{spec.name}'"
            if spec.source_node:
                desc = f"[n8n Canvas: {spec.source_node}] {desc}"

            wrapped = StructuredTool.from_function(
                func=_sync_not_supported,
                coroutine=_coro,
                name=spec.name,
                description=desc[:1024],
                args_schema=args_model,
            )
            tools[spec.name] = wrapped
            if spec.is_mcp:
                mcp_names.add(spec.name)
        return tools, mcp_names

    def to_status_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        end_ref = self.completed_at_mono if self.completed_at_mono is not None else now
        idle_seconds = (
            max(0.0, now - self.last_progress_at_mono)
            if self.status in ("running", "staged")
            else 0.0
        )
        pending_call = self.next_pending_tool_call() if self.status == "running" else None
        return {
            "job_id": self.job_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "is_alive": self.status in ("staged", "running") and not self.is_stalled,
            "is_stalled": self.is_stalled,
            "heartbeat_seq": self.heartbeat_seq,
            "progress_seq": self.progress_seq,
            "current_phase": self.current_phase,
            "last_event": self.last_event,
            "turn_count": self.turn_count,
            "steps_completed": self.steps_completed,
            "elapsed_seconds": round(max(0.0, end_ref - self.started_at_mono), 2),
            "idle_seconds": round(idle_seconds, 2),
            "stall_timeout_seconds": self.stall_timeout_seconds,
            "pending_tool_call": pending_call,
            "canvas_mcp_tools": list(self.canvas_mcp_tools),
            "canvas_skills": list(self.canvas_skills),
            "warnings": list(self.warnings),
            "error": self.error,
            "result": self.result,
        }


class JobManager:
    """In-memory lifecycle manager for async agent jobs, heartbeats, stall detection, and cleanup."""

    def __init__(
        self,
        *,
        default_stall_timeout_seconds: float | None = None,
        retention_seconds: float = DEFAULT_JOB_RETENTION_SECONDS,
    ) -> None:
        self.default_stall_timeout_seconds = (
            default_stall_timeout_seconds
            if default_stall_timeout_seconds is not None
            else DEFAULT_STALL_TIMEOUT_SECONDS
        )
        self.retention_seconds = retention_seconds
        self._jobs: dict[str, AgentJob] = {}

    def _prune_old_jobs(self) -> None:
        now = time.monotonic()
        stale_ids = [
            jid
            for jid, job in self._jobs.items()
            if job.completed_at_mono is not None
            and (now - job.completed_at_mono) > self.retention_seconds
        ]
        for jid in stale_ids:
            self._jobs.pop(jid, None)

    def create_job(
        self,
        *,
        thread_id: str,
        endpoint: Literal["run", "resume"],
        request_payload: dict[str, Any],
        stall_timeout_seconds: float | None = None,
        bridge_tool_timeout_seconds: float | None = None,
    ) -> AgentJob:
        self._prune_old_jobs()
        job_id = f"job-{uuid.uuid4().hex[:14]}"
        job_token = secrets.token_urlsafe(24)
        stall_s = (
            float(stall_timeout_seconds)
            if stall_timeout_seconds is not None
            else self.default_stall_timeout_seconds
        )
        job = AgentJob(
            job_id=job_id,
            job_token=job_token,
            thread_id=thread_id,
            endpoint=endpoint,
            request_payload=dict(request_payload),
            stall_timeout_seconds=max(0.2, stall_s),
            bridge_tool_timeout_seconds=max(
                1.0,
                float(bridge_tool_timeout_seconds)
                if bridge_tool_timeout_seconds is not None
                else DEFAULT_BRIDGE_TOOL_TIMEOUT_SECONDS,
            ),
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> AgentJob | None:
        return self._jobs.get(job_id)

    def verify_job_token(self, job_id: str, provided_token: str | None) -> bool:
        job = self._jobs.get(job_id)
        if job is None or not provided_token:
            return False
        return secrets.compare_digest(job.job_token, provided_token.strip())

    def launch_job(
        self,
        job: AgentJob,
        runner: Callable[[AgentJob], Awaitable[dict[str, Any]]],
    ) -> None:
        """Start background execution and stall watchdog for ``job``."""
        if job.status != "staged":
            return
        job.status = "running"
        now = time.monotonic()
        job.started_at_mono = now
        job.last_progress_at_mono = now
        job.last_heartbeat_at_mono = now
        job.record_progress({"phase": "starting", "detail": "LangGraph job started"})

        async def _run() -> None:
            try:
                result = await runner(job)
                if job.status in ("stalled", "cancelled"):
                    return
                job.result = result
                res_status = str(result.get("status") or "completed")
                if res_status == "awaiting_approval":
                    job.status = "awaiting_approval"
                elif res_status == "error":
                    job.status = "error"
                    job.error = str(result.get("error") or "Agent reported an error.")
                else:
                    job.status = "completed"
                job.current_phase = job.status
                job.last_event = f"Job finished with status '{job.status}'"
                job.completed_at_mono = time.monotonic()
                job.signal_update()
            except asyncio.CancelledError:
                if job.status == "running":
                    job.status = "cancelled"
                    job.error = "Job was cancelled."
                    job.completed_at_mono = time.monotonic()
                job.fail_pending_calls(job.error or "Job cancelled.")
                job.signal_update()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Async job %s failed", job.job_id)
                if job.status not in ("stalled", "cancelled"):
                    job.status = "error"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.current_phase = "error"
                    job.last_event = job.error[:240]
                    job.completed_at_mono = time.monotonic()
                job.fail_pending_calls(job.error or "Job failed.")
                job.signal_update()
            finally:
                if job._watchdog_task is not None and not job._watchdog_task.done():
                    job._watchdog_task.cancel()

        async def _watchdog() -> None:
            check_interval = min(1.0, max(0.05, job.stall_timeout_seconds / 4.0))
            try:
                while job.status == "running":
                    await asyncio.sleep(check_interval)
                    if job.status != "running":
                        break
                    idle_s = time.monotonic() - job.last_progress_at_mono
                    if idle_s >= job.stall_timeout_seconds:
                        job.is_stalled = True
                        job.status = "stalled"
                        job.error = (
                            f"Agent stalled: no forward progress for {idle_s:.1f}s "
                            f"(limit={job.stall_timeout_seconds:.1f}s, phase={job.current_phase}, "
                            f"turn={job.turn_count}, last_event={job.last_event})."
                        )
                        job.completed_at_mono = time.monotonic()
                        job.fail_pending_calls(job.error)
                        job.signal_update()
                        if job._runner_task is not None and not job._runner_task.done():
                            job._runner_task.cancel()
                        break
            except asyncio.CancelledError:
                return

        job._runner_task = asyncio.create_task(_run(), name=f"agent-job-{job.job_id}")
        job._watchdog_task = asyncio.create_task(_watchdog(), name=f"agent-watchdog-{job.job_id}")

    async def poll_job(self, job_id: str, *, wait_ms: int = 0) -> dict[str, Any] | None:
        """Long-poll a job's heartbeat status; wakes immediately on tool calls or completion."""
        job = self._jobs.get(job_id)
        if job is None:
            return None

        if wait_ms > 0 and job.status == "running" and job.next_pending_tool_call() is None:
            job._update_event.clear()
            # Re-check after clear to avoid lost wakeup race
            if job.status == "running" and job.next_pending_tool_call() is None:
                try:
                    await asyncio.wait_for(
                        job._update_event.wait(),
                        timeout=min(wait_ms, 15_000) / 1000.0,
                    )
                except TimeoutError:
                    pass

        job.heartbeat_seq += 1
        job.last_heartbeat_at_mono = time.monotonic()
        return job.to_status_payload()

    def cancel_job(self, job_id: str, reason: str = "Cancelled by client.") -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("completed", "awaiting_approval", "error", "stalled", "cancelled"):
            return True
        job.status = "cancelled"
        job.error = reason
        job.completed_at_mono = time.monotonic()
        job.fail_pending_calls(reason)
        job.signal_update()
        if job._runner_task is not None and not job._runner_task.done():
            job._runner_task.cancel()
        if job._watchdog_task is not None and not job._watchdog_task.done():
            job._watchdog_task.cancel()
        return True

    async def shutdown(self) -> None:
        for jid in list(self._jobs.keys()):
            self.cancel_job(jid, reason="Server shutting down.")
        tasks = [
            t
            for job in self._jobs.values()
            for t in (job._runner_task, job._watchdog_task)
            if t is not None and not t.done()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def health_summary(self) -> dict[str, Any]:
        active = [j for j in self._jobs.values() if j.status in ("staged", "running")]
        return {
            "active_jobs": len(active),
            "tracked_jobs": len(self._jobs),
            "default_stall_timeout_seconds": self.default_stall_timeout_seconds,
        }


def format_bridge_tool_output(raw: Any) -> str:
    """Serialize any n8n tool return value cleanly into string output for LangGraph."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(raw)
