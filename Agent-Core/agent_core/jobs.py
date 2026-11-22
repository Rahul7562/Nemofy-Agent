"""Async Agent Job Manager, Progress Heartbeat, and Stall Watchdog (Pure Python)."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_STALL_TIMEOUT_SECONDS = float(os.getenv("AGENT_STALL_TIMEOUT_SECONDS", "180"))
DEFAULT_JOB_RETENTION_SECONDS = 3600.0

ProgressCallback = Callable[[dict[str, Any]], None]

current_progress_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "current_progress_callback", default=None
)
current_job: ContextVar[AgentJob | None] = ContextVar("current_job", default=None)


def get_current_job() -> AgentJob | None:
    """Return the active AgentJob bound to the current async context, if any."""
    return current_job.get()


def get_active_timeout_seconds(default: float = DEFAULT_STALL_TIMEOUT_SECONDS) -> float:
    """Return the active job's timeout budget, falling back to ``default`` outside async jobs."""
    job = current_job.get()
    if job is not None and job.stall_timeout_seconds > 0:
        return float(job.stall_timeout_seconds)
    return float(default)


def _is_pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def emit_progress(phase: str, **fields: Any) -> None:
    """Report forward progress to the active job watchdog (no-op for non-job calls)."""
    cb = current_progress_callback.get()
    if cb is None:
        return
    try:
        cb({"phase": phase, **fields})
    except Exception:  # noqa: BLE001
        logger.debug("Progress callback raised an exception", exc_info=True)


@contextmanager
def track_active_task(
    kind: str,
    label: str,
    *,
    task_id: str | None = None,
    pid: int | None = None,
    expected_seconds: float | None = None,
    liveness_check: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """Register an active tool/subagent/background process/timer on the current job so the stall watchdog keeps it alive."""
    tid = task_id or f"{kind}-{uuid.uuid4().hex[:10]}"
    job = current_job.get()
    if job is not None:
        job.register_active_task(
            tid,
            kind=kind,
            label=label,
            pid=pid,
            expected_seconds=expected_seconds,
            liveness_check=liveness_check,
        )
    try:
        yield tid
    finally:
        if job is not None:
            job.unregister_active_task(tid)


class DynamicSkillSpec(BaseModel):
    """Per-request Markdown skill attached to an API job."""

    name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    content: str = Field(..., min_length=1, max_length=64_000)
    source: str = Field(default="api", max_length=128)


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
    mcp_tools: list[str] = field(default_factory=list)
    dynamic_skills: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    thinking_segments: list[str] = field(default_factory=list)
    live_thinking_buffer: str = ""
    activity_feed: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    file_changes: list[dict[str, Any]] = field(default_factory=list)
    context_usage: dict[str, Any] | None = None
    _active_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    _update_event: asyncio.Event = field(default_factory=asyncio.Event)
    _runner_task: asyncio.Task[None] | None = None
    _watchdog_task: asyncio.Task[None] | None = None

    def signal_update(self) -> None:
        self._update_event.set()

    def register_active_task(
        self,
        task_id: str,
        *,
        kind: str,
        label: str,
        pid: int | None = None,
        expected_seconds: float | None = None,
        liveness_check: Callable[[], bool] | None = None,
    ) -> None:
        now = time.monotonic()
        self._active_tasks[task_id] = {
            "id": task_id,
            "kind": kind,
            "label": str(label or kind)[:160],
            "started_at_mono": now,
            "pid": pid,
            "expected_seconds": expected_seconds,
            "liveness_check": liveness_check,
        }
        self.last_progress_at_mono = now
        self.last_heartbeat_at_mono = now
        self.signal_update()

    def unregister_active_task(self, task_id: str) -> None:
        if task_id in self._active_tasks:
            self._active_tasks.pop(task_id, None)
            now = time.monotonic()
            # Grant a fresh progress timestamp when a long-running background task finishes
            # so the subsequent LLM turn has a full stall timeout budget.
            self.last_progress_at_mono = now
            self.last_heartbeat_at_mono = now
            self.signal_update()

    def prune_dead_active_tasks(self) -> None:
        dead_ids: list[str] = []
        for tid, info in list(self._active_tasks.items()):
            check = info.get("liveness_check")
            alive = True
            if callable(check):
                try:
                    alive = bool(check())
                except Exception:  # noqa: BLE001
                    alive = False
            elif info.get("pid") is not None:
                alive = _is_pid_alive(info.get("pid"))
            if not alive:
                dead_ids.append(tid)
        for tid in dead_ids:
            self._active_tasks.pop(tid, None)

    def has_active_background_work(self) -> bool:
        self.prune_dead_active_tasks()
        return bool(self._active_tasks)

    def get_active_tasks_summary(self) -> list[dict[str, Any]]:
        self.prune_dead_active_tasks()
        now = time.monotonic()
        summary: list[dict[str, Any]] = []
        for info in self._active_tasks.values():
            entry: dict[str, Any] = {
                "id": info["id"],
                "kind": info["kind"],
                "label": info["label"],
                "elapsed_seconds": round(max(0.0, now - float(info["started_at_mono"])), 1),
            }
            if info.get("expected_seconds") is not None:
                entry["expected_seconds"] = info["expected_seconds"]
            summary.append(entry)
        return summary

    def record_progress(self, event: dict[str, Any]) -> None:
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

        # Live streaming thinking chunks arrive incrementally while the LLM is generating
        thinking_delta = event.get("thinking_delta")
        if isinstance(thinking_delta, str) and thinking_delta:
            self.live_thinking_buffer += thinking_delta

        # Finalized turn thinking replaces the live buffer for that turn cleanly
        thinking_text = event.get("thinking")
        if isinstance(thinking_text, str) and thinking_text.strip():
            clean_thought = thinking_text.strip()
            self.live_thinking_buffer = ""
            if not self.thinking_segments or self.thinking_segments[-1] != clean_thought:
                self.thinking_segments.append(clean_thought)
        elif phase in ("plan_llm_done", "act_tool_start", "observe") and self.live_thinking_buffer.strip():
            committed = self.live_thinking_buffer.strip()
            self.live_thinking_buffer = ""
            if not self.thinking_segments or self.thinking_segments[-1] != committed:
                self.thinking_segments.append(committed)

        tool_event = event.get("tool_event")
        if isinstance(tool_event, dict) and tool_event.get("tool"):
            ev_id = str(tool_event.get("id") or f"tool-{len(self.tool_events) + 1}")
            existing = next((t for t in self.tool_events if t.get("id") == ev_id), None)
            if existing is not None:
                existing.update({k: v for k, v in tool_event.items() if v is not None})
            else:
                self.tool_events.append({"id": ev_id, **tool_event})

        changes = event.get("file_changes")
        if isinstance(changes, list) and changes:
            for item in changes:
                if isinstance(item, dict) and item.get("path"):
                    self.file_changes.append(dict(item))

        ctx_usage = event.get("context_usage")
        if isinstance(ctx_usage, dict) and ctx_usage.get("context_window"):
            self.context_usage = dict(ctx_usage)

        detail = str(event.get("detail") or event.get("tool") or phase).strip()
        self.last_event = detail[:240]

        # Record meaningful non-duplicate human-readable step narration in activity_feed
        activity_note = event.get("activity_note")
        note_text = (
            str(activity_note).strip()
            if isinstance(activity_note, str) and activity_note.strip()
            else (detail if not thinking_delta and phase not in ("starting", "staged") else "")
        )
        if note_text:
            last_entry = self.activity_feed[-1] if self.activity_feed else None
            # Update repeating rate-limit or background-heartbeat entries in place so the feed stays tidy
            if (
                last_entry is not None
                and phase in ("rate_limited", "background_task")
                and last_entry.get("phase") == phase
            ):
                last_entry["text"] = note_text[:240]
                last_entry["timestamp"] = datetime.now(UTC).isoformat()
            elif last_entry is None or note_text != last_entry.get("text"):
                self.activity_feed.append(
                    {
                        "id": f"act-{len(self.activity_feed) + 1}",
                        "phase": phase,
                        "text": note_text[:240],
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        self.signal_update()

    def to_status_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        end_ref = self.completed_at_mono if self.completed_at_mono is not None else now
        active_tasks = self.get_active_tasks_summary()
        idle_seconds = (
            0.0
            if active_tasks
            else (
                max(0.0, now - self.last_progress_at_mono)
                if self.status in ("running", "staged")
                else 0.0
            )
        )
        thinking_parts = list(self.thinking_segments)
        if self.live_thinking_buffer.strip():
            thinking_parts.append(self.live_thinking_buffer.strip())
        merged_thinking = "\n\n".join(thinking_parts).strip() or None
        if not merged_thinking and isinstance(self.result, dict):
            res_think = self.result.get("thinking")
            if isinstance(res_think, str) and res_think.strip():
                merged_thinking = res_think.strip()

        merged_tools = list(self.tool_events)
        if not merged_tools and isinstance(self.result, dict) and isinstance(self.result.get("tool_events"), list):
            merged_tools = list(self.result["tool_events"])

        merged_files = list(self.file_changes)
        if not merged_files and isinstance(self.result, dict) and isinstance(self.result.get("file_changes"), list):
            merged_files = list(self.result["file_changes"])

        merged_context_usage = self.context_usage
        if (
            isinstance(self.result, dict)
            and isinstance(self.result.get("context_usage"), dict)
            and self.result["context_usage"].get("context_window")
        ):
            merged_context_usage = dict(self.result["context_usage"])

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
            "active_tasks": active_tasks,
            "mcp_tools": list(self.mcp_tools),
            "dynamic_skills": list(self.dynamic_skills),
            "warnings": list(self.warnings),
            "thinking": merged_thinking,
            "activity_feed": list(self.activity_feed),
            "tool_events": merged_tools,
            "file_changes": merged_files,
            "context_usage": merged_context_usage,
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
        if job.status != "staged":
            return
        job.status = "running"
        now = time.monotonic()
        job.started_at_mono = now
        job.last_progress_at_mono = now
        job.last_heartbeat_at_mono = now
        job.record_progress({"phase": "starting", "detail": "LangGraph job started"})

        async def _run() -> None:
            job_tok = current_job.set(job)
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
                    job.error = "Stopped by user."
                    job.current_phase = "cancelled"
                    job.last_event = "Stopped by user."
                    job.completed_at_mono = time.monotonic()
                job.signal_update()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Async job %s failed", job.job_id)
                if job.status not in ("stalled", "cancelled"):
                    job.status = "error"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.current_phase = "error"
                    job.last_event = job.error[:240]
                    job.completed_at_mono = time.monotonic()
                job.signal_update()
            finally:
                current_job.reset(job_tok)
                if job._watchdog_task is not None and not job._watchdog_task.done():
                    job._watchdog_task.cancel()

        async def _watchdog() -> None:
            check_interval = min(1.0, max(0.05, job.stall_timeout_seconds / 4.0))
            last_bg_heartbeat = time.monotonic()
            try:
                while job.status == "running":
                    await asyncio.sleep(check_interval)
                    if job.status != "running":
                        break
                    now_mono = time.monotonic()

                    # Keep the agent alive whenever a real background task, timer,
                    # running tool, or sub-agent is actively executing.
                    if job.has_active_background_work():
                        job.last_progress_at_mono = now_mono
                        if (now_mono - last_bg_heartbeat) >= 5.0:
                            last_bg_heartbeat = now_mono
                            active_list = job.get_active_tasks_summary()
                            if active_list:
                                primary_task = active_list[0]
                                extra_count = len(active_list) - 1
                                suffix = f" (+{extra_count} more)" if extra_count > 0 else ""
                                job.heartbeat_seq += 1
                                job.last_heartbeat_at_mono = now_mono
                                job.last_event = (
                                    f"Running {primary_task['label']}{suffix} "
                                    f"({primary_task['elapsed_seconds']:.0f}s elapsed)"
                                )[:240]
                                job.signal_update()
                        continue

                    idle_s = now_mono - job.last_progress_at_mono
                    if idle_s >= job.stall_timeout_seconds:
                        job.is_stalled = True
                        job.status = "stalled"
                        job.error = (
                            f"Agent stalled: no forward progress or active background tasks for {idle_s:.1f}s "
                            f"(limit={job.stall_timeout_seconds:.1f}s, phase={job.current_phase}, "
                            f"turn={job.turn_count}, last_event={job.last_event})."
                        )
                        job.completed_at_mono = time.monotonic()
                        job.signal_update()
                        if job._runner_task is not None and not job._runner_task.done():
                            job._runner_task.cancel()
                        break
            except asyncio.CancelledError:
                return

        job._runner_task = asyncio.create_task(_run(), name=f"agent-job-{job.job_id}")
        job._watchdog_task = asyncio.create_task(_watchdog(), name=f"agent-watchdog-{job.job_id}")

    async def poll_job(self, job_id: str, *, wait_ms: int = 0) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None

        if wait_ms > 0 and job.status == "running":
            job._update_event.clear()
            if job.status == "running":
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
        job.current_phase = "cancelled"
        job.last_event = reason[:240]
        job.completed_at_mono = time.monotonic()
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
