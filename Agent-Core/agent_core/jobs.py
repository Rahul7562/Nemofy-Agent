"""Async Agent Job Manager, Progress Heartbeat, and Stall Watchdog (Pure Python)."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
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


def emit_progress(phase: str, **fields: Any) -> None:
    """Report forward progress to the active job watchdog (no-op for non-job calls)."""
    cb = current_progress_callback.get()
    if cb is None:
        return
    try:
        cb({"phase": phase, **fields})
    except Exception:  # noqa: BLE001
        logger.debug("Progress callback raised an exception", exc_info=True)


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
    _update_event: asyncio.Event = field(default_factory=asyncio.Event)
    _runner_task: asyncio.Task[None] | None = None
    _watchdog_task: asyncio.Task[None] | None = None

    def signal_update(self) -> None:
        self._update_event.set()

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
        detail = str(event.get("detail") or event.get("tool") or phase).strip()
        self.last_event = detail[:240]
        self.signal_update()

    def to_status_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        end_ref = self.completed_at_mono if self.completed_at_mono is not None else now
        idle_seconds = (
            max(0.0, now - self.last_progress_at_mono)
            if self.status in ("running", "staged")
            else 0.0
        )
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
            "mcp_tools": list(self.mcp_tools),
            "dynamic_skills": list(self.dynamic_skills),
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
