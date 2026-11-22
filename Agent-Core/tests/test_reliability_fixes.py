"""Comprehensive tests for:
1. HTTP 429 retry every 3 seconds until success, or timeout with both timeout + 429 error detail.
2. Smart stall watchdog: stays alive during active background tasks/timers (e.g. 180s timer),
   and only stalls when genuinely stuck (idle with no active background work).
3. Async HITL approval resume via POST /agent/jobs/resume with 409 stale-approval detection.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent_core.api import create_app
from agent_core.jobs import AgentJob, JobManager, current_progress_callback
from agent_core.models import (
    ModelRateLimitError,
    invoke_with_rate_limit_retry,
    is_rate_limit_exception,
)
from agent_core.sandbox import SandboxConfig, SandboxManager


class Fake429Error(Exception):
    def __init__(self, message: str = "Rate limit exceeded", status_code: int = 429) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAsyncLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeAsyncLLM:
        return self

    async def ainvoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        if self._idx >= len(self._responses):
            return AIMessage(content="Done.")
        msg = self._responses[self._idx]
        self._idx += 1
        return msg


class TestReliabilityFixes(unittest.IsolatedAsyncioTestCase):
    async def test_429_retries_until_success_and_emits_progress(self) -> None:
        attempts = 0
        progress_events: list[dict[str, Any]] = []

        async def _flaky_call() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise Fake429Error("upstream 429: too many requests")
            return "recovered-ok"

        tok = current_progress_callback.set(progress_events.append)
        try:
            result = await invoke_with_rate_limit_retry(
                _flaky_call,
                label="primary model",
                retry_interval_seconds=0.05,
                max_wait_seconds=2.0,
            )
        finally:
            current_progress_callback.reset(tok)

        self.assertEqual(result, "recovered-ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(len(progress_events), 2)
        self.assertTrue(all(ev.get("phase") == "rate_limited" for ev in progress_events))
        self.assertIn("429", progress_events[0].get("detail", ""))

    async def test_429_timeout_includes_both_timeout_and_429_detail(self) -> None:
        attempts = 0

        async def _always_429() -> str:
            nonlocal attempts
            attempts += 1
            raise Fake429Error("Provider quota exceeded (429)")

        with self.assertRaises(ModelRateLimitError) as ctx:
            await invoke_with_rate_limit_retry(
                _always_429,
                label="primary model",
                retry_interval_seconds=0.05,
                max_wait_seconds=0.16,
            )

        msg = str(ctx.exception)
        self.assertGreaterEqual(attempts, 2)
        self.assertIn("timed out", msg.lower())
        self.assertIn("429", msg)
        self.assertIn("Provider quota exceeded", msg)

    async def test_non_429_errors_fail_immediately_without_retry(self) -> None:
        attempts = 0

        async def _conn_error() -> str:
            nonlocal attempts
            attempts += 1
            raise ConnectionError("network unreachable")

        self.assertFalse(is_rate_limit_exception(ConnectionError("network unreachable")))
        with self.assertRaises(ConnectionError):
            await invoke_with_rate_limit_retry(
                _conn_error,
                label="primary model",
                retry_interval_seconds=0.05,
                max_wait_seconds=1.0,
            )
        self.assertEqual(attempts, 1)

    async def test_watchdog_keeps_job_alive_during_long_background_task(self) -> None:
        jm = JobManager()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sandbox = SandboxManager(
                    SandboxConfig(
                        workspace_dir=Path(tmpdir),
                        backend="local",
                        timeout_seconds=10,
                    )
                )

                async def _runner(job: AgentJob) -> dict[str, Any]:
                    # Sleep 0.75s in a real subprocess while stall timeout is only 0.25s
                    res = await sandbox.exec_command("sleep 0.75 && echo timer_done", timeout=5)
                    return {
                        "status": "completed",
                        "final_response": res.stdout.strip(),
                        "plan": [],
                        "active_skills": [],
                        "turn_count": 1,
                    }

                job = jm.create_job(
                    thread_id="thread-bg-timer",
                    endpoint="run",
                    request_payload={"task": "run timer"},
                    stall_timeout_seconds=0.25,
                )
                jm.launch_job(job, _runner)

                # Wait while background command runs past stall_timeout_seconds
                await asyncio.sleep(0.45)
                mid_status = job.to_status_payload()
                self.assertEqual(mid_status["status"], "running")
                self.assertGreaterEqual(len(mid_status["active_tasks"]), 1)

                # Wait for completion
                for _ in range(40):
                    polled = await jm.poll_job(job.job_id, wait_ms=100)
                    assert polled is not None
                    if polled["status"] != "running":
                        break
                final_status = job.to_status_payload()
                self.assertEqual(final_status["status"], "completed")
                self.assertEqual(final_status["result"]["final_response"], "timer_done")
        finally:
            await jm.shutdown()

    async def test_watchdog_stalls_when_genuinely_stuck_with_no_background_tasks(self) -> None:
        jm = JobManager()
        try:

            async def _stuck_runner(job: AgentJob) -> dict[str, Any]:
                # No progress and no active background tasks -> genuinely stuck!
                await asyncio.sleep(5.0)
                return {"status": "completed", "final_response": "should not reach"}

            job = jm.create_job(
                thread_id="thread-stuck",
                endpoint="run",
                request_payload={"task": "hang forever"},
                stall_timeout_seconds=0.25,
            )
            jm.launch_job(job, _stuck_runner)

            for _ in range(30):
                polled = await jm.poll_job(job.job_id, wait_ms=100)
                assert polled is not None
                if polled["status"] != "running":
                    break

            final_status = job.to_status_payload()
            self.assertEqual(final_status["status"], "stalled")
            self.assertIn("stalled", (final_status["error"] or "").lower())
        finally:
            await jm.shutdown()

    async def test_watchdog_keeps_job_alive_during_429_retries_then_fails_at_timeout(self) -> None:
        jm = JobManager()
        try:

            async def _rate_limited_runner(job: AgentJob) -> dict[str, Any]:
                async def _always_429() -> str:
                    raise Fake429Error("upstream 429 rate limit")

                await invoke_with_rate_limit_retry(
                    _always_429,
                    label="primary model",
                    retry_interval_seconds=0.06,
                    max_wait_seconds=0.25,
                )
                return {"status": "completed", "final_response": "never"}

            job = jm.create_job(
                thread_id="thread-429-job",
                endpoint="run",
                request_payload={"task": "test 429"},
                stall_timeout_seconds=0.35,
            )
            jm.launch_job(job, _rate_limited_runner)

            for _ in range(40):
                polled = await jm.poll_job(job.job_id, wait_ms=100)
                assert polled is not None
                if polled["status"] != "running":
                    break

            final_status = job.to_status_payload()
            self.assertEqual(final_status["status"], "error")
            err = final_status["error"] or ""
            self.assertIn("timed out", err.lower())
            self.assertIn("429", err)
        finally:
            await jm.shutdown()

    async def test_track_active_task_prunes_dead_tasks(self) -> None:
        jm = JobManager()
        try:
            job = jm.create_job(
                thread_id="thread-prune",
                endpoint="run",
                request_payload={},
                stall_timeout_seconds=1.0,
            )
            alive_flag = True
            tid = "task-mock-1"
            job.register_active_task(
                tid,
                kind="background_process",
                label="mock worker",
                liveness_check=lambda: alive_flag,
            )
            self.assertTrue(job.has_active_background_work())
            alive_flag = False
            self.assertFalse(job.has_active_background_work())
            self.assertEqual(job.get_active_tasks_summary(), [])
            job.unregister_active_task(tid)
        finally:
            await jm.shutdown()

    def test_async_resume_job_endpoint_and_409_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ws_dir = tmp_path / "ws"
            ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / "temp.txt").write_text("hello", encoding="utf-8")
            sandbox = SandboxManager(
                SandboxConfig(workspace_dir=ws_dir, backend="local", timeout_seconds=10)
            )
            fake_llm = _FakeAsyncLLM(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-approve-1",
                                "name": "bash_exec",
                                "args": {"command": "rm -f temp.txt && echo approved_hello"},
                            }
                        ],
                    ),
                    AIMessage(content="Command executed: approved_hello"),
                ]
            )
            app = create_app(
                api_token="test-secret-token",
                primary_model=fake_llm,  # type: ignore[arg-type]
                worker_model=fake_llm,  # type: ignore[arg-type]
                checkpoint_db_path=tmp_path / "checkpoints.sqlite",
                sandbox=sandbox,
                connect_mcp=False,
            )
            headers = {"Authorization": "Bearer test-secret-token"}
            with TestClient(app) as client:
                # 1. Start an async run that pauses for approval
                r_start = client.post(
                    "/agent/jobs/run",
                    json={"task": "Remove temp.txt", "thread_id": "thread-hitl-1"},
                    headers=headers,
                )
                self.assertEqual(r_start.status_code, 202)
                run_job_id = r_start.json()["job_id"]

                status_data: dict[str, Any] = {}
                for _ in range(40):
                    r_poll = client.get(
                        f"/agent/jobs/{run_job_id}?wait_ms=100",
                        headers=headers,
                    )
                    self.assertEqual(r_poll.status_code, 200)
                    status_data = r_poll.json()
                    if status_data["status"] != "running":
                        break
                self.assertEqual(status_data["status"], "awaiting_approval")

                # 2. Resume asynchronously via POST /agent/jobs/resume
                r_resume = client.post(
                    "/agent/jobs/resume",
                    json={"thread_id": "thread-hitl-1", "decision": "approve"},
                    headers=headers,
                )
                self.assertEqual(r_resume.status_code, 202)
                resume_job_id = r_resume.json()["job_id"]

                resume_status: dict[str, Any] = {}
                for _ in range(40):
                    r_poll = client.get(
                        f"/agent/jobs/{resume_job_id}?wait_ms=100",
                        headers=headers,
                    )
                    self.assertEqual(r_poll.status_code, 200)
                    resume_status = r_poll.json()
                    if resume_status["status"] != "running":
                        break

                self.assertEqual(resume_status["status"], "completed")
                self.assertIn(
                    "approved_hello",
                    resume_status["result"]["final_response"],
                )
                self.assertFalse((ws_dir / "temp.txt").exists())

                # 3. Calling /agent/jobs/resume again on an already-resolved thread returns 409
                r_dup = client.post(
                    "/agent/jobs/resume",
                    json={"thread_id": "thread-hitl-1", "decision": "approve"},
                    headers=headers,
                )
                self.assertEqual(r_dup.status_code, 409)


if __name__ == "__main__":
    unittest.main()
