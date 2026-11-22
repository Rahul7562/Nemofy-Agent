"""Regression tests for Phase 9 Bug Hunter fixes (BUG-5, BUG-6, BUG-7, BUG-8, BUG-9).

Compatible with both pytest and Python's standard library unittest runner.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_core.api import create_app
from agent_core.jobs import AgentJob
from agent_core.models import ModelSpec, build_chat_model, save_persisted_llm_config
from agent_core.sandbox import SandboxConfig, SandboxManager, SandboxSecurityError


class BugHunterRegressionsTest(unittest.TestCase):
    def test_bug5_build_chat_model_unknown_model_without_max_output(self) -> None:
        """BUG-5: Custom/unregistered model returning (None, 'unknown') max_output_tokens must not crash with TypeError."""
        spec = ModelSpec(
            role="primary",
            provider="custom",
            model="my-org/unregistered-custom-model-v99",
            base_url="http://127.0.0.1:11434/v1",
            api_key="test-key",
            temperature=0.2,
            max_tokens=2048,
            max_output_tokens=None,
        )
        llm = build_chat_model("primary", spec=spec, require_api_key=False)
        self.assertEqual(llm.max_tokens, 2048)

    def test_bug6_save_persisted_llm_config_atomic_and_clean(self) -> None:
        """BUG-6: save_persisted_llm_config writes valid JSON atomically without leaving temp files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "llm_config.json"
            save_persisted_llm_config(
                {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "secret-should-strip",
                    },
                    "max_turns": 150,
                    "access_mode": "full_access",
                },
                config_path=cfg_path,
            )
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["primary"]["model"], "gpt-4o")
            self.assertNotIn("api_key", loaded["primary"])
            self.assertEqual(loaded["max_turns"], 150)
            self.assertEqual(loaded["access_mode"], "full_access")
            self.assertEqual(list(tmp_path.glob("*.tmp")), [])

    def test_bug8_agent_job_preserves_context_usage_when_context_window_is_none(self) -> None:
        """BUG-8: context_usage must be retained in record_progress and to_status_payload when context_window is None."""
        job = AgentJob(
            job_id="job-unknown-ctx",
            job_token="tok-1",
            thread_id="thread-unknown-ctx",
            endpoint="run",
            request_payload={"task": "hello"},
            stall_timeout_seconds=30.0,
        )
        snapshot = {
            "model": "my-custom/model",
            "provider": "custom",
            "context_window": None,
            "context_window_source": "unknown",
            "used_tokens": 1234,
            "free_tokens": None,
            "usage_percent": None,
        }
        job.record_progress({"phase": "thinking", "detail": "Working", "context_usage": snapshot})
        self.assertIsNotNone(job.context_usage)
        assert job.context_usage is not None
        self.assertEqual(job.context_usage["used_tokens"], 1234)
        self.assertIsNone(job.context_usage["context_window"])

        payload = job.to_status_payload()
        self.assertIsNotNone(payload["context_usage"])
        self.assertEqual(payload["context_usage"]["used_tokens"], 1234)
        self.assertIsNone(payload["context_usage"]["context_window"])

        # Also verify merging from final job.result when context_window is None
        job.context_usage = None
        job.result = {"status": "completed", "context_usage": {**snapshot, "used_tokens": 2345}}
        payload_after = job.to_status_payload()
        self.assertIsNotNone(payload_after["context_usage"])
        self.assertEqual(payload_after["context_usage"]["used_tokens"], 2345)

    def test_bug9_sandbox_auto_mode_fails_closed_even_when_bwrap_exists_on_path(self) -> None:
        """BUG-9: Presence of bwrap binary on PATH must not bypass allow_unsafe_local_bash when bwrap is not invoked."""
        import agent_core.sandbox as sandbox_mod

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            cfg = SandboxConfig(
                workspace_dir=ws,
                backend="auto",
                allow_unsafe_local_bash=False,
                timeout_seconds=5,
            )
            mgr = SandboxManager(cfg, docker_client=None)
            mgr._active_backend = "local_isolated"

            with patch.object(
                sandbox_mod.shutil,
                "which",
                side_effect=lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
            ):
                with self.assertRaisesRegex(
                    SandboxSecurityError, "unisolated local bash fallback is disabled"
                ):
                    asyncio.run(mgr.exec_command("echo should-be-blocked"))

    def test_bug7_api_jobs_run_rejects_concurrent_active_job_for_same_thread(self) -> None:
        """BUG-7: POST /agent/jobs/run returns HTTP 409 when a job is already active for the same thread_id."""

        class SlowFakeRuntime:
            async def run_task(self, *args, **kwargs):
                await asyncio.sleep(1.5)
                return {"status": "completed", "final_response": "done"}

            async def inspect_thread(self, thread_id: str):
                return {"thread_id": thread_id, "status": "idle"}

        app = create_app(api_token="test-token", checkpoint_db_path=":memory:")
        headers = {"Authorization": "Bearer test-token"}

        with TestClient(app) as client:
            app.state.runtime = SlowFakeRuntime()

            first = client.post(
                "/agent/jobs/run",
                json={"task": "first run", "thread_id": "thread-concurrent-1"},
                headers=headers,
            )
            self.assertEqual(first.status_code, 202)
            first_job_id = first.json()["job_id"]

            second = client.post(
                "/agent/jobs/run",
                json={"task": "second concurrent run", "thread_id": "thread-concurrent-1"},
                headers=headers,
            )
            self.assertEqual(second.status_code, 409)
            self.assertIn(first_job_id, second.json()["detail"])

            # Cleanup
            client.post(f"/agent/jobs/{first_job_id}/cancel", headers=headers)


if __name__ == "__main__":
    unittest.main()
