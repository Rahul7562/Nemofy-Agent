"""Authenticated FastAPI, CLI, and Runtime LLM Configuration End-to-End Verification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from agent_core.api import create_app
from agent_core.sandbox import SandboxManager
from agent_core.skills import SkillRegistry
from tests.conftest import ScriptedChatModel
from tests.test_hitl_guardrails import _make_tracked_email_tool

load_dotenv(".env")


def test_fastapi_auth_health_run_resume_and_thread_continuity(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    email_tool, executions = _make_tracked_email_tool()
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Hello from turn 1."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "api-mail-1",
                        "name": "gmail_send_email",
                        "args": {
                            "to": "team@example.com",
                            "subject": "API Test",
                            "body": "Hello team",
                        },
                    }
                ],
            ),
            AIMessage(content="API email sent after HTTP resume approval."),
        ]
    )

    app = create_app(
        api_token="test-secret-api-token",
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool],
    )

    with TestClient(app) as client:
        # 1. /health reports dependency status cleanly without leaking secrets
        health = client.get("/health")
        assert health.status_code == 200
        h_json = health.json()
        assert h_json["status"] == "ok"
        assert h_json["agent_ready"] is True
        assert h_json["auth_configured"] is True
        assert "test-secret-api-token" not in json.dumps(h_json)

        # 2. Unauthenticated & wrong-token requests must be rejected with 401
        unauth = client.post("/agent/run", json={"task": "Hi"})
        assert unauth.status_code == 401
        wrong = client.post(
            "/agent/run",
            headers={"Authorization": "Bearer wrong-token"},
            json={"task": "Hi"},
        )
        assert wrong.status_code == 401

        auth_headers = {"Authorization": "Bearer test-secret-api-token"}

        # 3. Malformed/blank task rejected with 422
        blank = client.post("/agent/run", headers=auth_headers, json={"task": "   "})
        assert blank.status_code == 422

        # 4. Authenticated /agent/run works and preserves thread_id
        r1 = client.post(
            "/agent/run",
            headers=auth_headers,
            json={"task": "Start session", "thread_id": "api-thread-1"},
        )
        assert r1.status_code == 200
        assert r1.json()["thread_id"] == "api-thread-1"
        assert r1.json()["final_response"] == "Hello from turn 1."

        # 5. Follow-up on same thread triggers HITL approval -> resume over HTTP
        r2 = client.post(
            "/agent/run",
            headers=auth_headers,
            json={"task": "Send API email", "thread_id": "api-thread-1"},
        )
        assert r2.status_code == 200
        assert r2.json()["status"] == "awaiting_approval"
        assert len(executions) == 0

        r3 = client.post(
            "/agent/resume",
            headers=auth_headers,
            json={"thread_id": "api-thread-1", "decision": "approve"},
        )
        assert r3.status_code == 200
        assert r3.json()["status"] == "completed"
        assert len(executions) == 1


def test_fastapi_dynamic_model_override_and_presets(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify dynamic per-request primary & sub-agent model overrides and /models/presets."""
    seen_model_specs: list[tuple[str, str, str]] = []
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="Dynamic model run succeeded."),
        ]
    )

    def _override_resolver(role: str, spec: Any) -> Any:
        seen_model_specs.append((role, spec.provider, spec.model))
        return model

    app = create_app(
        api_token="override-token",
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        model_override_resolver=_override_resolver,
    )

    with TestClient(app) as client:
        presets = client.get("/models/presets")
        assert presets.status_code == 200
        p_json = presets.json()
        assert "openrouter" in p_json["providers"]
        assert "openai" in p_json["providers"]

        run = client.post(
            "/agent/run",
            headers={"Authorization": "Bearer override-token"},
            json={
                "task": "Run with custom models",
                "thread_id": "dyn-thread-1",
                "primary_model": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-3.7-sonnet",
                },
                "worker_model": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                },
            },
        )
        assert run.status_code == 200
        body = run.json()
        assert body["status"] == "completed"
        assert body["active_models"]["primary"]["provider"] == "openrouter"
        assert body["active_models"]["worker"]["provider"] == "openai"
        assert ("primary", "openrouter", "anthropic/claude-3.7-sonnet") in seen_model_specs
        assert ("worker", "openai", "gpt-4o-mini") in seen_model_specs


def test_cli_check_command_runs_cleanly() -> None:
    """Verify `python Agent-Core/main.py --check` reports real environment readiness JSON cleanly."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"Agent-Core{os.pathsep}."
    proc = subprocess.run(
        [sys.executable, "Agent-Core/main.py", "--check"],
        cwd=str(Path.cwd()),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert "primary_model_probe" in data
    assert "sandbox" in data
    assert "skills" in data
    assert "mcps" in data
    assert "deep_research" in data["skills"]
    assert "code_and_data" in data["skills"]
