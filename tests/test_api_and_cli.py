"""Phase 11I & Phase 8: Authenticated FastAPI, CLI, and Live n8n Gateway End-to-End Verification."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from src.api import create_app
from src.sandbox import SandboxManager
from src.skills import SkillRegistry
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


def test_cli_check_command_runs_cleanly() -> None:
    """Verify `python main.py --check` reports real environment readiness JSON cleanly."""
    proc = subprocess.run(
        [sys.executable, "main.py", "--check"],
        cwd=str(Path.cwd()),
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
    assert "n8n_mcp" in data
    assert "deep_research" in data["skills"]
    assert "code_and_data" in data["skills"]


@pytest.mark.asyncio
async def test_live_n8n_gateway_workflow_end_to_end(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """REAL end-to-end test: HTTP -> n8n Gateway Webhook -> live Python FastAPI (:8100) -> LangGraph -> n8n Response."""
    agent_token = os.getenv("AGENT_API_TOKEN", "").strip()
    assert agent_token, "AGENT_API_TOKEN must be present in .env for n8n Gateway E2E verification."

    # Verify gateway workflow exists and is active in n8n SQLite database
    sqlite_path = Path.cwd() / ".n8n-data/.n8n/database.sqlite"
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT id, name, active FROM workflow_entity WHERE id = 'lgAgentGateway01'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[1] == "LangGraph Autonomous AI Agent Gateway"
    assert row[2] == 1

    # Spin up live FastAPI server on 127.0.0.1:8100 (exact URL targeted by n8n Gateway workflow)
    turn_log: list[int] = []

    def gw_handler(msgs: list, idx: int) -> AIMessage:
        turn_log.append(len(msgs))
        if idx == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "gw-write-1",
                        "name": "write_file",
                        "args": {"path": "/workspace/from_gateway.txt", "content": "gateway-ok"},
                    }
                ],
            )
        if idx == 1:
            return AIMessage(content="Gateway task 1 completed: wrote /workspace/from_gateway.txt.")
        return AIMessage(
            content=f"Gateway follow-up resumed same thread cleanly (history_len={len(msgs)})."
        )

    seen_model_specs: list[tuple[str, str, str]] = []

    def _test_override_resolver(role: str, spec: Any) -> Any:
        seen_model_specs.append((role, spec.provider, spec.model))
        return gw_model

    gw_model = ScriptedChatModel(handler=gw_handler)
    gw_app = create_app(
        api_token=agent_token,
        primary_model=gw_model,
        worker_model=gw_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=True,
        model_override_resolver=_test_override_resolver,
    )

    server_config = uvicorn.Config(gw_app, host="127.0.0.1", port=8100, log_level="warning")
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        # Wait until FastAPI :8100 is accepting requests
        ready = False
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _ in range(40):
                try:
                    r = await client.get("http://127.0.0.1:8100/health")
                    if r.status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        assert ready, "Live FastAPI server on :8100 did not become ready."

        webhook_url = "http://127.0.0.1:5678/webhook/langgraph-agent/run"
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 0. /models/presets exposes multi-provider presets without leaking secrets
            presets_resp = await client.get("http://127.0.0.1:8100/models/presets")
            assert presets_resp.status_code == 200
            presets_json = presets_resp.json()
            assert "openrouter" in presets_json["providers"]
            assert "openai" in presets_json["providers"]

            # 1. Input validation check via real n8n Gateway
            bad_resp = await client.post(webhook_url, json={"task": ""})
            assert bad_resp.status_code == 200
            bad_body = bad_resp.json()
            assert bad_body["ok"] is False
            assert bad_body["status"] == "validation_error"

            # 2. Valid task using n8n LLM Model Selector defaults -> hits Python API -> LangGraph -> local tool
            run1 = await client.post(
                webhook_url,
                json={"task": "Create /workspace/from_gateway.txt"},
            )
            assert run1.status_code == 200, run1.text
            b1 = run1.json()
            assert b1["ok"] is True
            assert b1["status"] == "completed"
            assert b1["thread_id"].startswith("n8n-")
            assert "from_gateway.txt" in b1["final_response"]
            assert isolated_sandbox.read_file("/workspace/from_gateway.txt") == "gateway-ok"
            assert b1["active_models"]["primary"]["provider"] == "river_ai"
            assert ("primary", "river_ai", "deepseek-ai/DeepSeek-V4.1-Flash") in seen_model_specs
            assert ("worker", "river_ai", "deepseek-ai/DeepSeek-V4.1-Flash") in seen_model_specs

            # 3. Follow-up request dynamically shifting Brain to OpenRouter & Sub-Agent to OpenAI via n8n!
            run2 = await client.post(
                webhook_url,
                json={
                    "task": "/brain openrouter:anthropic/claude-3.7-sonnet /subagent openai:gpt-4o-mini Follow up on the previous file",
                    "thread_id": b1["thread_id"],
                },
            )
            assert run2.status_code == 200, run2.text
            b2 = run2.json()
            assert b2["ok"] is True
            assert b2["thread_id"] == b1["thread_id"]
            assert "resumed same thread" in b2["final_response"].lower()
            assert b2["active_models"]["primary"]["provider"] == "openrouter"
            assert b2["active_models"]["primary"]["model"] == "anthropic/claude-3.7-sonnet"
            assert b2["active_models"]["worker"]["provider"] == "openai"
            assert b2["active_models"]["worker"]["model"] == "gpt-4o-mini"
            assert ("primary", "openrouter", "anthropic/claude-3.7-sonnet") in seen_model_specs
            assert ("worker", "openai", "gpt-4o-mini") in seen_model_specs
            # Conversation history grew across the two n8n webhook calls
            assert turn_log[-1] > turn_log[0]
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        await asyncio.sleep(0.2)

    # 4. API offline error handling check: when :8100 is stopped, n8n Gateway returns structured error cleanly
    async with httpx.AsyncClient(timeout=15.0) as client:
        err_resp = await client.post(
            "http://127.0.0.1:5678/webhook/langgraph-agent/run",
            json={"task": "Ping while Python API is offline", "thread_id": "n8n-err-test"},
        )
        assert err_resp.status_code == 200
        eb = err_resp.json()
        assert eb["ok"] is False
        assert eb["status"] == "error"
        assert eb["thread_id"] == "n8n-err-test"
        assert eb["error"]


def test_gateway_workflow_uses_native_openai_chat_model_nodes() -> None:
    """Verify Gateway workflow exposes two native OpenAI Chat Model nodes and zero n8n AI Agent nodes."""
    from src.tools.n8n_gateway_sync import RIVER_OPENAI_CREDENTIAL_ID, build_gateway_workflow

    wf = build_gateway_workflow(agent_api_base_url="http://127.0.0.1:8100")
    nodes_by_name = {node["name"]: node for node in wf["nodes"]}

    brain_node = nodes_by_name["Brain LLM (OpenAI Chat Model)"]
    subagent_node = nodes_by_name["Sub-Agent LLM (OpenAI Chat Model)"]

    for node in (brain_node, subagent_node):
        assert node["type"] == "@n8n/n8n-nodes-langchain.lmChatOpenAi"
        assert node["parameters"]["model"]["mode"] == "list"
        assert node["credentials"]["openAiApi"]["id"] == RIVER_OPENAI_CREDENTIAL_ID

    assert nodes_by_name["Read Brain LLM"]["type"] == "@n8n/n8n-nodes-langchain.code"
    assert nodes_by_name["Read Sub-Agent LLM"]["type"] == "@n8n/n8n-nodes-langchain.code"

    conns = wf["connections"]
    assert (
        conns["Brain LLM (OpenAI Chat Model)"]["ai_languageModel"][0][0]["node"] == "Read Brain LLM"
    )
    assert (
        conns["Sub-Agent LLM (OpenAI Chat Model)"]["ai_languageModel"][0][0]["node"]
        == "Read Sub-Agent LLM"
    )

    # Strict 3-system separation: no LLM Agent or Chain runs inside n8n
    node_types = {node["type"] for node in wf["nodes"]}
    assert "@n8n/n8n-nodes-langchain.agent" not in node_types
    assert "@n8n/n8n-nodes-langchain.chainLlm" not in node_types
