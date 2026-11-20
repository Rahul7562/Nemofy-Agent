"""Phase 11J, 11K & 11L: Failure Injection, Concurrency, and Infrastructure Preservation Tests."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage
from src.graph import create_agent_runtime
from src.sandbox import SandboxConfig, SandboxManager
from src.skills import SkillRegistry
from src.tools.mcp_n8n import N8nMCPConfig, N8nMCPManager
from tests.conftest import ScriptedChatModel


@pytest.mark.asyncio
async def test_failure_injection_model_unavailable_empty_response_and_n8n_offline(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify safe degradation when model fails, returns empty content, or n8n is offline."""
    # 1. Model raises connection error -> handled cleanly without unhandled crash
    broken_model = ScriptedChatModel(responses=[ConnectionError("River AI unreachable")])
    async with create_agent_runtime(
        primary_model=broken_model,
        worker_model=broken_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("Hello", thread_id="fail-model-1")
        assert res["status"] == "error"
        assert "River AI unreachable" in (res["error"] or "")

    # 2. Empty model response handled safely
    empty_model = ScriptedChatModel(responses=[AIMessage(content="")])
    async with create_agent_runtime(
        primary_model=empty_model,
        worker_model=empty_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("Return empty", thread_id="fail-empty-1")
        assert res["status"] == "completed"
        assert res["final_response"]

    # 3. Local execution remains fully functional while n8n MCP is offline
    offline_mcp = N8nMCPManager(
        N8nMCPConfig(
            url="http://127.0.0.1:19998/mcp-server/http",
            token="x",
            timeout_seconds=1.0,
            max_reconnect_attempts=1,
        )
    )
    local_ok_model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-local-offline",
                        "name": "write_file",
                        "args": {"path": "/workspace/offline_ok.txt", "content": "local-works"},
                    }
                ],
            ),
            AIMessage(content="Local tool succeeded even though n8n MCP is offline."),
        ]
    )
    async with create_agent_runtime(
        primary_model=local_ok_model,
        worker_model=local_ok_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        mcp_manager=offline_mcp,
        connect_mcp=True,
    ) as runtime:
        assert offline_mcp.health_status()["connected"] is False
        res = await runtime.run_task("Write local file", thread_id="n8n-offline-local-ok")
        assert res["status"] == "completed"
        assert isolated_sandbox.read_file("/workspace/offline_ok.txt") == "local-works"

    # 4. Unknown tool / repeated invalid tool arguments hit bounded retry limit cleanly
    bad_tool_model = ScriptedChatModel(
        handler=lambda _m, i: (
            AIMessage(
                content="",
                tool_calls=[{"id": f"bad-{i}", "name": "nonexistent_tool", "args": {"x": 1}}],
            )
            if i < 3
            else AIMessage(content="Stopped retrying unknown tool.")
        )
    )
    async with create_agent_runtime(
        primary_model=bad_tool_model,
        worker_model=bad_tool_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        max_tool_retries=2,
    ) as runtime:
        res = await runtime.run_task("Call unknown tool", thread_id="bad-tool-retry")
        assert res["status"] == "completed"


@pytest.mark.asyncio
async def test_concurrent_independent_sessions_and_sqlite_wal_stability(
    tmp_workspace: Path,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify multiple concurrent thread IDs stay isolated across SQLite checkpoints & sandbox."""
    sbx = SandboxManager(
        SandboxConfig(workspace_dir=tmp_workspace, backend="local_isolated", timeout_seconds=10),
        isolate_session_dirs=True,
    )

    def concurrent_handler(msgs: list, _idx: int) -> AIMessage:
        # First turn per thread writes its unique token; second turn responds
        last_content = str(msgs[-1].content)
        if last_content.startswith("Write token "):
            tok = last_content.replace("Write token ", "").strip()
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"tc-{tok}",
                        "name": "write_file",
                        "args": {"path": "/workspace/token.txt", "content": tok},
                    }
                ],
            )
        return AIMessage(content=f"Done: {last_content}")

    model = ScriptedChatModel(handler=concurrent_handler)
    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=sbx,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        tasks = [
            runtime.run_task(f"Write token T{i}", thread_id=f"conc-thread-{i}") for i in range(6)
        ]
        results = await asyncio.gather(*tasks)

    assert len(results) == 6
    for i, res in enumerate(results):
        assert res["status"] == "completed"
        assert res["thread_id"] == f"conc-thread-{i}"
        assert sbx.read_file("/workspace/token.txt", session_id=f"conc-thread-{i}") == f"T{i}"
    sbx.cleanup_all()


@pytest.mark.asyncio
async def test_phase_11l_existing_n8n_infrastructure_preserved() -> None:
    """Verify Phase 0 audit invariants: existing n8n instance, workflow, DB backup, and 5678->3000 mapping."""
    # 1. Proxy mapping 5678 -> 3000 alive & healthy
    async with httpx.AsyncClient(timeout=5.0) as client:
        proxy_status = await client.get("http://127.0.0.1:3000/_n8n_proxy/status")
        assert proxy_status.status_code == 200
        ps = proxy_status.json()
        assert ps["phase"] == "ready"
        assert ps["n8nPort"] == 5678
        assert ps["proxyPort"] == 3000

        n8n_health = await client.get("http://127.0.0.1:5678/healthz")
        assert n8n_health.status_code == 200

    # 2. Original workflow & new gateway workflow both present & active
    db_path = Path.cwd() / ".n8n-data/.n8n/database.sqlite"
    assert db_path.exists()
    backup_path = Path.cwd() / ".n8n-data/backups/database.sqlite.pre-upgrade.bak"
    assert backup_path.exists()

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        workflows = {
            row[0]: (row[1], row[2])
            for row in conn.execute("SELECT id, name, active FROM workflow_entity")
        }
    finally:
        conn.close()

    assert "dsV41MultiAgent01" in workflows
    assert (
        workflows["dsV41MultiAgent01"][0]
        == "Multi-Agent Cognitive Orchestrator (Universal LLM Toggle)"
    )
    assert workflows["dsV41MultiAgent01"][1] == 1

    assert "lgAgentGateway01" in workflows
    assert workflows["lgAgentGateway01"][0] == "LangGraph Autonomous AI Agent Gateway"
    assert workflows["lgAgentGateway01"][1] == 1
