"""Failure Injection, Concurrency, and Pure-Python MCP Degradation Tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from MCPs.manager import MCPRegistryManager, MCPServerConfig

from agent_core.graph import create_agent_runtime
from agent_core.sandbox import SandboxConfig, SandboxManager
from agent_core.skills import SkillRegistry
from tests.conftest import ScriptedChatModel


@pytest.mark.asyncio
async def test_failure_injection_model_unavailable_empty_response_and_mcp_offline(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify safe degradation when model fails, returns empty content, or MCP server is offline."""
    # 1. Persistent connection error across all retry attempts -> handled cleanly without unhandled crash
    monkeypatch.setenv("MODEL_MAX_RETRIES", "0")
    broken_model = ScriptedChatModel(
        handler=lambda _msgs, _idx: ConnectionError("River AI unreachable"),
    )
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

    # 3. Local execution remains fully functional while an external MCP server is offline
    offline_mcp = MCPRegistryManager(
        servers=[
            MCPServerConfig(
                name="offline_ext",
                url="https://offline-mcp.example.invalid/mcp",
                transport="streamable_http",
                timeout_seconds=2.0,
            )
        ]
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
            AIMessage(content="Local tool succeeded even though external MCP is offline."),
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
        res = await runtime.run_task("Write local file", thread_id="mcp-offline-local-ok")
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
