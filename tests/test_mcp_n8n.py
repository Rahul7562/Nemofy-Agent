"""Phase 11G: n8n MCP Integration Verification (Live Streamable HTTP + Caching + Offline Handling)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from dotenv import load_dotenv
from src.tools.mcp_n8n import N8nMCPConfig, N8nMCPManager

load_dotenv(".env")


@pytest.mark.asyncio
async def test_live_n8n_mcp_discovery_caching_allowlist_and_readonly_invocation() -> None:
    """Connect to the REAL running n8n MCP server (http://127.0.0.1:5678/mcp-server/http)."""
    token = os.getenv("N8N_MCP_TOKEN", "").strip()
    url = os.getenv("N8N_MCP_URL", "http://127.0.0.1:5678/mcp-server/http").strip()
    assert token, "N8N_MCP_TOKEN must be provisioned in .env for live n8n MCP verification."

    # 1. Full discovery & startup schema caching
    cfg = N8nMCPConfig(url=url, token=token, timeout_seconds=20.0)
    mgr = N8nMCPManager(cfg)
    tools = await mgr.connect()

    health = mgr.health_status()
    assert health["connected"] is True
    assert health["discovered_tools_count"] >= 1
    assert len(tools) == health["exposed_tools_count"]
    assert "search_workflows" in mgr.get_tool_names()

    # Verify schemas are cached and structurally valid
    schemas = mgr.get_cached_schemas()
    assert "search_workflows" in schemas
    assert schemas["search_workflows"]["name"] == "search_workflows"

    # Calling connect() again without force_refresh returns cached list cleanly
    cached = await mgr.connect(force_refresh=False)
    assert len(cached) == len(tools)

    # 2. Harmless read-only live invocation (`search_workflows`)
    search_tool = next(t for t in tools if t.name == "search_workflows")
    live_res = await search_tool.ainvoke({})
    live_text = str(live_res)
    assert "dsV41MultiAgent01" in live_text
    assert "lgAgentGateway01" in live_text

    # 3. Allowlist filtering (`ALLOWED_N8N_TOOLS`)
    filtered_cfg = N8nMCPConfig(
        url=url,
        token=token,
        allowed_tools=frozenset({"search_workflows", "get_workflow_details"}),
        timeout_seconds=20.0,
    )
    filtered_mgr = N8nMCPManager(filtered_cfg)
    filtered_tools = await filtered_mgr.connect()
    assert {t.name for t in filtered_tools} == {"search_workflows", "get_workflow_details"}
    assert filtered_mgr.health_status()["discovered_tools_count"] > 2
    assert filtered_mgr.health_status()["exposed_tools_count"] == 2


@pytest.mark.asyncio
async def test_n8n_mcp_invalid_token_and_offline_graceful_handling() -> None:
    """Verify invalid Bearer auth, unreachable endpoint, and timeouts degrade gracefully."""
    # 1. Invalid token against real n8n MCP endpoint
    bad_auth = N8nMCPManager(
        N8nMCPConfig(
            url="http://127.0.0.1:5678/mcp-server/http",
            token="invalid.jwt.token",
            timeout_seconds=5.0,
            max_reconnect_attempts=1,
        )
    )
    tools = await bad_auth.connect()
    assert tools == []
    assert bad_auth.health_status()["connected"] is False
    assert bad_auth.health_status()["last_error"] is not None
    assert "invalid.jwt.token" not in str(bad_auth.health_status()["last_error"])

    # 2. Unreachable offline n8n port
    offline = N8nMCPManager(
        N8nMCPConfig(
            url="http://127.0.0.1:19999/mcp-server/http",
            token="unused",
            timeout_seconds=2.0,
            max_reconnect_attempts=1,
        )
    )
    offline_tools = await offline.connect()
    assert offline_tools == []
    assert offline.health_status()["connected"] is False


@pytest.mark.asyncio
async def test_n8n_mcp_timeout_and_reconnection_recovery() -> None:
    """Verify timeout handling and reconnect() succeeds once the MCP server recovers."""
    state = {"attempt": 0}

    class FlakyClient:
        async def get_tools(self) -> list[Any]:
            state["attempt"] += 1
            if state["attempt"] == 1:
                await asyncio.sleep(5)  # triggers timeout
            return []

    mgr = N8nMCPManager(
        N8nMCPConfig(
            url="http://n8n:5678/mcp-server/http",
            token="secret-token-xyz",
            timeout_seconds=0.1,
            max_reconnect_attempts=1,
            reconnect_backoff_seconds=0.05,
        ),
        client_factory=lambda _conn: FlakyClient(),
    )

    first = await mgr.connect()
    assert first == []
    assert mgr.health_status()["connected"] is False
    assert "Timed out" in str(mgr.health_status()["last_error"])

    # Reconnect recovers on second call
    second = await mgr.reconnect()
    assert second == []
    assert mgr.health_status()["connected"] is True
