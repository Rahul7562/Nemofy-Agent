"""Pure-Python Multi-Server MCP Manager.

Loads external Model Context Protocol servers (Streamable HTTP or SSE) from
``MCPs/servers.json`` or per-request configuration, discovers LangChain tools
dynamically via ``langchain-mcp-adapters``, and surfaces honest health metadata
with pure-Python async transport support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG_PATH = Path(os.getenv("MCP_CONFIG_PATH", "./MCPs/servers.json"))


class MCPServerConfig(BaseModel):
    """Configuration for a direct pure-Python MCP server connection."""

    name: str = Field(..., min_length=1, max_length=64)
    url: str = Field(..., min_length=1, max_length=1024)
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    enabled: bool = True
    bearer_token: str | None = Field(default=None, max_length=1024)
    headers: dict[str, str] = Field(default_factory=dict)
    include_tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    timeout_seconds: float = Field(default=30.0, ge=2.0, le=180.0)


def load_mcp_servers_config(config_path: Path | str | None = None) -> list[MCPServerConfig]:
    """Read configured MCP servers from ``MCPs/servers.json``."""
    target = Path(config_path) if config_path is not None else DEFAULT_MCP_CONFIG_PATH
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        entries = raw.get("servers", []) if isinstance(raw, dict) else []
        return [MCPServerConfig.model_validate(item) for item in entries if isinstance(item, dict)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse MCP config at %s: %s", target, exc)
        return []


def save_mcp_servers_config(
    servers: list[MCPServerConfig],
    config_path: Path | str | None = None,
) -> Path:
    """Persist MCP server definitions cleanly to disk."""
    target = Path(config_path) if config_path is not None else DEFAULT_MCP_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"servers": [s.model_dump() for s in servers]}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


async def load_mcp_tools_from_configs(
    servers: list[MCPServerConfig] | None,
) -> tuple[dict[str, BaseTool], set[str], list[str]]:
    """Connect enabled external MCP servers and return ``(tools_map, mcp_names, warnings)``."""
    if not servers:
        return {}, set(), []

    tools_map: dict[str, BaseTool] = {}
    mcp_names: set[str] = set()
    warnings: list[str] = []

    for srv in servers:
        if not srv.enabled:
            continue
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
            warnings.append(f"MCP server '{srv.name}' unavailable: {err_msg}")
            logger.warning("MCP server '%s' connection failed: %s", srv.name, err_msg)

    return tools_map, mcp_names, warnings


class MCPRegistryManager:
    """Manages configured pure-Python MCP server connections for Agent-Core."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        *,
        servers: list[MCPServerConfig] | None = None,
    ) -> None:
        self.config_path = Path(config_path) if config_path is not None else DEFAULT_MCP_CONFIG_PATH
        self._inline_servers: list[MCPServerConfig] | None = list(servers) if servers is not None else None
        self._tools: list[BaseTool] = []
        self._warnings: list[str] = []
        self._connected: bool = False

    def list_servers(self) -> list[MCPServerConfig]:
        if self._inline_servers is not None:
            return list(self._inline_servers)
        return load_mcp_servers_config(self.config_path)

    async def connect(self) -> list[BaseTool]:
        servers = self.list_servers()
        enabled = [s for s in servers if s.enabled]
        if not enabled:
            self._tools = []
            self._warnings = []
            self._connected = False
            return []

        tools_map, _names, warnings = await load_mcp_tools_from_configs(enabled)
        self._tools = list(tools_map.values())
        self._warnings = warnings
        self._connected = len(self._tools) > 0
        return list(self._tools)

    def health_status(self) -> dict[str, Any]:
        servers = self.list_servers()
        return {
            "config_path": str(self.config_path),
            "configured_servers": len(servers),
            "enabled_servers": sum(1 for s in servers if s.enabled),
            "connected": self._connected,
            "tool_count": len(self._tools),
            "tool_names": [t.name for t in self._tools],
            "warnings": list(self._warnings),
        }
