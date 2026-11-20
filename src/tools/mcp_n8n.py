"""Phase 5: n8n MCP SaaS Integration (Streamable HTTP + MultiServerMCPClient).

Connects the LangGraph supervisor to the existing n8n instance's MCP server using:
- Streamable HTTP transport (`transport="streamable_http"`)
- `langchain-mcp-adapters` (`MultiServerMCPClient`)
- Bearer-token authentication (`N8N_MCP_TOKEN`)
- Startup tool discovery & schema caching
- Optional `ALLOWED_N8N_TOOLS` allowlist filtering
- Timeouts, reconnection, and graceful offline degradation (preserving local tools)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.models import mask_secret

logger = logging.getLogger(__name__)

DEFAULT_N8N_MCP_URL = "http://localhost:5678/mcp-server/http"
DEFAULT_N8N_MCP_TIMEOUT = 30.0


def parse_allowed_tools(raw: str | None) -> frozenset[str] | None:
    """Parse comma-separated ALLOWED_N8N_TOOLS into a normalized allowlist (or None if unset)."""
    if raw is None or not raw.strip():
        return None
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return frozenset(items) if items else None


@dataclass(frozen=True)
class N8nMCPConfig:
    """Configuration for connecting to the n8n Streamable HTTP MCP server."""

    url: str = DEFAULT_N8N_MCP_URL
    token: str | None = None
    allowed_tools: frozenset[str] | None = None
    timeout_seconds: float = DEFAULT_N8N_MCP_TIMEOUT
    max_reconnect_attempts: int = 2
    reconnect_backoff_seconds: float = 1.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> N8nMCPConfig:
        src = dict(os.environ) if env is None else env
        url = (src.get("N8N_MCP_URL") or DEFAULT_N8N_MCP_URL).strip()
        token = (src.get("N8N_MCP_TOKEN") or "").strip() or None
        allowed = parse_allowed_tools(src.get("ALLOWED_N8N_TOOLS"))
        try:
            timeout = max(2.0, float(src.get("N8N_MCP_TIMEOUT_SECONDS") or DEFAULT_N8N_MCP_TIMEOUT))
        except ValueError:
            timeout = DEFAULT_N8N_MCP_TIMEOUT
        return cls(
            url=url,
            token=token,
            allowed_tools=allowed,
            timeout_seconds=timeout,
        )

    def __repr__(self) -> str:
        return (
            f"N8nMCPConfig(url={self.url!r}, token={mask_secret(self.token)!r}, "
            f"allowed_tools={sorted(self.allowed_tools) if self.allowed_tools else None!r}, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )


def extract_tool_schema(tool_obj: BaseTool) -> dict[str, Any]:
    """Extract a JSON-serializable schema snapshot from a discovered MCP BaseTool."""
    args_schema = getattr(tool_obj, "args_schema", None)
    schema_dict: dict[str, Any]
    if isinstance(args_schema, dict):
        schema_dict = args_schema
    elif args_schema is not None and hasattr(args_schema, "model_json_schema"):
        schema_dict = args_schema.model_json_schema()
    else:
        schema_dict = {}

    return {
        "name": tool_obj.name,
        "description": getattr(tool_obj, "description", "") or "",
        "args_schema": schema_dict,
    }


def filter_tools_by_allowlist(
    tools: list[BaseTool],
    allowed_tools: frozenset[str] | None,
) -> list[BaseTool]:
    """Filter discovered MCP tools using the optional allowlist."""
    if allowed_tools is None:
        return list(tools)
    return [t for t in tools if t.name in allowed_tools]


@dataclass
class N8nMCPState:
    """Observable runtime state of the n8n MCP client."""

    connected: bool = False
    discovered_count: int = 0
    exposed_count: int = 0
    last_connected_at: str | None = None
    last_error: str | None = None
    cached_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)


class N8nMCPManager:
    """Manages Streamable HTTP MCP connectivity, tool caching, and offline fallback for n8n."""

    def __init__(
        self,
        config: N8nMCPConfig | None = None,
        *,
        client_factory: Any | None = None,
    ) -> None:
        self.config = config or N8nMCPConfig.from_env()
        self._client_factory = client_factory
        self._client: MultiServerMCPClient | None = None
        self._all_tools: list[BaseTool] = []
        self._exposed_tools: list[BaseTool] = []
        self._state = N8nMCPState()
        self._lock = asyncio.Lock()

    def build_connection_dict(self) -> dict[str, Any]:
        """Build the MultiServerMCPClient Streamable HTTP connection dictionary."""
        headers: dict[str, str] = {"Accept": "application/json, text/event-stream"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        return {
            "n8n": {
                "transport": "streamable_http",
                "url": self.config.url,
                "headers": headers,
                "timeout": timedelta(seconds=self.config.timeout_seconds),
                "sse_read_timeout": timedelta(seconds=max(self.config.timeout_seconds * 2, 60.0)),
            }
        }

    def _create_client(self) -> Any:
        connections = self.build_connection_dict()
        if self._client_factory is not None:
            return self._client_factory(connections)
        return MultiServerMCPClient(connections)

    async def connect(self, *, force_refresh: bool = False) -> list[BaseTool]:
        """Connect to n8n MCP, discover tools, cache schemas, and apply allowlist filtering.

        Never raises on n8n offline/auth failures; instead records the error cleanly and
        returns an empty list so local agent functionality remains available.
        """
        async with self._lock:
            if self._state.connected and self._exposed_tools and not force_refresh:
                return list(self._exposed_tools)

            if not self.config.url:
                self._state.connected = False
                self._state.last_error = "N8N_MCP_URL is not configured."
                self._all_tools = []
                self._exposed_tools = []
                return []

            last_exc: Exception | None = None
            attempts = max(1, self.config.max_reconnect_attempts)

            for attempt in range(1, attempts + 1):
                try:
                    client = self._create_client()
                    discovered = await asyncio.wait_for(
                        client.get_tools(),
                        timeout=self.config.timeout_seconds,
                    )
                    if not isinstance(discovered, list):
                        raise ValueError("MCP client returned invalid tool list payload.")

                    self._client = client
                    self._all_tools = list(discovered)
                    filtered = filter_tools_by_allowlist(
                        self._all_tools,
                        self.config.allowed_tools,
                    )
                    self._exposed_tools = filtered
                    self._state.connected = True
                    self._state.discovered_count = len(self._all_tools)
                    self._state.exposed_count = len(self._exposed_tools)
                    self._state.last_connected_at = datetime.now(UTC).isoformat()
                    self._state.last_error = None
                    self._state.cached_schemas = {
                        t.name: extract_tool_schema(t) for t in self._exposed_tools
                    }
                    logger.info(
                        "Connected to n8n MCP at %s (discovered=%d, exposed=%d).",
                        self.config.url,
                        self._state.discovered_count,
                        self._state.exposed_count,
                    )
                    return list(self._exposed_tools)
                except TimeoutError as exc:
                    last_exc = TimeoutError(
                        f"Timed out connecting to n8n MCP at {self.config.url} "
                        f"after {self.config.timeout_seconds}s"
                    )
                    logger.warning("n8n MCP attempt %d/%d timed out: %s", attempt, attempts, exc)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    safe_msg = str(exc)
                    if self.config.token:
                        safe_msg = safe_msg.replace(self.config.token, "[REDACTED]")
                    logger.warning("n8n MCP attempt %d/%d failed: %s", attempt, attempts, safe_msg)

                if attempt < attempts:
                    await asyncio.sleep(self.config.reconnect_backoff_seconds * attempt)

            # Graceful offline fallback: retain empty active list and record error
            self._state.connected = False
            self._all_tools = []
            self._exposed_tools = []
            err_text = str(last_exc) if last_exc else "Unknown n8n MCP connection failure."
            if self.config.token:
                err_text = err_text.replace(self.config.token, "[REDACTED]")
            self._state.last_error = err_text
            return []

    async def reconnect(self) -> list[BaseTool]:
        """Force reconnection and re-discovery of n8n MCP tools."""
        return await self.connect(force_refresh=True)

    def get_cached_tools(self) -> list[BaseTool]:
        """Return currently cached, allowlist-filtered MCP tools without network I/O."""
        return list(self._exposed_tools)

    def get_cached_schemas(self) -> dict[str, dict[str, Any]]:
        """Return startup-cached JSON schemas for all exposed MCP tools."""
        return dict(self._state.cached_schemas)

    def get_tool_names(self) -> set[str]:
        """Return names of all currently exposed n8n MCP tools."""
        return {t.name for t in self._exposed_tools}

    def health_status(self) -> dict[str, Any]:
        """Return credential-safe n8n MCP diagnostics for /health and CLI status."""
        return {
            "url": self.config.url,
            "token_configured": bool(self.config.token),
            "connected": self._state.connected,
            "discovered_tools_count": self._state.discovered_count,
            "exposed_tools_count": self._state.exposed_count,
            "exposed_tools": sorted(self._state.cached_schemas.keys()),
            "allowlist": sorted(self.config.allowed_tools) if self.config.allowed_tools else None,
            "last_connected_at": self._state.last_connected_at,
            "last_error": self._state.last_error,
        }
