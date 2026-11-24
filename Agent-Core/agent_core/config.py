"""Generalized, composable configuration primitives for embedding Agent-Core.

This module makes ``agent_core`` cleanly reusable across future LangGraph applications
without hardcoding Nemofy-specific personas, tool policies, or sub-agent roles, while
keeping Nemofy Agent's defaults 100% backward compatible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from agent_core.guardrails import ToolClassification

ExecutionAccessMode = Literal["default", "full_access"]

PromptBuilder = Callable[..., str]
ToolClassifierHook = Callable[[str, dict[str, Any], bool], "ToolClassification | None"]


@dataclass(frozen=True)
class PromptConfig:
    """Customizable system prompt and persona configuration for the supervisor agent."""

    agent_name: str = "Nemo"
    product_name: str = "Nemofy Agent"
    persona: str | None = None
    extra_instructions: str | None = None
    settings_hint: str = "Settings -> LLM Config"
    prompt_builder: PromptBuilder | None = None
    include_workspace_rules: bool = True
    include_firecrawl_rules: bool = True


@dataclass(frozen=True)
class GuardrailPolicy:
    """Composable tool safety and HITL approval policy.

    Callers embedding ``agent_core`` can extend safe/sensitive tool sets, provide a
    custom classifier hook, or override the runtime's default execution access mode
    without mutating process-wide globals.
    """

    extra_safe_tools: frozenset[str] = field(default_factory=frozenset)
    extra_sensitive_tools: frozenset[str] = field(default_factory=frozenset)
    default_access_mode: ExecutionAccessMode | None = None
    custom_classifier: ToolClassifierHook | None = None

    @classmethod
    def create(
        cls,
        *,
        extra_safe_tools: Iterable[str] = (),
        extra_sensitive_tools: Iterable[str] = (),
        default_access_mode: ExecutionAccessMode | None = None,
        custom_classifier: ToolClassifierHook | None = None,
    ) -> GuardrailPolicy:
        return cls(
            extra_safe_tools=frozenset(t.strip().lower() for t in extra_safe_tools if t and t.strip()),
            extra_sensitive_tools=frozenset(
                t.strip().lower() for t in extra_sensitive_tools if t and t.strip()
            ),
            default_access_mode=default_access_mode,
            custom_classifier=custom_classifier,
        )


@dataclass(frozen=True)
class SubagentRoleSpec:
    """Declarative specification for an isolated worker sub-agent role."""

    name: str
    description: str = ""
    default_tools: tuple[str, ...] = ()
    max_turns: int = 8
    instructions: str = ""


class SubagentRegistry:
    """Extensible registry mapping role names to :class:`SubagentRoleSpec` definitions."""

    def __init__(self, roles: dict[str, SubagentRoleSpec] | None = None) -> None:
        self._roles: dict[str, SubagentRoleSpec] = {}
        if roles:
            for key, spec in roles.items():
                self.register(spec, alias=key)

    def register(self, spec: SubagentRoleSpec, *, alias: str | None = None) -> None:
        key = (alias or spec.name).strip().lower()
        if not key:
            raise ValueError("Subagent role name cannot be empty.")
        self._roles[key] = spec

    def get(self, role: str) -> SubagentRoleSpec | None:
        return self._roles.get((role or "").strip().lower())

    def default_tools_for(self, role: str, fallback: Sequence[str] = ()) -> list[str]:
        spec = self.get(role)
        if spec is not None:
            return list(spec.default_tools)
        return list(fallback)

    def list_roles(self) -> list[str]:
        return sorted(self._roles.keys())

    def clone(self) -> SubagentRegistry:
        return SubagentRegistry(dict(self._roles))


@dataclass
class AgentConfig:
    """Unified, framework-level configuration for ``create_agent_runtime``.

    All fields are optional with Nemofy-compatible defaults so downstream projects
    can embed Agent-Core with custom tools, personas, guardrails, sub-agent roles,
    or custom LangGraph checkpointers (including ``":memory:"``).
    """

    prompt: PromptConfig = field(default_factory=PromptConfig)
    guardrails: GuardrailPolicy | None = None
    subagents: SubagentRegistry | None = None
    default_access_mode: ExecutionAccessMode | None = None
    max_turns: int | None = None
    max_tool_retries: int = 3
    compaction_threshold: int = 4000
    include_local_tools: bool = True
    include_skill_tools: bool = True
    include_subagent_tool: bool = True
    snapshot_workspace: bool = True
    extra_tools: list[BaseTool] = field(default_factory=list)
    mcp_tool_names: set[str] = field(default_factory=set)
    checkpoint_db_path: str | Path | None = None
    checkpointer: BaseCheckpointSaver[Any] | None = None
    read_only_tools: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "read_file",
                "list_dir",
                "web_search",
                "web_fetch",
                "web_map",
                "web_crawl",
                "web_extract",
                "load_skill",
                "inspect_media",
            }
        )
    )
