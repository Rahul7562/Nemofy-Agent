"""Reusable configuration primitives for Agent-Core LangGraph runtimes.

Provides clean, typed, decoupled configuration objects so ``agent_core`` can be
embedded into any Python application (not only Nemofy Agent):
- ``AgentPersona``: customize agent identity, system prompt, or custom prompt builder.
- ``ExecutionLimits``: bounded turns, tool retries, compaction thresholds, and wall-clock caps.
- ``AgentHooks``: optional lifecycle callbacks (sync or async, defensively isolated).
- ``AgentConfig``: top-level runtime configuration combining persona, limits, guardrails,
  feature toggles (local tools, web tools, skills, sub-agents, workspace diffs), and optional
  custom LangGraph checkpointer injection.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_core.context_manager import AUTO_COMPACT_THRESHOLD
from agent_core.guardrails import (
    DEFAULT_GUARDRAIL_POLICY,
    ExecutionAccessMode,
    GuardrailPolicy,
)
from agent_core.state import (
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_MAX_TOOL_RETRIES,
    DEFAULT_MAX_TURNS,
)

logger = logging.getLogger(__name__)

PromptBuilderFn = Callable[..., str]
HookFn = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class AgentPersona:
    """Identity and prompt configuration for the LangGraph supervisor."""

    name: str = "Nemo"
    organization: str = "Nemofy Agent"
    role_description: str = (
        "You are diligent, honest, warm, and friendly, capable of helping with everything "
        "from quick everyday questions and writing to complex multi-step research, data "
        "analysis, coding, and workspace automation."
    )
    workspace_label: str = "/workspace"
    extra_instructions: str = ""
    custom_system_prompt: str | None = None
    system_prompt_builder: PromptBuilderFn | None = None
    is_default_nemofy: bool = False


def default_nemofy_persona() -> AgentPersona:
    """Return the default Nemofy Agent persona preserving exact production prompt copy."""
    return AgentPersona(is_default_nemofy=True)


@dataclass(frozen=True)
class ExecutionLimits:
    """Hard safety and efficiency limits enforced during LangGraph execution."""

    max_turns: int = DEFAULT_MAX_TURNS
    max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES
    tool_output_compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD
    auto_compact_threshold: float = AUTO_COMPACT_THRESHOLD
    max_wall_clock_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}.")
        if self.max_tool_retries < 0:
            raise ValueError(f"max_tool_retries must be >= 0, got {self.max_tool_retries}.")
        if self.tool_output_compaction_threshold < 256:
            raise ValueError(
                "tool_output_compaction_threshold must be >= 256, "
                f"got {self.tool_output_compaction_threshold}."
            )
        if not (0.10 <= self.auto_compact_threshold <= 0.99):
            raise ValueError(
                f"auto_compact_threshold must be in [0.10, 0.99], got {self.auto_compact_threshold}."
            )
        if self.max_wall_clock_seconds is not None and self.max_wall_clock_seconds <= 0:
            raise ValueError(
                f"max_wall_clock_seconds must be > 0 when set, got {self.max_wall_clock_seconds}."
            )


@dataclass(frozen=True)
class AgentHooks:
    """Optional lifecycle callbacks invoked defensively during supervisor execution.

    Callbacks may be synchronous or async callables accepting a single event ``dict``.
    Exceptions raised inside hooks are logged and never crash an agent run.
    """

    on_turn_start: HookFn | None = None
    on_tool_start: HookFn | None = None
    on_tool_end: HookFn | None = None
    on_compaction: HookFn | None = None
    on_run_complete: HookFn | None = None


async def invoke_agent_hook(hook: HookFn | None, event: dict[str, Any]) -> None:
    """Invoke a sync or async AgentHooks callback defensively."""
    if hook is None:
        return
    try:
        res = hook(event)
        if inspect.isawaitable(res):
            await res
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent hook %r raised an error (ignored): %s", hook, exc)


@dataclass(frozen=True)
class AgentConfig:
    """Top-level configuration for creating a generalized Agent-Core LangGraph runtime."""

    persona: AgentPersona = field(default_factory=default_nemofy_persona)
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    guardrail_policy: GuardrailPolicy = DEFAULT_GUARDRAIL_POLICY
    hooks: AgentHooks = field(default_factory=AgentHooks)
    default_access_mode: ExecutionAccessMode | None = None
    include_default_tools: bool = True
    enable_web_tools: bool = True
    enable_skills: bool = True
    enable_subagents: bool = True
    enable_workspace_diff: bool = True
    enable_auto_compaction: bool = True
    custom_checkpointer: BaseCheckpointSaver[Any] | None = None
    app_title: str = "Nemofy Agent-Core API"
    app_description: str = "Pure-Python LangGraph Autonomous AI Agent API (Sandbox + Skills + MCPs)."
