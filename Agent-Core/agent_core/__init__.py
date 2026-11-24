"""Pure-Python LangGraph Autonomous AI Agent Core.

Public library facade for embedding or extending the LangGraph supervisor engine,
sandbox isolation, human-in-the-loop guardrails, skills, sub-agents, context management,
and FastAPI service in any Python application.
"""

from __future__ import annotations

from agent_core.api import create_app
from agent_core.config import (
    AgentConfig,
    AgentHooks,
    AgentPersona,
    ExecutionLimits,
    default_nemofy_persona,
)
from agent_core.graph import (
    AgentRuntime,
    ModelOverrideResolver,
    build_supervisor_graph,
    create_agent_runtime,
)
from agent_core.guardrails import (
    DEFAULT_GUARDRAIL_POLICY,
    ApprovalDecision,
    ExecutionAccessMode,
    GuardrailPolicy,
    SafetyLevel,
    ToolClassification,
    classify_tool_call,
    execute_tool_with_guardrails,
)
from agent_core.models import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelOverride,
    ModelProviderError,
    ModelRateLimitError,
    ModelSpec,
    build_chat_model,
    load_model_spec,
)
from agent_core.sandbox import (
    ExecResult,
    SandboxConfig,
    SandboxExecutionError,
    SandboxManager,
    SandboxSecurityError,
    SandboxTimeoutError,
)
from agent_core.skills import DynamicSkill, SkillMetadata, SkillRegistry
from agent_core.state import AgentState, PlanStep, make_initial_state
from agent_core.subagents import SubagentReport, make_spawn_subagent_tool, run_isolated_subagent

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_GUARDRAIL_POLICY",
    "AgentConfig",
    "AgentHooks",
    "AgentPersona",
    "AgentRuntime",
    "AgentState",
    "ApprovalDecision",
    "DynamicSkill",
    "ExecResult",
    "ExecutionAccessMode",
    "ExecutionLimits",
    "GuardrailPolicy",
    "ModelAuthenticationError",
    "ModelConfigurationError",
    "ModelConnectionError",
    "ModelOverride",
    "ModelOverrideResolver",
    "ModelProviderError",
    "ModelRateLimitError",
    "ModelSpec",
    "PlanStep",
    "SafetyLevel",
    "SandboxConfig",
    "SandboxExecutionError",
    "SandboxManager",
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "SkillMetadata",
    "SkillRegistry",
    "SubagentReport",
    "ToolClassification",
    "__version__",
    "build_chat_model",
    "build_supervisor_graph",
    "classify_tool_call",
    "create_agent_runtime",
    "create_app",
    "default_nemofy_persona",
    "execute_tool_with_guardrails",
    "load_model_spec",
    "make_initial_state",
    "make_spawn_subagent_tool",
    "run_isolated_subagent",
]
