"""Typed AgentState, structured plan schemas, and context compaction.

Provides the canonical LangGraph state definition and the 4,000-character context
compaction pipeline that preserves raw tool outputs under /workspace/.logs/.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

PlanStepStatus = Literal["pending", "in_progress", "completed", "failed", "skipped"]

DEFAULT_MAX_TURNS = int(os.getenv("MAX_TURNS", "30"))
DEFAULT_MAX_TOOL_RETRIES = int(os.getenv("MAX_TOOL_RETRIES", "2"))
DEFAULT_COMPACTION_THRESHOLD = int(os.getenv("CONTEXT_COMPACTION_THRESHOLD", "4000"))


class PlanStep(TypedDict):
    """A single structured step in the supervisor's execution plan."""

    step_id: str
    description: str
    status: PlanStepStatus
    tool_or_action: str
    result: str


class PendingApproval(TypedDict):
    """Metadata describing a sensitive tool invocation awaiting human approval."""

    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    expected_effect: str
    risk_reason: str
    source: str
    edited: bool


class AgentState(TypedDict, total=False):
    """Typed LangGraph state for the cyclic Plan -> Act -> Observe -> Reflect supervisor."""

    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[PlanStep]
    active_skills: list[str]
    turn_count: int
    pending_approval: PendingApproval | None
    thread_id: str
    max_turns: int
    current_step_id: str | None
    reflection: str
    next_action: Literal["continue", "complete"]
    final_response: str
    tool_retry_counts: dict[str, int]
    last_observations: list[dict[str, Any]]
    subagent_reports: list[dict[str, Any]]
    error: str | None


class PlanStepModel(BaseModel):
    """Pydantic validator for structured plan steps parsed from LLM JSON output."""

    step_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: PlanStepStatus = "pending"
    tool_or_action: str = Field(default="direct_response")
    result: str = ""

    def to_typed_dict(self) -> PlanStep:
        return PlanStep(
            step_id=self.step_id,
            description=self.description,
            status=self.status,
            tool_or_action=self.tool_or_action,
            result=self.result,
        )


def _sanitize_filename_token(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip().lower()).strip("_")
    return cleaned[:48] or "tool"


def compact_tool_output(
    tool_name: str,
    raw_output: str,
    *,
    workspace_dir: Path | str | None = None,
    threshold: int | None = None,
) -> tuple[str, str | None]:
    """Compact tool output exceeding the character threshold and archive the raw payload."""
    limit = DEFAULT_COMPACTION_THRESHOLD if threshold is None else threshold
    if len(raw_output) <= limit:
        return raw_output, None

    ws_root = Path(
        workspace_dir if workspace_dir is not None else os.getenv("WORKSPACE_DIR", "./workspace")
    ).resolve()
    logs_dir = ws_root / ".logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    token = _sanitize_filename_token(tool_name)
    unique = uuid.uuid4().hex[:8]
    filename = f"{timestamp}_{token}_{unique}.log"
    disk_path = logs_dir / filename
    disk_path.write_text(raw_output, encoding="utf-8")

    workspace_rel_path = f"/workspace/.logs/{filename}"
    lines = raw_output.splitlines()
    head_chars = min(1200, max(400, limit // 4))
    tail_chars = min(600, max(200, limit // 6))
    head = raw_output[:head_chars].rstrip()
    tail = raw_output[-tail_chars:].lstrip()

    summary = (
        f"[COMPACTED OUTPUT — {len(raw_output):,} chars / {len(lines):,} lines]\n"
        f"Tool: {tool_name}\n"
        f"Full raw output archived at: {workspace_rel_path}\n"
        f"Use read_file('{workspace_rel_path}') to inspect the complete output if needed.\n"
        f"\n--- HEAD PREVIEW ---\n{head}\n"
        f"\n--- TAIL PREVIEW ---\n{tail}\n"
    )
    return summary, workspace_rel_path


def make_initial_state(
    user_messages: list[BaseMessage],
    *,
    thread_id: str,
    max_turns: int | None = None,
) -> AgentState:
    """Create a well-typed initial AgentState for a new turn or session."""
    return AgentState(
        messages=user_messages,
        plan=[],
        active_skills=[],
        turn_count=0,
        pending_approval=None,
        thread_id=thread_id,
        max_turns=DEFAULT_MAX_TURNS if max_turns is None else max_turns,
        current_step_id=None,
        reflection="",
        next_action="continue",
        final_response="",
        tool_retry_counts={},
        last_observations=[],
        subagent_reports=[],
        error=None,
    )
