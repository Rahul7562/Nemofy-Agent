"""Central Guardrails and Human-in-the-Loop (HITL) Approval System.

All tool executions—from both the LangGraph supervisor and isolated sub-agents—
must pass through this central safety classifier and approval gate.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import ValidationError

SafetyLevel = Literal["SAFE", "SENSITIVE"]

EXPLICIT_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "web_search",
        "web_fetch",
        "load_skill",
        "write_file",
        "edit_file",
        "spawn_subagent",
    }
)

SENSITIVE_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|_)(?:send|email|mail|gmail)(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:slack|message|notify|dm|post|chat_post)(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:calendar|event|schedule|invite)(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:sheet|sheets|spreadsheet|append|row)(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:webhook|http_post|http_put|http_delete|mutate)(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:delete|remove|drop|purge|destroy|truncate|wipe)(?:_|$)", re.IGNORECASE),
    re.compile(r"(?:^|_)(?:create|update|modify|patch|insert|upsert)(?:_|$)", re.IGNORECASE),
)

DESTRUCTIVE_BASH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\brm\s+[^\n]*-[a-zA-Z]*[rf]", re.IGNORECASE),
        "Recursive or forced file deletion",
    ),
    (re.compile(r"\brm\s+", re.IGNORECASE), "File deletion via rm"),
    (re.compile(r"\b(?:rmdir|shred|unlink)\b", re.IGNORECASE), "Filesystem deletion command"),
    (
        re.compile(r"\b(?:dd|mkfs|fdisk|parted|mount|umount)\b", re.IGNORECASE),
        "Low-level disk operation",
    ),
    (
        re.compile(r"\b(?:shutdown|reboot|poweroff|halt|killall|pkill)\b", re.IGNORECASE),
        "Process/system termination",
    ),
    (re.compile(r"\bkill\s+-9\b", re.IGNORECASE), "Forced process kill"),
    (re.compile(r"\bchmod\s+777\b", re.IGNORECASE), "Overly permissive file permissions"),
    (
        re.compile(r"\bgit\s+(?:push|reset\s+--hard|clean\s+-[a-zA-Z]*f)", re.IGNORECASE),
        "Destructive or external git mutation",
    ),
    (
        re.compile(
            r"\bcurl\b[^\n]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--request\s+(?:POST|PUT|PATCH|DELETE)|-d\s|--data|--upload-file)",
            re.IGNORECASE,
        ),
        "External HTTP mutation via curl",
    ),
    (
        re.compile(
            r"\bwget\b[^\n]*(?:--post-data|--post-file|--method=(?:POST|PUT|PATCH|DELETE))",
            re.IGNORECASE,
        ),
        "External HTTP mutation via wget",
    ),
)


@dataclass(frozen=True)
class ToolClassification:
    tool_name: str
    safety_level: SafetyLevel
    risk_reason: str
    expected_effect: str


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    arguments: dict[str, Any]
    edited: bool = False
    reason: str = ""
    approval_id: str = ""


def _describe_expected_effect(tool_name: str, arguments: dict[str, Any]) -> str:
    lower = tool_name.lower()
    if lower == "bash_exec":
        cmd = str(arguments.get("command", "")).strip()
        preview = cmd[:160] + ("..." if len(cmd) > 160 else "")
        return f"Execute shell command inside sandbox: `{preview}`"
    if any(tok in lower for tok in ("gmail", "email", "mail")):
        to = arguments.get("to") or arguments.get("recipient") or "recipient"
        subject = arguments.get("subject") or "(no subject)"
        return f"Send or modify email to {to!r} with subject {subject!r}."
    if "slack" in lower:
        channel = arguments.get("channel") or arguments.get("channel_id") or "channel"
        return f"Send Slack message to {channel!r}."
    if "calendar" in lower:
        summary = arguments.get("summary") or arguments.get("title") or "calendar event"
        return f"Create or modify calendar event {summary!r}."
    if "sheet" in lower:
        return f"Modify external spreadsheet data via '{tool_name}'."
    if "delete" in lower or "remove" in lower:
        return f"Delete or remove resource via '{tool_name}'."
    return f"Invoke sensitive external or mutating tool '{tool_name}' with provided arguments."


def classify_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    is_mcp_tool: bool = False,
) -> ToolClassification:
    """Centrally classify any tool call as SAFE or SENSITIVE."""
    args = normalize_tool_arguments(tool_name, arguments or {})
    normalized = tool_name.strip().lower()

    if normalized == "bash_exec":
        cmd = str(args.get("command", ""))
        for pattern, reason in DESTRUCTIVE_BASH_PATTERNS:
            if pattern.search(cmd):
                return ToolClassification(
                    tool_name=tool_name,
                    safety_level="SENSITIVE",
                    risk_reason=f"Destructive or mutating shell command detected: {reason}",
                    expected_effect=_describe_expected_effect(tool_name, args),
                )
        return ToolClassification(
            tool_name=tool_name,
            safety_level="SAFE",
            risk_reason="Non-destructive sandbox shell execution inside /workspace.",
            expected_effect=_describe_expected_effect(tool_name, args),
        )

    if normalized in EXPLICIT_SAFE_TOOLS:
        return ToolClassification(
            tool_name=tool_name,
            safety_level="SAFE",
            risk_reason="Explicitly registered non-mutating or workspace-confined tool.",
            expected_effect=_describe_expected_effect(tool_name, args),
        )

    for pattern in SENSITIVE_TOOL_PATTERNS:
        if pattern.search(normalized):
            return ToolClassification(
                tool_name=tool_name,
                safety_level="SENSITIVE",
                risk_reason=f"Tool '{tool_name}' matches sensitive action policy ({pattern.pattern}).",
                expected_effect=_describe_expected_effect(tool_name, args),
            )

    if is_mcp_tool:
        if any(normalized.endswith(f"_{safe}") for safe in EXPLICIT_SAFE_TOOLS) or re.search(
            r"(?:^|_)(?:get|list|search|read|find|fetch|describe|lookup|query|inspect|count|check|status|ping|echo|health|whoami|info)(?:_|$)",
            normalized,
        ):
            return ToolClassification(
                tool_name=tool_name,
                safety_level="SAFE",
                risk_reason="Read-only MCP discovery/query tool.",
                expected_effect=_describe_expected_effect(tool_name, args),
            )
        return ToolClassification(
            tool_name=tool_name,
            safety_level="SENSITIVE",
            risk_reason=f"External MCP tool '{tool_name}' defaults to SENSITIVE for safety.",
            expected_effect=_describe_expected_effect(tool_name, args),
        )

    return ToolClassification(
        tool_name=tool_name,
        safety_level="SENSITIVE",
        risk_reason=f"Unclassified tool '{tool_name}' requires approval by fail-safe default.",
        expected_effect=_describe_expected_effect(tool_name, args),
    )


_BUILTIN_ARG_ALIASES: dict[str, dict[str, str]] = {
    "bash_exec": {"cmd": "command", "script": "command", "bash": "command", "shell": "command"},
    "read_file": {
        "file_path": "path",
        "filepath": "path",
        "filename": "path",
        "start": "start_line",
        "end": "end_line",
    },
    "write_file": {
        "file_path": "path",
        "filepath": "path",
        "filename": "path",
        "text": "content",
        "file_content": "content",
        "body": "content",
        "code": "content",
        "data": "content",
    },
    "edit_file": {
        "file_path": "path",
        "filepath": "path",
        "filename": "path",
        "old_str": "old_text",
        "new_str": "new_text",
        "old_string": "old_text",
        "new_string": "new_text",
        "old": "old_text",
        "new": "new_text",
        "search": "old_text",
        "replace": "new_text",
        "target": "old_text",
        "replacement": "new_text",
    },
    "list_dir": {"dir_path": "path", "directory": "path", "folder": "path"},
    "load_skill": {"skill": "name", "skill_name": "name"},
    "web_search": {
        "q": "query",
        "search_query": "query",
        "keywords": "query",
        "term": "query",
        "limit": "max_results",
        "num_results": "max_results",
        "count": "max_results",
    },
    "web_fetch": {"uri": "url", "link": "url"},
}


def normalize_tool_arguments(tool_name: str, arguments: Any) -> dict[str, Any]:
    """Normalize JSON-string args or common LLM parameter aliases for built-in tools."""
    if isinstance(arguments, str):
        raw_str = arguments.strip()
        if not raw_str:
            return {}
        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, dict):
                arguments = parsed
            else:
                return {"__invalid_json__": raw_str[:240]}
        except json.JSONDecodeError as exc:
            return {"__invalid_json__": f"{exc.msg}: {raw_str[:200]}"}
    if not isinstance(arguments, dict):
        return {}
    aliases = _BUILTIN_ARG_ALIASES.get(tool_name.strip().lower())
    if not aliases:
        return dict(arguments)
    normalized = dict(arguments)
    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)
    return normalized


def validate_tool_arguments(
    tool_obj: BaseTool | None,
    arguments: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(arguments, dict):
        return None, "Edited arguments must be a valid JSON object (dictionary)."
    if "__invalid_json__" in arguments:
        return (
            None,
            f"Malformed JSON arguments for '{getattr(tool_obj, 'name', 'tool')}': "
            f"{str(arguments.get('__invalid_json__'))[:240]}",
        )

    if tool_obj is None:
        return dict(arguments), None

    normalized_args = normalize_tool_arguments(tool_obj.name, arguments)
    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is None:
        return normalized_args, None

    if hasattr(args_schema, "model_validate"):
        try:
            validated = args_schema.model_validate(normalized_args)
            return validated.model_dump(), None
        except ValidationError as exc:
            return None, f"Invalid arguments for '{tool_obj.name}': {exc}"
    if isinstance(args_schema, dict):
        required_fields = args_schema.get("required", [])
        properties = args_schema.get("properties", {})
        if isinstance(required_fields, list):
            missing = [f for f in required_fields if f not in normalized_args]
            if missing:
                return (
                    None,
                    f"Missing required arguments for '{tool_obj.name}': {', '.join(missing)}",
                )
        if isinstance(properties, dict) and args_schema.get("additionalProperties") is False:
            extra = [k for k in normalized_args if k not in properties]
            if extra:
                return None, f"Unexpected arguments for '{tool_obj.name}': {', '.join(extra)}"
        return normalized_args, None

    return normalized_args, None


def _parse_human_response(raw_response: Any) -> tuple[str, dict[str, Any] | None, str]:
    if isinstance(raw_response, str):
        action = raw_response.strip().lower()
        return action, None, ""
    if isinstance(raw_response, dict):
        action = (
            str(
                raw_response.get("decision")
                or raw_response.get("action")
                or raw_response.get("status")
                or ""
            )
            .strip()
            .lower()
        )
        edited_args = raw_response.get("arguments")
        if isinstance(edited_args, str):
            try:
                edited_args = json.loads(edited_args)
            except json.JSONDecodeError:
                edited_args = {"__invalid_json__": edited_args}
        reason = str(raw_response.get("reason") or raw_response.get("comment") or "").strip()
        return action, edited_args if isinstance(edited_args, dict) else None, reason
    return "reject", None, "Unrecognized approval payload; defaulting to reject."


def request_human_approval(
    *,
    classification: ToolClassification,
    arguments: dict[str, Any],
    tool_obj: BaseTool | None = None,
    source: str = "supervisor",
) -> ApprovalDecision:
    approval_id = f"apr-{uuid.uuid4().hex[:10]}"
    current_args = dict(arguments)
    edited = False
    validation_error: str | None = None

    while True:
        payload: dict[str, Any] = {
            "type": "human_approval_required",
            "approval_id": approval_id,
            "tool_name": classification.tool_name,
            "arguments": current_args,
            "arguments_json": json.dumps(current_args, indent=2, sort_keys=True, default=str),
            "expected_effect": _describe_expected_effect(classification.tool_name, current_args),
            "risk_reason": classification.risk_reason,
            "source": source,
            "edited": edited,
            "allowed_decisions": ["approve", "reject", "edit"],
        }
        if validation_error:
            payload["validation_error"] = validation_error

        human_reply = interrupt(payload)
        action, candidate_args, reason = _parse_human_response(human_reply)

        if action in ("approve", "approved", "yes", "y"):
            validated_args, err = validate_tool_arguments(tool_obj, current_args)
            if err is not None or validated_args is None:
                validation_error = err or "Argument validation failed."
                continue
            return ApprovalDecision(
                approved=True,
                arguments=validated_args,
                edited=edited,
                reason=reason or "Approved by human operator.",
                approval_id=approval_id,
            )

        if action in ("reject", "rejected", "deny", "no", "n", "cancel"):
            return ApprovalDecision(
                approved=False,
                arguments=current_args,
                edited=edited,
                reason=reason or "Rejected by human operator.",
                approval_id=approval_id,
            )

        if action == "edit":
            if candidate_args is None or "__invalid_json__" in candidate_args:
                validation_error = "Edited arguments must be a valid JSON object."
                continue
            validated_args, err = validate_tool_arguments(tool_obj, candidate_args)
            if err is not None or validated_args is None:
                validation_error = err or "Edited arguments failed validation."
                continue
            current_args = validated_args
            edited = True
            validation_error = None
            continue

        validation_error = (
            f"Unsupported decision {action!r}. Respond with 'approve', 'reject', or 'edit'."
        )


async def execute_tool_with_guardrails(
    tool_obj: BaseTool,
    arguments: dict[str, Any],
    *,
    is_mcp_tool: bool = False,
    source: str = "supervisor",
) -> tuple[str, dict[str, Any]]:
    classification = classify_tool_call(
        tool_obj.name,
        arguments,
        is_mcp_tool=is_mcp_tool,
    )

    final_args = dict(arguments)
    approval_record: dict[str, Any] = {
        "tool_name": tool_obj.name,
        "safety_level": classification.safety_level,
        "source": source,
        "approved": True,
        "edited": False,
        "executed": False,
    }

    if classification.safety_level == "SENSITIVE":
        decision = request_human_approval(
            classification=classification,
            arguments=final_args,
            tool_obj=tool_obj,
            source=source,
        )
        approval_record["approval_id"] = decision.approval_id
        approval_record["approved"] = decision.approved
        approval_record["edited"] = decision.edited
        approval_record["reason"] = decision.reason

        if not decision.approved:
            approval_record["executed"] = False
            return (
                f"ACTION REJECTED BY HUMAN: Tool '{tool_obj.name}' was NOT executed. "
                f"Reason: {decision.reason}",
                approval_record,
            )
        final_args = decision.arguments

    validated_args, val_err = validate_tool_arguments(tool_obj, final_args)
    if val_err is not None or validated_args is None:
        approval_record["executed"] = False
        approval_record["error"] = val_err
        return f"ERROR: {val_err}", approval_record

    approval_record["final_arguments"] = validated_args
    raw_result = await tool_obj.ainvoke(validated_args)
    approval_record["executed"] = True
    return str(raw_result), approval_record
