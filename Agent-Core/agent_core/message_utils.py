"""Pure message inspection, reasoning extraction, history sanitization, and recovery utilities.

Decoupled from the LangGraph execution loop so both supervisor and sub-agent graphs
(as well as external consumers of ``agent_core``) can reuse deterministic message utilities.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from agent_core.state import PlanStep

THINK_TAG_RE = re.compile(r"<think>([\s\S]*?)</think>", flags=re.IGNORECASE)
UNCLOSED_THINK_TAG_RE = re.compile(r"<think>([\s\S]*)$", flags=re.IGNORECASE)
BASH_EXIT_CODE_RE = re.compile(r"^exit_code:\s*(-?\d+)\b", re.MULTILINE)
KNOWN_FINISH_REASONS: tuple[str, ...] = (
    "length",
    "max_tokens",
    "stop",
    "end_turn",
    "tool_calls",
    "function_call",
    "content_filter",
)


def summarize_tool_args(args: dict[str, Any]) -> str:
    """Return a compact, single-line summary of tool arguments for telemetry/activity feeds."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if k in ("content", "new_str", "old_str") and isinstance(v, str) and len(v) > 48:
            parts.append(f"{k}=<{len(v)} chars>")
            continue
        val_str = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
        val_str = val_str.replace("\n", " ").strip()
        if len(val_str) > 64:
            val_str = val_str[:61] + "..."
        parts.append(f"{k}={val_str}")
        if len(parts) >= 3:
            break
    summary = ", ".join(parts)
    return summary[:160]


def latest_human_text(messages: list[BaseMessage]) -> str:
    """Return the most recent HumanMessage content, or a safe fallback goal."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return "Complete the user task."


def latest_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Return the most recent AIMessage in ``messages`` or ``None``."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def normalize_finish_reason(raw: Any) -> str:
    """Normalize provider finish_reason values, including duplicated chunk merges like 'lengthlength'."""
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if s in KNOWN_FINISH_REASONS:
        return "length" if s == "max_tokens" else ("stop" if s == "end_turn" else s)
    for known in KNOWN_FINISH_REASONS:
        klen = len(known)
        if len(s) % klen == 0 and s == known * (len(s) // klen):
            return "length" if known == "max_tokens" else ("stop" if known == "end_turn" else known)
    if "length" in s or "max_tokens" in s:
        return "length"
    if "tool_calls" in s:
        return "tool_calls"
    if "content_filter" in s:
        return "content_filter"
    if "stop" in s or "end_turn" in s:
        return "stop"
    return s


def get_finish_reason(ai_msg: AIMessage | None) -> str:
    """Extract and normalize finish_reason from an AIMessage's response_metadata."""
    if ai_msg is None:
        return ""
    meta = getattr(ai_msg, "response_metadata", None) or {}
    return normalize_finish_reason(
        meta.get("finish_reason") or meta.get("normalized_finish_reason")
    )


def is_tool_output_error(tool_name: str, raw_output: str) -> bool:
    """Detect tool execution failures including non-zero exit codes or timeouts from bash_exec."""
    if raw_output.startswith("ERROR:"):
        return True
    if tool_name == "bash_exec":
        if "status: TIMED_OUT" in raw_output:
            return True
        match = BASH_EXIT_CODE_RE.search(raw_output)
        if match is not None:
            try:
                if int(match.group(1)) != 0:
                    return True
            except ValueError:
                pass
    return False


def extract_reasoning_from_extra(extra: dict[str, Any]) -> list[str]:
    """Extract reasoning/thinking text segments from OpenAI/LangChain additional_kwargs."""
    found: list[str] = []
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            found.append(val.strip())
    if not found:
        details = extra.get("reasoning_details")
        if isinstance(details, list):
            parts: list[str] = []
            for item in details:
                if isinstance(item, dict):
                    item_type = str(item.get("type") or "").lower()
                    if item_type and "encrypted" in item_type:
                        continue
                    t = item.get("text") or item.get("summary") or item.get("content")
                    if isinstance(t, str) and t:
                        parts.append(t)
            joined = "".join(parts).strip()
            if joined:
                found.append(joined)
    return found


def extract_text_and_thinking(ai_msg: AIMessage) -> tuple[str, str | None]:
    """Safely extract clean response text and optional reasoning trace from an AIMessage."""
    raw_content = ai_msg.content
    thinking_parts: list[str] = []

    if isinstance(raw_content, list):
        text_chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                text_chunks.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if item_type in ("thinking", "reasoning"):
                    val = item.get("thinking") or item.get("reasoning") or item.get("text")
                    if isinstance(val, str) and val.strip():
                        thinking_parts.append(val.strip())
                elif isinstance(item.get("text"), str):
                    text_chunks.append(item["text"])
        text = "\n".join(text_chunks)
    else:
        text = str(raw_content or "")

    extra = getattr(ai_msg, "additional_kwargs", None) or {}
    if isinstance(extra, dict):
        thinking_parts.extend(extract_reasoning_from_extra(extra))

    inline_thinks = THINK_TAG_RE.findall(text)
    if inline_thinks:
        for block in inline_thinks:
            if block.strip():
                thinking_parts.append(block.strip())
        text = THINK_TAG_RE.sub("", text)

    unclosed_match = UNCLOSED_THINK_TAG_RE.search(text)
    if unclosed_match:
        unclosed_inner = unclosed_match.group(1).strip()
        if unclosed_inner:
            thinking_parts.append(unclosed_inner)
        text = text[: unclosed_match.start()]

    clean_text = text.strip()
    thinking = "\n\n".join(thinking_parts).strip() or None
    return clean_text, thinking


def extract_chunk_thinking_delta(chunk: Any) -> str:
    """Extract incremental reasoning/thinking text from a streamed AIMessageChunk."""
    parts: list[str] = []
    extra = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(extra, dict):
        found_plain = False
        for key in ("reasoning_content", "reasoning", "thinking"):
            val = extra.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
                found_plain = True
        if not found_plain:
            details = extra.get("reasoning_details")
            if isinstance(details, list):
                for item in details:
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "").lower()
                        if item_type and "encrypted" in item_type:
                            continue
                        t = item.get("text") or item.get("summary") or item.get("content")
                        if isinstance(t, str) and t:
                            parts.append(t)
    raw_content = getattr(chunk, "content", None)
    if isinstance(raw_content, list):
        for item in raw_content:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if item_type in ("thinking", "reasoning"):
                    val = item.get("thinking") or item.get("reasoning") or item.get("text")
                    if isinstance(val, str) and val:
                        parts.append(val)
    return "".join(parts)


def sanitize_ai_message_for_history(ai_msg: AIMessage) -> AIMessage:
    """Ensure saved AIMessages never retain invalid_tool_calls or mismatched raw tool_calls."""
    valid_tcs = list(getattr(ai_msg, "tool_calls", None) or [])
    extra = dict(getattr(ai_msg, "additional_kwargs", None) or {})
    inv_tcs = list(getattr(ai_msg, "invalid_tool_calls", None) or [])
    has_raw_ak = "tool_calls" in extra or "function_call" in extra

    if not has_raw_ak and not inv_tcs:
        return ai_msg

    extra.pop("tool_calls", None)
    extra.pop("function_call", None)
    cleaned_kwargs: dict[str, Any] = {
        "content": ai_msg.content,
        "additional_kwargs": extra,
        "response_metadata": dict(getattr(ai_msg, "response_metadata", None) or {}),
        "tool_calls": valid_tcs,
        "invalid_tool_calls": [],
    }
    if getattr(ai_msg, "id", None):
        cleaned_kwargs["id"] = ai_msg.id
    if getattr(ai_msg, "usage_metadata", None):
        cleaned_kwargs["usage_metadata"] = ai_msg.usage_metadata
    return AIMessage(**cleaned_kwargs)


def sanitize_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Repair orphaned assistant tool_calls AND drop orphaned ToolMessages before calling an LLM."""
    sanitized: list[BaseMessage] = []
    n = len(messages)
    active_tool_ids: set[str] | None = None
    active_allow_any_id = False

    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tid = str(getattr(msg, "tool_call_id", "") or "")
            if active_tool_ids is not None:
                if active_allow_any_id:
                    sanitized.append(msg)
                elif tid and tid in active_tool_ids:
                    active_tool_ids.remove(tid)
                    sanitized.append(msg)
            continue

        active_tool_ids = None
        active_allow_any_id = False

        if not isinstance(msg, AIMessage):
            sanitized.append(msg)
            continue

        tcs = list(getattr(msg, "tool_calls", None) or [])
        ak = dict(getattr(msg, "additional_kwargs", None) or {})
        inv_tcs = list(getattr(msg, "invalid_tool_calls", None) or [])
        has_raw_ak_tc = bool(ak.get("tool_calls") or ak.get("function_call"))
        if not tcs and not has_raw_ak_tc and not inv_tcs:
            sanitized.append(msg)
            continue

        expected_ids = {
            str(tc.get("id")) for tc in tcs if isinstance(tc, dict) and tc.get("id")
        }
        seen_ids: set[str] = set()
        j = idx + 1
        while j < n and isinstance(messages[j], ToolMessage):
            tid = str(getattr(messages[j], "tool_call_id", "") or "")
            if tid:
                seen_ids.add(tid)
            j += 1

        if tcs and (not expected_ids or expected_ids.issubset(seen_ids)):
            sanitized.append(sanitize_ai_message_for_history(msg))
            active_tool_ids = set(expected_ids)
            active_allow_any_id = len(expected_ids) == 0
        else:
            ak.pop("tool_calls", None)
            ak.pop("function_call", None)
            clean_text, _ = extract_text_and_thinking(msg)
            sanitized.append(
                AIMessage(
                    content=clean_text or "(Previous tool call was interrupted before execution.)",
                    additional_kwargs=ak,
                    response_metadata=dict(getattr(msg, "response_metadata", None) or {}),
                    tool_calls=[],
                    invalid_tool_calls=[],
                )
            )
    return sanitized


def build_recovery_messages(
    prompt_messages: list[BaseMessage],
    nudge_text: str,
) -> list[BaseMessage]:
    """Construct role-alternation-safe retry messages accepted by strict OpenAI/Anthropic/Gemini gateways."""
    msgs = list(prompt_messages)
    if not msgs:
        return [HumanMessage(content=nudge_text)]

    last = msgs[-1]
    if isinstance(last, HumanMessage):
        if isinstance(last.content, list):
            merged_content: Any = [*last.content, {"type": "text", "text": f"\n\n{nudge_text}"}]
        else:
            existing = str(last.content or "").rstrip()
            merged_content = f"{existing}\n\n{nudge_text}" if existing else nudge_text
        msgs[-1] = HumanMessage(
            content=merged_content,
            additional_kwargs=dict(getattr(last, "additional_kwargs", None) or {}),
        )
        return msgs

    if isinstance(last, ToolMessage):
        msgs.append(AIMessage(content="I reviewed the tool outputs above."))
        msgs.append(HumanMessage(content=nudge_text))
        return msgs

    msgs.append(HumanMessage(content=nudge_text))
    return msgs


def build_empty_output_diagnostic(
    *,
    last_ai: AIMessage | None,
    plan: list[PlanStep],
    turn_count: int,
) -> str:
    """Return a clear, actionable user-facing diagnostic when the model returns an empty output."""
    finish_reason = get_finish_reason(last_ai)
    completed_steps = [s for s in plan if s.get("status") == "completed"]
    parts: list[str] = []

    if finish_reason == "length":
        parts.append(
            "I hit the configured **Max Tokens** output limit before finishing my response or tool call "
            "even after retrying."
        )
        parts.append(
            "**How to fix this:** Open **Settings → LLM Config** and increase **Max Tokens** "
            "(for example, to `16384`), or lower the **Thinking Mode** level for large file generation tasks."
        )
    else:
        parts.append(
            "The model returned an empty completion after retries without producing a visible message or valid tool call."
        )
        parts.append(
            "You can send a follow-up message (for example, *\"Continue\"*) or check your provider settings."
        )

    if completed_steps:
        step_lines = "\n".join(
            f"- **{s['step_id']}** ({s['tool_or_action']}): {s['result'][:140] or s['description'][:140]}"
            for s in completed_steps[-5:]
        )
        parts.append(f"**Completed before stopping (turn {turn_count}):**\n{step_lines}")

    return "\n\n".join(parts)
