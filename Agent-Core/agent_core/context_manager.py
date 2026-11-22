"""Model-specific context window resolution, categorized token accounting, and 80%-only auto-compaction.

Provides:
- Dynamic model-specific context window lookup (explicit live catalog value -> model registry -> fallback)
- Categorized token accounting across System Prompt, Tools & MCPs, Skills, User Messages,
  Assistant & Reasoning, Tool Outputs & Sub-Agents, and Compacted Memory
- Strict >= 80% auto-compaction trigger (`AUTO_COMPACT_THRESHOLD = 0.80`) that summarizes
  older conversation turns safely without splitting AIMessage(tool_calls) / ToolMessage pairs
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

AUTO_COMPACT_THRESHOLD = 0.80
AUTO_COMPACT_THRESHOLD_PERCENT = 80
DEFAULT_CONTEXT_WINDOW = 128_000
MIN_CONTEXT_WINDOW = 512
DEFAULT_MAX_OUTPUT_TOKENS = 16_384
MIN_MAX_OUTPUT_TOKENS = 256

# Ordered from most specific model prefixes/substrings to general families.
_MODEL_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    # Google Gemini
    ("gemini-2.5-flash-image", 32_768),
    ("gemini-1.5-pro", 2_000_000),
    ("gemini-1.5-flash", 1_000_000),
    ("gemini-2.5-pro", 1_048_576),
    ("gemini-2.5-flash", 1_048_576),
    ("gemini-2.0", 1_048_576),
    ("gemini-3", 1_048_576),
    ("gemini-4", 1_048_576),
    ("gemini", 1_048_576),
    # Meta Llama 4 & 3
    ("llama-4-scout", 10_000_000),
    ("llama-4-maverick", 1_000_000),
    ("llama-4", 1_000_000),
    ("llama-3.3", 131_072),
    ("llama-3.2", 131_072),
    ("llama-3.1", 131_072),
    ("llama3.3", 131_072),
    ("llama3.1", 131_072),
    # OpenAI GPT-5 / GPT-4.1 / GPT-4o / o-series
    ("gpt-5.6", 1_050_000),
    ("gpt-5.5", 1_050_000),
    ("gpt-5.4", 1_050_000),
    ("gpt-5.2", 400_000),
    ("gpt-5.1", 400_000),
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_047_576),
    ("o4-mini", 200_000),
    ("o3-mini", 200_000),
    ("o3", 200_000),
    ("o1", 200_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-3.5", 16_385),
    # Anthropic Claude
    ("claude-opus-4.8", 1_000_000),
    ("claude-opus-4.7", 1_000_000),
    ("claude-opus-4.6", 1_000_000),
    ("claude-opus-4.5", 200_000),
    ("claude-opus-4.1", 200_000),
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4.6", 1_000_000),
    ("claude-sonnet-4.5", 1_000_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3.7", 200_000),
    ("claude-3.5", 200_000),
    ("claude-3", 200_000),
    ("claude", 200_000),
    # Moonshot Kimi
    ("kimi-k3", 1_048_576),
    ("kimi-k2.7", 262_144),
    ("kimi-k2.6", 262_144),
    ("kimi-k2.5", 262_144),
    ("kimi-k2-thinking", 262_144),
    ("kimi-k2", 131_072),
    ("kimi", 131_072),
    # NVIDIA Nemotron
    ("nemotron-3.5-content-safety", 131_072),
    ("nemotron-3.5", 1_048_576),
    ("nemotron-3-ultra-550b-a55b:free", 1_000_000),
    ("nemotron-3-ultra", 262_144),
    ("nemotron-3-super", 262_144),
    ("nemotron-3-nano-omni", 256_000),
    ("nemotron-3-nano", 131_072),
    ("nemotron", 131_072),
    # DeepSeek
    ("deepseek-v4", 1_048_576),
    ("deepseek-r1-0528", 163_840),
    ("deepseek-v3.2", 163_840),
    ("deepseek-v3.1", 163_840),
    ("deepseek-v3", 128_000),
    ("deepseek-r1", 128_000),
    ("deepseek-chat", 128_000),
    ("deepseek-reasoner", 128_000),
    ("deepseek", 128_000),
    # Zhipu GLM
    ("glm-5", 1_048_576),
    ("glm-4.7", 204_800),
    ("glm-4.6", 204_800),
    ("glm-4.5", 131_072),
    ("glm-4", 128_000),
    # MiniMax
    ("minimax-m3", 1_048_576),
    ("minimax-m2.7", 196_608),
    ("minimax-m2.5", 198_000),
    ("minimax-m2", 196_608),
    ("minimax", 196_608),
    # Qwen
    ("qwen3.8-27b", 1_024_000),
    ("qwen3.8", 1_000_000),
    ("qwen3.5", 262_144),
    ("qwen3-coder", 262_144),
    ("qwen3-max", 262_144),
    ("qwen3-next", 262_144),
    ("qwen3", 131_072),
    ("qwen2.5", 131_072),
    ("qwq", 131_072),
    ("qwen", 131_072),
    # xAI Grok
    ("grok-4.20", 2_000_000),
    ("grok-4.3", 1_000_000),
    ("grok-4.7", 500_000),
    ("grok-4.6", 500_000),
    ("grok-4.5", 500_000),
    ("grok-4", 256_000),
    ("grok-3", 131_072),
    ("grok-2", 131_072),
    ("grok", 131_072),
    # Mistral
    ("mistral-large-4", 1_048_576),
    ("mistral-large-2512", 262_144),
    ("mistral-large-2407", 131_072),
    ("mistral-large", 128_000),
    ("mistral-small-3", 128_000),
    ("pixtral", 128_000),
    ("codestral", 256_000),
    ("mistral", 128_000),
    # Upstage Solar
    ("solar-pro4", 524_288),
    # Baidu ERNIE
    ("ernie-5", 128_000),
    ("ernie-4", 128_000),
    ("ernie", 128_000),
    # Gemma / Phi
    ("gemma-4", 262_144),
    ("gemma-3", 128_000),
    ("gemma", 32_768),
    ("phi-4", 128_000),
    ("phi-3.5", 128_000),
)


def _build_model_match_candidates(raw_model: str | None) -> tuple[str, ...]:
    clean = (raw_model or "").strip().lower()
    if not clean:
        return ()

    def _norm(s: str) -> str:
        s = re.sub(r"v(\d+)p(\d+)", r"\1.\2", s)
        s = re.sub(r"(\d)-(\d)(?=$|[-_/:])", r"\1.\2", s)
        return s

    bare_with_tag = clean.split("/")[-1]
    bare_no_tag = bare_with_tag.split(":")[0]
    seen: list[str] = []
    for item in (
        clean,
        _norm(clean),
        bare_with_tag,
        _norm(bare_with_tag),
        bare_no_tag,
        _norm(bare_no_tag),
    ):
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def resolve_model_context_window(
    provider: str | None,
    model: str | None,
    explicit_context_window: int | None = None,
) -> tuple[int, str]:
    """Return `(context_window_tokens, source)` for the specified model."""
    if isinstance(explicit_context_window, (int, float)) and explicit_context_window >= MIN_CONTEXT_WINDOW:
        return int(explicit_context_window), "provider_catalog"

    candidates = _build_model_match_candidates(model)
    if candidates:
        for pattern, limit in _MODEL_CONTEXT_WINDOWS:
            if any(pattern in target for target in candidates):
                return limit, "model_registry"

    clean_provider = (provider or "").strip().lower()
    if clean_provider in ("ollama", "local", "vllm"):
        return 32_768, "provider_default"

    return DEFAULT_CONTEXT_WINDOW, "default"


# Ordered from most specific model prefixes/substrings to general families.
_MODEL_MAX_OUTPUT_TOKENS: tuple[tuple[str, int], ...] = (
    # Google Gemini
    ("gemini-2.5-flash-image", 8_192),
    ("gemini-2.5-pro", 65_536),
    ("gemini-2.5-flash", 65_536),
    ("gemini-3", 65_536),
    ("gemini-4", 65_536),
    ("gemini-2.0", 8_192),
    ("gemini-1.5", 8_192),
    ("gemini", 65_536),
    # OpenAI GPT-5 / o-series / GPT-4.1 / GPT-4o
    ("gpt-5.6", 128_000),
    ("gpt-5.5", 128_000),
    ("gpt-5.4", 128_000),
    ("gpt-5.2", 128_000),
    ("gpt-5", 128_000),
    ("o4-mini", 100_000),
    ("o3-mini", 100_000),
    ("o3", 100_000),
    ("o1", 100_000),
    ("gpt-4.1", 32_768),
    ("gpt-4o", 16_384),
    ("gpt-4-turbo", 4_096),
    ("gpt-3.5", 4_096),
    # Anthropic Claude
    ("claude-opus-4.8", 128_000),
    ("claude-opus-4.7", 128_000),
    ("claude-opus-4.6", 128_000),
    ("claude-opus-4.5", 64_000),
    ("claude-opus-4", 32_000),
    ("claude-sonnet-4.6", 128_000),
    ("claude-sonnet-4.5", 64_000),
    ("claude-sonnet-4", 64_000),
    ("claude-haiku-4", 64_000),
    ("claude-3.7", 64_000),
    ("claude-3.5", 8_192),
    ("claude-3", 4_096),
    ("claude", 64_000),
    # DeepSeek
    ("deepseek-v4", 131_072),
    ("deepseek-r1", 32_768),
    ("deepseek-reasoner", 32_768),
    ("deepseek-v3", 8_192),
    ("deepseek-chat", 8_192),
    ("deepseek", 16_384),
    # Moonshot Kimi
    ("kimi-k3", 131_072),
    ("kimi-k2.7", 131_072),
    ("kimi-k2.6", 131_072),
    ("kimi-k2.5", 65_536),
    ("kimi-k2-thinking", 98_304),
    ("kimi-k2", 32_768),
    ("kimi", 32_768),
    # NVIDIA Nemotron
    ("nemotron-3.5-content-safety", 32_768),
    ("nemotron-3.5", 131_072),
    ("nemotron-3-ultra", 65_536),
    ("nemotron-3-super", 65_536),
    ("nemotron-3-nano-omni", 65_536),
    ("nemotron-3-nano", 32_768),
    ("nemotron", 32_768),
    # MiniMax
    ("minimax-m3", 131_072),
    ("minimax-m2.7", 131_072),
    ("minimax-m2.5", 32_768),
    ("minimax", 32_768),
    # Qwen
    ("qwen3.8", 131_072),
    ("qwen3.5", 65_536),
    ("qwen3-coder", 65_536),
    ("qwen3-max", 65_536),
    ("qwen3-next", 65_536),
    ("qwen3", 32_768),
    ("qwq", 32_768),
    ("qwen2.5", 8_192),
    ("qwen", 16_384),
    # xAI Grok
    ("grok-4.20", 128_000),
    ("grok-4.3", 128_000),
    ("grok-4.7", 128_000),
    ("grok-4.6", 128_000),
    ("grok-4.5", 128_000),
    ("grok-4", 64_000),
    ("grok-3", 32_768),
    ("grok-2", 16_384),
    ("grok", 16_384),
    # Meta Llama 4 & 3
    ("llama-4", 32_768),
    ("llama-3.3", 8_192),
    ("llama-3.2", 8_192),
    ("llama-3.1", 8_192),
    ("llama3.3", 8_192),
    ("llama3.1", 8_192),
    # Mistral
    ("mistral-large-4", 262_144),
    ("mistral-large-2512", 131_072),
    ("codestral", 32_768),
    ("mistral-large", 32_768),
    ("mistral-small-3", 16_384),
    ("pixtral", 16_384),
    ("mistral", 16_384),
    # Zhipu GLM
    ("glm-5", 131_072),
    ("glm-4.7", 131_072),
    ("glm-4.6", 131_072),
    ("glm-4.5", 32_768),
    ("glm-4", 8_192),
    # Upstage Solar
    ("solar-pro4", 131_072),
    # Baidu ERNIE
    ("ernie-5", 16_384),
    ("ernie-4", 8_192),
    ("ernie", 8_192),
    # Gemma / Phi
    ("gemma-4", 32_768),
    ("gemma-3", 8_192),
    ("gemma", 8_192),
    ("phi-4", 16_384),
    ("phi-3.5", 8_192),
)


def resolve_model_max_output_tokens(
    provider: str | None,
    model: str | None,
    explicit_max_output_tokens: int | None = None,
) -> tuple[int, str]:
    """Return `(max_output_tokens, source)` for the specified model."""
    if (
        isinstance(explicit_max_output_tokens, (int, float))
        and explicit_max_output_tokens >= MIN_MAX_OUTPUT_TOKENS
    ):
        return int(explicit_max_output_tokens), "provider_catalog"

    candidates = _build_model_match_candidates(model)
    if candidates:
        for pattern, limit in _MODEL_MAX_OUTPUT_TOKENS:
            if any(pattern in target for target in candidates):
                return limit, "model_registry"

    clean_provider = (provider or "").strip().lower()
    if clean_provider in ("ollama", "local", "vllm"):
        return 8_192, "provider_default"

    return DEFAULT_MAX_OUTPUT_TOKENS, "default"


def estimate_text_tokens(text: str | None) -> int:
    """Fast, deterministic token estimator (~3.7 chars/token for mixed prose & code)."""
    if not text:
        return 0
    s = str(text)
    if not s:
        return 0
    # Code/JSON punctuation tokenizes slightly denser than English prose
    base = len(s) / 3.7
    return max(1, int(math.ceil(base)))


def estimate_message_content_tokens(content: Any) -> int:
    """Estimate tokens for string or multimodal block content."""
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += estimate_text_tokens(block)
            elif isinstance(block, dict):
                btype = str(block.get("type") or "").lower()
                if btype in ("text", "output_text", "input_text"):
                    total += estimate_text_tokens(str(block.get("text") or ""))
                elif btype == "image_url":
                    total += 1_024
                elif btype == "input_audio":
                    total += 1_536
                elif btype == "video_url":
                    total += 4_096
                elif btype in ("thinking", "reasoning"):
                    total += estimate_text_tokens(
                        str(block.get("thinking") or block.get("text") or block.get("reasoning") or "")
                    )
                else:
                    total += estimate_text_tokens(str(block.get("text") or ""))
        return total
    return estimate_text_tokens(str(content))


def estimate_tools_schema_tokens(tools: list[BaseTool] | None) -> int:
    """Estimate token overhead of bound tool schemas (local + MCP tools)."""
    if not tools:
        return 0
    total = 0
    for tool in tools:
        name = getattr(tool, "name", "") or ""
        desc = getattr(tool, "description", "") or ""
        schema_str = ""
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                if hasattr(args_schema, "model_json_schema"):
                    schema_str = json.dumps(args_schema.model_json_schema(), default=str)
                elif isinstance(args_schema, dict):
                    schema_str = json.dumps(args_schema, default=str)
            except Exception:  # noqa: BLE001
                schema_str = ""
        total += estimate_text_tokens(f"{name}\n{desc}\n{schema_str}") + 24
    return total


def classify_context_usage(usage_percent: float) -> dict[str, str]:
    """Classify context window fill level into clear user-facing tiers."""
    pct = max(0.0, float(usage_percent))
    if pct >= AUTO_COMPACT_THRESHOLD_PERCENT:
        return {
            "level": "critical",
            "label": "Auto-Compacting",
            "description": "Reached the 80% threshold — older turns are automatically compacted to preserve memory.",
        }
    if pct >= 70.0:
        return {
            "level": "high",
            "label": "High",
            "description": "Approaching the 80% auto-compaction threshold.",
        }
    if pct >= 50.0:
        return {
            "level": "moderate",
            "label": "Moderate",
            "description": "Healthy active context with plenty of headroom before 80% compaction.",
        }
    return {
        "level": "optimal",
        "label": "Optimal",
        "description": "Light context footprint with full recall clarity.",
    }


def extract_provider_input_tokens(ai_message: BaseMessage | None) -> int | None:
    """Extract authoritative prompt/input token count from provider usage_metadata if present."""
    if ai_message is None:
        return None
    usage = getattr(ai_message, "usage_metadata", None)
    if isinstance(usage, dict):
        for key in ("input_tokens", "prompt_tokens"):
            val = usage.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
    resp_meta = getattr(ai_message, "response_metadata", None)
    if isinstance(resp_meta, dict):
        token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
        if isinstance(token_usage, dict):
            for key in ("prompt_tokens", "input_tokens"):
                val = token_usage.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
    return None


def compute_context_usage(
    *,
    provider: str | None,
    model: str | None,
    explicit_context_window: int | None = None,
    explicit_max_output_tokens: int | None = None,
    configured_max_tokens: int | None = None,
    system_prompt_text: str = "",
    skill_catalog_text: str = "",
    loaded_skills_text: str = "",
    tools: list[BaseTool] | None = None,
    messages: list[BaseMessage] | None = None,
    compacted_summary: str = "",
    provider_input_tokens: int | None = None,
    compaction_count: int = 0,
    last_compacted_at: str | None = None,
    tokens_saved_last_compaction: int = 0,
) -> dict[str, Any]:
    """Compute a complete, categorized model-specific context usage snapshot."""
    context_window, source = resolve_model_context_window(
        provider, model, explicit_context_window=explicit_context_window
    )
    max_output_tokens, max_output_source = resolve_model_max_output_tokens(
        provider, model, explicit_max_output_tokens=explicit_max_output_tokens
    )
    effective_max_tokens = (
        min(int(configured_max_tokens), max_output_tokens)
        if isinstance(configured_max_tokens, (int, float)) and configured_max_tokens >= 64
        else max_output_tokens
    )

    skills_raw_tokens = estimate_text_tokens(skill_catalog_text) + estimate_text_tokens(
        loaded_skills_text
    )
    compacted_tokens = estimate_text_tokens(compacted_summary)
    # Subtract skills & compacted summary if embedded inside system_prompt_text so we never double-count
    full_sys_tokens = estimate_text_tokens(system_prompt_text)
    system_only_tokens = max(0, full_sys_tokens - skills_raw_tokens - compacted_tokens)
    tools_tokens = estimate_tools_schema_tokens(tools)

    user_tokens = 0
    assistant_tokens = 0
    tool_output_tokens = 0

    for msg in messages or []:
        if isinstance(msg, SystemMessage):
            system_only_tokens += estimate_message_content_tokens(msg.content) + 4
        elif isinstance(msg, HumanMessage):
            user_tokens += estimate_message_content_tokens(msg.content) + 6
        elif isinstance(msg, AIMessage):
            assistant_tokens += estimate_message_content_tokens(msg.content) + 6
            tcs = getattr(msg, "tool_calls", None) or []
            if tcs:
                assistant_tokens += estimate_text_tokens(json.dumps(tcs, default=str))
            add_kw = getattr(msg, "additional_kwargs", None) or {}
            if isinstance(add_kw, dict):
                reasoning = add_kw.get("reasoning_content") or add_kw.get("reasoning")
                if isinstance(reasoning, str) and reasoning.strip():
                    assistant_tokens += estimate_text_tokens(reasoning)
        elif isinstance(msg, ToolMessage):
            tool_output_tokens += estimate_message_content_tokens(msg.content) + 8
        else:
            assistant_tokens += estimate_message_content_tokens(getattr(msg, "content", "")) + 4

    raw_counts = {
        "system_prompt": system_only_tokens,
        "tools": tools_tokens,
        "skills": skills_raw_tokens,
        "user_messages": user_tokens,
        "assistant_messages": assistant_tokens,
        "tool_outputs": tool_output_tokens,
        "compacted_summary": compacted_tokens,
    }
    estimated_total = sum(raw_counts.values())

    calibrated = False
    if (
        isinstance(provider_input_tokens, (int, float))
        and provider_input_tokens > 0
        and estimated_total > 0
    ):
        target_total = int(provider_input_tokens)
        scale = target_total / estimated_total
        # Only calibrate when within a sane ratio (avoid wildly skewed counts from streaming deltas)
        if 0.25 <= scale <= 4.0:
            scaled: dict[str, int] = {}
            running = 0
            keys = list(raw_counts.keys())
            for k in keys:
                val = int(round(raw_counts[k] * scale)) if raw_counts[k] > 0 else 0
                scaled[k] = val
                running += val
            diff = target_total - running
            if diff != 0:
                largest_key = max(keys, key=lambda k: scaled[k])
                scaled[largest_key] = max(0, scaled[largest_key] + diff)
            raw_counts = scaled
            estimated_total = sum(raw_counts.values())
            calibrated = True

    used_tokens = max(0, int(estimated_total))
    free_tokens = max(0, context_window - used_tokens)
    usage_percent = round(min(100.0, (used_tokens / max(1, context_window)) * 100.0), 1)
    classification = classify_context_usage(usage_percent)

    category_meta = [
        ("system_prompt", "System Prompt", "Core agent instructions, guardrails, and active plan"),
        ("tools", "Tools & MCPs", "Bound sandbox, web, sub-agent, and MCP tool definitions"),
        ("skills", "Skills", "Skill catalog and active domain skill guidelines"),
        ("user_messages", "User Messages", "Conversation prompts and attached media overhead"),
        ("assistant_messages", "Assistant & Reasoning", "Agent responses, tool calls, and retained reasoning"),
        ("tool_outputs", "Tool Outputs & Sub-Agents", "File reads, bash output, web results, and sub-agent reports"),
        ("compacted_summary", "Compacted Memory", "Rolling memory summary preserved from 80% auto-compaction"),
    ]

    categories = []
    for key, label, description in category_meta:
        toks = int(raw_counts[key])
        pct_of_window = round((toks / max(1, context_window)) * 100.0, 1)
        pct_of_used = round((toks / max(1, used_tokens)) * 100.0, 1) if used_tokens > 0 else 0.0
        categories.append(
            {
                "key": key,
                "label": label,
                "description": description,
                "tokens": toks,
                "percent": pct_of_window,
                "share_of_used": pct_of_used,
            }
        )

    return {
        "model": (model or "").strip(),
        "provider": (provider or "").strip(),
        "context_window": context_window,
        "context_window_source": source,
        "max_output_tokens": max_output_tokens,
        "max_output_tokens_source": max_output_source,
        "configured_max_tokens": effective_max_tokens,
        "used_tokens": used_tokens,
        "free_tokens": free_tokens,
        "usage_percent": usage_percent,
        "auto_compact_threshold_percent": AUTO_COMPACT_THRESHOLD_PERCENT,
        "classification": classification,
        "categories": categories,
        "compacted_count": max(0, int(compaction_count)),
        "last_compacted_at": last_compacted_at,
        "tokens_saved_last_compaction": max(0, int(tokens_saved_last_compaction)),
        "calibrated_from_provider": calibrated,
    }


def should_auto_compact(usage_ratio: float, message_count: int) -> bool:
    """Strictly trigger auto-compaction ONLY when context usage reaches >= 80%."""
    return float(usage_ratio) >= AUTO_COMPACT_THRESHOLD and message_count >= 2


def find_safe_compaction_split(messages: list[BaseMessage], keep_recent_turns: int = 2) -> int:
    """Return index `split_idx` so `messages[:split_idx]` can be compacted safely.

    Guarantees:
    - Never splits an `AIMessage(tool_calls)` away from its matching `ToolMessage`(s).
    - Always keeps at least the latest user turn (or latest message) intact in `messages[split_idx:]`.
    - Returns `0` when no safe non-empty prefix can be compacted.
    """
    n = len(messages)
    if n <= 1:
        return 0

    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) >= keep_recent_turns:
        candidate = human_indices[-keep_recent_turns]
    elif len(human_indices) == 1:
        # Single long turn (e.g. many tool calls inside one turn): keep at least the user prompt
        # or if the user prompt is at 0, compact older tool cycles after index 1 while preserving
        # the latest AI/Tool tail.
        candidate = max(0, n - 4)
    else:
        candidate = max(0, n - 2)

    if candidate <= 0:
        # If there are >= 2 human messages and keep_recent_turns==2 landed at 0, keep only the latest 1 human turn
        if len(human_indices) >= 2 and human_indices[-1] > 0:
            candidate = human_indices[-1]
        elif n >= 4:
            candidate = n - 2
        else:
            return 0

    # Never land on a ToolMessage: prefer walking forward past the current tool-result block,
    # and if the tool-result block extends all the way to the end of `messages`, walk backward
    # to include the initiating AIMessage(tool_calls) in the preserved suffix.
    if isinstance(messages[candidate], ToolMessage):
        forward = candidate
        while forward < n and isinstance(messages[forward], ToolMessage):
            forward += 1
        if forward < n:
            candidate = forward
        else:
            backward = candidate
            while backward > 0 and isinstance(messages[backward], ToolMessage):
                backward -= 1
            candidate = backward

    # If the message right before candidate is an AIMessage with tool_calls and a ToolMessage
    # immediately follows candidate, step back so the AIMessage stays paired with its ToolMessages.
    if candidate > 0 and isinstance(messages[candidate - 1], AIMessage):
        prev_tcs = getattr(messages[candidate - 1], "tool_calls", None) or []
        if prev_tcs and candidate < n and isinstance(messages[candidate], ToolMessage):
            candidate -= 1

    if candidate <= 0 or candidate >= n or isinstance(messages[candidate], ToolMessage):
        return 0
    return candidate


def _format_messages_for_summary(messages: list[BaseMessage], max_chars: int = 24_000) -> str:
    """Serialize older messages into a clean transcript for the compaction summarizer."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "USER"
        elif isinstance(msg, AIMessage):
            role = "ASSISTANT"
        elif isinstance(msg, ToolMessage):
            role = f"TOOL({getattr(msg, 'name', '') or 'tool'})"
        else:
            role = type(msg).__name__.upper()

        content = msg.content
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(str(b.get("text") or ""))
                elif isinstance(b, dict):
                    parts.append(f"[{b.get('type', 'media')}]")
                else:
                    parts.append(str(b))
            text = " ".join(parts).strip()
        else:
            text = str(content or "").strip()

        text = re.sub(r"\s+", " ", text)[:1_200]
        if text:
            lines.append(f"{role}: {text}")

    joined = "\n".join(lines)
    if len(joined) > max_chars:
        half = max_chars // 2
        joined = joined[:half] + "\n... [earlier lines omitted] ...\n" + joined[-half:]
    return joined


def _extractive_fallback_summary(
    messages: list[BaseMessage],
    existing_summary: str = "",
) -> str:
    """Deterministic fallback summary when the summarizer LLM call fails or is unavailable."""
    bullets: list[str] = []
    if existing_summary.strip():
        bullets.append(existing_summary.strip()[:1_500])

    user_points: list[str] = []
    assistant_points: list[str] = []
    tool_points: list[str] = []

    for msg in messages:
        raw = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        compact = re.sub(r"\s+", " ", raw).strip()
        if not compact:
            continue
        if isinstance(msg, HumanMessage):
            user_points.append(compact[:220])
        elif isinstance(msg, AIMessage):
            assistant_points.append(compact[:240])
        elif isinstance(msg, ToolMessage):
            tname = getattr(msg, "name", "") or "tool"
            tool_points.append(f"{tname}: {compact[:160]}")

    if user_points:
        bullets.append("User goals & requests:\n- " + "\n- ".join(user_points[-6:]))
    if tool_points:
        bullets.append("Key tool findings & workspace actions:\n- " + "\n- ".join(tool_points[-6:]))
    if assistant_points:
        bullets.append("Assistant conclusions:\n- " + "\n- ".join(assistant_points[-4:]))

    return "\n\n".join(bullets).strip() or "Prior conversation turns were compacted to save context."


async def compact_conversation_history(
    messages: list[BaseMessage],
    *,
    summarizer_model: BaseChatModel | None = None,
    existing_summary: str = "",
    keep_recent_turns: int = 2,
) -> tuple[list[RemoveMessage], list[BaseMessage], str, int]:
    """Compact older conversation messages when context usage reaches >= 80%.

    Returns:
        (remove_messages, remaining_messages, new_compacted_summary, estimated_tokens_saved)
    """
    split_idx = find_safe_compaction_split(messages, keep_recent_turns=keep_recent_turns)
    if split_idx <= 0:
        return [], list(messages), existing_summary, 0

    to_compact = messages[:split_idx]
    remaining = messages[split_idx:]

    before_tokens = sum(
        estimate_message_content_tokens(m.content) + 6 for m in to_compact
    ) + estimate_text_tokens(existing_summary)

    transcript = _format_messages_for_summary(to_compact)
    new_summary = ""

    if summarizer_model is not None and transcript.strip():
        prompt = (
            "You are compacting an active autonomous agent conversation because context usage "
            "reached 80% of the model's context window.\n"
            "Produce a concise, high-density structured memory summary preserving:\n"
            "1. User goals, constraints, and preferences\n"
            "2. Key decisions, facts, numbers, and verified conclusions\n"
            "3. Files created/modified in /workspace and important tool/sub-agent outputs\n"
            "4. Open items or next steps still in progress\n"
            "Keep the summary under 350 words and strictly factual."
        )
        user_body = (
            f"EXISTING MEMORY SUMMARY:\n{existing_summary.strip() or '(none)'}\n\n"
            f"EARLIER CONVERSATION TURNS TO COMPACT:\n{transcript}"
        )
        try:
            from agent_core.models import invoke_with_rate_limit_retry

            reply = await invoke_with_rate_limit_retry(
                lambda: summarizer_model.ainvoke(
                    [SystemMessage(content=prompt), HumanMessage(content=user_body)]
                ),
                label="context summarizer",
            )
            raw_content = getattr(reply, "content", "")
            if isinstance(raw_content, str):
                new_summary = raw_content.strip()
            elif isinstance(raw_content, list):
                new_summary = "\n".join(
                    str(b.get("text") or "")
                    for b in raw_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM conversation compaction failed; using deterministic summary: %s", exc)

    if not new_summary:
        new_summary = _extractive_fallback_summary(to_compact, existing_summary=existing_summary)

    after_tokens = estimate_text_tokens(new_summary)
    tokens_saved = max(0, before_tokens - after_tokens)

    removals: list[RemoveMessage] = []
    for msg in to_compact:
        msg_id = getattr(msg, "id", None)
        if isinstance(msg_id, str) and msg_id.strip():
            removals.append(RemoveMessage(id=msg_id))

    return removals, remaining, new_summary, tokens_saved


def build_compacted_memory_block(compacted_summary: str) -> str:
    """Format the rolling compacted memory summary for injection into the supervisor prompt."""
    clean = (compacted_summary or "").strip()
    if not clean:
        return ""
    return (
        "\n\n## Compacted Conversation Memory (Auto-Compacted at 80% Context)\n"
        "Earlier turns in this thread were compacted to stay within this model's context window. "
        "Treat the summary below as authoritative prior context:\n\n"
        f"{clean}\n"
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
