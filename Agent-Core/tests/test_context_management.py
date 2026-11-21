"""Unit and integration tests for model-specific context accounting and 80%-only auto-compaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_core.context_manager import (
    AUTO_COMPACT_THRESHOLD,
    classify_context_usage,
    compute_context_usage,
    find_safe_compaction_split,
    resolve_model_context_window,
    resolve_model_max_output_tokens,
    should_auto_compact,
)
from agent_core.graph import create_agent_runtime
from agent_core.models import ModelSpec, build_chat_model
from agent_core.sandbox import SandboxConfig, SandboxManager
from agent_core.skills import SkillRegistry


class _TinyWindowChatModel(BaseChatModel):
    """Deterministic chat model with a configurable context_window for testing 80% compaction."""

    model_name: str = "test-tiny-ctx-model"
    context_window: int = 3_200
    call_count: int = 0
    saw_compacted_memory: bool = False

    @property
    def _llm_type(self) -> str:
        return "tiny-window-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _TinyWindowChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        # Detect if this is the summarizer call
        first_sys = str(messages[0].content or "") if messages else ""
        if "compacting an active autonomous agent conversation" in first_sys.lower():
            summary_msg = AIMessage(
                content=(
                    "- User asked about earlier design milestones and project facts.\n"
                    "- Earlier turns were safely compacted at the 80% threshold."
                )
            )
            return ChatResult(generations=[ChatGeneration(message=summary_msg)])

        if "Compacted Conversation Memory" in first_sys:
            self.saw_compacted_memory = True

        reply = AIMessage(content=f"Acknowledged turn {self.call_count}.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_resolve_model_context_window_is_model_specific() -> None:
    win, src = resolve_model_context_window("gemini", "gemini-2.5-pro")
    assert win == 1_048_576
    assert src == "model_registry"

    win, src = resolve_model_context_window("anthropic", "claude-sonnet-4-6")
    assert win == 200_000
    assert src == "model_registry"

    win, src = resolve_model_context_window("openai", "gpt-5.2")
    assert win == 400_000
    assert src == "model_registry"

    win, src = resolve_model_context_window("river_ai", "deepseek-ai/DeepSeek-V4.1-Flash")
    assert win == 128_000
    assert src == "model_registry"

    # Explicit live catalog context_length always wins (including 1M+ windows)
    win, src = resolve_model_context_window("openrouter", "custom/model", explicit_context_window=65_536)
    assert win == 65_536
    assert src == "provider_catalog"


def test_resolve_model_max_output_tokens_is_model_specific() -> None:
    out, src = resolve_model_max_output_tokens("gemini", "gemini-2.5-pro")
    assert out == 65_536
    assert src == "model_registry"

    out, src = resolve_model_max_output_tokens("openai", "gpt-5.2")
    assert out == 128_000
    assert src == "model_registry"

    out, src = resolve_model_max_output_tokens("anthropic", "claude-opus-4-6")
    assert out == 128_000
    assert src == "model_registry"

    out, src = resolve_model_max_output_tokens("river_ai", "moonshotai/Kimi-K3.1-Pro")
    assert out == 32_768
    assert src == "model_registry"

    out, src = resolve_model_max_output_tokens("deepseek", "deepseek-chat")
    assert out == 8_192
    assert src == "model_registry"

    out, src = resolve_model_max_output_tokens("openrouter", "custom/long-out", explicit_max_output_tokens=200_000)
    assert out == 200_000
    assert src == "provider_catalog"


def test_build_chat_model_honors_large_model_output_and_clamps_oversized_requests() -> None:
    # Model supporting 128k output gets full 128k without artificial 16k cap
    gpt52_spec = ModelSpec(
        role="primary",
        provider="openai",
        model="gpt-5.2",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key",
        max_tokens=128_000,
    )
    gpt52_llm = build_chat_model(spec=gpt52_spec)
    assert gpt52_llm.max_tokens == 128_000

    # Oversized max_tokens requested on an 8k-output model safely clamps to 8,192
    ds_chat_spec = ModelSpec(
        role="primary",
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test-key",
        max_tokens=128_000,
    )
    ds_chat_llm = build_chat_model(spec=ds_chat_spec)
    assert ds_chat_llm.max_tokens == 8_192


def test_should_auto_compact_triggers_strictly_at_80_percent_only() -> None:
    assert AUTO_COMPACT_THRESHOLD == 0.80
    assert should_auto_compact(0.0, 10) is False
    assert should_auto_compact(0.50, 10) is False
    assert should_auto_compact(0.75, 10) is False
    assert should_auto_compact(0.7999, 10) is False
    assert should_auto_compact(0.80, 10) is True
    assert should_auto_compact(0.95, 10) is True
    # Single message cannot be compacted away
    assert should_auto_compact(0.90, 1) is False


def test_classify_context_usage_tiers() -> None:
    assert classify_context_usage(12.0)["level"] == "optimal"
    assert classify_context_usage(55.0)["level"] == "moderate"
    assert classify_context_usage(74.5)["level"] == "high"
    assert classify_context_usage(80.0)["level"] == "critical"
    assert classify_context_usage(80.0)["label"] == "Auto-Compacting"


def test_compute_context_usage_categorizes_and_calibrates() -> None:
    msgs: list[BaseMessage] = [
        HumanMessage(content="Summarize workspace architecture in detail."),
        AIMessage(
            content="Inspecting files now.",
            tool_calls=[{"id": "c1", "name": "read_file", "args": {"path": "README.md"}}],
        ),
        ToolMessage(content="README content " * 80, tool_call_id="c1", name="read_file"),
        AIMessage(content="Here is the complete summary of the workspace architecture."),
    ]
    usage = compute_context_usage(
        provider="anthropic",
        model="claude-sonnet-4-6",
        system_prompt_text="You are Nemo, an autonomous assistant.\n" + "Skill catalog text here.",
        skill_catalog_text="Skill catalog text here.",
        loaded_skills_text="",
        messages=msgs,
        compacted_summary="Earlier summary memory.",
    )
    assert usage["context_window"] == 200_000
    assert usage["max_output_tokens"] == 64_000
    assert usage["max_output_tokens_source"] == "model_registry"
    assert usage["auto_compact_threshold_percent"] == 80
    assert usage["used_tokens"] > 0
    assert usage["free_tokens"] == usage["context_window"] - usage["used_tokens"]

    by_key = {c["key"]: c["tokens"] for c in usage["categories"]}
    assert set(by_key.keys()) == {
        "system_prompt",
        "tools",
        "skills",
        "user_messages",
        "assistant_messages",
        "tool_outputs",
        "compacted_summary",
    }
    assert by_key["user_messages"] > 0
    assert by_key["assistant_messages"] > 0
    assert by_key["tool_outputs"] > 0
    assert by_key["compacted_summary"] > 0
    assert sum(by_key.values()) == usage["used_tokens"]

    # Calibration with provider input_tokens matches total exactly
    calibrated = compute_context_usage(
        provider="anthropic",
        model="claude-sonnet-4-6",
        system_prompt_text="System prompt",
        messages=msgs,
        provider_input_tokens=int(usage["used_tokens"] * 1.15),
    )
    assert calibrated["calibrated_from_provider"] is True
    assert calibrated["used_tokens"] == int(usage["used_tokens"] * 1.15)


def test_find_safe_compaction_split_never_orphans_tool_messages() -> None:
    msgs: list[BaseMessage] = [
        HumanMessage(id="h1", content="First question"),
        AIMessage(
            id="a1",
            content="Calling tool",
            tool_calls=[{"id": "tc1", "name": "read_file", "args": {"path": "a.py"}}],
        ),
        ToolMessage(id="t1", content="file a", tool_call_id="tc1", name="read_file"),
        AIMessage(id="a2", content="Done with first question"),
        HumanMessage(id="h2", content="Second question"),
        AIMessage(id="a3", content="Done with second question"),
        HumanMessage(id="h3", content="Third question"),
    ]
    split_idx = find_safe_compaction_split(msgs, keep_recent_turns=2)
    assert split_idx == 4
    # Kept tail starts cleanly at HumanMessage h2, never a ToolMessage
    assert isinstance(msgs[split_idx], HumanMessage)
    assert msgs[split_idx].id == "h2"


@pytest.mark.asyncio
async def test_runtime_auto_compacts_strictly_at_80_percent_and_not_below(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    db_path = tmp_path / "checkpoints.db"

    model = _TinyWindowChatModel(context_window=6_000)
    sandbox = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=db_path,
        sandbox=sandbox,
        skills=skills,
        connect_mcp=False,
        model_override_resolver=lambda role, spec: _TinyWindowChatModel(
            model_name=spec.model,
            context_window=spec.context_window or 6_000,
        ),
    ) as runtime:
        thread_id = "ctx-auto-compact-test"

        # Turn 1: Well below 80% of 6,000 tokens -> MUST NOT compact
        spec = ModelSpec(
            role="primary",
            provider="custom",
            model="tiny-ctx-test",
            base_url="http://localhost:9999/v1",
            api_key="test",
            context_window=6_000,
        )
        r1 = await runtime.run_task(
            "Hello! Remember project codename APOLLO-42.",
            thread_id=thread_id,
            primary_model_override=spec,
        )
        assert r1["status"] == "completed"
        ctx1 = r1["context_usage"]
        assert ctx1 is not None
        assert ctx1["context_window"] == 6_000
        assert ctx1["usage_percent"] < 80.0
        assert ctx1["compacted_count"] == 0

        # Turn 2: Add a medium message still below 80% -> MUST NOT compact
        r2 = await runtime.run_task(
            "Second turn note: keep tracking metrics. " + ("alpha beta " * 40),
            thread_id=thread_id,
            primary_model_override=spec,
        )
        ctx2 = r2["context_usage"]
        assert ctx2["usage_percent"] < 80.0
        assert ctx2["compacted_count"] == 0

        # Turn 3: Push total context >= 80% of 6,000 tokens -> MUST auto-compact!
        large_payload = "Detailed benchmark record line with diagnostics and telemetry values. " * 140
        r3 = await runtime.run_task(
            f"Third turn heavy input:\n{large_payload}",
            thread_id=thread_id,
            primary_model_override=spec,
        )
        assert r3["status"] == "completed"
        ctx3 = r3["context_usage"]
        assert ctx3 is not None
        assert ctx3["compacted_count"] >= 1
        assert ctx3["last_compacted_at"] is not None
        assert any(ev.get("tool") == "auto_compact_context" for ev in r3.get("tool_events", []))

        by_key = {c["key"]: c["tokens"] for c in ctx3["categories"]}
        assert by_key["compacted_summary"] > 0
