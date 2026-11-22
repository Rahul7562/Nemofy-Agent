"""Verification tests for Phases 3, 4, and 5 in pure-Python Agent-Core.

Runnable via `.venv/bin/python -m unittest Agent-Core/tests/test_audit_features.py`.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_CORE_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_CORE_DIR))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from agent_core.api import (  # noqa: E402
    AccessModeUpdateRequest,
    AgentResumeRequest,
    AgentRunRequest,
    LLMConfigUpdateRequest,
)
from agent_core.context_manager import (  # noqa: E402
    compute_context_usage,
    resolve_model_context_window,
)
from agent_core.graph import AgentRuntime  # noqa: E402
from agent_core.guardrails import (  # noqa: E402
    bind_run_access_mode,
    current_access_mode,
    execute_tool_with_guardrails,
    get_effective_access_mode,
    get_runtime_access_mode,
    normalize_access_mode,
    set_runtime_access_mode,
)
from agent_core.state import (  # noqa: E402
    MAX_MAX_TURNS,
    MIN_MAX_TURNS,
    make_initial_state,
)


class _DummyShellSchema(BaseModel):
    command: str = Field(..., min_length=1)


class _DummySensitiveTool:
    name = "run_shell"
    args_schema = _DummyShellSchema

    def __init__(self) -> None:
        self.invocations: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.invocations.append(dict(args))
        return f"executed:{args.get('command')}"


class TestAuditFeatures(unittest.TestCase):
    def tearDown(self) -> None:
        set_runtime_access_mode("default")

    def test_max_turns_range_1_to_180(self) -> None:
        self.assertEqual(MIN_MAX_TURNS, 1)
        self.assertEqual(MAX_MAX_TURNS, 180)

        # Valid boundaries
        s_min = make_initial_state("hi", thread_id="t1", max_turns=1)
        self.assertEqual(s_min["max_turns"], 1)
        s_max = make_initial_state("hi", thread_id="t1", max_turns=180)
        self.assertEqual(s_max["max_turns"], 180)

        # Clamping in make_initial_state
        self.assertEqual(make_initial_state("hi", thread_id="t1", max_turns=0)["max_turns"], 1)
        self.assertEqual(make_initial_state("hi", thread_id="t1", max_turns=999)["max_turns"], 180)

        # Recursion limit scales cleanly for 180 turns
        stub = AgentRuntime.__new__(AgentRuntime)
        stub.max_turns = 25
        cfg_180 = stub.make_config("t1", max_turns=180)
        self.assertGreaterEqual(cfg_180["recursion_limit"], 180 * 6)

        # API Pydantic schema validation rejects out-of-range values
        self.assertEqual(AgentRunRequest(task="ok", max_turns=1).max_turns, 1)
        self.assertEqual(AgentRunRequest(task="ok", max_turns=180).max_turns, 180)
        with self.assertRaises(ValidationError):
            AgentRunRequest(task="ok", max_turns=0)
        with self.assertRaises(ValidationError):
            AgentRunRequest(task="ok", max_turns=-5)
        with self.assertRaises(ValidationError):
            AgentRunRequest(task="ok", max_turns=181)

        with self.assertRaises(ValidationError):
            LLMConfigUpdateRequest(max_turns=0)
        with self.assertRaises(ValidationError):
            LLMConfigUpdateRequest(max_turns=181)
        self.assertEqual(LLMConfigUpdateRequest(max_turns=180).max_turns, 180)

    def test_access_mode_default_vs_full_access_and_mid_run_switch(self) -> None:
        self.assertEqual(normalize_access_mode("default"), "default")
        self.assertEqual(normalize_access_mode("full_access"), "full_access")
        self.assertEqual(normalize_access_mode("unknown"), "default")

        # API schemas accept both modes and reject invalid ones
        self.assertEqual(AccessModeUpdateRequest(access_mode="full_access").access_mode, "full_access")
        self.assertEqual(
            AgentResumeRequest(thread_id="t1", decision="approve", access_mode="default").access_mode,
            "default",
        )
        with self.assertRaises(ValidationError):
            AccessModeUpdateRequest(access_mode="god_mode")  # type: ignore[arg-type]

        # Mid-run live switching: bind run in 'default', switch runtime to 'full_access' mid-run
        set_runtime_access_mode("default")
        token = bind_run_access_mode("default")
        try:
            self.assertEqual(get_effective_access_mode(), "default")

            # In default mode, sensitive tool MUST invoke interrupt() for HITL approval
            tool = _DummySensitiveTool()
            interrupt_calls: list[dict] = []

            def fake_interrupt(payload: dict) -> dict:
                interrupt_calls.append(payload)
                return {"decision": "approve"}

            with patch("agent_core.guardrails.interrupt", side_effect=fake_interrupt):
                out, audit1 = asyncio.run(
                    execute_tool_with_guardrails(tool, {"command": "echo hi"})
                )
            self.assertEqual(len(interrupt_calls), 1)
            self.assertEqual(out, "executed:echo hi")
            self.assertEqual(audit1.get("access_mode"), "default")
            self.assertEqual(audit1.get("approved"), True)
            self.assertEqual(audit1.get("auto_approved"), False)
            self.assertEqual(audit1.get("executed"), True)

            # Now switch runtime access mode to 'full_access' mid-run (no restart!)
            set_runtime_access_mode("full_access")
            self.assertEqual(get_runtime_access_mode(), "full_access")
            self.assertEqual(get_effective_access_mode(), "full_access")

            # Sensitive tool MUST auto-approve without calling interrupt()
            interrupt_calls.clear()
            with patch("agent_core.guardrails.interrupt", side_effect=fake_interrupt):
                out2, audit2 = asyncio.run(
                    execute_tool_with_guardrails(tool, {"command": "echo auto"})
                )
            self.assertEqual(len(interrupt_calls), 0)
            self.assertEqual(out2, "executed:echo auto")
            self.assertEqual(audit2.get("access_mode"), "full_access")
            self.assertEqual(audit2.get("approved"), True)
            self.assertEqual(audit2.get("auto_approved"), True)
            self.assertEqual(audit2.get("executed"), True)

            # Security invariant: Full Access STILL enforces argument validation & sandbox checks!
            with patch("agent_core.guardrails.interrupt", side_effect=fake_interrupt):
                out_invalid, audit_invalid = asyncio.run(
                    execute_tool_with_guardrails(tool, {"command": ""})
                )
            self.assertIn("error", str(out_invalid).lower())
            self.assertEqual(audit_invalid.get("executed"), False)
            self.assertTrue(audit_invalid.get("error"))
        finally:
            current_access_mode.reset(token)

    def test_context_window_detection_no_fake_128k_fallback(self) -> None:
        # Known models across providers resolve exact windows
        self.assertEqual(resolve_model_context_window("gemini", "gemini-2.5-pro")[0], 1_048_576)
        self.assertEqual(resolve_model_context_window("anthropic", "claude-opus-4-6")[0], 1_000_000)
        self.assertEqual(resolve_model_context_window("openai", "gpt-5.2")[0], 400_000)
        self.assertEqual(resolve_model_context_window("fireworks", "accounts/fireworks/models/deepseek-v3")[0], 128_000)

        # Unknown model MUST return (None, "unknown") instead of pretending 128K
        window, source = resolve_model_context_window("custom", "unregistered-proprietary-model-999")
        self.assertIsNone(window)
        self.assertEqual(source, "unknown")

        snap = compute_context_usage(
            messages=[],
            provider="custom",
            model="unregistered-proprietary-model-999",
            explicit_context_window=None,
        )
        self.assertIsNone(snap["context_window"])
        self.assertEqual(snap["context_window_source"], "unknown")
        self.assertIsNone(snap["usage_percent"])
        self.assertIsNone(snap["free_tokens"])
        self.assertEqual(snap["classification"]["level"], "unknown")


if __name__ == "__main__":
    unittest.main()
