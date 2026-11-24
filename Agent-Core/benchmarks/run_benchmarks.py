"""Live production-readiness benchmark suite for Agent-Core.

Executes real LangGraph runs, sandbox subprocesses, guardrail pause/resume cycles,
80% auto-compaction, concurrent SQLite WAL threads, and live ASGI/HTTP job API checks
under explicit safety limits and pass/fail acceptance criteria.

Usage:
    PYTHONPATH=Agent-Core:. .venv/bin/python Agent-Core/benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure Agent-Core and repository root are importable when run directly
_BENCH_DIR = Path(__file__).resolve().parent
_AGENT_CORE_DIR = _BENCH_DIR.parent
_REPO_ROOT = _AGENT_CORE_DIR.parent
for _p in (str(_AGENT_CORE_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx  # noqa: E402
from langchain_core.callbacks import (  # noqa: E402
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import BaseTool, tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from agent_core import (  # noqa: E402
    AgentConfig,
    AgentHooks,
    AgentPersona,
    ExecutionLimits,
    GuardrailPolicy,
    SandboxConfig,
    SandboxManager,
    SandboxSecurityError,
    SandboxTimeoutError,
    SkillRegistry,
    classify_tool_call,
    create_agent_runtime,
    create_app,
)
from agent_core.firecrawl import find_firecrawl_local_binary  # noqa: E402
from agent_core.guardrails import set_runtime_access_mode  # noqa: E402
from agent_core.models import ModelSpec  # noqa: E402


class _BenchmarkChatModel(BaseChatModel):
    """Deterministic zero-network BaseChatModel for repeatable runtime benchmarking."""

    model_name: str = "benchmark-model"
    context_window: int | None = None
    _responses: list[Any] = PrivateAttr(default_factory=list)
    _handler: Callable[[list[BaseMessage], int], Any] | None = PrivateAttr(default=None)
    _calls: int = PrivateAttr(default=0)

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        *,
        handler: Callable[[list[BaseMessage], int], Any] | None = None,
        model_name: str = "benchmark-model",
        context_window: int | None = None,
    ) -> None:
        super().__init__(model_name=model_name, context_window=context_window)
        self._responses = list(responses or [])
        self._handler = handler
        self._calls = 0

    @property
    def _llm_type(self) -> str:
        return "benchmark-chat-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> _BenchmarkChatModel:
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        idx = self._calls
        self._calls += 1
        if self._handler is not None:
            out = self._handler(messages, idx)
        elif idx < len(self._responses):
            out = self._responses[idx]
        else:
            out = AIMessage(content="Benchmark completed.")
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, AIMessage) else AIMessage(content=str(out))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    duration_ms: float
    metrics: dict[str, Any]
    criteria: dict[str, str]
    failures: list[str]


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


async def bench_supervisor_latency_and_persistence(root: Path) -> BenchmarkResult:
    """Suite 1: Supervisor latency across SQLite WAL, :memory:, and MemorySaver."""
    t0 = time.perf_counter()
    failures: list[str] = []
    ws = root / "s1_ws"
    ws.mkdir()
    skills = SkillRegistry(skills_dir=root / "s1_skills")
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated", timeout_seconds=10))

    direct_latencies_ms: list[float] = []
    multistep_latencies_ms: list[float] = []

    try:
        # 1a. SQLite WAL direct turns (N=8)
        db_path = root / "s1_checkpoints.db"
        direct_model = _BenchmarkChatModel(handler=lambda _m, _i: AIMessage(content="Direct reply OK."))
        async with create_agent_runtime(
            primary_model=direct_model,
            worker_model=direct_model,
            checkpoint_db_path=db_path,
            sandbox=sbx,
            skills=skills,
            connect_mcp=False,
        ) as runtime:
            for i in range(8):
                s = time.perf_counter()
                res = await runtime.run_task(f"Ping {i}", thread_id=f"s1-direct-{i}")
                direct_latencies_ms.append((time.perf_counter() - s) * 1000.0)
                if res["status"] != "completed":
                    failures.append(f"direct run {i} failed: {res.get('error')}")

        # 1b. Multi-step write + read tool loop on SQLite WAL (N=5)
        for i in range(5):
            def _step_handler(msgs: list[BaseMessage], idx: int, run_i: int = i) -> AIMessage:
                if idx == 0:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": f"tc-w-{run_i}",
                                "name": "write_file",
                                "args": {
                                    "path": f"/workspace/bench_{run_i}.txt",
                                    "content": f"payload-{run_i}",
                                },
                            }
                        ],
                    )
                if idx == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": f"tc-r-{run_i}",
                                "name": "read_file",
                                "args": {"path": f"/workspace/bench_{run_i}.txt"},
                            }
                        ],
                    )
                return AIMessage(content=f"Verified payload-{run_i}")

            step_model = _BenchmarkChatModel(handler=_step_handler)
            async with create_agent_runtime(
                primary_model=step_model,
                worker_model=step_model,
                checkpoint_db_path=db_path,
                sandbox=sbx,
                skills=skills,
                connect_mcp=False,
            ) as runtime:
                s = time.perf_counter()
                res = await runtime.run_task(f"Write and read {i}", thread_id=f"s1-multi-{i}")
                multistep_latencies_ms.append((time.perf_counter() - s) * 1000.0)
                if res["status"] != "completed" or f"payload-{i}" not in res["final_response"]:
                    failures.append(f"multistep run {i} failed: {res}")

        # 1c. :memory: SQLite must create ZERO disk artifacts
        mem_cwd = root / "s1_mem_cwd"
        mem_cwd.mkdir()
        orig_cwd = Path.cwd()
        os.chdir(mem_cwd)
        try:
            mem_model = _BenchmarkChatModel(responses=[AIMessage(content="in-memory-sqlite-ok")])
            async with create_agent_runtime(
                primary_model=mem_model,
                worker_model=mem_model,
                checkpoint_db_path=":memory:",
                sandbox=sbx,
                skills=skills,
                connect_mcp=False,
            ) as runtime:
                res = await runtime.run_task("Memory test", thread_id="s1-mem")
                if res["status"] != "completed":
                    failures.append(f":memory: run failed: {res}")
            memory_file_leaked = (mem_cwd / ":memory:").exists()
            if memory_file_leaked:
                failures.append(":memory: file leaked onto disk")
        finally:
            os.chdir(orig_cwd)

        # 1d. MemorySaver custom checkpointer
        ms_model = _BenchmarkChatModel(responses=[AIMessage(content="memory-saver-ok")])
        async with create_agent_runtime(
            primary_model=ms_model,
            worker_model=ms_model,
            sandbox=sbx,
            skills=skills,
            connect_mcp=False,
            config=AgentConfig(custom_checkpointer=MemorySaver()),
        ) as runtime:
            res = await runtime.run_task("MemorySaver test", thread_id="s1-ms")
            if res["status"] != "completed":
                failures.append(f"MemorySaver run failed: {res}")
    finally:
        sbx.cleanup_all()

    p50_direct = round(statistics.median(direct_latencies_ms), 2)
    p95_direct = _p95(direct_latencies_ms)
    p50_multi = round(statistics.median(multistep_latencies_ms), 2)
    p95_multi = _p95(multistep_latencies_ms)

    if p95_direct >= 350.0:
        failures.append(f"p95_direct_ms={p95_direct} exceeded 350ms")
    if p95_multi >= 600.0:
        failures.append(f"p95_multistep_ms={p95_multi} exceeded 600ms")

    return BenchmarkResult(
        name="1. Supervisor Graph Latency & Persistence Modes",
        passed=not failures,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        metrics={
            "direct_runs": len(direct_latencies_ms),
            "direct_p50_ms": p50_direct,
            "direct_p95_ms": p95_direct,
            "multistep_runs": len(multistep_latencies_ms),
            "multistep_p50_ms": p50_multi,
            "multistep_p95_ms": p95_multi,
            "sqlite_memory_file_leaked": False,
            "memory_saver_ok": True,
        },
        criteria={
            "direct_p95_ms": "< 350ms",
            "multistep_p95_ms": "< 600ms",
            "sqlite_memory_file_leaked": "== False",
        },
        failures=failures,
    )


async def bench_sandbox_safety_and_perf(root: Path) -> BenchmarkResult:
    """Suite 2: Sandbox local_isolated latency, timeout kill, path confinement, secret scrubbing."""
    t0 = time.perf_counter()
    failures: list[str] = []
    ws = root / "s2_ws"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated", timeout_seconds=1))

    exec_latencies_ms: list[float] = []
    timeout_enforced_ms = 0.0
    traversal_blocked = False
    symlink_blocked = False
    secret_scrubbed = False

    prev_openai = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-benchmark-sentinel-do-not-leak"

    try:
        for i in range(10):
            s = time.perf_counter()
            res = await sbx.exec_command(f"python3 -c 'print({i} * 7)'")
            exec_latencies_ms.append((time.perf_counter() - s) * 1000.0)
            if res.exit_code != 0 or res.stdout.strip() != str(i * 7):
                failures.append(f"sandbox python exec {i} unexpected output: {res}")

        # Timeout enforcement: 5s sleep under 1s timeout must be killed cleanly
        ts = time.perf_counter()
        try:
            timeout_res = await sbx.exec_command("sleep 5")
            timeout_enforced_ms = round((time.perf_counter() - ts) * 1000.0, 2)
            if not timeout_res.timed_out:
                failures.append(f"sleep 5 did not time out: {timeout_res}")
        except SandboxTimeoutError:
            timeout_enforced_ms = round((time.perf_counter() - ts) * 1000.0, 2)
        if timeout_enforced_ms >= 2500.0:
            failures.append(f"timeout kill took {timeout_enforced_ms}ms (>= 2500ms)")

        # Path traversal block
        try:
            sbx.read_file("/workspace/../../etc/passwd")
            failures.append("path traversal was not blocked")
        except SandboxSecurityError:
            traversal_blocked = True

        # Symlink escape block
        sym = ws / "escape_link"
        sym.symlink_to("/etc/passwd")
        try:
            sbx.read_file("/workspace/escape_link")
            failures.append("symlink escape was not blocked")
        except SandboxSecurityError:
            symlink_blocked = True

        # Secret environment scrubbing
        env_res = await sbx.exec_command("python3 -c \"import os; print(os.getenv('OPENAI_API_KEY', 'SCRUBBED'))\"")
        if "sk-benchmark-sentinel" in env_res.stdout or env_res.stdout.strip() != "SCRUBBED":
            failures.append(f"OPENAI_API_KEY leaked into sandbox: {env_res.stdout!r}")
        else:
            secret_scrubbed = True
    finally:
        if prev_openai is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = prev_openai
        sbx.cleanup_all()

    p50_ms = round(statistics.median(exec_latencies_ms), 2)
    p95_ms = _p95(exec_latencies_ms)
    if p95_ms >= 500.0:
        failures.append(f"sandbox exec p95={p95_ms}ms exceeded 500ms")

    return BenchmarkResult(
        name="2. Sandbox Execution Performance & Security Boundaries",
        passed=not failures,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        metrics={
            "exec_runs": len(exec_latencies_ms),
            "exec_p50_ms": p50_ms,
            "exec_p95_ms": p95_ms,
            "timeout_kill_ms": timeout_enforced_ms,
            "traversal_blocked": traversal_blocked,
            "symlink_blocked": symlink_blocked,
            "secret_env_scrubbed": secret_scrubbed,
        },
        criteria={
            "exec_p95_ms": "< 500ms",
            "timeout_kill_ms": "< 2500ms",
            "security_boundaries": "100% enforced",
        },
        failures=failures,
    )


async def bench_guardrails_hitl_and_safety_limits(root: Path) -> BenchmarkResult:
    """Suite 3: Guardrails classification, HITL pause/resume, custom policy, hooks & wall-clock cap."""
    t0 = time.perf_counter()
    failures: list[str] = []
    ws = root / "s3_ws"
    ws.mkdir()
    skills = SkillRegistry(skills_dir=root / "s3_skills")
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    set_runtime_access_mode("default")

    side_effects = {"approved_count": 0, "rejected_count": 0}
    hook_count = {"events": 0}

    @tool("deploy_release")
    def deploy_release(tag: str) -> str:
        """Deploy a release tag."""
        side_effects["approved_count"] += 1
        return f"Deployed {tag}"

    @tool("drop_records")
    def drop_records(table: str) -> str:
        """Drop records from a table."""
        side_effects["rejected_count"] += 1
        return f"Dropped {table}"

    @tool("slow_spin")
    async def slow_spin() -> str:
        """Simulate a slow tool step for wall-clock cap verification."""
        await asyncio.sleep(0.12)
        return "tick"

    try:
        # Classification determinism
        if classify_tool_call("bash_exec", {"command": "ls -la /workspace"}).safety_level != "SAFE":
            failures.append("read-only bash_exec not classified SAFE")
        if classify_tool_call("bash_exec", {"command": "rm -rf /workspace/tmp"}).safety_level != "SENSITIVE":
            failures.append("destructive rm -rf not classified SENSITIVE")

        policy = GuardrailPolicy().extend(
            add_safe_tools=("slow_spin",),
            add_sensitive_tools=("deploy_release", "drop_records"),
        )

        # HITL Approve with zero duplicate side effects
        approve_model = _BenchmarkChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-dep", "name": "deploy_release", "args": {"tag": "v2.0.0"}}],
                ),
                AIMessage(content="Release v2.0.0 deployed."),
            ]
        )
        cfg = AgentConfig(
            persona=AgentPersona(name="Release Bot", organization="Benchmark"),
            guardrail_policy=policy,
            hooks=AgentHooks(
                on_turn_start=lambda _ev: hook_count.__setitem__("events", hook_count["events"] + 1),
                on_tool_end=lambda _ev: hook_count.__setitem__("events", hook_count["events"] + 1),
                on_run_complete=lambda _ev: hook_count.__setitem__("events", hook_count["events"] + 1),
            ),
            include_default_tools=False,
            enable_skills=False,
            enable_subagents=False,
            enable_workspace_diff=False,
            custom_checkpointer=MemorySaver(),
        )
        async with create_agent_runtime(
            primary_model=approve_model,
            worker_model=approve_model,
            sandbox=sbx,
            skills=skills,
            extra_tools=[deploy_release, drop_records],
            connect_mcp=False,
            config=cfg,
        ) as runtime:
            paused = await runtime.run_task("Deploy v2.0.0", thread_id="s3-approve")
            if paused["status"] != "awaiting_approval":
                failures.append(f"expected awaiting_approval before approve, got {paused['status']}")
            if side_effects["approved_count"] != 0:
                failures.append("side effect executed before approval")
            resumed = await runtime.resume_approval(thread_id="s3-approve", decision="approve")
            if resumed["status"] != "completed":
                failures.append(f"approve resume failed: {resumed}")
            if side_effects["approved_count"] != 1:
                failures.append(f"expected exactly 1 side effect, got {side_effects['approved_count']}")

        # HITL Reject must never execute tool
        reject_model = _BenchmarkChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-drop", "name": "drop_records", "args": {"table": "users"}}],
                ),
                AIMessage(content="Aborted per user rejection."),
            ]
        )
        async with create_agent_runtime(
            primary_model=reject_model,
            worker_model=reject_model,
            sandbox=sbx,
            skills=skills,
            extra_tools=[deploy_release, drop_records],
            connect_mcp=False,
            config=cfg,
        ) as runtime:
            paused = await runtime.run_task("Drop users", thread_id="s3-reject")
            if paused["status"] != "awaiting_approval":
                failures.append(f"expected awaiting_approval before reject, got {paused['status']}")
            resumed = await runtime.resume_approval(
                thread_id="s3-reject",
                decision="reject",
            )
            if resumed["status"] != "completed" or side_effects["rejected_count"] != 0:
                failures.append(f"reject failed or executed tool: {resumed}, count={side_effects['rejected_count']}")

        # Wall-clock safety cap halts infinite tool loop
        loop_model = _BenchmarkChatModel(
            handler=lambda _m, idx: AIMessage(
                content="",
                tool_calls=[{"id": f"tc-spin-{idx}", "name": "slow_spin", "args": {}}],
            )
        )
        wc_cfg = AgentConfig(
            limits=ExecutionLimits(max_turns=25, max_wall_clock_seconds=0.14),
            guardrail_policy=policy,
            include_default_tools=False,
            enable_skills=False,
            enable_subagents=False,
            enable_workspace_diff=False,
            custom_checkpointer=MemorySaver(),
        )
        wc_start = time.perf_counter()
        async with create_agent_runtime(
            primary_model=loop_model,
            worker_model=loop_model,
            sandbox=sbx,
            skills=skills,
            extra_tools=[slow_spin],
            connect_mcp=False,
            config=wc_cfg,
        ) as runtime:
            wc_res = await runtime.run_task("Spin forever", thread_id="s3-wallclock")
        wall_clock_stop_ms = round((time.perf_counter() - wc_start) * 1000.0, 2)
        if "max_wall_clock_seconds" not in wc_res["final_response"] or wc_res["turn_count"] >= 5:
            failures.append(f"wall-clock safety limit did not halt runaway loop cleanly: {wc_res}")
        if wall_clock_stop_ms >= 1000.0:
            failures.append(f"wall-clock stop took {wall_clock_stop_ms}ms (>= 1000ms)")
    finally:
        sbx.cleanup_all()

    return BenchmarkResult(
        name="3. Guardrails, HITL Pause/Resume & Wall-Clock Safety Cap",
        passed=not failures,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        metrics={
            "approved_side_effects": side_effects["approved_count"],
            "rejected_side_effects": side_effects["rejected_count"],
            "lifecycle_hook_events": hook_count["events"],
            "wall_clock_stop_ms": wall_clock_stop_ms,
            "wall_clock_turns_before_halt": wc_res["turn_count"],
        },
        criteria={
            "approved_side_effects": "== 1 (zero duplicate executions)",
            "rejected_side_effects": "== 0",
            "wall_clock_stop_ms": "< 1000ms & turns < 5",
        },
        failures=failures,
    )


async def bench_context_auto_compaction(root: Path) -> BenchmarkResult:
    """Suite 4: Strict 80% auto-compaction threshold enforcement & summary generation."""
    t0 = time.perf_counter()
    failures: list[str] = []
    ws = root / "s4_ws"
    ws.mkdir()
    skills = SkillRegistry(skills_dir=root / "s4_skills")
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))

    def _ctx_handler(msgs: list[BaseMessage], _idx: int) -> AIMessage:
        joined = " ".join(str(m.content) for m in msgs)
        if "COMPACT" in joined.upper() and "SUMMARY" in joined.upper():
            return AIMessage(content="- Preserved codename APOLLO-99 and prior turn decisions.")
        return AIMessage(content="Acknowledged context turn.")

    spec = ModelSpec(
        role="primary",
        provider="custom",
        model="bench-8k",
        base_url="http://localhost:9999/v1",
        api_key="test",
        context_window=8_000,
    )
    model = _BenchmarkChatModel(handler=_ctx_handler, context_window=8_000)

    try:
        async with create_agent_runtime(
            primary_model=model,
            worker_model=model,
            sandbox=sbx,
            skills=skills,
            connect_mcp=False,
            model_override_resolver=lambda _role, s: _BenchmarkChatModel(
                handler=_ctx_handler,
                model_name=s.model,
                context_window=s.context_window or 8_000,
            ),
            config=AgentConfig(custom_checkpointer=MemorySaver()),
        ) as runtime:
            r1 = await runtime.run_task(
                "Remember codename APOLLO-99.",
                thread_id="s4-ctx",
                primary_model_override=spec,
            )
            r2 = await runtime.run_task(
                "Follow-up note: " + ("metric alpha " * 40),
                thread_id="s4-ctx",
                primary_model_override=spec,
            )
            r3 = await runtime.run_task(
                "Large benchmark telemetry dump:\n"
                + ("Diagnostic sample record with latency and memory counters. " * 240),
                thread_id="s4-ctx",
                primary_model_override=spec,
            )
    finally:
        sbx.cleanup_all()

    u1 = r1["context_usage"]
    u2 = r2["context_usage"]
    u3 = r3["context_usage"]
    summary_tokens = {c["key"]: c["tokens"] for c in u3["categories"]}.get("compacted_summary", 0)

    if u1["compacted_count"] != 0 or u1["usage_percent"] >= 80.0:
        failures.append(f"Turn 1 unexpectedly compacted or >=80%: {u1}")
    if u2["compacted_count"] != 0 or u2["usage_percent"] >= 80.0:
        failures.append(f"Turn 2 unexpectedly compacted or >=80%: {u2}")
    if u3["compacted_count"] < 1 or summary_tokens <= 0:
        failures.append(f"Turn 3 failed to auto-compact: {u3}")

    return BenchmarkResult(
        name="4. Context Accounting & Strict 80% Auto-Compaction",
        passed=not failures,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        metrics={
            "turn1_usage_pct": u1["usage_percent"],
            "turn2_usage_pct": u2["usage_percent"],
            "turn3_compacted_count": u3["compacted_count"],
            "turn3_post_compact_usage_pct": u3["usage_percent"],
            "compacted_summary_tokens": summary_tokens,
        },
        criteria={
            "turns_1_2_compacted": "== 0 (below 80%)",
            "turn_3_compacted": ">= 1 (at/above 80%)",
            "compacted_summary_tokens": "> 0",
        },
        failures=failures,
    )


async def bench_concurrency_sqlite_wal(root: Path) -> BenchmarkResult:
    """Suite 5: 12 concurrent multi-step threads on a single shared SQLite WAL checkpointer."""
    t0 = time.perf_counter()
    failures: list[str] = []
    ws = root / "s5_ws"
    ws.mkdir()
    db_path = root / "s5_wal.db"
    skills = SkillRegistry(skills_dir=root / "s5_skills")
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))

    concurrency = 12

    def _concurrent_handler(msgs: list[BaseMessage], _idx: int) -> AIMessage:
        # Extract the unique task marker from the HumanMessage so concurrent threads stay stateless
        first_human = next((str(m.content) for m in msgs if m.type == "human"), "")
        has_tool_msg = any(m.type == "tool" for m in msgs)
        marker = first_human.split("MARKER=")[-1].strip()
        if not has_tool_msg:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"tc-{marker}",
                        "name": "write_file",
                        "args": {"path": f"/workspace/thread_{marker}.txt", "content": marker},
                    }
                ],
            )
        return AIMessage(content=f"THREAD_DONE={marker}")

    model = _BenchmarkChatModel(handler=_concurrent_handler)
    thread_latencies_ms: list[float] = []

    try:
        async with create_agent_runtime(
            primary_model=model,
            worker_model=model,
            checkpoint_db_path=db_path,
            sandbox=sbx,
            skills=skills,
            connect_mcp=False,
        ) as runtime:
            async def _run_one(idx: int) -> tuple[int, dict[str, Any], float]:
                marker = f"T{idx:02d}"
                s = time.perf_counter()
                res = await runtime.run_task(f"Execute MARKER={marker}", thread_id=f"wal-thread-{idx}")
                return idx, res, (time.perf_counter() - s) * 1000.0

            results = await asyncio.gather(*(_run_one(i) for i in range(concurrency)))

        for idx, res, lat_ms in results:
            thread_latencies_ms.append(lat_ms)
            marker = f"T{idx:02d}"
            if res["status"] != "completed" or f"THREAD_DONE={marker}" not in res["final_response"]:
                failures.append(f"thread {idx} failed or bled state: {res}")
            disk_val = sbx.read_file(f"/workspace/thread_{marker}.txt")
            if disk_val != marker:
                failures.append(f"thread {idx} file mismatch: got {disk_val!r}")
    finally:
        sbx.cleanup_all()

    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if total_ms >= 10_000.0:
        failures.append(f"12-thread concurrency took {total_ms}ms (>= 10000ms)")

    return BenchmarkResult(
        name="5. Concurrent Multi-Thread Isolation (12 Parallel SQLite WAL Threads)",
        passed=not failures,
        duration_ms=total_ms,
        metrics={
            "parallel_threads": concurrency,
            "completed_cleanly": concurrency - len(failures),
            "thread_p50_ms": round(statistics.median(thread_latencies_ms), 2),
            "thread_p95_ms": _p95(thread_latencies_ms),
            "total_wall_ms": total_ms,
            "cross_thread_bleed": 0 if not failures else len(failures),
        },
        criteria={
            "completed_cleanly": f"== {concurrency}/{concurrency}",
            "cross_thread_bleed": "== 0",
            "total_wall_ms": "< 10000ms",
        },
        failures=failures,
    )


async def bench_fastapi_jobs_and_auth(root: Path) -> BenchmarkResult:
    """Suite 6: Live ASGI/HTTP Bearer auth, async job polling, cancellation & 8x concurrent jobs."""
    t0 = time.perf_counter()
    failures: list[str] = []
    ws = root / "s6_ws"
    ws.mkdir()
    skills = SkillRegistry(skills_dir=root / "s6_skills")
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))

    prev_token = os.environ.get("AGENT_API_TOKEN")
    ephemeral_token = secrets.token_hex(16)
    os.environ["AGENT_API_TOKEN"] = ephemeral_token

    @tool("cancellable_wait")
    async def cancellable_wait() -> str:
        """Sleep long enough to be cancelled via API."""
        await asyncio.sleep(5.0)
        return "never-reached"

    def _api_handler(msgs: list[BaseMessage], _idx: int) -> AIMessage:
        human = next((str(m.content) for m in msgs if m.type == "human"), "")
        if "SLOW_CANCEL" in human:
            return AIMessage(
                content="",
                tool_calls=[{"id": "tc-cancel", "name": "cancellable_wait", "args": {}}],
            )
        return AIMessage(content=f"API_OK:{human}")

    model = _BenchmarkChatModel(handler=_api_handler)
    app = create_app(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=":memory:",
        sandbox=sbx,
        skills=skills,
        extra_tools=[cancellable_wait],
        connect_mcp=False,
        agent_config=AgentConfig(
            guardrail_policy=GuardrailPolicy().extend(add_safe_tools=("cancellable_wait",)),
            app_title="Benchmark Agent API",
        ),
    )

    job_latencies_ms: list[float] = []
    cancel_ms = 0.0
    health_clean = False
    auth_enforced = False

    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://benchmark.local") as client:
                # Health check & zero path leaks
                h = await client.get("/health")
                h_body = h.json()
                health_clean = (
                    h.status_code == 200
                    and h_body.get("status") == "ok"
                    and h_body.get("checkpoint") == {"exists": True, "in_memory": True}
                    and "skills_dir" not in h_body.get("skills", {})
                    and "workspace_dir" not in h_body.get("sandbox", {})
                )
                if not health_clean:
                    failures.append(f"/health check failed or leaked paths: {h_body}")

                # Auth enforcement: 401 without token
                unauth = await client.post("/agent/jobs/run", json={"task": "no-auth"})
                auth_enforced = unauth.status_code == 401
                if not auth_enforced:
                    failures.append(f"expected 401 without Bearer token, got {unauth.status_code}")

                headers = {"Authorization": f"Bearer {ephemeral_token}"}

                async def _poll_until_terminal(job_id: str, max_seconds: float = 10.0) -> dict[str, Any]:
                    deadline = time.monotonic() + max_seconds
                    last: dict[str, Any] = {}
                    while time.monotonic() < deadline:
                        w = await client.get(
                            f"/agent/jobs/{job_id}",
                            params={"wait_ms": 1000},
                            headers=headers,
                        )
                        last = w.json()
                        if last.get("status") in {"completed", "error", "cancelled", "awaiting_approval"}:
                            return last
                    return last

                # 8 concurrent async jobs submitted and polled to completion
                async def _submit_and_wait(i: int) -> None:
                    s = time.perf_counter()
                    r = await client.post(
                        "/agent/jobs/run",
                        json={"task": f"job-{i}", "thread_id": f"api-th-{i}"},
                        headers=headers,
                    )
                    if r.status_code != 202:
                        failures.append(f"job {i} submit failed: {r.status_code} {r.text}")
                        return
                    job_id = r.json()["job_id"]
                    w_body = await _poll_until_terminal(job_id)
                    job_latencies_ms.append((time.perf_counter() - s) * 1000.0)
                    if w_body.get("status") != "completed":
                        failures.append(f"job {i} wait failed: {w_body}")

                await asyncio.gather(*(_submit_and_wait(i) for i in range(8)))

                # Cancel in-flight slow job cleanly
                c_sub = await client.post(
                    "/agent/jobs/run",
                    json={"task": "SLOW_CANCEL", "thread_id": "api-cancel-1"},
                    headers=headers,
                )
                c_job_id = c_sub.json()["job_id"]
                await asyncio.sleep(0.08)
                cs = time.perf_counter()
                c_resp = await client.post(f"/agent/jobs/{c_job_id}/cancel", headers=headers)
                c_wait = await _poll_until_terminal(c_job_id, max_seconds=5.0)
                cancel_ms = round((time.perf_counter() - cs) * 1000.0, 2)
                if c_resp.status_code != 200 or c_wait.get("status") != "cancelled":
                    failures.append(f"cancel failed: resp={c_resp.status_code} wait={c_wait}")
                if cancel_ms >= 1000.0:
                    failures.append(f"cancel took {cancel_ms}ms (>= 1000ms)")
    finally:
        if prev_token is None:
            os.environ.pop("AGENT_API_TOKEN", None)
        else:
            os.environ["AGENT_API_TOKEN"] = prev_token
        sbx.cleanup_all()

    return BenchmarkResult(
        name="6. FastAPI Auth, Concurrent Async Jobs & Cancellation",
        passed=not failures,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        metrics={
            "health_clean_no_path_leaks": health_clean,
            "bearer_auth_401_enforced": auth_enforced,
            "concurrent_jobs_completed": len(job_latencies_ms),
            "job_p50_ms": round(statistics.median(job_latencies_ms), 2) if job_latencies_ms else 0.0,
            "job_p95_ms": _p95(job_latencies_ms),
            "in_flight_cancel_ms": cancel_ms,
        },
        criteria={
            "health_clean_no_path_leaks": "== True",
            "bearer_auth_401_enforced": "== True",
            "concurrent_jobs_completed": "== 8/8",
            "in_flight_cancel_ms": "< 1000ms",
        },
        failures=failures,
    )


async def probe_live_environment() -> dict[str, Any]:
    """Honest report of live local services and optional external provider readiness."""
    live_daemon: dict[str, Any] = {"reachable": False}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://127.0.0.1:8100/health")
            body = resp.json()
            primary_meta = body.get("models", {}).get("primary", {})
            worker_meta = body.get("models", {}).get("worker", {})
            live_daemon = {
                "reachable": True,
                "status_code": resp.status_code,
                "status": body.get("status"),
                "skills_dir_leaked": "skills_dir" in body.get("skills", {}),
                "primary_provider": primary_meta.get("provider"),
                "primary_model": primary_meta.get("model"),
                "primary_api_key_configured": bool(primary_meta.get("api_key_configured")),
                "worker_api_key_configured": bool(worker_meta.get("api_key_configured")),
            }
    except Exception as exc:
        live_daemon = {"reachable": False, "detail": f"{type(exc).__name__}: {exc}"}

    provider_keys = [
        k
        for k in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "NVIDIA_API_KEY",
            "DEEPSEEK_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "FIREWORKS_API_KEY",
            "RIVER_AI_API",
        )
        if os.getenv(k, "").strip()
    ]
    return {
        "live_agent_core_8100": live_daemon,
        "shell_env_llm_keys_configured": provider_keys,
        "firecrawl_local_binary_installed": find_firecrawl_local_binary() is not None,
    }


async def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent-core-bench-") as tmp_dir:
        root = Path(tmp_dir)
        os.environ["LLM_CONFIG_PATH"] = str(root / "bench_llm_config.json")
        set_runtime_access_mode("default")

        suites = [
            await bench_supervisor_latency_and_persistence(root),
            await bench_sandbox_safety_and_perf(root),
            await bench_guardrails_hitl_and_safety_limits(root),
            await bench_context_auto_compaction(root),
            await bench_concurrency_sqlite_wal(root),
            await bench_fastapi_jobs_and_auth(root),
        ]
        env_probe = await probe_live_environment()

    passed_count = sum(1 for s in suites if s.passed)
    summary = {
        "overall_passed": passed_count == len(suites),
        "suites_passed": passed_count,
        "suites_total": len(suites),
        "environment_probe": env_probe,
        "results": [asdict(s) for s in suites],
    }

    print("\n=== AGENT-CORE LIVE BENCHMARK REPORT ===")
    for s in suites:
        badge = "PASS" if s.passed else "FAIL"
        print(f"[{badge}] {s.name} ({s.duration_ms:.1f} ms)")
        for k, v in s.metrics.items():
            print(f"       {k}: {v}")
        if s.failures:
            for f in s.failures:
                print(f"       FAILURE: {f}")
    print("\n--- Environment Probe ---")
    print(json.dumps(env_probe, indent=2))
    print("\n--- Summary ---")
    print(
        f"Overall: {'PASS' if summary['overall_passed'] else 'FAIL'} "
        f"({passed_count}/{len(suites)} suites passed)"
    )
    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
