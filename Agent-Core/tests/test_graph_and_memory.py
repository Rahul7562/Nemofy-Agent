"""Phase 11C: LangGraph Brain, Cyclic Orchestration, Compaction, MAX_TURNS, and Checkpoint Recovery."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from agent_core.graph import create_agent_runtime
from agent_core.sandbox import SandboxManager
from agent_core.skills import SkillRegistry
from agent_core.state import compact_tool_output
from tests.conftest import ScriptedChatModel


@pytest.mark.asyncio
async def test_cyclic_plan_act_observe_reflect_multistep_and_local_tools(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify multi-step planning, local tool execution, observation, reflection, and completion."""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-write",
                        "name": "write_file",
                        "args": {"path": "/workspace/calc.py", "content": "print(21 * 2)\n"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-exec",
                        "name": "bash_exec",
                        "args": {"command": "python3 /workspace/calc.py"},
                    }
                ],
            ),
            AIMessage(content="Created /workspace/calc.py and verified output is 42."),
        ]
    )

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task(
            "Create calc.py printing 21*2 and execute it.",
            thread_id="brain-multistep-1",
        )

    assert res["status"] == "completed"
    assert "42" in res["final_response"]
    assert res["turn_count"] == 3
    assert len(res["plan"]) >= 1
    assert all(step["status"] == "completed" for step in res["plan"])
    assert isolated_sandbox.read_file("/workspace/calc.py") == "print(21 * 2)\n"


@pytest.mark.asyncio
async def test_failed_tool_recovery_and_replanning(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify agent observes tool error, appends recovery step, re-plans, and succeeds."""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-missing",
                        "name": "read_file",
                        "args": {"path": "/workspace/does_not_exist.txt"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-create",
                        "name": "write_file",
                        "args": {"path": "/workspace/recovered.txt", "content": "ok"},
                    }
                ],
            ),
            AIMessage(content="Recovered from missing file and created /workspace/recovered.txt."),
        ]
    )

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("Read or create file", thread_id="brain-replan-1")

    assert res["status"] == "completed"
    statuses = [s["status"] for s in res["plan"]]
    assert "failed" in statuses
    assert "completed" in statuses
    assert isolated_sandbox.read_file("/workspace/recovered.txt") == "ok"


@pytest.mark.asyncio
async def test_max_turns_enforcement_prevents_infinite_loops(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify MAX_TURNS strictly bounds cyclic execution when the model keeps calling tools."""
    looping_model = ScriptedChatModel(
        handler=lambda _msgs, i: AIMessage(
            content="",
            tool_calls=[{"id": f"tc-loop-{i}", "name": "list_dir", "args": {"path": "/workspace"}}],
        )
    )

    async with create_agent_runtime(
        primary_model=looping_model,
        worker_model=looping_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        max_turns=3,
    ) as runtime:
        res = await runtime.run_task("Loop forever", thread_id="brain-maxturns-1")

    assert res["status"] == "completed"
    assert res["turn_count"] == 3
    assert "turn limit" in res["final_response"].lower()


@pytest.mark.asyncio
async def test_context_compaction_archives_over_4000_chars_to_workspace_logs(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
    tmp_workspace: Path,
) -> None:
    """Verify >4,000 char tool outputs are saved in /workspace/.logs/ and summarized."""
    huge_payload = "LINE_DATA_" + ("A" * 6500) + "_END_MARKER"

    # Direct helper check
    summary, saved_vpath = compact_tool_output(
        "huge_tool",
        huge_payload,
        workspace_dir=tmp_workspace,
        threshold=4000,
    )
    assert saved_vpath is not None
    assert saved_vpath.startswith("/workspace/.logs/")
    assert len(summary) < len(huge_payload)
    assert saved_vpath in summary
    assert isolated_sandbox.read_file(saved_vpath) == huge_payload

    # End-to-end inside LangGraph ACT node
    @tool("emit_large_output")
    def emit_large_output() -> str:
        """Return a >4,000 character payload."""
        return huge_payload

    # Note: custom tool name defaults to SENSITIVE; test with read_file on a >4,000 char file (SAFE)
    isolated_sandbox.write_file("/workspace/large_input.txt", huge_payload)

    seen_tool_msg_content: list[str] = []

    def handler(msgs: list, call_idx: int) -> AIMessage:
        if call_idx == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-big",
                        "name": "read_file",
                        "args": {"path": "/workspace/large_input.txt"},
                    }
                ],
            )
        seen_tool_msg_content.append(str(msgs[-1].content))
        return AIMessage(content="Inspected compacted summary cleanly.")

    model = ScriptedChatModel(handler=handler)
    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("Read large file", thread_id="brain-compact-1")

    assert res["status"] == "completed"
    assert len(seen_tool_msg_content) == 1
    assert "COMPACTED OUTPUT" in seen_tool_msg_content[0]
    assert "/workspace/.logs/" in seen_tool_msg_content[0]
    assert len(seen_tool_msg_content[0]) < 3500


@pytest.mark.asyncio
async def test_sqlite_checkpoint_survives_restart_and_never_reexecutes_completed_steps(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify session persistence across full runtime restart and zero duplicate step runs."""
    side_effect_counter = {"count": 0}

    # Step 1 in Runtime instance #1: write file via bash_exec (SAFE)
    model_1 = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-inc",
                        "name": "bash_exec",
                        "args": {"command": "echo run1 >> /workspace/counter.log"},
                    }
                ],
            ),
            AIMessage(content="Turn 1 done."),
        ]
    )

    async with create_agent_runtime(
        primary_model=model_1,
        worker_model=model_1,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime_1:
        res1 = await runtime_1.run_task("Log run1", thread_id="persist-thread-1")
        assert res1["status"] == "completed"

    assert isolated_sandbox.read_file("/workspace/counter.log").strip().splitlines() == ["run1"]

    # Restart completely with Runtime instance #2 on same checkpoints.db & thread_id
    seen_history_lengths: list[int] = []

    def handler_2(msgs: list, _idx: int) -> AIMessage:
        seen_history_lengths.append(len(msgs))
        side_effect_counter["count"] += 1
        return AIMessage(content="Resumed session with full prior context.")

    model_2 = ScriptedChatModel(handler=handler_2)
    async with create_agent_runtime(
        primary_model=model_2,
        worker_model=model_2,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime_2:
        res2 = await runtime_2.run_task("Follow up question", thread_id="persist-thread-1")

    assert res2["status"] == "completed"
    # History carried over: system + turn1 human + ai(tool) + tool + ai(done) + turn2 human >= 6
    assert seen_history_lengths[0] >= 6
    # Step 1 did NOT re-execute on resume
    assert isolated_sandbox.read_file("/workspace/counter.log").strip().splitlines() == ["run1"]
