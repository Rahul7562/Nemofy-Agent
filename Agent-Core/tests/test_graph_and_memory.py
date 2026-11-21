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


@pytest.mark.asyncio
async def test_live_activity_tracks_thinking_tools_and_workspace_file_dir_changes(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify thinking, tool_events, and workspace file/dir create/modify/delete are captured live."""
    from agent_core.jobs import AgentJob

    job = AgentJob(
        job_id="job-activity-test",
        job_token="tok",
        thread_id="activity-thread-1",
        endpoint="run",
        request_payload={},
        stall_timeout_seconds=30.0,
        status="running",
    )

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="<think>First create a reports directory and write a file inside it.</think>",
                tool_calls=[
                    {
                        "id": "tc-write-1",
                        "name": "write_file",
                        "args": {"path": "/workspace/reports/summary.txt", "content": "v1\n"},
                    }
                ],
            ),
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "Now update summary.txt and clean up a temp file."},
                tool_calls=[
                    {
                        "id": "tc-edit-1",
                        "name": "write_file",
                        "args": {"path": "/workspace/reports/summary.txt", "content": "v2 updated\n"},
                    },
                    {
                        "id": "tc-mkdir-1",
                        "name": "bash_exec",
                        "args": {"command": "mkdir -p /workspace/temp_dir"},
                    },
                    {
                        "id": "tc-del-1",
                        "name": "bash_exec",
                        "args": {
                            "command": (
                                "python3 -c \"import os, shutil; "
                                "os.remove('/workspace/reports/summary.txt'); "
                                "shutil.rmtree('/workspace/temp_dir')\""
                            )
                        },
                    },
                ],
            ),
            AIMessage(content="Finished creating, updating, and cleaning up workspace files."),
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
            "Create reports/summary.txt, update it, then delete it.",
            thread_id="activity-thread-1",
            progress_callback=job.record_progress,
        )

    assert res["status"] == "completed"
    # Thinking accumulated from both <think> tag and reasoning_content
    assert res["thinking"] is not None
    assert "reports directory" in res["thinking"]
    assert "clean up a temp file" in res["thinking"]

    # Tools recorded cleanly with statuses
    tools = res["tool_events"]
    assert len(tools) == 4
    assert [t["tool"] for t in tools] == ["write_file", "write_file", "bash_exec", "bash_exec"]
    assert all(t["status"] == "ok" for t in tools)

    # Workspace file and directory changes detected
    changes = res["file_changes"]
    tuples = {(c["kind"], c["action"], c["path"]) for c in changes}
    assert ("dir", "created", "reports") in tuples
    assert ("file", "created", "reports/summary.txt") in tuples
    assert ("file", "modified", "reports/summary.txt") in tuples
    assert ("dir", "created", "temp_dir") in tuples
    assert ("file", "deleted", "reports/summary.txt") in tuples
    assert ("dir", "deleted", "temp_dir") in tuples

    # Job status payload mirrors live activity
    status_payload = job.to_status_payload()
    assert status_payload["thinking"] is not None
    assert len(status_payload["tool_events"]) == 4
    assert len(status_payload["file_changes"]) >= 6


def test_reasoning_chat_openai_preserves_reasoning_content_and_streams_to_job() -> None:
    """Verify ReasoningChatOpenAI preserves reasoning_content chunks and AgentJob streams thinking_delta + activity_feed."""
    from langchain_core.messages import AIMessageChunk
    from agent_core.jobs import AgentJob
    from agent_core.models import ReasoningChatOpenAI

    model = ReasoningChatOpenAI(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        api_key="test-key",
        base_url="https://api.river.ai/v1",
    )
    gen_chunk = model._convert_chunk_to_generation_chunk(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "reasoning_content": "Step 1: check files. "},
                    "finish_reason": None,
                }
            ],
        },
        AIMessageChunk,
        None,
    )
    assert gen_chunk is not None
    assert gen_chunk.message.additional_kwargs.get("reasoning_content") == "Step 1: check files. "

    # Verify incremental thinking_delta streaming & activity_feed on AgentJob
    job = AgentJob(
        job_id="job-stream-think",
        job_token="tok",
        thread_id="t-stream",
        endpoint="run",
        request_payload={},
        stall_timeout_seconds=30.0,
        status="running",
    )
    job.record_progress({"phase": "plan_start", "detail": "Reviewing your request"})
    job.record_progress({"phase": "thinking", "thinking_delta": "Step 1: check files. "})
    job.record_progress({"phase": "thinking", "thinking_delta": "Step 2: compute 17 * 23 = 391."})

    mid_payload = job.to_status_payload()
    assert mid_payload["thinking"] == "Step 1: check files. Step 2: compute 17 * 23 = 391."
    assert len(mid_payload["activity_feed"]) >= 1
    assert mid_payload["activity_feed"][0]["text"] == "Reviewing your request"


@pytest.mark.asyncio
async def test_self_healing_recovery_on_truncated_tool_call_and_length_limit(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify finish_reason=length mid-tool-call recovers automatically without poisoning thread history."""
    from agent_core.graph import _sanitize_messages_for_llm
    from langchain_core.messages import HumanMessage

    seen_prompts: list[list] = []

    def handler(msgs: list, idx: int) -> AIMessage:
        seen_prompts.append(list(msgs))
        if idx == 0:
            # Exact failure mode from thread-c90299d12d: hit max_tokens mid write_file JSON
            return AIMessage(
                content="",
                tool_calls=[],
                invalid_tool_calls=[
                    {
                        "name": "write_file",
                        "args": '{"path": "/workspace/tb.v", "content": "module tb;',
                        "id": "call-truncated-1",
                        "error": "Unterminated string starting at line 1",
                        "type": "invalid_tool_call",
                    }
                ],
                additional_kwargs={
                    "reasoning_content": "Writing comprehensive self-checking testbench...",
                    "tool_calls": [
                        {
                            "id": "call-truncated-1",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": '{"path": "/workspace/tb.v"'},
                        }
                    ],
                },
                response_metadata={"finish_reason": "length"},
            )
        if idx == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-valid-1",
                        "name": "write_file",
                        "args": {"path": "/workspace/tb.v", "content": "module tb; endmodule\n"},
                    }
                ],
            )
        return AIMessage(content="Successfully created `/workspace/tb.v` after chunking.")

    model = ScriptedChatModel(handler=handler)
    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("Create tb.v", thread_id="recover-truncated-1")

    assert res["status"] == "completed"
    assert "tb.v" in res["final_response"]
    assert "No additional output" not in res["final_response"]
    assert isolated_sandbox.read_file("/workspace/tb.v") == "module tb; endmodule\n"
    # Recovery retry received a targeted SYSTEM RECOVERY prompt mentioning write_file and max_tokens
    assert "SYSTEM RECOVERY" in str(seen_prompts[1][-1].content)
    assert "write_file" in str(seen_prompts[1][-1].content)

    # Verify history sanitizer repairs orphaned tool_calls from legacy interrupted checkpoints
    dirty_history = [
        HumanMessage(content="Hi"),
        AIMessage(
            content="",
            tool_calls=[{"id": "orphan-1", "name": "write_file", "args": {"path": "x", "content": "y"}}],
            additional_kwargs={"tool_calls": [{"id": "orphan-1"}]},
        ),
        HumanMessage(content="Continue"),
    ]
    cleaned = _sanitize_messages_for_llm(dirty_history)
    assert cleaned[1].tool_calls == []
    assert "tool_calls" not in (cleaned[1].additional_kwargs or {})


@pytest.mark.asyncio
async def test_thinking_only_recovery_and_persistent_length_diagnostic(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify thinking-only replies recover and persistent truncation produces an actionable diagnostic."""
    # 1. Thinking-only output followed by real answer on retry
    think_only_model = ScriptedChatModel(
        responses=[
            AIMessage(content="<think>Let me verify the answer internally first.</think>"),
            AIMessage(content="The final verified answer is **42**."),
        ]
    )
    async with create_agent_runtime(
        primary_model=think_only_model,
        worker_model=think_only_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("What is the answer?", thread_id="think-only-recover-1")

    assert res["status"] == "completed"
    assert "42" in res["final_response"]
    assert "turn limit" not in res["final_response"].lower()

    # 2. Persistent finish_reason=length across all retries -> clear actionable diagnostic
    always_length_model = ScriptedChatModel(
        handler=lambda _msgs, _i: AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "Still thinking..."},
            response_metadata={"finish_reason": "length"},
        )
    )
    async with create_agent_runtime(
        primary_model=always_length_model,
        worker_model=always_length_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res2 = await runtime.run_task("Write huge module", thread_id="always-length-1")

    assert res2["status"] == "completed"
    assert "Max Tokens" in res2["final_response"]
    assert "No additional output was returned by the model." not in res2["final_response"]


@pytest.mark.asyncio
async def test_tool_argument_alias_normalization_and_guardrail_safety(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify common LLM arg aliases work smoothly while destructive aliases stay SENSITIVE."""
    from agent_core.guardrails import classify_tool_call

    # Destructive command passed via alias `cmd` MUST still be classified SENSITIVE
    cls = classify_tool_call("bash_exec", {"cmd": "rm -rf /workspace/old"})
    assert cls.safety_level == "SENSITIVE"

    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-alias-write",
                        "name": "write_file",
                        "args": {"file_path": "/workspace/alias.txt", "text": "hello world\n"},
                    },
                    {
                        "id": "tc-alias-edit",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/workspace/alias.txt",
                            "old_str": "world",
                            "new_str": "nemofy",
                        },
                    },
                    {
                        "id": "tc-alias-bash",
                        "name": "bash_exec",
                        "args": {"cmd": "cat /workspace/alias.txt"},
                    },
                ],
            ),
            AIMessage(content="Aliases normalized and executed cleanly."),
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
        res = await runtime.run_task("Use aliased tool params", thread_id="alias-tools-1")

    assert res["status"] == "completed"
    assert isolated_sandbox.read_file("/workspace/alias.txt") == "hello nemofy\n"
    assert all(ev["status"] == "ok" for ev in res["tool_events"])



