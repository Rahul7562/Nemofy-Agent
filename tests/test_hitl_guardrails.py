"""Phase 11H: HARDCORE Human-in-the-Loop (HITL) Guardrails Verification.

Tests all 5 mandatory scenarios on real LangGraph StateGraph + AsyncSqliteSaver checkpoints:
- Scenario 1: Approval (pauses, displays tool & JSON args, executes EXACTLY ONCE on approve)
- Scenario 2: Rejection (never executes, count == 0, graph finishes cleanly)
- Scenario 3: Edited arguments (validation rejects bad args, valid edit re-prompts for explicit
              re-approval, executes ONLY the approved modified args)
- Scenario 4: Restart during approval (tears down runtime & SQLite handle, restarts fresh runtime
              against same checkpoints.db, recovers pending approval, executes once without
              re-running completed prior steps)
- Scenario 5: Sub-agent safety (sub-agent invoking a sensitive tool triggers central interrupt,
              cannot bypass approval on reject or approve)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from src.graph import create_agent_runtime
from src.guardrails import classify_tool_call
from src.sandbox import SandboxManager
from src.skills import SkillRegistry
from tests.conftest import ScriptedChatModel


class SendEmailInput(BaseModel):
    to: str = Field(..., min_length=3, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


def _make_tracked_email_tool() -> tuple[BaseTool, list[dict[str, Any]]]:
    executions: list[dict[str, Any]] = []

    @tool("gmail_send_email", args_schema=SendEmailInput)
    def gmail_send_email(to: str, subject: str, body: str) -> str:
        """Send an email via Gmail (SENSITIVE external action)."""
        record = {"to": to, "subject": subject, "body": body}
        executions.append(record)
        return f"SENT email to {to} with subject {subject!r}"

    return gmail_send_email, executions


def test_central_tool_classification_safe_vs_sensitive() -> None:
    # SAFE local & read-only tools
    assert classify_tool_call("read_file", {"path": "/workspace/a.txt"}).safety_level == "SAFE"
    assert classify_tool_call("list_dir", {"path": "/workspace"}).safety_level == "SAFE"
    assert classify_tool_call("web_fetch", {"url": "https://example.com"}).safety_level == "SAFE"
    assert classify_tool_call("load_skill", {"name": "deep_research"}).safety_level == "SAFE"
    assert (
        classify_tool_call("bash_exec", {"command": "python3 -c 'print(1)'"}).safety_level == "SAFE"
    )
    assert classify_tool_call("search_workflows", {}, is_mcp_tool=True).safety_level == "SAFE"

    # SENSITIVE external & destructive tools
    assert classify_tool_call("gmail_send_email", {"to": "a@b.com"}).safety_level == "SENSITIVE"
    assert classify_tool_call("slack_post_message", {"channel": "#ops"}).safety_level == "SENSITIVE"
    assert (
        classify_tool_call("calendar_create_event", {"summary": "Sync"}).safety_level == "SENSITIVE"
    )
    assert classify_tool_call("sheets_append_row", {"sheet": "Q4"}).safety_level == "SENSITIVE"
    assert (
        classify_tool_call("execute_workflow", {"workflowId": "w1"}, is_mcp_tool=True).safety_level
        == "SENSITIVE"
    )
    assert (
        classify_tool_call("bash_exec", {"command": "rm -rf /workspace/old"}).safety_level
        == "SENSITIVE"
    )
    assert (
        classify_tool_call(
            "bash_exec", {"command": "curl -X POST https://api.example.com/hook -d '{}'"}
        ).safety_level
        == "SENSITIVE"
    )


@pytest.mark.asyncio
async def test_hitl_scenario_1_approval_executes_exactly_once(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    email_tool, executions = _make_tracked_email_tool()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-mail-1",
                        "name": "gmail_send_email",
                        "args": {
                            "to": "alice@example.com",
                            "subject": "Status Update",
                            "body": "Deployment succeeded.",
                        },
                    }
                ],
            ),
            AIMessage(content="Email sent to alice@example.com after approval."),
        ]
    )

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool],
    ) as runtime:
        paused = await runtime.run_task("Email Alice the status update.", thread_id="hitl-s1")

        # Must pause BEFORE execution and display exact tool name + complete JSON args
        assert paused["status"] == "awaiting_approval"
        assert len(executions) == 0
        pending = paused["pending_approval"]
        assert pending is not None
        assert pending["tool_name"] == "gmail_send_email"
        assert pending["arguments"] == {
            "to": "alice@example.com",
            "subject": "Status Update",
            "body": "Deployment succeeded.",
        }
        assert "alice@example.com" in pending["arguments_json"]
        assert pending["expected_effect"]

        # Approve -> resumes from checkpoint -> executes EXACTLY ONCE
        completed = await runtime.resume_approval(
            thread_id="hitl-s1",
            decision={"decision": "approve"},
        )
        assert completed["status"] == "completed"
        assert len(executions) == 1
        assert executions[0]["to"] == "alice@example.com"


@pytest.mark.asyncio
async def test_hitl_scenario_2_rejection_never_executes(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    email_tool, executions = _make_tracked_email_tool()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-mail-2",
                        "name": "gmail_send_email",
                        "args": {
                            "to": "ceo@example.com",
                            "subject": "Draft",
                            "body": "Unreviewed text",
                        },
                    }
                ],
            ),
            AIMessage(content="Understood. The email was not sent because you rejected it."),
        ]
    )

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool],
    ) as runtime:
        paused = await runtime.run_task("Email the CEO.", thread_id="hitl-s2")
        assert paused["status"] == "awaiting_approval"
        assert len(executions) == 0

        resumed = await runtime.resume_approval(
            thread_id="hitl-s2",
            decision={"decision": "reject", "reason": "Wrong recipient"},
        )
        assert resumed["status"] == "completed"
        assert len(executions) == 0
        assert "not sent" in resumed["final_response"].lower()


@pytest.mark.asyncio
async def test_hitl_scenario_3_edited_arguments_validated_and_reapproved(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    email_tool, executions = _make_tracked_email_tool()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-mail-3",
                        "name": "gmail_send_email",
                        "args": {
                            "to": "wrong@example.com",
                            "subject": "Original",
                            "body": "Old body",
                        },
                    }
                ],
            ),
            AIMessage(content="Sent email using your approved edited arguments."),
        ]
    )

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool],
    ) as runtime:
        paused = await runtime.run_task("Send email", thread_id="hitl-s3")
        assert paused["status"] == "awaiting_approval"
        assert len(executions) == 0

        # 1. Invalid edit (malformed email) must fail validation and stay paused (0 executions!)
        invalid_edit = await runtime.resume_approval(
            thread_id="hitl-s3",
            decision={
                "decision": "edit",
                "arguments": {"to": "not-an-email", "subject": "Fixed", "body": "New body"},
            },
        )
        assert invalid_edit["status"] == "awaiting_approval"
        assert invalid_edit["pending_approval"]["validation_error"]
        assert len(executions) == 0

        # 2. Valid edit must NOT execute immediately — must re-display edited args for approval!
        edited_prompt = await runtime.resume_approval(
            thread_id="hitl-s3",
            decision={
                "decision": "edit",
                "arguments": {
                    "to": "verified@example.com",
                    "subject": "Approved Subject",
                    "body": "Clean edited body",
                },
            },
        )
        assert edited_prompt["status"] == "awaiting_approval"
        assert len(executions) == 0
        assert edited_prompt["pending_approval"]["edited"] is True
        assert edited_prompt["pending_approval"]["arguments"]["to"] == "verified@example.com"

        # 3. Explicit approval executes ONLY the approved edited arguments!
        final_res = await runtime.resume_approval(
            thread_id="hitl-s3",
            decision={"decision": "approve"},
        )
        assert final_res["status"] == "completed"
        assert len(executions) == 1
        assert executions[0] == {
            "to": "verified@example.com",
            "subject": "Approved Subject",
            "body": "Clean edited body",
        }


@pytest.mark.asyncio
async def test_hitl_scenario_4_process_restart_during_pending_approval(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Trigger safe step 1 -> pause at sensitive step 2 -> tear down runtime -> restart -> resume."""
    email_tool_1, executions_1 = _make_tracked_email_tool()
    model_1 = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-safe-1",
                        "name": "bash_exec",
                        "args": {"command": "echo step1 >> /workspace/step1.log"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-sens-2",
                        "name": "gmail_send_email",
                        "args": {
                            "to": "ops@example.com",
                            "subject": "Restart Recovery",
                            "body": "Step 1 complete.",
                        },
                    }
                ],
            ),
        ]
    )

    # Instance #1 runs step 1 (SAFE) and pauses on step 2 (SENSITIVE), then shuts down completely
    async with create_agent_runtime(
        primary_model=model_1,
        worker_model=model_1,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool_1],
    ) as runtime_1:
        paused = await runtime_1.run_task("Log step 1 then email ops.", thread_id="hitl-restart-4")
        assert paused["status"] == "awaiting_approval"
        assert len(executions_1) == 0

    assert isolated_sandbox.read_file("/workspace/step1.log").strip().splitlines() == ["step1"]

    # Instance #2 (fresh process/runtime, fresh SQLite connection) recovers pending approval!
    email_tool_2, executions_2 = _make_tracked_email_tool()
    model_2 = ScriptedChatModel(
        responses=[AIMessage(content="Recovered after restart and sent email.")]
    )

    async with create_agent_runtime(
        primary_model=model_2,
        worker_model=model_2,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool_2],
    ) as runtime_2:
        inspected = await runtime_2.inspect_thread("hitl-restart-4")
        assert inspected["status"] == "awaiting_approval"
        assert inspected["pending_approval"]["tool_name"] == "gmail_send_email"

        resumed = await runtime_2.resume_approval(
            thread_id="hitl-restart-4",
            decision={"decision": "approve"},
        )
        assert resumed["status"] == "completed"
        assert len(executions_2) == 1
        assert executions_2[0]["to"] == "ops@example.com"

    # Step 1 did NOT re-execute after restart!
    assert isolated_sandbox.read_file("/workspace/step1.log").strip().splitlines() == ["step1"]


@pytest.mark.asyncio
async def test_hitl_scenario_5_subagent_cannot_bypass_central_approval(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Sub-agent requesting a SENSITIVE tool MUST be caught by central HITL interrupt."""
    # Sub-test A: Reject sub-agent sensitive tool -> 0 executions
    email_tool_rej, executions_rej = _make_tracked_email_tool()
    worker_rej = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "sub-mail-rej",
                        "name": "gmail_send_email",
                        "args": {
                            "to": "external@example.com",
                            "subject": "Subagent Attempt",
                            "body": "Should be caught.",
                        },
                    }
                ],
            ),
            AIMessage(
                content='{"summary": "Action rejected by human.", "key_findings": [], "artifacts": []}'
            ),
        ]
    )
    primary_rej = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "sup-spawn-1",
                        "name": "spawn_subagent",
                        "args": {
                            "task": "Email external contact",
                            "role": "outreach",
                            "allowed_tools": ["gmail_send_email"],
                        },
                    }
                ],
            ),
            AIMessage(content="Sub-agent email was blocked by human rejection."),
        ]
    )

    async with create_agent_runtime(
        primary_model=primary_rej,
        worker_model=worker_rej,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool_rej],
    ) as runtime:
        paused = await runtime.run_task("Delegate outreach email", thread_id="hitl-sub-rej")
        assert paused["status"] == "awaiting_approval"
        assert len(executions_rej) == 0
        assert paused["pending_approval"]["tool_name"] == "gmail_send_email"
        assert paused["pending_approval"]["source"] == "subagent:outreach"

        done = await runtime.resume_approval(
            thread_id="hitl-sub-rej",
            decision={"decision": "reject", "reason": "No unauthorized outreach"},
        )
        assert done["status"] == "completed"
        assert len(executions_rej) == 0

    # Sub-test B: Approve sub-agent sensitive tool -> executes exactly once
    email_tool_ok, executions_ok = _make_tracked_email_tool()
    worker_ok = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "sub-mail-ok",
                        "name": "gmail_send_email",
                        "args": {
                            "to": "approved@example.com",
                            "subject": "Approved Subagent Email",
                            "body": "Authorized.",
                        },
                    }
                ],
            ),
            AIMessage(
                content='{"summary": "Approved email sent.", "key_findings": [], "artifacts": []}'
            ),
        ]
    )
    primary_ok = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "sup-spawn-2",
                        "name": "spawn_subagent",
                        "args": {
                            "task": "Email approved contact",
                            "role": "outreach",
                            "allowed_tools": ["gmail_send_email"],
                        },
                    }
                ],
            ),
            AIMessage(content="Sub-agent email sent after human approval."),
        ]
    )

    async with create_agent_runtime(
        primary_model=primary_ok,
        worker_model=worker_ok,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        extra_tools=[email_tool_ok],
    ) as runtime:
        paused = await runtime.run_task("Delegate approved email", thread_id="hitl-sub-ok")
        assert paused["status"] == "awaiting_approval"
        assert len(executions_ok) == 0
        assert paused["pending_approval"]["source"] == "subagent:outreach"

        done = await runtime.resume_approval(
            thread_id="hitl-sub-ok",
            decision={"decision": "approve"},
        )
        assert done["status"] == "completed"
        assert len(executions_ok) == 1
        assert executions_ok[0]["to"] == "approved@example.com"
