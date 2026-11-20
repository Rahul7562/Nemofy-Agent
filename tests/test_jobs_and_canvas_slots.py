"""Unit & integration tests for Async Jobs, Progress Heartbeat, Stall Watchdog, and n8n Canvas Slots."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage
from src.api import create_app
from src.sandbox import SandboxConfig, SandboxManager
from src.skills import SkillRegistry
from src.tools.mcp_n8n import N8nMCPConfig, N8nMCPManager
from src.tools.n8n_gateway_sync import (
    N8N_MCP_BEARER_CREDENTIAL_ID,
    RIVER_OPENAI_CREDENTIAL_ID,
    build_gateway_workflow,
)


class _ScriptedSlowProgressModel:
    """Simulates a multi-step task whose total wall time exceeds stall_timeout while advancing every step."""

    def __init__(self, *, step_delay: float = 0.35, total_tool_turns: int = 3) -> None:
        self.step_delay = step_delay
        self.total_tool_turns = total_tool_turns
        self.calls = 0

    def bind_tools(self, _tools: list[Any]) -> _ScriptedSlowProgressModel:
        return self

    async def ainvoke(self, _messages: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        await asyncio.sleep(self.step_delay)
        if self.calls <= self.total_tool_turns:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"tc-slow-{self.calls}",
                        "name": "list_dir",
                        "args": {"path": "/workspace"},
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content=f"Completed all {self.total_tool_turns} slow steps cleanly.")


class _HangingModel:
    """Simulates a genuinely stuck/hung LLM call that makes zero forward progress."""

    def bind_tools(self, _tools: list[Any]) -> _HangingModel:
        return self

    async def ainvoke(self, _messages: list[BaseMessage]) -> AIMessage:
        await asyncio.sleep(30.0)
        return AIMessage(content="Should never arrive")


class _CanvasBridgeAndSkillModel:
    """Exercises both a plugged-in n8n canvas Skill (load_skill) and an n8n canvas MCP Tool."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_system_prompts: list[str] = []
        self.seen_tool_names: list[str] = []

    def bind_tools(self, tools: list[Any]) -> _CanvasBridgeAndSkillModel:
        self.seen_tool_names = sorted(t.name for t in tools)
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        if messages:
            self.seen_system_prompts.append(str(messages[0].content))
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-load-canvas-skill",
                        "name": "load_skill",
                        "args": {"name": "canvas_crm_audit"},
                        "type": "tool_call",
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-canvas-mcp",
                        "name": "CRM_MCP_lookup_customer",
                        "args": {"customer_id": "CUST-42"},
                        "type": "tool_call",
                    }
                ],
            )
        last_tool_output = str(messages[-1].content)
        return AIMessage(content=f"Canvas audit complete. Tool result: {last_tool_output}")


@pytest.mark.asyncio
async def test_long_running_task_exceeds_stall_limit_total_time_without_timing_out(
    tmp_path: Path,
) -> None:
    """A job advancing across 3 turns (total ~1.1s) must NOT stall even when stall_timeout=0.7s."""
    sandbox = SandboxManager(SandboxConfig(workspace_dir=tmp_path / "ws", backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")
    mcp = N8nMCPManager(N8nMCPConfig(url="", token=None))
    model = _ScriptedSlowProgressModel(step_delay=0.32, total_tool_turns=3)

    app = create_app(
        api_token="job-secret",
        primary_model=model,  # type: ignore[arg-type]
        worker_model=model,  # type: ignore[arg-type]
        checkpoint_db_path=tmp_path / "ckpt.db",
        sandbox=sandbox,
        skills=skills,
        mcp_manager=mcp,
        connect_mcp=False,
    )

    headers = {"Authorization": "Bearer job-secret"}
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = await client.post(
                "/agent/jobs/run",
                headers=headers,
                json={
                    "task": "Run a multi-step long task",
                    "thread_id": "long-progress-thread",
                    "stall_timeout_seconds": 0.75,
                },
            )
            assert start.status_code == 202, start.text
            job_id = start.json()["job_id"]
            job_token = start.json()["job_token"]

            final_poll: dict[str, Any] = {}
            heartbeats = 0
            for _ in range(30):
                poll = await client.get(
                    f"/agent/jobs/{job_id}?wait_ms=400",
                    headers={"X-Job-Token": job_token},
                )
                assert poll.status_code == 200
                final_poll = poll.json()
                heartbeats += 1
                if final_poll["status"] not in ("staged", "running"):
                    break

            assert final_poll["status"] == "completed", final_poll
            assert final_poll["is_stalled"] is False
            assert final_poll["progress_seq"] >= 6
            assert final_poll["elapsed_seconds"] >= 0.9
            assert "Completed all 3 slow steps cleanly." in final_poll["result"]["final_response"]
            assert heartbeats >= 1


@pytest.mark.asyncio
async def test_stuck_agent_detected_by_stall_watchdog(tmp_path: Path) -> None:
    """When an agent hangs with zero forward progress past stall_timeout, watchdog aborts cleanly."""
    sandbox = SandboxManager(SandboxConfig(workspace_dir=tmp_path / "ws", backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")
    mcp = N8nMCPManager(N8nMCPConfig(url="", token=None))
    hanging = _HangingModel()

    app = create_app(
        api_token="stall-secret",
        primary_model=hanging,  # type: ignore[arg-type]
        worker_model=hanging,  # type: ignore[arg-type]
        checkpoint_db_path=tmp_path / "ckpt.db",
        sandbox=sandbox,
        skills=skills,
        mcp_manager=mcp,
        connect_mcp=False,
    )

    headers = {"Authorization": "Bearer stall-secret"}
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            start = await client.post(
                "/agent/jobs/run",
                headers=headers,
                json={
                    "task": "Trigger stuck model",
                    "thread_id": "stall-thread",
                    "stall_timeout_seconds": 0.45,
                },
            )
            assert start.status_code == 202
            job_id = start.json()["job_id"]
            job_token = start.json()["job_token"]

            final_poll: dict[str, Any] = {}
            for _ in range(15):
                poll = await client.get(
                    f"/agent/jobs/{job_id}?wait_ms=500",
                    headers={"X-Job-Token": job_token},
                )
                assert poll.status_code == 200
                final_poll = poll.json()
                if final_poll["status"] not in ("staged", "running"):
                    break

            assert final_poll["status"] == "stalled", final_poll
            assert final_poll["is_stalled"] is True
            assert final_poll["is_alive"] is False
            assert "stalled" in str(final_poll["error"]).lower()


@pytest.mark.asyncio
async def test_n8n_canvas_mcp_tool_bridge_and_canvas_skills_end_to_end(tmp_path: Path) -> None:
    """Verify staged job -> canvas skill catalog + load_skill -> live n8n bridge tool callback."""
    sandbox = SandboxManager(SandboxConfig(workspace_dir=tmp_path / "ws", backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")
    mcp = N8nMCPManager(N8nMCPConfig(url="", token=None))
    model = _CanvasBridgeAndSkillModel()

    app = create_app(
        api_token="bridge-secret",
        primary_model=model,  # type: ignore[arg-type]
        worker_model=model,  # type: ignore[arg-type]
        checkpoint_db_path=tmp_path / "ckpt.db",
        sandbox=sandbox,
        skills=skills,
        mcp_manager=mcp,
        connect_mcp=False,
    )

    headers = {"Authorization": "Bearer bridge-secret"}
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Stage job with a Canvas Skill
            create_resp = await client.post(
                "/agent/jobs/create",
                headers=headers,
                json={
                    "endpoint": "run",
                    "task": "Audit customer CUST-42 using the canvas CRM skill and CRM MCP tool.",
                    "thread_id": "canvas-slot-thread",
                    "canvas_skills": [
                        {
                            "name": "canvas_crm_audit",
                            "description": "Audit customer accounts via CRM MCP tools.",
                            "content": (
                                "---\nname: canvas_crm_audit\n---\n"
                                "# CRM Audit Checklist\nAlways look up customer_id via CRM_MCP_lookup_customer."
                            ),
                            "source": "n8n_canvas:Skill: CRM Audit",
                        }
                    ],
                },
            )
            assert create_resp.status_code == 202, create_resp.text
            created = create_resp.json()
            job_id = created["job_id"]
            job_token = created["job_token"]
            assert created["status"] == "staged"

            # 2. Launch job registering a connected n8n Canvas MCP tool
            start_resp = await client.post(
                f"/agent/jobs/{job_id}/start",
                headers={"X-Job-Token": job_token},
                json={
                    "n8n_bridge_tools": [
                        {
                            "name": "CRM_MCP_lookup_customer",
                            "description": "Look up a customer record by customer_id.",
                            "args_schema": {
                                "type": "object",
                                "properties": {
                                    "customer_id": {
                                        "type": "string",
                                        "description": "Unique customer ID",
                                    }
                                },
                                "required": ["customer_id"],
                            },
                            "source_node": "MCP Server: CRM",
                            "is_mcp": True,
                        }
                    ]
                },
            )
            assert start_resp.status_code == 200, start_resp.text

            # 3. Poll heartbeat & respond to live bridge tool callback
            tool_executed_args: dict[str, Any] | None = None
            final_poll: dict[str, Any] = {}
            for _ in range(20):
                poll = await client.get(
                    f"/agent/jobs/{job_id}?wait_ms=1000",
                    headers={"X-Job-Token": job_token},
                )
                assert poll.status_code == 200
                final_poll = poll.json()

                pending = final_poll.get("pending_tool_call")
                if pending:
                    assert pending["tool_name"] == "CRM_MCP_lookup_customer"
                    tool_executed_args = pending["arguments"]
                    res_post = await client.post(
                        f"/agent/jobs/{job_id}/tool-result",
                        headers={"X-Job-Token": job_token},
                        json={
                            "call_id": pending["call_id"],
                            "ok": True,
                            "output": {
                                "customer_id": "CUST-42",
                                "tier": "Enterprise",
                                "arr": 95000,
                            },
                        },
                    )
                    assert res_post.status_code == 200
                    continue

                if final_poll["status"] not in ("staged", "running"):
                    break

            assert tool_executed_args == {"customer_id": "CUST-42"}
            assert final_poll["status"] == "completed", final_poll
            assert "CRM_MCP_lookup_customer" in model.seen_tool_names
            assert "canvas_crm_audit" in model.seen_system_prompts[0]
            assert "CRM Audit Checklist" in model.seen_system_prompts[1]
            result = final_poll["result"]
            assert "canvas_crm_audit" in result["active_skills"]
            assert "Enterprise" in result["final_response"]
            assert "95000" in result["final_response"]


def test_gateway_workflow_has_all_four_canvas_slots_and_no_120s_timeout() -> None:
    """Verify Gateway JSON exposes Models, Skills, MCP Servers slots, Heartbeat Bridge, and 0 AI Agents."""
    wf = build_gateway_workflow(agent_api_base_url="http://127.0.0.1:8100")
    nodes = {n["name"]: n for n in wf["nodes"]}
    conns = wf["connections"]

    # 1. Native OpenAI Chat Model slots
    assert nodes["Brain LLM (OpenAI Chat Model)"]["type"] == "@n8n/n8n-nodes-langchain.lmChatOpenAi"
    assert (
        nodes["Sub-Agent LLM (OpenAI Chat Model)"]["type"]
        == "@n8n/n8n-nodes-langchain.lmChatOpenAi"
    )
    assert (
        nodes["Brain LLM (OpenAI Chat Model)"]["credentials"]["openAiApi"]["id"]
        == RIVER_OPENAI_CREDENTIAL_ID
    )

    # 2. Plug-and-play Skills slot (unlimited ai_document connections)
    skills_hub = nodes["Read Canvas Skills"]
    assert skills_hub["type"] == "@n8n/n8n-nodes-langchain.code"
    skill_inputs = skills_hub["parameters"]["inputs"]["input"]
    assert {"type": "ai_document", "maxConnections": -1, "required": False} in skill_inputs
    assert (
        nodes["Skill: Executive Brief"]["type"]
        == "@n8n/n8n-nodes-langchain.documentDefaultDataLoader"
    )
    assert conns["Skill: Executive Brief"]["ai_document"][0][0]["node"] == "Read Canvas Skills"

    # 3. Plug-and-play MCP Servers slot (unlimited ai_tool connections)
    bridge = nodes["Agent Heartbeat & MCP Bridge"]
    assert bridge["type"] == "@n8n/n8n-nodes-langchain.code"
    bridge_inputs = bridge["parameters"]["inputs"]["input"]
    assert {"type": "ai_tool", "maxConnections": -1, "required": False} in bridge_inputs
    mcp_node = nodes["MCP Server: n8n Canvas"]
    assert mcp_node["type"] == "@n8n/n8n-nodes-langchain.mcpClientTool"
    assert mcp_node["credentials"]["httpBearerAuth"]["id"] == N8N_MCP_BEARER_CREDENTIAL_ID
    assert (
        conns["MCP Server: n8n Canvas"]["ai_tool"][0][0]["node"] == "Agent Heartbeat & MCP Bridge"
    )

    # 4. Async Job Start (no 120s blocking /agent/run call)
    start_node = nodes["Start Python Agent Job"]
    assert "/agent/jobs/create" in start_node["parameters"]["url"]
    assert start_node["parameters"]["options"]["timeout"] <= 30000

    # 5. Strict 3-system separation: no n8n AI Agent or Chain node
    node_types = {n["type"] for n in wf["nodes"]}
    assert "@n8n/n8n-nodes-langchain.agent" not in node_types
    assert "@n8n/n8n-nodes-langchain.chainLlm" not in node_types
