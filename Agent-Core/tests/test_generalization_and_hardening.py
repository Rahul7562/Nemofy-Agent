"""Regression tests for bottom-up hardening fixes and generalized Agent-Core reuse."""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import agent_core
from agent_core import (
    AgentConfig,
    AgentHooks,
    AgentPersona,
    ExecutionLimits,
    GuardrailPolicy,
    SandboxConfig,
    SandboxManager,
    SkillRegistry,
    classify_tool_call,
    create_agent_runtime,
    create_app,
    default_nemofy_persona,
)
from agent_core.firecrawl import FirecrawlClient, FirecrawlConfig
from agent_core.guardrails import validate_tool_arguments
from agent_core.models import ModelSpec, load_persisted_llm_config, save_persisted_llm_config
from agent_core.tools.local import build_local_tools
from tests.conftest import ScriptedChatModel


def test_public_facade_exports_generalized_api() -> None:
    """Verify agent_core exposes a clean public facade for standalone reuse."""
    expected = {
        "AgentConfig",
        "AgentHooks",
        "AgentPersona",
        "AgentRuntime",
        "ExecutionLimits",
        "GuardrailPolicy",
        "SafetyLevel",
        "SandboxConfig",
        "SandboxManager",
        "SkillRegistry",
        "create_agent_runtime",
        "create_app",
        "default_nemofy_persona",
    }
    assert expected.issubset(set(agent_core.__all__))
    for name in expected:
        assert hasattr(agent_core, name)


@pytest.mark.asyncio
async def test_in_memory_sqlite_checkpointer_creates_no_disk_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using checkpoint_db_path=':memory:' must never create a ':memory:' file on disk."""
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")
    model = ScriptedChatModel(responses=[AIMessage(content="Memory-only response.")])

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=":memory:",
        sandbox=sbx,
        skills=skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("Hello from memory", thread_id="mem-only-1")
        assert res["status"] == "completed"
        assert "Memory-only response." in res["final_response"]

    assert not (tmp_path / ":memory:").exists()
    sbx.cleanup_all()


def test_api_health_in_memory_checkpointer_and_no_host_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastAPI /health must support ':memory:' without creating files and never leak host paths."""
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")
    model = ScriptedChatModel(responses=[AIMessage(content="ok")])

    app = create_app(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=":memory:",
        sandbox=sbx,
        skills=skills,
        connect_mcp=False,
        agent_config=AgentConfig(
            app_title="Custom Reusable Agent",
            app_description="Generalized LangGraph service",
        ),
    )

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checkpoint"] == {"exists": True, "in_memory": True}
        assert "skills_dir" not in body["skills"]
        assert "directory" not in body["skills"]
        assert "workspace_dir" not in body["sandbox"]
        assert "config_path" not in body["mcps"]
        assert app.title == "Custom Reusable Agent"

    assert not (tmp_path / ":memory:").exists()
    sbx.cleanup_all()


def test_save_persisted_llm_config_honors_dynamic_env_and_firecrawl_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save_persisted_llm_config must follow dynamic LLM_CONFIG_PATH and validate firecrawl_api_url."""
    custom_cfg = tmp_path / "nested" / "dynamic_llm_config.json"
    monkeypatch.setenv("LLM_CONFIG_PATH", str(custom_cfg))

    saved_path = save_persisted_llm_config(
        {
            "access_mode": "default",
            "firecrawl_api_url": "http://127.0.0.1:3002",
            "primary": {
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
        }
    )
    assert saved_path == custom_cfg
    assert custom_cfg.exists()

    loaded = load_persisted_llm_config()
    assert loaded["firecrawl_api_url"] == "http://127.0.0.1:3002"
    assert loaded["primary"]["model"] == "gpt-4o-mini"

    save_persisted_llm_config({"firecrawl_api_url": "file:///etc/passwd"})
    assert "firecrawl_api_url" not in load_persisted_llm_config()


@pytest.mark.asyncio
async def test_web_extract_schema_json_alias_has_zero_pydantic_warnings(tmp_path: Path) -> None:
    """Building and invoking web_extract with schema_json must emit zero Pydantic UserWarnings."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "data": {"company": "Nemofy", "founded": 2025}},
        )

    client = FirecrawlClient(
        FirecrawlConfig(api_url="http://127.0.0.1:3002", api_key="", max_retries=0),
        transport=httpx.MockTransport(mock_handler),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tools = build_local_tools(sbx, firecrawl_client=client)
        web_extract = {t.name: t for t in tools}["web_extract"]
        validated_args, validation_err = validate_tool_arguments(
            web_extract,
            {
                "urls": ["https://example.com"],
                "prompt": "Extract company info",
                "schema_json": '{"type":"object"}',
            },
        )
        assert validation_err is None
        assert validated_args["json_schema"] == '{"type":"object"}'
        out = await web_extract.ainvoke(
            {
                "urls": ["https://example.com"],
                "prompt": "Extract company info",
                "schema_json": '{"type":"object"}',
            }
        )

    assert "Nemofy" in out
    pydantic_shadow_warnings = [
        w for w in caught if "schema_json" in str(w.message) or "shadows an attribute" in str(w.message)
    ]
    assert pydantic_shadow_warnings == []
    sbx.cleanup_all()


@pytest.mark.asyncio
async def test_generalized_agent_config_persona_tools_policy_memory_saver_and_hooks(
    tmp_path: Path,
) -> None:
    """Verify AgentConfig allows building a completely custom domain agent on agent_core."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")

    @tool("lookup_inventory")
    def lookup_inventory(sku: str) -> str:
        """Look up warehouse stock for a given SKU."""
        return f"SKU {sku}: 42 units available"

    @tool("purge_inventory")
    def purge_inventory(sku: str) -> str:
        """Delete an inventory record."""
        return f"Purged {sku}"

    custom_policy = GuardrailPolicy().extend(
        add_safe_tools=("lookup_inventory",),
        add_sensitive_tools=("purge_inventory",),
    )
    assert classify_tool_call("lookup_inventory", {"sku": "A-1"}, policy=custom_policy).safety_level == "SAFE"
    assert classify_tool_call("purge_inventory", {"sku": "A-1"}, policy=custom_policy).safety_level == "SENSITIVE"

    hook_events: list[str] = []

    def on_turn_start(payload: dict[str, Any]) -> None:
        hook_events.append(f"turn:{payload['turn_count']}")

    async def on_tool_start(payload: dict[str, Any]) -> None:
        hook_events.append(f"tool_start:{payload['tool']}")

    def on_tool_end(payload: dict[str, Any]) -> None:
        hook_events.append(f"tool_end:{payload['tool']}:{payload['status']}")
        # Intentionally raise inside a hook to prove defensive error isolation
        raise RuntimeError("hook failure must not crash agent run")

    async def on_run_complete(payload: dict[str, Any]) -> None:
        hook_events.append(f"done:{payload['status']}")

    seen_system_prompts: list[str] = []

    def handler(msgs: list[BaseMessage], call_idx: int) -> AIMessage:
        if call_idx == 0:
            seen_system_prompts.append(str(msgs[0].content))
            return AIMessage(
                content="",
                tool_calls=[{"id": "tc-inv-1", "name": "lookup_inventory", "args": {"sku": "WIDGET-9"}}],
            )
        return AIMessage(content="WIDGET-9 has 42 units in stock.")

    model = ScriptedChatModel(handler=handler)
    memory_saver = MemorySaver()

    cfg = AgentConfig(
        persona=AgentPersona(
            name="Warehouse Ops Agent",
            organization="Acme Logistics",
            role_description="an inventory operations specialist",
            extra_instructions="Always cite exact SKU stock counts.",
        ),
        limits=ExecutionLimits(max_turns=8, max_tool_retries=2, max_wall_clock_seconds=30.0),
        guardrail_policy=custom_policy,
        hooks=AgentHooks(
            on_turn_start=on_turn_start,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_run_complete=on_run_complete,
        ),
        include_default_tools=False,
        enable_skills=False,
        enable_subagents=False,
        enable_workspace_diff=False,
        custom_checkpointer=memory_saver,
    )

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        sandbox=sbx,
        skills=skills,
        extra_tools=[lookup_inventory, purge_inventory],
        connect_mcp=False,
        config=cfg,
    ) as runtime:
        assert sorted(t.name for t in runtime.tools) == ["lookup_inventory", "purge_inventory"]
        res = await runtime.run_task("How many WIDGET-9 units do we have?", thread_id="ops-1")

    assert res["status"] == "completed"
    assert "42 units" in res["final_response"]
    assert seen_system_prompts
    assert "Warehouse Ops Agent" in seen_system_prompts[0]
    assert "Nemofy" not in seen_system_prompts[0]
    assert "turn:1" in hook_events
    assert "tool_start:lookup_inventory" in hook_events
    assert "tool_end:lookup_inventory:ok" in hook_events
    assert "done:completed" in hook_events
    assert default_nemofy_persona().name == "Nemo"
    sbx.cleanup_all()


@pytest.mark.asyncio
async def test_enable_web_tools_false_excludes_firecrawl_tools(tmp_path: Path) -> None:
    """Setting AgentConfig(enable_web_tools=False) keeps local sandbox tools without web tools."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")
    model = ScriptedChatModel(responses=[AIMessage(content="offline-ok")])

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        sandbox=sbx,
        skills=skills,
        connect_mcp=False,
        config=AgentConfig(
            enable_web_tools=False,
            custom_checkpointer=MemorySaver(),
        ),
    ) as runtime:
        names = {t.name for t in runtime.tools}
        assert "read_file" in names
        assert "bash_exec" in names
        assert "web_search" not in names
        assert "web_fetch" not in names
        assert "web_extract" not in names

    sbx.cleanup_all()


@pytest.mark.asyncio
async def test_fallback_on_timeout_routes_through_model_override_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When primary model times out, fallback_on_timeout must invoke model_override_resolver."""
    monkeypatch.setenv("MODEL_MAX_RETRIES", "0")
    ws = tmp_path / "workspace"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")

    resolved_specs: list[tuple[str, str]] = []

    def resolver(role: str, spec: ModelSpec) -> ScriptedChatModel:
        resolved_specs.append((role, spec.model))
        if spec.model == "slow-primary":
            return ScriptedChatModel(
                model_name="slow-primary",
                handler=lambda _m, _i: TimeoutError("upstream read timed out"),
            )
        return ScriptedChatModel(
            model_name=spec.model,
            responses=[AIMessage(content="Recovered via fallback model!")],
        )

    primary_spec = ModelSpec(
        role="primary",
        provider="openai",
        model="slow-primary",
        base_url="http://localhost:9999/v1",
        api_key="test",
        max_retries=0,
        fallback_on_timeout=True,
        fallback_provider="openai",
        fallback_model="fast-fallback",
    )

    dummy = ScriptedChatModel(responses=[AIMessage(content="unused")])
    async with create_agent_runtime(
        primary_model=dummy,
        worker_model=dummy,
        sandbox=sbx,
        skills=skills,
        connect_mcp=False,
        model_override_resolver=resolver,
        config=AgentConfig(custom_checkpointer=MemorySaver()),
    ) as runtime:
        res = await runtime.run_task(
            "Trigger timeout fallback",
            thread_id="timeout-fb-1",
            primary_model_override=primary_spec,
        )

    assert res["status"] == "completed"
    assert "Recovered via fallback model!" in res["final_response"]
    assert ("primary", "slow-primary") in resolved_specs
    assert ("primary", "fast-fallback") in resolved_specs
    sbx.cleanup_all()


@pytest.mark.asyncio
async def test_execution_limits_wall_clock_cap_stops_runaway_loop(tmp_path: Path) -> None:
    """ExecutionLimits.max_wall_clock_seconds must terminate runaway tool loops cleanly."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
    skills = SkillRegistry(skills_dir=tmp_path / "skills")

    @tool("slow_step")
    async def slow_step() -> str:
        """Sleep briefly and return a status string."""
        await asyncio.sleep(0.15)
        return "step done"

    def handler(_msgs: list[BaseMessage], call_idx: int) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"id": f"tc-{call_idx}", "name": "slow_step", "args": {}}],
        )

    model = ScriptedChatModel(handler=handler)
    cfg = AgentConfig(
        limits=ExecutionLimits(max_turns=20, max_wall_clock_seconds=0.12),
        guardrail_policy=GuardrailPolicy().extend(add_safe_tools=("slow_step",)),
        include_default_tools=False,
        enable_skills=False,
        enable_subagents=False,
        enable_workspace_diff=False,
        custom_checkpointer=MemorySaver(),
    )

    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        sandbox=sbx,
        skills=skills,
        extra_tools=[slow_step],
        connect_mcp=False,
        config=cfg,
    ) as runtime:
        res = await runtime.run_task("Run forever", thread_id="wall-clock-1")

    assert res["status"] == "completed"
    assert "max_wall_clock_seconds" in res["final_response"]
    assert res["turn_count"] < 5
    sbx.cleanup_all()
