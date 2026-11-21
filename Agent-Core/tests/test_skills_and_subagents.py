"""Phase 11E & 11F: Dynamic Skills and Isolated Sub-Agent Verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from agent_core.graph import create_agent_runtime
from agent_core.sandbox import SandboxManager
from agent_core.skills import SkillRegistry, load_skill
from agent_core.subagents import run_isolated_subagent
from agent_core.tools.local import build_local_tools
from tests.conftest import ScriptedChatModel


def test_skill_discovery_compact_catalog_and_dynamic_loading(tmp_path: Path) -> None:
    """Verify startup injects ONLY name + 1-line description; full bodies load on demand."""
    repo_reg = SkillRegistry(skills_dir=Path.cwd() / "Skills")
    names = {s.name for s in repo_reg.list_skills()}
    assert {"deep_research", "code_and_data"}.issubset(names)

    catalog = repo_reg.format_catalog_for_prompt()
    assert "deep_research:" in catalog
    assert "code_and_data:" in catalog
    # Full instructional body headings must NOT be injected at startup
    assert "## 1. Multi-Source Research Strategy" not in catalog
    assert "## 3. Safe File Operations" not in catalog

    # Dynamic load brings full instructions
    dr_body = repo_reg.load_skill("deep_research")
    assert "Multi-Source Research Strategy" in dr_body
    assert "Source Attribution" in dr_body

    cd_body = load_skill("code_and_data", skills_dir=Path.cwd() / "Skills")
    assert "Repository Inspection" in cd_body
    assert "Data Analysis" in cd_body

    # Adding a brand-new Markdown skill dynamically is discovered without engine changes
    custom_dir = tmp_path / "custom_skills"
    custom_dir.mkdir()
    custom_reg = SkillRegistry(skills_dir=custom_dir)
    assert custom_reg.list_skills() == []

    (custom_dir / "incident_triage.md").write_text(
        "---\nname: incident_triage\ndescription: Structured incident response checklist.\n---\n\n# Steps\n1. Isolate.\n",
        encoding="utf-8",
    )
    loaded = custom_reg.load_skill("incident_triage")
    assert "1. Isolate." in loaded

    # Malformed Markdown without frontmatter still parses safely
    (custom_dir / "malformed.md").write_text("\n\n   # Broken Heading Only\n", encoding="utf-8")
    custom_reg.refresh()
    assert any(s.name == "malformed" for s in custom_reg.list_skills())

    # Invalid & missing skill names raise clean errors
    with pytest.raises(ValueError, match="Invalid skill name"):
        custom_reg.load_skill("../etc/passwd")
    with pytest.raises(FileNotFoundError, match="not found"):
        custom_reg.load_skill("nonexistent_skill")


@pytest.mark.asyncio
async def test_supervisor_loads_skill_dynamically_into_active_skills(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify startup prompt omits full skill body until load_skill is called."""
    seen_system_prompts: list[str] = []

    def handler(msgs: list, call_idx: int) -> AIMessage:
        seen_system_prompts.append(str(msgs[0].content))
        if call_idx == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc-skill", "name": "load_skill", "args": {"name": "deep_research"}}
                ],
            )
        return AIMessage(content="Loaded deep_research skill and completed plan.")

    model = ScriptedChatModel(handler=handler)
    async with create_agent_runtime(
        primary_model=model,
        worker_model=model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task("Research quantum error correction", thread_id="skill-dyn-1")

    assert res["status"] == "completed"
    assert "deep_research" in res["active_skills"]
    # Turn 1 system prompt has catalog only, NOT full body
    assert "deep_research:" in seen_system_prompts[0]
    assert "## 1. Multi-Source Research Strategy" not in seen_system_prompts[0]
    # Turn 2 system prompt includes loaded skill body
    assert "## 1. Multi-Source Research Strategy" in seen_system_prompts[1]


@pytest.mark.asyncio
async def test_subagent_isolation_worker_model_tool_permissions_and_structured_report(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify isolated history, WORKER_MODEL usage, tool permission restriction, and structured report."""
    isolated_sandbox.write_file("/workspace/notes.txt", "Quarterly revenue grew 18%.")

    # Worker tries an unassigned tool (`write_file`) first -> must get PERMISSION DENIED, then uses `read_file`
    worker_calls_received: list[list] = []

    def worker_handler(msgs: list, idx: int) -> AIMessage:
        worker_calls_received.append(msgs)
        if idx == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "w-denied",
                        "name": "write_file",
                        "args": {"path": "/workspace/unauthorized.txt", "content": "nope"},
                    }
                ],
            )
        if idx == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "w-read", "name": "read_file", "args": {"path": "/workspace/notes.txt"}}
                ],
            )
        return AIMessage(
            content=json.dumps(
                {
                    "summary": "Revenue grew 18% per workspace notes.",
                    "key_findings": ["Growth rate: 18%"],
                    "artifacts": ["/workspace/notes.txt"],
                }
            )
        )

    worker_model = ScriptedChatModel(handler=worker_handler, model_name="worker-fast")
    local_tools = {t.name: t for t in build_local_tools(isolated_sandbox)}

    report = await run_isolated_subagent(
        task="Summarize /workspace/notes.txt",
        role="data_analysis",
        worker_model=worker_model,
        available_tools=local_tools,
        allowed_tool_names=["read_file", "list_dir"],
        workspace_dir=str(isolated_sandbox.config.workspace_dir),
    )

    assert report.status == "completed"
    assert "18%" in report.summary
    assert report.tools_used == ["read_file"]
    # Unauthorized write_file must NEVER have executed
    assert not (isolated_sandbox.config.workspace_dir / "unauthorized.txt").exists()
    # Worker saw PERMISSION DENIED in its own isolated message history
    assert "PERMISSION DENIED" in str(worker_calls_received[1][-1].content)
    # Worker bound only its explicitly assigned tools
    assert set(worker_model.bound_tool_names) == {"read_file", "list_dir"}

    # Now test supervisor delegating multiple tasks via spawn_subagent using WORKER_MODEL
    parent_secret_phrase = "PARENT_PRIVATE_CONTEXT_DO_NOT_LEAK"
    delegated_worker_inputs: list[str] = []

    def delegated_worker_handler(msgs: list, _idx: int) -> AIMessage:
        delegated_worker_inputs.append(" ".join(str(m.content) for m in msgs))
        return AIMessage(
            content=json.dumps(
                {
                    "summary": "Worker completed delegated analysis.",
                    "key_findings": ["Finding A"],
                    "artifacts": [],
                }
            )
        )

    supervisor_worker_model = ScriptedChatModel(
        handler=delegated_worker_handler,
        model_name="worker-economical",
    )
    primary_model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-sub-1",
                        "name": "spawn_subagent",
                        "args": {
                            "task": "Review module A",
                            "role": "code_review",
                            "allowed_tools": ["read_file"],
                        },
                    },
                    {
                        "id": "tc-sub-2",
                        "name": "spawn_subagent",
                        "args": {
                            "task": "Draft docs for module A",
                            "role": "documentation",
                            "allowed_tools": ["list_dir"],
                        },
                    },
                ],
            ),
            AIMessage(content="Both delegated worker reports synthesized by supervisor."),
        ],
        model_name="primary-supervisor",
    )

    async with create_agent_runtime(
        primary_model=primary_model,
        worker_model=supervisor_worker_model,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
    ) as runtime:
        res = await runtime.run_task(
            f"Run code review and docs. Note: {parent_secret_phrase}",
            thread_id="subagent-multi-1",
        )

    assert res["status"] == "completed"
    assert len(res["subagent_reports"]) == 2
    assert len(delegated_worker_inputs) == 2
    # Parent conversation history was NOT leaked into isolated workers
    assert all(parent_secret_phrase not in txt for txt in delegated_worker_inputs)


@pytest.mark.asyncio
async def test_subagent_failure_recovery_returns_failed_report(
    isolated_sandbox: SandboxManager,
) -> None:
    crashing_worker = ScriptedChatModel(responses=[RuntimeError("worker provider exploded")])
    local_tools = {t.name: t for t in build_local_tools(isolated_sandbox)}

    report = await run_isolated_subagent(
        task="Run failing task",
        role="code_review",
        worker_model=crashing_worker,
        available_tools=local_tools,
        allowed_tool_names=["read_file"],
    )
    assert report.status == "failed"
    assert "worker provider exploded" in (report.error or "")


@pytest.mark.asyncio
async def test_per_request_dynamic_brain_and_subagent_model_switching(
    isolated_sandbox: SandboxManager,
    repo_skills: SkillRegistry,
    tmp_checkpoint_db: Path,
) -> None:
    """Verify Brain and Sub-Agents dynamically switch to per-request selected LLMs without bleeding."""
    default_brain = ScriptedChatModel(
        responses=[AIMessage(content="Default brain response.")],
        model_name="default-brain",
    )
    default_worker = ScriptedChatModel(
        responses=[
            AIMessage(content='{"summary": "default worker", "key_findings": [], "artifacts": []}')
        ],
        model_name="default-worker",
    )

    override_brain_hits: list[str] = []
    override_worker_hits: list[str] = []

    override_brain = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-dyn-sub",
                        "name": "spawn_subagent",
                        "args": {
                            "task": "Summarize workspace",
                            "role": "research",
                            "allowed_tools": ["list_dir"],
                        },
                    }
                ],
            ),
            AIMessage(content="Override brain synthesized override worker report."),
        ],
        model_name="claude-3.7-sonnet",
    )
    override_worker = ScriptedChatModel(
        responses=[
            AIMessage(
                content='{"summary": "Generated by override worker model.", "key_findings": ["dynamic"], "artifacts": []}'
            )
        ],
        model_name="gpt-4o-mini",
    )

    def _resolver(role: str, spec: Any) -> Any:
        if role == "primary":
            override_brain_hits.append(f"{spec.provider}:{spec.model}")
            return override_brain
        override_worker_hits.append(f"{spec.provider}:{spec.model}")
        return override_worker

    async with create_agent_runtime(
        primary_model=default_brain,
        worker_model=default_worker,
        checkpoint_db_path=tmp_checkpoint_db,
        sandbox=isolated_sandbox,
        skills=repo_skills,
        connect_mcp=False,
        model_override_resolver=_resolver,
    ) as runtime:
        # Run 1: dynamically select OpenRouter Brain + OpenAI Sub-Agent
        res_override = await runtime.run_task(
            "Delegate workspace summary",
            thread_id="dyn-model-thread-1",
            primary_model_override={
                "provider": "openrouter",
                "model": "anthropic/claude-3.7-sonnet",
                "api_key": "sk-or-fake",
            },
            worker_model_override={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-oa-fake",
            },
        )
        assert res_override["status"] == "completed"
        assert "Override brain synthesized" in res_override["final_response"]
        assert (
            res_override["subagent_reports"][0]["summary"] == "Generated by override worker model."
        )
        assert override_brain_hits == ["openrouter:anthropic/claude-3.7-sonnet"]
        assert override_worker_hits == ["openai:gpt-4o-mini"]
        assert res_override["active_models"]["primary"]["provider"] == "openrouter"
        assert res_override["active_models"]["worker"]["provider"] == "openai"

        # Run 2: without overrides, reverts cleanly to default brain (zero cross-request bleed)
        res_default = await runtime.run_task(
            "Simple follow-up",
            thread_id="dyn-model-thread-2",
        )
        assert res_default["final_response"] == "Default brain response."


@pytest.mark.asyncio
async def test_subagent_unclosed_think_recovery_and_final_turn_synthesis(
    isolated_sandbox: SandboxManager,
) -> None:
    """Verify subagent recovers from unclosed <think>/length truncation with role-safe nudges and synthesizes on final turn."""
    from langchain_core.messages import HumanMessage

    isolated_sandbox.write_file("/workspace/metrics.txt", "throughput=420 req/s\n")
    local_tools = {t.name: t for t in build_local_tools(isolated_sandbox)}
    seen_worker_prompts: list[list[Any]] = []

    def worker_handler(msgs: list[Any], idx: int) -> AIMessage:
        seen_worker_prompts.append(list(msgs))
        if idx == 0:
            # Truncated mid-thought with unclosed <think>
            return AIMessage(
                content="<think>Let me inspect metrics.txt first...",
                response_metadata={"finish_reason": "length"},
            )
        if idx == 1:
            # Recovered on retry: calls read_file
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "w-read-1", "name": "read_file", "args": {"path": "/workspace/metrics.txt"}}
                ],
            )
        # Final turn (turn 2 of max_turns=2) synthesizes report cleanly
        return AIMessage(
            content=json.dumps(
                {
                    "summary": "Measured throughput is 420 req/s.",
                    "key_findings": ["throughput=420 req/s"],
                    "artifacts": ["/workspace/metrics.txt"],
                }
            )
        )

    worker_model = ScriptedChatModel(handler=worker_handler, model_name="worker-recover")
    report = await run_isolated_subagent(
        task="Read metrics.txt and summarize",
        role="data_analysis",
        worker_model=worker_model,
        available_tools=local_tools,
        allowed_tool_names=["read_file"],
        max_turns=2,
        workspace_dir=str(isolated_sandbox.config.workspace_dir),
    )

    assert report.status == "completed"
    assert "420 req/s" in report.summary
    assert report.tools_used == ["read_file"]
    # Verify retry prompt had no consecutive HumanMessages
    for i in range(len(seen_worker_prompts[1]) - 1):
        assert not (
            isinstance(seen_worker_prompts[1][i], HumanMessage)
            and isinstance(seen_worker_prompts[1][i + 1], HumanMessage)
        )
    # Verify final-turn synthesis prompt instructed worker to synthesize without more tools
    assert "final worker turn" in str(seen_worker_prompts[2][-1].content).lower()

