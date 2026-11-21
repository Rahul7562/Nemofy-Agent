"""Structured planning, prompt assembly, and reflection parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from agent_core.state import PlanStep, PlanStepModel, PlanStepStatus


def extract_json_block(raw_text: str) -> dict[str, Any] | None:
    """Extract a JSON object from raw model output (direct JSON, fenced block, or braces)."""
    text = raw_text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            parsed = json.loads(text[first : last + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def parse_structured_plan(
    raw_output: str,
    *,
    existing_plan: list[PlanStep] | None = None,
    fallback_goal: str = "Fulfill the user request",
) -> tuple[list[PlanStep], str | None]:
    existing_by_id: dict[str, PlanStep] = {step["step_id"]: step for step in (existing_plan or [])}

    parsed = extract_json_block(raw_output)
    steps: list[PlanStep] = []
    rationale: str | None = None

    if parsed:
        raw_rationale = parsed.get("rationale") or parsed.get("summary")
        if isinstance(raw_rationale, str) and raw_rationale.strip():
            rationale = raw_rationale.strip()

        raw_steps = parsed.get("plan") or parsed.get("steps")
        if isinstance(raw_steps, list):
            for idx, item in enumerate(raw_steps, start=1):
                if not isinstance(item, dict):
                    continue
                step_id = str(item.get("step_id") or item.get("id") or f"step-{idx}").strip()
                desc = str(item.get("description") or "").strip()
                if not desc:
                    continue
                tool_or_action = str(
                    item.get("tool_or_action") or item.get("tool") or item.get("action") or "reason"
                ).strip()
                status_raw = str(item.get("status") or "pending").strip().lower()
                status: PlanStepStatus
                if status_raw == "in_progress":
                    status = "in_progress"
                elif status_raw == "completed":
                    status = "completed"
                elif status_raw == "failed":
                    status = "failed"
                elif status_raw == "skipped":
                    status = "skipped"
                else:
                    status = "pending"
                result_str = str(item.get("result") or "")

                prev = existing_by_id.get(step_id)
                if prev and prev["status"] == "completed":
                    steps.append(prev)
                    continue

                validated = PlanStepModel(
                    step_id=step_id,
                    description=desc,
                    status=status,
                    tool_or_action=tool_or_action,
                    result=result_str or (prev["result"] if prev else ""),
                )
                steps.append(validated.to_typed_dict())

    if not steps:
        if existing_plan:
            return list(existing_plan), rationale
        steps = [
            PlanStep(
                step_id="step-1",
                description=fallback_goal[:240],
                status="pending",
                tool_or_action="direct_or_tool",
                result="",
            )
        ]

    return steps, rationale


def get_next_pending_step(plan: list[PlanStep]) -> PlanStep | None:
    for step in plan:
        if step["status"] in ("in_progress", "pending"):
            return step
    return None


def update_plan_step(
    plan: list[PlanStep],
    step_id: str,
    *,
    status: PlanStepStatus,
    result: str | None = None,
) -> list[PlanStep]:
    updated: list[PlanStep] = []
    for step in plan:
        if step["step_id"] == step_id:
            next_step: PlanStep = {
                "step_id": step["step_id"],
                "description": step["description"],
                "status": status,
                "tool_or_action": step["tool_or_action"],
                "result": step["result"] if result is None else result,
            }
            updated.append(next_step)
        else:
            updated.append(step)
    return updated


def format_plan_for_prompt(plan: list[PlanStep]) -> str:
    if not plan:
        return "(no active plan yet)"
    lines: list[str] = []
    for step in plan:
        result_snippet = f" -> {step['result'][:160]}" if step.get("result") else ""
        lines.append(
            f"- [{step['status'].upper()}] {step['step_id']} ({step['tool_or_action']}): "
            f"{step['description']}{result_snippet}"
        )
    return "\n".join(lines)


def build_supervisor_system_prompt(
    *,
    skill_catalog_text: str,
    loaded_skills_text: str,
    available_tool_names: list[str],
    plan: list[PlanStep],
    turn_count: int,
    max_turns: int,
) -> str:
    """Build the LangGraph supervisor system prompt with compact skill catalog injection."""
    tools_line = ", ".join(sorted(available_tool_names)) if available_tool_names else "(none)"
    plan_text = format_plan_for_prompt(plan)

    sections = [
        (
            "You are Nemo, the AI assistant working inside Nemofy Agent. "
            "You are diligent, honest, warm, and friendly, capable of helping with everything "
            "from quick everyday questions and writing to complex multi-step research, data "
            "analysis, coding, and workspace automation. Never introduce yourself as a 'pure-Python "
            "agent' or describe internal graph plumbing unless the user specifically asks about "
            "the underlying architecture."
        ),
        (
            "Workspace & capabilities:\n"
            "- Workspace tools: bash_exec, read_file, write_file, edit_file, list_dir, and "
            "web_fetch operate safely inside the workspace sandbox. File tools accept relative paths "
            "(e.g. 'src/app.py') or '/workspace/...'. Inside scripts or code run via bash_exec, the "
            "current working directory (CWD) is already the workspace root and $WORKSPACE points to "
            "it — use relative paths (e.g. 'data.txt') or os.environ.get('WORKSPACE', '.') rather "
            "than hardcoding literal '/workspace/...' paths inside script source code.\n"
            "- External integrations: Connected Model Context Protocol (MCP) tools handle live "
            "external services when configured.\n"
            "- Skills: Call load_skill(name) FIRST whenever a task matches a skill in the catalog "
            "and follow its instructions carefully.\n"
            "- Delegation: Use spawn_subagent for large independent workstreams (deep research, "
            "repo audits, code review, data analysis, documentation).\n"
            "- Safety & honesty: Never fabricate tool outputs, file contents, or facts. If "
            "something is uncertain or fails, say so honestly and clearly. Sensitive actions pause "
            "for explicit human approval."
        ),
        f"Turn budget: {turn_count}/{max_turns}\nAvailable tools: {tools_line}",
        f"Available skills (metadata only — call load_skill(name) to load full instructions):\n"
        f"{skill_catalog_text}",
    ]

    if loaded_skills_text:
        sections.append(f"Currently loaded skill instructions:\n{loaded_skills_text}")

    sections.append(f"Current execution plan:\n{plan_text}")
    sections.append(
        "Execution, accuracy & presentation rules:\n"
        "1. Do not re-execute steps already marked COMPLETED.\n"
        "2. Verify non-trivial math, statistics, parsing, or code logic with bash_exec (Python) "
        "instead of guessing mentally.\n"
        "3. Never fabricate tool outputs, file contents, or external facts; ground every claim "
        "in observed tool output or mark uncertainty explicitly.\n"
        "4. Only call tools needed for the active step; if a tool fails, inspect the error and "
        "recover within bounded retries.\n"
        "5. Keep individual write_file calls concise so they never truncate mid-JSON; for large "
        "files (>200 lines), write a clean base with write_file and extend with edit_file or a "
        "short Python script via bash_exec. Before compiling/simulating in specialized toolchains, "
        "check tool availability with `which` and if unavailable, report honestly or verify with "
        "a Python reference model/testbench.\n"
        "6. Format final responses clearly in GitHub-Flavored Markdown: use **bold** for key "
        "points, concise headings, bullet/numbered lists, Markdown data tables (`| col | col |`) "
        "for comparisons or structured datasets, fenced code blocks with language tags for code, "
        "and clean monospace ASCII charts/diagrams (fenced with ```ascii or ```chart) whenever "
        "visualizing trends, architectures, or distributions helps the user."
    )
    return "\n\n".join(sections)


def parse_reflection_decision(
    raw_output: str,
    *,
    has_pending_steps: bool,
    turn_count: int,
    max_turns: int,
) -> tuple[bool, str]:
    if turn_count >= max_turns:
        return False, f"Reached MAX_TURNS ({max_turns}); terminating cleanly."

    parsed = extract_json_block(raw_output)
    if parsed:
        decision = str(parsed.get("decision") or parsed.get("next_action") or "").strip().lower()
        note = str(parsed.get("reflection") or parsed.get("reason") or raw_output).strip()
        if decision in {"complete", "done", "finish", "stop"}:
            return False, note
        if decision in {"continue", "replan", "retry"}:
            return True, note

    lowered = raw_output.strip().lower()
    if "complete" in lowered and "continue" not in lowered and not has_pending_steps:
        return False, raw_output.strip() or "All planned steps complete."
    return has_pending_steps, raw_output.strip() or (
        "Continuing remaining plan steps." if has_pending_steps else "Task completed."
    )
