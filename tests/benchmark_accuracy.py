"""Live Model Accuracy & Benchmark Suite for the Autonomous LangGraph Agent + n8n Gateway.

Runs 6 deterministic, objectively scored benchmarks through the live n8n Gateway webhook
(exercising native OpenAI Chat Model nodes, Heartbeat & Stall Watchdog, Canvas Skills slot,
Canvas MCP Client Tool slot, Isolated Sandbox computation, and Sub-Agent delegation):

1. Exact Algorithmic Computation (primes + SHA-256 via sandbox Python verification)
2. Multi-Step Workspace File Engineering (write CSV, compute exact stats, write JSON)
3. Plug-and-Play Canvas Skill Activation (`executive_brief` via `load_skill` + format compliance)
4. Plug-and-Play Canvas MCP Server Live Invocation (real `search_workflows` via n8n MCP Client Tool)
5. Isolated Sub-Agent Delegation & Synthesis (`spawn_subagent` with verified numeric findings)
6. Strict Structured JSON & Anti-Hallucination Precision
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

GATEWAY_URL = "http://127.0.0.1:5678/webhook/langgraph-agent/run"


@dataclass
class BenchmarkCase:
    id: str
    name: str
    prompt: str
    evaluator: Any


def _expected_prime_sum_and_hash() -> tuple[int, str]:
    primes = [n for n in range(2, 500) if all(n % d != 0 for d in range(2, int(n**0.5) + 1))]
    total = sum(primes)
    digest = hashlib.sha256(str(total).encode("utf-8")).hexdigest()[:8]
    return total, digest


def build_benchmark_cases() -> list[BenchmarkCase]:
    prime_sum, prime_hash = _expected_prime_sum_and_hash()

    def eval_compute(resp: dict[str, Any]) -> tuple[bool, str]:
        text = str(resp.get("final_response") or resp.get("output") or "")
        ok = str(prime_sum) in text and prime_hash.lower() in text.lower()
        return ok, f"expected sum={prime_sum} & sha256[:8]={prime_hash}"

    def eval_workspace(resp: dict[str, Any]) -> tuple[bool, str]:
        text = str(resp.get("final_response") or resp.get("output") or "")
        # Revenue values: 120, 250, 380, 150 -> sum=900, mean=225, max=380
        ok = "900" in text and "225" in text and "380" in text
        return ok, "expected sum=900, mean=225, max=380"

    def eval_canvas_skill(resp: dict[str, Any]) -> tuple[bool, str]:
        text = str(resp.get("final_response") or resp.get("output") or "")
        lower = text.lower()
        skills = [str(s).lower() for s in (resp.get("active_skills") or [])]
        loaded = "executive_brief" in skills
        has_sections = (
            "tl;dr" in lower
            and ("finding" in lower or "metric" in lower)
            and "risk" in lower
            and "next step" in lower
        )
        return (loaded and has_sections), f"active_skills={skills}, sections_ok={has_sections}"

    def eval_canvas_mcp(resp: dict[str, Any]) -> tuple[bool, str]:
        text = str(resp.get("final_response") or resp.get("output") or "")
        tools = list(resp.get("canvas_mcp_tools") or [])
        has_gateway = "LangGraph Autonomous AI Agent Gateway" in text
        has_orch = "Multi-Agent Cognitive Orchestrator" in text
        has_tool = any("search_workflows" in t for t in tools)
        return (
            has_tool and has_gateway and has_orch,
            f"canvas_mcp_tools={tools}, found_both_workflows={has_gateway and has_orch}",
        )

    def eval_subagent(resp: dict[str, Any]) -> tuple[bool, str]:
        text = str(resp.get("final_response") or resp.get("output") or "")
        reports = resp.get("subagent_reports") or []
        # 17 * 23 = 391, 391 + 109 = 500
        ok = len(reports) >= 1 and "391" in text and "500" in text
        return (
            ok,
            f"subagent_reports={len(reports)}, exact_values_in_answer={('391' in text and '500' in text)}",
        )

    def eval_strict_json(resp: dict[str, Any]) -> tuple[bool, str]:
        text = str(resp.get("final_response") or resp.get("output") or "").strip()
        cleaned = text
        if "```" in cleaned:
            parts = cleaned.split("```")
            for p in parts:
                s = p.strip()
                if s.lower().startswith("json"):
                    s = s[4:].strip()
                if s.startswith("{") and s.endswith("}"):
                    cleaned = s
                    break
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"
        # 144 / 12 = 12, 12^3 = 1728
        ok = (
            isinstance(obj, dict)
            and obj.get("quotient") == 12
            and obj.get("cubed") == 1728
            and obj.get("verified") is True
        )
        return ok, f"parsed={obj}"

    return [
        BenchmarkCase(
            id="B1_COMPUTE",
            name="Sandbox Algorithmic Accuracy",
            prompt=(
                "Use bash_exec with Python to compute the exact sum of all prime numbers strictly "
                "less than 500, and the first 8 hex characters of the SHA-256 hash of that decimal "
                "integer string. Report both exact values."
            ),
            evaluator=eval_compute,
        ),
        BenchmarkCase(
            id="B2_WORKSPACE",
            name="Multi-Step Workspace File + Stats",
            prompt=(
                "Create /workspace/bench_sales.csv with header 'region,revenue' and four rows: "
                "North,120; South,250; East,380; West,150. Then use bash_exec to compute the exact "
                "sum, mean, and maximum revenue, save them to /workspace/bench_summary.json, and "
                "report all three numbers."
            ),
            evaluator=eval_workspace,
        ),
        BenchmarkCase(
            id="B3_CANVAS_SKILL",
            name="Plug-and-Play Canvas Skill Activation",
            prompt=(
                "Load the `executive_brief` skill using load_skill first, then write an executive "
                "brief summarizing a 99.95% uptime SLA migration with 12ms p95 latency and zero "
                "downtime."
            ),
            evaluator=eval_canvas_skill,
        ),
        BenchmarkCase(
            id="B4_CANVAS_MCP",
            name="Plug-and-Play Canvas MCP Tool Execution",
            prompt=(
                "Use the connected canvas MCP tool `MCP_Server__n8n_Canvas_search_workflows` "
                "(or whichever search_workflows MCP tool is available) to list the active n8n "
                "workflows and state their exact names."
            ),
            evaluator=eval_canvas_mcp,
        ),
        BenchmarkCase(
            id="B5_SUBAGENT",
            name="Isolated Sub-Agent Delegation",
            prompt=(
                "Delegate a data_analysis sub-agent via spawn_subagent (allowing bash_exec) to "
                "compute 17 * 23 and then add 109 to that product. Report both intermediate (391) "
                "and final (500) verified numbers."
            ),
            evaluator=eval_subagent,
        ),
        BenchmarkCase(
            id="B6_STRICT_JSON",
            name="Strict JSON & Verification Precision",
            prompt=(
                "Compute 144 / 12 as `quotient` and quotient**3 as `cubed`. Return ONLY a raw "
                'JSON object with exact keys {"quotient": <int>, "cubed": <int>, "verified": true} '
                "and no extra prose."
            ),
            evaluator=eval_strict_json,
        ),
    ]


def run_benchmark_suite(model_override: str | None = None) -> dict[str, Any]:
    cases = build_benchmark_cases()
    results: list[dict[str, Any]] = []
    passed = 0
    total_latency_ms = 0
    total_turns = 0

    with httpx.Client(timeout=180.0) as client:
        for case in cases:
            prompt = f"/model {model_override} {case.prompt}" if model_override else case.prompt
            thread_id = f"bench-{case.id.lower()}-{int(time.time())}"
            t0 = time.monotonic()
            resp = client.post(
                GATEWAY_URL,
                json={"task": prompt, "thread_id": thread_id, "max_turns": 12},
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            body = resp.json() if resp.status_code == 200 else {"ok": False, "error": resp.text}
            ok, detail = (
                case.evaluator(body)
                if body.get("ok")
                else (False, f"HTTP/error: {body.get('error')}")
            )
            turns = int(body.get("turn_count") or 0)
            heartbeats = int((body.get("job_monitor") or {}).get("heartbeat_count") or 0)
            if ok:
                passed += 1
            total_latency_ms += elapsed_ms
            total_turns += turns
            results.append(
                {
                    "id": case.id,
                    "name": case.name,
                    "passed": ok,
                    "turns": turns,
                    "heartbeats": heartbeats,
                    "latency_ms": elapsed_ms,
                    "detail": detail,
                }
            )

    accuracy_pct = round((passed / len(cases)) * 100.0, 1) if cases else 0.0
    summary = {
        "model": model_override or "n8n-canvas-default (deepseek-ai/DeepSeek-V4.1-Flash)",
        "passed": passed,
        "total": len(cases),
        "accuracy_pct": accuracy_pct,
        "avg_turns": round(total_turns / len(cases), 2) if cases else 0.0,
        "avg_latency_ms": int(total_latency_ms / len(cases)) if cases else 0,
        "total_latency_ms": total_latency_ms,
        "results": results,
    }
    return summary


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    report = run_benchmark_suite(model_override=model_arg)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] == report["total"] else 1)
