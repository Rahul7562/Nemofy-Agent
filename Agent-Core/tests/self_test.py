"""Comprehensive self-test runner executing static analysis + full pytest verification suite."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_step(label: str, cmd: list[str]) -> int:
    print(f"\n=== {label} ===")
    print("+", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"Agent-Core{os.pathsep}."
    res = subprocess.run(cmd, cwd=str(Path.cwd()), env=env, check=False)
    print(f"-> exit code: {res.returncode}")
    return res.returncode


def main() -> int:
    py = sys.executable
    venv_bin = Path(py).parent
    ruff_bin = str(venv_bin / "ruff")
    mypy_bin = str(venv_bin / "mypy")
    pytest_bin = str(venv_bin / "pytest")

    steps: list[tuple[str, list[str]]] = [
        (
            "Syntax Compile Check",
            [
                py,
                "-m",
                "py_compile",
                "Agent-Core/main.py",
                "Agent-Core/agent_core/models.py",
                "Agent-Core/agent_core/state.py",
                "Agent-Core/agent_core/planner.py",
                "Agent-Core/agent_core/graph.py",
                "Agent-Core/agent_core/skills.py",
                "Agent-Core/agent_core/subagents.py",
                "Agent-Core/agent_core/sandbox.py",
                "Agent-Core/agent_core/guardrails.py",
                "Agent-Core/agent_core/jobs.py",
                "Agent-Core/agent_core/api.py",
                "Agent-Core/agent_core/tools/local.py",
                "MCPs/manager.py",
            ],
        ),
        ("Ruff Linter", [ruff_bin, "check", "Agent-Core", "MCPs"]),
        ("Mypy Type Checker", [mypy_bin, "Agent-Core/main.py", "Agent-Core/agent_core", "MCPs"]),
        ("Pytest Full Verification Suite", [pytest_bin, "Agent-Core/tests", "-v"]),
    ]

    failed = 0
    for label, cmd in steps:
        code = run_step(label, cmd)
        if code != 0:
            failed += 1

    print("\n=== SELF-TEST SUMMARY ===")
    print(f"Total checks: {len(steps)} | Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
