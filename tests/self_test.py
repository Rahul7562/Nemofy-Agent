"""Comprehensive self-test runner executing static analysis + full pytest verification suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(label: str, cmd: list[str]) -> int:
    print(f"\n=== {label} ===")
    print("+", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(Path.cwd()), check=False)
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
                "main.py",
                "src/models.py",
                "src/state.py",
                "src/planner.py",
                "src/graph.py",
                "src/skills.py",
                "src/subagents.py",
                "src/sandbox.py",
                "src/guardrails.py",
                "src/api.py",
                "src/tools/local.py",
                "src/tools/mcp_n8n.py",
                "src/tools/n8n_gateway_sync.py",
            ],
        ),
        ("Ruff Linter", [ruff_bin, "check", "main.py", "src", "tests"]),
        ("Mypy Type Checker", [mypy_bin, "main.py", "src"]),
        ("Pytest Full Verification Suite", [pytest_bin, "-v"]),
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
