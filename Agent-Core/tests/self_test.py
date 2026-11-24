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

    py_files = [
        "Agent-Core/main.py",
        *[str(p) for p in sorted(Path("Agent-Core/agent_core").rglob("*.py"))],
        "MCPs/manager.py",
    ]

    steps: list[tuple[str, list[str]]] = [
        ("Syntax Compile Check", [py, "-m", "py_compile", *py_files]),
        ("Ruff Linter", [ruff_bin, "check", "Agent-Core", "MCPs"]),
    ]
    if Path(mypy_bin).exists():
        steps.append(
            ("Mypy Type Checker", [mypy_bin, "Agent-Core/main.py", "Agent-Core/agent_core", "MCPs"])
        )
    steps.append(("Pytest Full Verification Suite", [pytest_bin, "Agent-Core/tests", "-v"]))

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
