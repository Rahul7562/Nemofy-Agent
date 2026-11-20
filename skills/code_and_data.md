---
name: code_and_data
description: Software engineering and data analysis workflow covering repository inspection, Python programming, debugging, testing, data processing, and safe file operations.
---

# Code & Data Engineering Skill

Use this skill for software development, debugging, automated testing, refactoring, script authoring, and tabular/statistical data analysis inside the sandbox.

## 1. Repository Inspection
- Start by inspecting directory layout (`list_dir`) and reading configuration/manifests (`read_file`) before editing code.
- Identify existing conventions, types, test runners, and dependencies so changes integrate cleanly without breaking working code.
- Prefer targeted searches (`grep`, `rg`, `find` via `bash_exec`) over dumping entire large files into context.

## 2. Python Programming Standards
- Write clean, modular Python with explicit type annotations, docstrings, and structured error handling.
- Avoid mutable default arguments, bare `except:`, and hardcoded credentials.
- Validate inputs at boundaries and close file/network/database handles cleanly with context managers.

## 3. Safe File Operations
- Keep all file reads, writes, edits, and generated artifacts strictly inside `/workspace`.
- Use `edit_file` for surgical modifications to existing files; use `write_file` for new files or deliberate full rewrites.
- Never run destructive recursive deletes (`rm -rf`) or overwrite user data without explicit verification and approval.

## 4. Debugging Workflow
- Reproduce the failure first with a minimal deterministic command or test case.
- Read full tracebacks and logs (including archived `/workspace/.logs/` files when tool output was compacted).
- Isolate the root cause before applying a fix—avoid superficial try/except masking.

## 5. Testing & Verification
- Run unit tests (`pytest`) and static checks (`ruff`, `mypy`, `python -m py_compile`) after every code change.
- Test edge cases: empty inputs, malformed data, boundary sizes, timeouts, and error paths.
- Report exact command outputs and pass/fail counts accurately.

## 6. Data Analysis & Processing
- Inspect schema, row counts, missing values, duplicates, and data types before running aggregations.
- Perform numerical computations programmatically in Python (e.g., `csv`, `json`, `statistics`, `pandas`, `numpy`) and verify totals/invariants.
- Save processed outputs (CSV, JSON, charts, reports) cleanly under `/workspace/` and summarize key metrics with units and sample sizes.
