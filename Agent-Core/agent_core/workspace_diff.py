"""Workspace filesystem snapshotting and diff tracking utilities.

Pure filesystem inspection helpers used to report created, modified, and deleted
workspace artifacts produced during agent tool executions.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IGNORED_WORKSPACE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".agent_logs",
        ".cache",
        ".venv",
        "venv",
        "node_modules",
    }
)


def snapshot_workspace(
    ws_dir: Path,
    *,
    ignored_dirs: frozenset[str] | set[str] = IGNORED_WORKSPACE_DIRS,
) -> dict[str, tuple[bool, int, int]]:
    """Return ``{rel_posix_path: (is_dir, size, mtime_ns)}`` for entries under ``ws_dir``."""
    snap: dict[str, tuple[bool, int, int]] = {}
    if not ws_dir.exists():
        return snap
    try:
        for root, dirs, files in os.walk(ws_dir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            root_path = Path(root)
            for dname in dirs:
                full = root_path / dname
                try:
                    rel = full.relative_to(ws_dir).as_posix()
                    st = full.stat()
                    snap[rel] = (True, 0, st.st_mtime_ns)
                except OSError:
                    continue
            for fname in files:
                full = root_path / fname
                try:
                    rel = full.relative_to(ws_dir).as_posix()
                    st = full.stat()
                    snap[rel] = (False, st.st_size, st.st_mtime_ns)
                except OSError:
                    continue
    except OSError:
        return snap
    return snap


def diff_workspace_snapshots(
    before: dict[str, tuple[bool, int, int]],
    after: dict[str, tuple[bool, int, int]],
    *,
    tool_name: str,
) -> list[dict[str, Any]]:
    """Compute a sorted list of created/modified/deleted file and directory change records."""
    now_iso = datetime.now(UTC).isoformat()
    changes: list[dict[str, Any]] = []

    for rel in sorted(after.keys() - before.keys()):
        is_dir = after[rel][0]
        changes.append(
            {
                "kind": "dir" if is_dir else "file",
                "action": "created",
                "path": rel,
                "tool": tool_name,
                "timestamp": now_iso,
            }
        )

    for rel in sorted(before.keys() & after.keys()):
        b_dir, b_size, b_mtime = before[rel]
        a_dir, a_size, a_mtime = after[rel]
        if b_dir != a_dir:
            changes.append(
                {
                    "kind": "dir" if a_dir else "file",
                    "action": "modified",
                    "path": rel,
                    "tool": tool_name,
                    "timestamp": now_iso,
                }
            )
        elif not a_dir and (b_size != a_size or b_mtime != a_mtime):
            changes.append(
                {
                    "kind": "file",
                    "action": "modified",
                    "path": rel,
                    "tool": tool_name,
                    "timestamp": now_iso,
                }
            )

    for rel in sorted(before.keys() - after.keys()):
        is_dir = before[rel][0]
        changes.append(
            {
                "kind": "dir" if is_dir else "file",
                "action": "deleted",
                "path": rel,
                "tool": tool_name,
                "timestamp": now_iso,
            }
        )

    return changes
