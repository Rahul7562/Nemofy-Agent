"""Phase 3A: Dynamic Markdown skill catalog and on-demand loader.

At startup, only skill names and one-line descriptions are extracted and injected
into the supervisor prompt. Full Markdown bodies are loaded lazily via ``load_skill``.
Adding a new skill requires only dropping a ``.md`` file into ``./skills/``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

DEFAULT_SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "./skills"))

_VALID_SKILL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(frozen=True)
class SkillMetadata:
    """Compact startup metadata for a discovered Markdown skill."""

    name: str
    description: str
    path: Path


def _parse_skill_header(path: Path) -> tuple[str, str]:
    """Extract skill name and one-line description from YAML frontmatter or Markdown header."""
    stem_name = path.stem.strip().lower()
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stem_name, "Skill file unreadable."

    name = stem_name
    description = ""

    # 1. Optional YAML frontmatter: --- \n name: ... \n description: ... \n ---
    fm_match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*(?:\n|$)", raw)
    body = raw
    if fm_match:
        frontmatter = fm_match.group(1)
        body = raw[fm_match.end() :]
        for line in frontmatter.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            k = key.strip().lower()
            v = val.strip().strip("\"'")
            if k == "name" and v and _VALID_SKILL_NAME_RE.match(v):
                name = v.lower()
            elif k == "description" and v:
                description = v

    # 2. Fallback: first non-empty non-heading line, or heading text
    if not description:
        heading_fallback = ""
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if not heading_fallback:
                    heading_fallback = stripped.lstrip("#").strip()
                continue
            description = stripped.lstrip("-* ").strip()
            break
        if not description:
            description = heading_fallback or f"Specialized instructions for {name}."

    # Keep one-line description compact
    one_line = " ".join(description.split())
    if len(one_line) > 180:
        one_line = one_line[:177].rstrip() + "..."
    return name, one_line


class SkillRegistry:
    """Scans ``skills/`` for Markdown files and loads full skill content on demand."""

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir is not None else DEFAULT_SKILLS_DIR
        self._catalog: dict[str, SkillMetadata] = {}
        self.refresh()

    def refresh(self) -> dict[str, SkillMetadata]:
        """Rescan the skills directory for all ``*.md`` skill definitions."""
        catalog: dict[str, SkillMetadata] = {}
        if self.skills_dir.exists() and self.skills_dir.is_dir():
            for md_file in sorted(self.skills_dir.glob("*.md")):
                if not md_file.is_file():
                    continue
                name, desc = _parse_skill_header(md_file)
                catalog[name] = SkillMetadata(
                    name=name,
                    description=desc,
                    path=md_file.resolve(),
                )
        self._catalog = catalog
        return dict(self._catalog)

    def list_skills(self) -> list[SkillMetadata]:
        """Return compact metadata for all discovered skills."""
        return list(self._catalog.values())

    def format_catalog_for_prompt(self) -> str:
        """Render ONLY skill names and one-line descriptions for startup injection."""
        if not self._catalog:
            return "(no skills installed in ./skills/)"
        return "\n".join(f"- {meta.name}: {meta.description}" for meta in self._catalog.values())

    def load_skill(self, name: str) -> str:
        """Load the full Markdown instructions for a skill on demand."""
        normalized = name.strip().lower()
        if not normalized or not _VALID_SKILL_NAME_RE.match(normalized):
            raise ValueError(
                f"Invalid skill name {name!r}. Use alphanumeric, underscore, or hyphen characters."
            )

        # Refresh on cache miss so newly dropped Markdown files work without restart
        if normalized not in self._catalog:
            self.refresh()

        meta = self._catalog.get(normalized)
        if meta is None:
            available = ", ".join(sorted(self._catalog.keys())) or "none"
            raise FileNotFoundError(
                f"Skill {normalized!r} not found. Available skills: {available}."
            )

        if not meta.path.exists():
            self.refresh()
            raise FileNotFoundError(f"Skill file for {normalized!r} no longer exists on disk.")

        content = meta.path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return f"# Skill: {meta.name}\n\n(Skill file is empty.)"
        return content


def make_load_skill_tool(registry: SkillRegistry) -> BaseTool:
    """Create the LangChain tool ``load_skill(name)`` bound to a ``SkillRegistry``."""

    @tool("load_skill")
    def load_skill(name: str) -> str:
        """Load full specialized instructions for a skill by name from ./skills/.

        Call this tool only when a task matches one of the advertised skills in the catalog.
        """
        try:
            return registry.load_skill(name)
        except (ValueError, FileNotFoundError) as exc:
            return f"ERROR: {exc}"

    return load_skill


def load_skill(name: str, *, skills_dir: Path | str | None = None) -> str:
    """Convenience module-level helper matching the Phase 3 ``load_skill(name)`` contract."""
    registry = SkillRegistry(skills_dir=skills_dir)
    return registry.load_skill(name)


def build_loaded_skills_context(
    active_skills: list[str],
    registry: SkillRegistry,
) -> str:
    """Assemble full bodies only for skills explicitly activated during the session."""
    blocks: list[str] = []
    seen: set[str] = set()
    for name in active_skills:
        norm = name.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            body = registry.load_skill(norm)
            blocks.append(f"=== SKILL: {norm} ===\n{body}")
        except (ValueError, FileNotFoundError):
            continue
    return "\n\n".join(blocks)


def get_skills_health(registry: SkillRegistry) -> dict[str, Any]:
    """Return compact health metadata for the FastAPI /health endpoint."""
    skills = registry.list_skills()
    return {
        "skills_dir": str(registry.skills_dir),
        "count": len(skills),
        "names": [s.name for s in skills],
    }
