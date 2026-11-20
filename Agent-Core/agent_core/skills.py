"""Dynamic Markdown skill catalog and on-demand loader.

At startup, only skill names and one-line descriptions are extracted and injected
into the supervisor prompt. Full Markdown bodies are loaded lazily via ``load_skill``.
Adding a new skill requires only dropping a ``.md`` file into ``./Skills/``.
"""

from __future__ import annotations

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

DEFAULT_SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "./Skills"))

_VALID_SKILL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(frozen=True)
class SkillMetadata:
    """Compact startup metadata for a discovered Markdown or dynamic skill."""

    name: str
    description: str
    path: Path | None = None
    source: str = "disk"


@dataclass(frozen=True)
class DynamicSkill:
    """Per-request dynamic skill supplied through the API payload."""

    name: str
    description: str
    content: str
    source: str = "api"


current_dynamic_skills: ContextVar[dict[str, DynamicSkill] | None] = ContextVar(
    "current_dynamic_skills", default=None
)


def _normalize_skill_name(raw_name: str, fallback: str = "custom_skill") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (raw_name or "").strip().lower()).strip("_-")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"skill_{cleaned}" if cleaned else fallback
    return cleaned[:64]


def _extract_skill_header_from_text(raw: str, *, fallback_name: str) -> tuple[str, str]:
    """Extract skill name and one-line description from YAML frontmatter or Markdown content."""
    name = _normalize_skill_name(fallback_name)
    description = ""

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
            if k == "name" and v:
                candidate = _normalize_skill_name(v, fallback=name)
                if _VALID_SKILL_NAME_RE.match(candidate):
                    name = candidate
            elif k == "description" and v:
                description = v

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

    one_line = " ".join(description.split())
    if len(one_line) > 180:
        one_line = one_line[:177].rstrip() + "..."
    return name, one_line


def _parse_skill_header(path: Path) -> tuple[str, str]:
    stem_name = path.stem.strip().lower()
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stem_name, "Skill file unreadable."
    return _extract_skill_header_from_text(raw, fallback_name=stem_name)


def parse_dynamic_skills_payload(raw_skills: list[dict[str, Any]] | None) -> dict[str, DynamicSkill]:
    """Normalize per-request dynamic skill entries supplied via the API."""
    if not raw_skills:
        return {}
    result: dict[str, DynamicSkill] = {}
    for idx, item in enumerate(raw_skills, start=1):
        if not isinstance(item, dict):
            continue
        content = str(
            item.get("content") or item.get("instructions") or item.get("markdown") or ""
        ).strip()
        if not content:
            continue
        raw_name = str(item.get("name") or item.get("skill_name") or "").strip()
        parsed_name, parsed_desc = _extract_skill_header_from_text(
            content,
            fallback_name=raw_name or f"custom_skill_{idx}",
        )
        final_name = (
            _normalize_skill_name(raw_name, fallback=parsed_name) if raw_name else parsed_name
        )
        explicit_desc = str(item.get("description") or "").strip()
        final_desc = " ".join(explicit_desc.split())[:180] if explicit_desc else parsed_desc
        source = str(item.get("source") or "api").strip() or "api"
        result[final_name] = DynamicSkill(
            name=final_name,
            description=final_desc,
            content=content,
            source=source,
        )
    return result


class SkillRegistry:
    """Scans ``Skills/`` for Markdown files and merges per-request dynamic skills."""

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir is not None else DEFAULT_SKILLS_DIR
        self._catalog: dict[str, SkillMetadata] = {}
        self.refresh()

    def refresh(self) -> dict[str, SkillMetadata]:
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
                    source="disk",
                )
        self._catalog = catalog
        return dict(self._catalog)

    def list_skills(self) -> list[SkillMetadata]:
        self.refresh()
        merged: dict[str, SkillMetadata] = dict(self._catalog)
        extra = current_dynamic_skills.get()
        if extra:
            for name, cs in extra.items():
                merged[name] = SkillMetadata(
                    name=cs.name,
                    description=cs.description,
                    path=None,
                    source=cs.source,
                )
        return list(merged.values())

    def format_catalog_for_prompt(self) -> str:
        skills = self.list_skills()
        if not skills:
            return "(no skills installed in ./Skills/)"
        lines: list[str] = []
        for meta in skills:
            tag = " [dynamic]" if meta.source != "disk" else ""
            lines.append(f"- {meta.name}{tag}: {meta.description}")
        return "\n".join(lines)

    def load_skill(self, name: str) -> str:
        normalized = name.strip().lower()
        if not normalized or not _VALID_SKILL_NAME_RE.match(normalized):
            raise ValueError(
                f"Invalid skill name {name!r}. Use alphanumeric, underscore, or hyphen characters."
            )

        extra = current_dynamic_skills.get() or {}
        if normalized in extra:
            body = extra[normalized].content.strip()
            return body or f"# Skill: {normalized}\n\n(Skill content is empty.)"

        if normalized not in self._catalog:
            self.refresh()

        meta = self._catalog.get(normalized)
        if meta is None or meta.path is None:
            available = ", ".join(sorted({*self._catalog.keys(), *extra.keys()})) or "none"
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
        """Load full specialized instructions for a skill by name from ./Skills/."""
        try:
            return registry.load_skill(name)
        except (ValueError, FileNotFoundError) as exc:
            return f"ERROR: {exc}"

    return load_skill


def load_skill(name: str, *, skills_dir: Path | str | None = None) -> str:
    registry = SkillRegistry(skills_dir=skills_dir)
    return registry.load_skill(name)


def build_loaded_skills_context(
    active_skills: list[str],
    registry: SkillRegistry,
) -> str:
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
    skills = registry.list_skills()
    return {
        "skills_dir": str(registry.skills_dir),
        "count": len(skills),
        "names": [s.name for s in skills],
    }
