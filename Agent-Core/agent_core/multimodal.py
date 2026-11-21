"""Multimodal capability detection, media payload encoding, and Accessory subagent routing.

When the user attaches images, videos, or audio:
- Modalities natively supported by the active Primary LLM are passed directly into
  the Primary LLM's workflow turn as OpenAI-compatible multimodal content blocks.
- Modalities NOT natively supported by the active Primary LLM are automatically
  delegated to a specialist Accessory subagent configured for that modality
  (Settings -> Accessories), which analyzes the media and reports its findings back
  to the Primary LLM so the Primary LLM can complete the user's task seamlessly.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MediaModality = Literal["image", "video", "audio"]
ALL_MEDIA_MODALITIES: tuple[MediaModality, ...] = ("image", "video", "audio")

DEFAULT_UPLOADS_DIR = (Path(__file__).resolve().parents[2] / "data" / "uploads").resolve()

MAX_MEDIA_BYTES: dict[MediaModality, int] = {
    "image": 15 * 1024 * 1024,  # 15 MB
    "audio": 25 * 1024 * 1024,  # 25 MB
    "video": 40 * 1024 * 1024,  # 40 MB
}

DEFAULT_ACCESSORY_PROVIDER = "openrouter"
DEFAULT_ACCESSORY_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
DEFAULT_ACCESSORY_BASE_URL = "https://openrouter.ai/api/v1"


def default_accessories_config() -> dict[str, dict[str, Any]]:
    """Return sensible default Accessory LLM configurations for image, video, and audio."""
    base = {
        "enabled": True,
        "provider": DEFAULT_ACCESSORY_PROVIDER,
        "model": DEFAULT_ACCESSORY_MODEL,
        "base_url": DEFAULT_ACCESSORY_BASE_URL,
        "temperature": 0.2,
        "max_tokens": 4096,
        "reasoning_effort": "low",
    }
    return {
        "image": dict(base),
        "video": dict(base),
        "audio": dict(base),
    }


class MediaAttachment(BaseModel):
    """Normalized descriptor for an uploaded image, video, or audio attachment."""

    id: str = Field(default_factory=lambda: f"att-{uuid.uuid4().hex[:10]}")
    name: str = Field(default="attachment", max_length=260)
    kind: MediaModality
    mime_type: str = Field(default="application/octet-stream", max_length=128)
    size: int = Field(default=0, ge=0)
    local_path: str | None = Field(default=None, max_length=1024)
    data_b64: str | None = Field(default=None)


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# Model family heuristics when provider /models does not supply explicit architecture.input_modalities
_OMNI_ALL_PATTERNS: tuple[str, ...] = (
    "omni",
    "gemini-1.5",
    "gemini-2",
    "gemini-2.5",
    "gemini-3",
    "gemini-4",
    "gemini-flash",
    "gemini-pro",
    "qwen2.5-omni",
    "qwen3-omni",
    "minicpm-o",
)

_TEXT_ONLY_OVERRIDES: tuple[str, ...] = (
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-coder",
    "qwen2.5-coder",
    "qwen3-coder",
    "o1-mini",
    "o3-mini",
    "llama-3.1",
    "llama-3.3",
    "gpt-3.5",
    "text-embedding",
    "embed",
    "rerank",
)

_VISION_PATTERNS: tuple[str, ...] = (
    "deepseek-v4",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4-vision",
    "gpt-4-turbo",
    "gpt-5",
    "claude-3",
    "claude-4",
    "claude-sonnet",
    "claude-opus",
    "claude-haiku",
    "gemma-3",
    "gemma-4",
    "llama-4",
    "llama-3.2-11b-vision",
    "llama-3.2-90b-vision",
    "pixtral",
    "mistral-small-3",
    "mistral-large-2",
    "grok-2-vision",
    "grok-3",
    "grok-4",
    "kimi-k2.5",
    "kimi-k3",
    "glm-4v",
    "glm-4.5v",
    "glm-5",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "-vl",
    "_vl",
    "vision",
    "multimodal",
    "internvl",
    "llava",
    "minicpm-v",
    "molmo",
    "cogvlm",
    "phi-3.5-vision",
    "phi-4-multimodal",
)

_AUDIO_ONLY_PLUS_PATTERNS: tuple[str, ...] = (
    "gpt-4o-audio",
    "gpt-audio",
    "voxtral",
    "phi-4-multimodal",
    "ultravox",
    "whisper",
)


def detect_model_input_modalities(
    provider: str | None,
    model: str | None,
    *,
    explicit_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
) -> set[str]:
    """Determine which input modalities ('text', 'image', 'audio', 'video') a model accepts natively."""
    if explicit_modalities is not None:
        normalized = {str(m).strip().lower() for m in explicit_modalities if str(m).strip()}
        valid = normalized.intersection({"text", "image", "audio", "video"})
        valid.add("text")
        return valid

    modalities: set[str] = {"text"}
    m = (model or "").strip().lower()
    if not m:
        return modalities

    # Check full-omni models first (e.g., nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free, Gemini)
    if any(pat in m for pat in _OMNI_ALL_PATTERNS):
        if "embedding" not in m and "imagen" not in m and "tts" not in m:
            modalities.update({"image", "audio", "video"})
            return modalities

    # Explicit text-only families (avoid false positives on generic patterns)
    if any(pat in m for pat in _TEXT_ONLY_OVERRIDES) and not any(
        tag in m for tag in ("vl", "vision", "omni", "audio", "multimodal")
    ):
        return modalities

    # OpenAI o1/o3/o4 (non-mini) accept image inputs
    if re.search(r"(?:^|/)(?:o1|o3|o4)(?:$|[-_:]|\b)", m) and "mini" not in m:
        modalities.add("image")

    if any(pat in m for pat in _VISION_PATTERNS):
        modalities.add("image")

    if any(pat in m for pat in _AUDIO_ONLY_PLUS_PATTERNS):
        modalities.add("audio")

    if "video" in m and "embed" not in m:
        modalities.add("video")

    return modalities


def partition_attachments_by_capability(
    attachments: list[MediaAttachment],
    primary_modalities: set[str],
) -> tuple[list[MediaAttachment], dict[MediaModality, list[MediaAttachment]]]:
    """Split attachments into those the primary model handles natively vs those needing Accessory subagents."""
    native: list[MediaAttachment] = []
    delegated: dict[MediaModality, list[MediaAttachment]] = {}
    for att in attachments:
        if att.kind in primary_modalities:
            native.append(att)
        else:
            delegated.setdefault(att.kind, []).append(att)
    return native, delegated


def audio_format_from_mime_or_name(mime_type: str, name: str) -> str:
    """Resolve an OpenAI/OpenRouter-compatible `input_audio.format` string."""
    mime = (mime_type or "").lower().split(";")[0].strip()
    mime_map = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/wave": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/opus": "ogg",
        "audio/flac": "flac",
        "audio/x-flac": "flac",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/m4a": "m4a",
        "audio/aac": "aac",
        "audio/webm": "webm",
    }
    if mime in mime_map:
        return mime_map[mime]
    ext = Path(name or "").suffix.lower().lstrip(".")
    if ext in {"wav", "mp3", "ogg", "flac", "m4a", "aac", "webm"}:
        return ext
    return "wav"


def _resolve_uploads_dir(allowed_uploads_dir: Path | None = None) -> Path:
    if allowed_uploads_dir is not None:
        return allowed_uploads_dir.resolve()
    env_dir = os.getenv("NEMOFY_UPLOADS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).resolve()
    return DEFAULT_UPLOADS_DIR


def load_attachment_base64(
    att: MediaAttachment,
    *,
    allowed_uploads_dir: Path | None = None,
) -> str:
    """Safely load base64 bytes for an attachment, enforcing path confinement and size caps."""
    max_bytes = MAX_MEDIA_BYTES.get(att.kind, 25 * 1024 * 1024)
    if att.data_b64:
        cleaned = att.data_b64.strip()
        if cleaned.startswith("data:") and "," in cleaned:
            cleaned = cleaned.split(",", 1)[1]
        # Approximate decoded size
        approx_bytes = (len(cleaned) * 3) // 4
        if approx_bytes > max_bytes:
            raise ValueError(
                f"Attachment '{att.name}' exceeds maximum {att.kind} size limit "
                f"({max_bytes // (1024 * 1024)} MB)."
            )
        return cleaned

    if not att.local_path:
        raise ValueError(f"Attachment '{att.name}' has neither data_b64 nor local_path.")

    uploads_root = _resolve_uploads_dir(allowed_uploads_dir)
    target = Path(att.local_path).resolve()
    try:
        target.relative_to(uploads_root)
    except ValueError as exc:
        raise ValueError(
            f"Attachment path for '{att.name}' is outside the permitted uploads directory."
        ) from exc

    if not target.is_file():
        raise ValueError(f"Attachment file '{att.name}' was not found on disk.")

    file_size = target.stat().st_size
    if file_size > max_bytes:
        raise ValueError(
            f"Attachment '{att.name}' ({file_size} bytes) exceeds maximum {att.kind} size limit "
            f"({max_bytes // (1024 * 1024)} MB)."
        )
    raw_bytes = target.read_bytes()
    return base64.b64encode(raw_bytes).decode("ascii")


def build_multimodal_content_blocks(
    text: str,
    attachments: list[MediaAttachment],
    *,
    allowed_uploads_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI/OpenRouter-compatible multimodal message content blocks."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for att in attachments:
        b64 = load_attachment_base64(att, allowed_uploads_dir=allowed_uploads_dir)
        mime = (att.mime_type or "").split(";")[0].strip()
        if att.kind == "image":
            img_mime = mime if mime.startswith("image/") else "image/png"
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_mime};base64,{b64}"},
                }
            )
        elif att.kind == "audio":
            fmt = audio_format_from_mime_or_name(mime, att.name)
            blocks.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": b64,
                        "format": fmt,
                    },
                }
            )
        elif att.kind == "video":
            vid_mime = mime if mime.startswith("video/") else "video/mp4"
            blocks.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:{vid_mime};base64,{b64}"},
                }
            )
    return blocks


def is_unsupported_modality_error(exc: Exception) -> bool:
    """Return True if a provider error indicates the primary model does not accept the attached media."""
    msg = str(exc).lower()
    modality_markers = (
        "image_url",
        "input_audio",
        "video_url",
        "does not support image",
        "does not support audio",
        "does not support video",
        "multimodal",
        "vision is not",
        "unsupported content type",
        "invalid content type",
        "image input not supported",
        "audio input not supported",
        "video input not supported",
        "cannot read image",
        "unknown variant `image_url`",
        "unknown variant `input_audio`",
        "unknown variant `video_url`",
        "expected string",
    )
    return any(marker in msg for marker in modality_markers)


def _extract_plain_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _THINK_TAG_RE.sub("", content).strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in (None, "text", "output_text"):
                val = item.get("text")
                if isinstance(val, str):
                    parts.append(val)
        return _THINK_TAG_RE.sub("", "\n".join(parts)).strip()
    return _THINK_TAG_RE.sub("", str(content)).strip()


_MODALITY_SYSTEM_PROMPTS: dict[MediaModality, str] = {
    "image": (
        "You are a specialist visual-analysis sub-agent reporting to a primary AI assistant. "
        "Carefully inspect every attached image and produce a thorough, factual, structured report "
        "covering: (1) direct answer to the user's question about the image(s), (2) all readable text/OCR, "
        "UI elements, numbers, charts, code, or labels verbatim, and (3) relevant visual details, layout, "
        "objects, and context. Be exact and never fabricate details not visible."
    ),
    "audio": (
        "You are a specialist audio-analysis sub-agent reporting to a primary AI assistant. "
        "Carefully listen to every attached audio clip and produce a thorough, factual, structured report "
        "covering: (1) accurate verbatim transcript of speech (with speaker labels/timestamps if discernible), "
        "(2) key non-speech sounds, music, tone, or audio cues relevant to the user's prompt, and "
        "(3) direct insights needed to answer the user's request. Never fabricate words not heard."
    ),
    "video": (
        "You are a specialist video-analysis sub-agent reporting to a primary AI assistant. "
        "Carefully inspect every attached video and produce a thorough, factual, structured report "
        "covering: (1) chronological scene-by-scene breakdown of visual events and on-screen text/actions, "
        "(2) spoken dialogue or audio track details if present, and (3) direct answers and key findings "
        "for the user's request. Be precise and never invent unseen events."
    ),
}


async def run_accessory_media_subagent(
    *,
    modality: MediaModality,
    attachments: list[MediaAttachment],
    user_task: str,
    model: BaseChatModel,
    provider_label: str,
    model_label: str,
    allowed_uploads_dir: Path | None = None,
) -> dict[str, Any]:
    """Run an isolated specialist Accessory subagent to analyze unsupported media and return a report."""
    subagent_id = f"accessory-{modality}-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(UTC).isoformat()
    file_names = [a.name for a in attachments]
    file_summary = ", ".join(f"{a.name} ({a.mime_type})" for a in attachments)

    prompt_text = (
        f"The primary assistant does not natively accept {modality} input and delegated these "
        f"{len(attachments)} {modality} attachment(s) to you: {file_summary}.\n\n"
        f"User's original request:\n{user_task}\n\n"
        f"Analyze the attached {modality} input thoroughly so the primary assistant has everything "
        f"it needs to answer the user accurately."
    )

    try:
        blocks = build_multimodal_content_blocks(
            prompt_text,
            attachments,
            allowed_uploads_dir=allowed_uploads_dir,
        )
        messages = [
            SystemMessage(content=_MODALITY_SYSTEM_PROMPTS[modality]),
            HumanMessage(content=blocks),
        ]
        reply = await model.ainvoke(messages)
        report_text = _extract_plain_text(getattr(reply, "content", ""))
        if not report_text:
            report_text = (
                f"The {modality} accessory model ({provider_label}: {model_label}) returned an empty response."
            )
            status = "error"
            error_msg: str | None = report_text
        else:
            status = "completed"
            error_msg = None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Accessory %s subagent (%s / %s) failed: %s",
            modality,
            provider_label,
            model_label,
            exc,
        )
        status = "error"
        error_msg = f"{type(exc).__name__}: {exc}"
        report_text = (
            f"Accessory {modality} analysis via {provider_label} ({model_label}) failed: {error_msg}"
        )

    return {
        "subagent_id": subagent_id,
        "role": f"accessory_{modality}",
        "modality": modality,
        "provider": provider_label,
        "model": model_label,
        "attachments": file_names,
        "status": status,
        "summary": report_text[:300],
        "findings": report_text,
        "error": error_msg,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }


def format_accessory_reports_for_primary(reports: list[dict[str, Any]]) -> str:
    """Format one or more Accessory subagent reports into a structured context block for the Primary LLM."""
    if not reports:
        return ""
    sections: list[str] = [
        "[MULTIMODAL ACCESSORY SUB-AGENT REPORTS]",
        "The user attached media whose modality is not natively accepted by the primary model. "
        "Specialist multimodal sub-agent(s) analyzed those attachments and produced the verified reports below:",
    ]
    for rep in reports:
        modality = str(rep.get("modality") or "media").upper()
        provider = str(rep.get("provider") or "accessory")
        model = str(rep.get("model") or "unknown")
        files = ", ".join(rep.get("attachments") or []) or "attachment"
        status = str(rep.get("status") or "completed")
        findings = str(rep.get("findings") or rep.get("summary") or "").strip()
        sections.append(
            f"\n--- {modality} SUB-AGENT REPORT ({provider}: {model}) | Files: {files} | Status: {status} ---\n"
            f"{findings}"
        )
    sections.append(
        "\n[END MULTIMODAL ACCESSORY SUB-AGENT REPORTS]\n"
        "Use the sub-agent findings above (along with any natively attached media) to answer the user directly."
    )
    return "\n".join(sections)
