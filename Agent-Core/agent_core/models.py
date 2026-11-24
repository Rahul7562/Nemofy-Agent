"""Provider-agnostic chat model adapter for Main Agent (Primary) and Sub-Agent (Worker).

Supports OpenAI-compatible endpoints (River AI, OpenRouter, OpenAI, Gemini, DeepSeek,
Groq, Together, Fireworks, Anthropic, local Ollama/vLLM, or custom base URLs)
configurable via environment variables, persisted runtime settings, or per-request overrides.
Credentials are never logged or exposed in reprs or diagnostics.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import random
import re
import socket
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphInterrupt
from pydantic import SecretStr

load_dotenv()


def reload_env_file(dotenv_path: Path | str | None = None) -> None:
    """Reload .env into os.environ so API keys saved via Settings apply immediately."""
    target = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    if target.exists():
        load_dotenv(dotenv_path=target, override=True)


logger = logging.getLogger(__name__)

_T = TypeVar("_T")
DEFAULT_RATE_LIMIT_RETRY_INTERVAL_SECONDS = max(
    0.05, float(os.getenv("RATE_LIMIT_RETRY_INTERVAL_SECONDS", "3.0"))
)
_RATE_LIMIT_TEXT_RE = re.compile(
    r"(?:\b429\b|rate[\s_-]*limit|too many requests|quota[\s_-]*exceeded)",
    re.IGNORECASE,
)

DEFAULT_RIVER_AI_BASE_URL = "https://api.river.ai/v1"
DEFAULT_PRIMARY_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_WORKER_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_LLM_CONFIG_PATH = Path(os.getenv("LLM_CONFIG_PATH", "./data/llm_config.json"))

PROVIDER_BASE_URLS: dict[str, str] = {
    "river_ai": DEFAULT_RIVER_AI_BASE_URL,
    "river": DEFAULT_RIVER_AI_BASE_URL,
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "nvidia_nim": "https://integrate.api.nvidia.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "local": "http://127.0.0.1:11434/v1",
}

PROVIDER_ENV_KEY_MAP: dict[str, tuple[str, ...]] = {
    "river_ai": ("RIVER_AI_API",),
    "river": ("RIVER_AI_API",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "nvidia_nim": ("NVIDIA_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": ("OLLAMA_API_KEY",),
    "vllm": ("VLLM_API_KEY",),
    "local": ("LOCAL_LLM_API_KEY",),
    "custom": ("PRIMARY_API_KEY", "WORKER_API_KEY"),
}

PROVIDER_ALIASES: dict[str, str] = {
    "river": "river_ai",
    "google": "gemini",
    "nvidia_nim": "nvidia",
    "nim": "nvidia",
    "local": "ollama",
}

LOCAL_NO_KEY_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "local"})

# Provider-specific default (connect_timeout_s, read_timeout_s) profiles.
# Reasoning models & planning steps frequently take > 90s before streaming tokens.
PROVIDER_TIMEOUT_PROFILES: dict[str, tuple[float, float]] = {
    "river_ai": (15.0, 180.0),
    "openai": (15.0, 180.0),
    "openrouter": (15.0, 180.0),
    "nvidia": (15.0, 180.0),
    "gemini": (15.0, 180.0),
    "deepseek": (15.0, 180.0),
    "anthropic": (15.0, 180.0),
    "groq": (12.0, 120.0),
    "together": (15.0, 150.0),
    "fireworks": (15.0, 150.0),
    "ollama": (15.0, 240.0),
    "vllm": (15.0, 240.0),
    "custom": (15.0, 180.0),
}
DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
DEFAULT_READ_TIMEOUT_SECONDS = 180.0
DEFAULT_WRITE_TIMEOUT_SECONDS = 30.0
DEFAULT_POOL_TIMEOUT_SECONDS = 15.0


def canonicalize_provider(provider: str | None) -> str:
    clean = (provider or "").strip().lower()
    if not clean:
        return "river_ai"
    return PROVIDER_ALIASES.get(clean, clean)


def is_custom_provider_id(provider: str | None) -> bool:
    clean = canonicalize_provider(provider)
    return clean == "custom" or clean.startswith("custom_")


def custom_provider_env_var(provider_id: str) -> str:
    safe = re.sub(r"[^A-Z0-9]+", "_", provider_id.strip().upper()).strip("_")
    return f"CUSTOM_PROVIDER_{safe}_API_KEY"


def get_provider_timeout_defaults(provider: str | None) -> tuple[float, float]:
    prov = canonicalize_provider(provider)
    if prov.startswith("custom_"):
        return PROVIDER_TIMEOUT_PROFILES["custom"]
    return PROVIDER_TIMEOUT_PROFILES.get(
        prov, (DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_READ_TIMEOUT_SECONDS)
    )


ModelRole = Literal["primary", "worker"]


class ModelConfigurationError(RuntimeError):
    """Raised when required model settings or provider credentials are invalid or missing."""


class ModelAuthenticationError(RuntimeError):
    """Raised when the upstream model provider rejects credentials (HTTP 401/403)."""


class ModelRateLimitError(RuntimeError):
    """Raised when the upstream model provider rate-limits requests (HTTP 429)."""


class ModelConnectionError(RuntimeError):
    """Raised when the upstream model provider is unreachable or times out."""


class ModelProviderError(RuntimeError):
    """Raised for unexpected upstream provider errors (HTTP 4xx/5xx)."""


def mask_secret(secret: str | None) -> str:
    """Return a safe, redacted representation of a secret suitable for logs."""
    if not secret:
        return "<unset>"
    cleaned = secret.strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:3]}...{cleaned[-4:]}"


def _parse_float(raw: str | None, default: float, *, min_val: float = 0.0) -> float:
    if raw is None or not str(raw).strip():
        return default
    try:
        parsed = float(str(raw).strip())
        if parsed < min_val:
            raise ValueError("value below minimum")
        return parsed
    except ValueError as exc:
        raise ModelConfigurationError(f"Invalid float setting {raw!r}: {exc}") from exc


def _parse_int(raw: str | None, default: int, *, min_val: int = 1) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        parsed = int(str(raw).strip())
        if parsed < min_val:
            raise ValueError("value below minimum")
        return parsed
    except ValueError as exc:
        raise ModelConfigurationError(f"Invalid integer setting {raw!r}: {exc}") from exc


@dataclass(frozen=True)
class ModelSpec:
    """Immutable, provider-agnostic configuration for a single LLM role."""

    role: ModelRole
    provider: str
    model: str
    base_url: str
    api_key: str | None
    temperature: float = 1.0
    max_tokens: int = 16384
    timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    write_timeout_seconds: float = DEFAULT_WRITE_TIMEOUT_SECONDS
    pool_timeout_seconds: float = DEFAULT_POOL_TIMEOUT_SECONDS
    max_retries: int = 2
    reasoning_effort: str | None = None
    context_window: int | None = None
    context_window_source: str | None = None
    manual_context_window: int | None = None
    max_output_tokens: int | None = None
    protocol: str = "openai_compatible"
    fallback_on_timeout: bool = False
    fallback_provider: str | None = None
    fallback_model: str | None = None

    def __repr__(self) -> str:
        return (
            f"ModelSpec(role={self.role!r}, provider={self.provider!r}, "
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key={mask_secret(self.api_key)!r}, "
            f"connect_timeout_seconds={self.connect_timeout_seconds!r}, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )

    def safe_summary(self) -> dict[str, Any]:
        """Return credential-free metadata for health checks, CLI headers, and Settings UI."""
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key and self.api_key.strip()),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "max_retries": self.max_retries,
            "reasoning_effort": self.reasoning_effort,
            "context_window": self.context_window,
            "context_window_source": self.context_window_source,
            "manual_context_window": self.manual_context_window,
            "max_output_tokens": self.max_output_tokens,
            "protocol": self.protocol,
            "fallback_on_timeout": self.fallback_on_timeout,
            "fallback_provider": self.fallback_provider,
            "fallback_model": self.fallback_model,
        }


def _resolve_api_key(
    provider: str,
    explicit_env_var: str | None,
    env: dict[str, str],
    *,
    allow_explicit_fallback: bool = False,
) -> str | None:
    prov = canonicalize_provider(provider)
    # Named custom providers (custom_<slug>) resolve their dedicated env var first
    if prov.startswith("custom_"):
        dedicated = env.get(custom_provider_env_var(prov), "").strip()
        if dedicated:
            return dedicated
        if allow_explicit_fallback and explicit_env_var:
            direct = env.get(explicit_env_var, "").strip()
            if direct:
                return direct
        return None

    # For legacy custom endpoints, check the role-specific env var (PRIMARY_API_KEY / WORKER_API_KEY) first
    if prov == "custom":
        if explicit_env_var:
            direct = env.get(explicit_env_var, "").strip()
            if direct:
                return direct
        for key_name in PROVIDER_ENV_KEY_MAP["custom"]:
            candidate = env.get(key_name, "").strip()
            if candidate:
                return candidate
        return None

    # For known providers, ALWAYS prefer the provider's dedicated API key first so a stale
    # PRIMARY_API_KEY from an earlier custom provider never hijacks another provider.
    for key_name in PROVIDER_ENV_KEY_MAP.get(prov, ()):
        candidate = env.get(key_name, "").strip()
        if candidate:
            return candidate

    if allow_explicit_fallback and explicit_env_var:
        direct = env.get(explicit_env_var, "").strip()
        if direct:
            return direct
    return None


def _resolve_llm_config_path(
    config_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    if config_path is not None:
        return config_path
    raw = (env.get("LLM_CONFIG_PATH") if env is not None else os.getenv("LLM_CONFIG_PATH")) or ""
    if raw.strip():
        return Path(raw.strip()).resolve()
    return DEFAULT_LLM_CONFIG_PATH


def find_persisted_custom_provider(
    provider_id: str,
    config_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Look up a persisted user-managed custom provider definition by ID."""
    clean_id = canonicalize_provider(provider_id)
    if not is_custom_provider_id(clean_id):
        return None
    persisted = load_persisted_llm_config(config_path, env=env)
    items = persisted.get("custom_providers")
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and str(item.get("id") or "").strip().lower() == clean_id:
            return item
    return None


def load_persisted_llm_config(
    config_path: Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load non-secret runtime LLM overrides saved from the Nemofy Agent Settings panel."""
    path = _resolve_llm_config_path(config_path, env=env)
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_persisted_llm_config(
    config: dict[str, Any],
    config_path: Path | None = None,
) -> Path:
    """Persist non-secret Main Agent, Sub-Agent, and Accessories LLM configuration cleanly to disk."""
    path = _resolve_llm_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip any accidental secret fields before persisting
    clean: dict[str, Any] = {}
    for key in ("primary", "worker"):
        if isinstance(config.get(key), dict):
            clean[key] = {
                k: v for k, v in config[key].items() if k != "api_key" and v is not None
            }
    if isinstance(config.get("accessories"), dict):
        acc_clean: dict[str, dict[str, Any]] = {}
        for mod in ("image", "video", "audio"):
            mod_cfg = config["accessories"].get(mod)
            if isinstance(mod_cfg, dict):
                acc_clean[mod] = {
                    k: v for k, v in mod_cfg.items() if k != "api_key" and v is not None
                }
        if acc_clean:
            clean["accessories"] = acc_clean
    if "max_turns" in config and config["max_turns"] is not None:
        clean["max_turns"] = max(1, min(180, int(config["max_turns"])))
    if "access_mode" in config and config["access_mode"] is not None:
        raw_mode = str(config["access_mode"]).strip().lower()
        clean["access_mode"] = "full_access" if raw_mode == "full_access" else "default"
    if "allow_unsafe_local_bash" in config and isinstance(config["allow_unsafe_local_bash"], bool):
        clean["allow_unsafe_local_bash"] = config["allow_unsafe_local_bash"]
    if "firecrawl_api_url" in config and isinstance(config["firecrawl_api_url"], str):
        fc_url = config["firecrawl_api_url"].strip().rstrip("/")
        if not fc_url or fc_url.startswith(("http://", "https://")):
            clean["firecrawl_api_url"] = fc_url
    if isinstance(config.get("custom_providers"), list):
        safe_custom: list[dict[str, Any]] = []
        for cp in config["custom_providers"]:
            if isinstance(cp, dict):
                safe_custom.append({k: v for k, v in cp.items() if k != "api_key"})
        clean["custom_providers"] = safe_custom
    if isinstance(config.get("planning_timeout"), dict):
        clean["planning_timeout"] = dict(config["planning_timeout"])
    payload_text = json.dumps(clean, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".llm_config.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(payload_text)
            tmp_file.flush()
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def load_persisted_access_mode(config_path: Path | None = None) -> Literal["default", "full_access"]:
    """Load persisted execution access mode, defaulting strictly to 'default'."""
    persisted = load_persisted_llm_config(config_path)
    raw = str(persisted.get("access_mode") or "").strip().lower()
    return "full_access" if raw == "full_access" else "default"


def load_persisted_accessories_config(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load persisted Accessories configuration merged over sensible defaults."""
    from agent_core.multimodal import default_accessories_config

    defaults = default_accessories_config()
    persisted = load_persisted_llm_config(config_path).get("accessories")
    if not isinstance(persisted, dict):
        return defaults
    merged: dict[str, dict[str, Any]] = {}
    for mod in ("image", "video", "audio"):
        base = dict(defaults[mod])
        raw_mod = persisted.get(mod)
        if isinstance(raw_mod, dict):
            for k, v in raw_mod.items():
                if k != "api_key" and v is not None:
                    base[k] = v
        merged[mod] = base
    return merged


def resolve_accessory_model_spec(
    modality: Literal["image", "video", "audio"],
    *,
    override: ModelOverride | dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[ModelSpec, bool]:
    """Resolve an Accessory ModelSpec and enabled flag for image, video, or audio sub-agents."""
    source_env = dict(os.environ) if env is None else env
    persisted = load_persisted_accessories_config().get(modality, {})
    merged: dict[str, Any] = dict(persisted)

    if isinstance(override, dict):
        for k, v in override.items():
            if v is not None and (not isinstance(v, str) or v.strip()):
                merged[k] = v
    elif isinstance(override, ModelOverride):
        for field_name in ("provider", "model", "base_url", "api_key", "temperature", "max_tokens", "reasoning_effort"):
            val = getattr(override, field_name)
            if val is not None and (not isinstance(val, str) or val.strip()):
                merged[field_name] = val

    enabled = bool(merged.get("enabled", True))
    provider = canonicalize_provider(str(merged.get("provider") or "openrouter"))
    default_base = PROVIDER_BASE_URLS.get(provider, "https://openrouter.ai/api/v1")
    base_url = str(merged.get("base_url") or default_base).strip().rstrip("/")
    model = str(merged.get("model") or "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free").strip()
    explicit_key = str(merged.get("api_key") or "").strip() or None
    api_key = explicit_key or _resolve_api_key(
        provider,
        "WORKER_API_KEY",
        source_env,
        allow_explicit_fallback=(provider == "custom"),
    )
    temperature = _parse_float(str(merged.get("temperature", 0.2)), 0.2, min_val=0.0)
    max_tokens = _parse_int(str(merged.get("max_tokens", 4096)), 4096, min_val=64)
    reasoning_effort = str(merged.get("reasoning_effort") or "").strip() or None
    timeout_seconds = _parse_float(source_env.get("MODEL_TIMEOUT_SECONDS"), default=90.0, min_val=1.0)
    max_retries = _parse_int(source_env.get("MODEL_MAX_RETRIES"), default=2, min_val=0)

    if not model:
        raise ModelConfigurationError(f"Accessory '{modality}' model must not be empty.")
    if not base_url.startswith(("http://", "https://")):
        raise ModelConfigurationError(
            f"Accessory '{modality}' base_url must be a valid http(s) URL, got {base_url!r}."
        )

    spec = ModelSpec(
        role="worker",
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
    )
    return spec, enabled


def load_model_spec(
    role: ModelRole = "primary",
    env: dict[str, str] | None = None,
    *,
    include_persisted: bool = False,
) -> ModelSpec:
    """Resolve Primary or Worker model specification from environment variables."""
    source_env = dict(os.environ) if env is None else env

    max_retries = _parse_int(
        source_env.get("MODEL_MAX_RETRIES"),
        default=2,
        min_val=0,
    )

    if role == "primary":
        raw_env_provider = (source_env.get("PRIMARY_PROVIDER") or "").strip().lower()
        provider = canonicalize_provider(raw_env_provider or "river_ai")
        default_base = PROVIDER_BASE_URLS.get(provider, DEFAULT_RIVER_AI_BASE_URL)
        base_url = (source_env.get("PRIMARY_BASE_URL") or default_base).strip().rstrip("/")
        model = (source_env.get("PRIMARY_MODEL") or DEFAULT_PRIMARY_MODEL).strip()
        api_key = _resolve_api_key(
            provider,
            "PRIMARY_API_KEY",
            source_env,
            allow_explicit_fallback=bool(raw_env_provider) or is_custom_provider_id(provider),
        )
        temperature = _parse_float(source_env.get("PRIMARY_TEMPERATURE"), 1.0, min_val=0.0)
        max_tokens = _parse_int(source_env.get("PRIMARY_MAX_TOKENS"), 16384, min_val=64)
        reasoning_effort = (source_env.get("PRIMARY_REASONING_EFFORT") or "").strip() or None
    else:
        primary_provider = canonicalize_provider(source_env.get("PRIMARY_PROVIDER") or "river_ai")
        raw_env_worker = (source_env.get("WORKER_PROVIDER") or "").strip().lower()
        provider = canonicalize_provider(raw_env_worker or primary_provider)
        default_base = PROVIDER_BASE_URLS.get(provider, DEFAULT_RIVER_AI_BASE_URL)
        base_url = (source_env.get("WORKER_BASE_URL") or default_base).strip().rstrip("/")
        model = (source_env.get("WORKER_MODEL") or DEFAULT_WORKER_MODEL).strip()
        api_key = _resolve_api_key(
            provider,
            "WORKER_API_KEY",
            source_env,
            allow_explicit_fallback=bool(raw_env_worker) or is_custom_provider_id(provider),
        )
        temperature = _parse_float(source_env.get("WORKER_TEMPERATURE"), 0.2, min_val=0.0)
        max_tokens = _parse_int(source_env.get("WORKER_MAX_TOKENS"), 8192, min_val=64)
        reasoning_effort = (source_env.get("WORKER_REASONING_EFFORT") or "").strip() or None

    prof_connect, prof_read = get_provider_timeout_defaults(provider)
    raw_read_env = source_env.get("MODEL_READ_TIMEOUT_SECONDS") or source_env.get(
        "MODEL_TIMEOUT_SECONDS"
    )
    timeout_seconds = _parse_float(raw_read_env, default=prof_read, min_val=1.0)
    connect_timeout_seconds = _parse_float(
        source_env.get("MODEL_CONNECT_TIMEOUT_SECONDS"),
        default=prof_connect,
        min_val=1.0,
    )
    write_timeout_seconds = _parse_float(
        source_env.get("MODEL_WRITE_TIMEOUT_SECONDS"),
        default=DEFAULT_WRITE_TIMEOUT_SECONDS,
        min_val=1.0,
    )
    pool_timeout_seconds = _parse_float(
        source_env.get("MODEL_POOL_TIMEOUT_SECONDS"),
        default=DEFAULT_POOL_TIMEOUT_SECONDS,
        min_val=1.0,
    )

    if not model:
        raise ModelConfigurationError(f"{role.upper()}_MODEL must not be empty.")
    if not base_url.startswith(("http://", "https://")):
        raise ModelConfigurationError(
            f"{role.upper()}_BASE_URL must be a valid http(s) URL, got {base_url!r}."
        )

    base_spec = ModelSpec(
        role=role,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        write_timeout_seconds=write_timeout_seconds,
        pool_timeout_seconds=pool_timeout_seconds,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
    )

    if include_persisted and env is None:
        persisted = load_persisted_llm_config()
        role_ov = persisted.get(role)
        if isinstance(role_ov, dict) and role_ov:
            try:
                return resolve_model_spec_with_override(role, override=role_ov, env=source_env)
            except ModelConfigurationError:
                return base_spec
    return base_spec


@dataclass(frozen=True)
class ModelOverride:
    """Per-request model override selected from UI Settings, Session Config, or API payload."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    context_window: int | None = None
    context_window_source: str | None = None
    manual_context_window: int | None = None
    max_output_tokens: int | None = None
    protocol: str | None = None
    timeout_seconds: float | None = None
    connect_timeout_seconds: float | None = None
    max_retries: int | None = None
    fallback_on_timeout: bool | None = None
    fallback_provider: str | None = None
    fallback_model: str | None = None

    def __repr__(self) -> str:
        return (
            f"ModelOverride(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={mask_secret(self.api_key)!r}, "
            f"temperature={self.temperature!r}, max_tokens={self.max_tokens!r})"
        )

    def has_any_override(self) -> bool:
        return any(
            v is not None and (not isinstance(v, str) or bool(v.strip()))
            for v in (
                self.provider,
                self.model,
                self.base_url,
                self.api_key,
                self.temperature,
                self.max_tokens,
                self.reasoning_effort,
                self.context_window,
                self.context_window_source,
                self.manual_context_window,
                self.max_output_tokens,
                self.protocol,
                self.timeout_seconds,
                self.connect_timeout_seconds,
                self.max_retries,
                self.fallback_on_timeout,
                self.fallback_provider,
                self.fallback_model,
            )
        )


def resolve_model_spec_with_override(
    role: ModelRole = "primary",
    *,
    override: ModelOverride | dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> ModelSpec:
    """Resolve a ModelSpec merging per-request or UI overrides over environment defaults."""
    source_env = dict(os.environ) if env is None else env
    base_spec = load_model_spec(role, env=source_env)

    if override is None:
        return base_spec

    if isinstance(override, dict):
        ov = ModelOverride(
            provider=(str(override["provider"]).strip() if override.get("provider") else None),
            model=(str(override["model"]).strip() if override.get("model") else None),
            base_url=(str(override["base_url"]).strip() if override.get("base_url") else None),
            api_key=(str(override["api_key"]).strip() if override.get("api_key") else None),
            temperature=(
                float(override["temperature"])
                if override.get("temperature") is not None
                and str(override.get("temperature")) != ""
                else None
            ),
            max_tokens=(
                int(override["max_tokens"])
                if override.get("max_tokens") is not None and str(override.get("max_tokens")) != ""
                else None
            ),
            reasoning_effort=(
                str(override["reasoning_effort"]).strip()
                if override.get("reasoning_effort")
                else None
            ),
            context_window=(
                int(override["context_window"])
                if override.get("context_window") is not None
                and str(override.get("context_window")) != ""
                else None
            ),
            context_window_source=(
                str(override["context_window_source"]).strip()
                if override.get("context_window_source")
                else None
            ),
            manual_context_window=(
                int(override["manual_context_window"])
                if override.get("manual_context_window") is not None
                and str(override.get("manual_context_window")) != ""
                else None
            ),
            max_output_tokens=(
                int(override["max_output_tokens"])
                if override.get("max_output_tokens") is not None
                and str(override.get("max_output_tokens")) != ""
                else None
            ),
            protocol=(str(override["protocol"]).strip() if override.get("protocol") else None),
            timeout_seconds=(
                float(override["timeout_seconds"])
                if override.get("timeout_seconds") is not None
                and str(override.get("timeout_seconds")) != ""
                else None
            ),
            connect_timeout_seconds=(
                float(override["connect_timeout_seconds"])
                if override.get("connect_timeout_seconds") is not None
                and str(override.get("connect_timeout_seconds")) != ""
                else None
            ),
            max_retries=(
                int(override["max_retries"])
                if override.get("max_retries") is not None and str(override.get("max_retries")) != ""
                else None
            ),
            fallback_on_timeout=(
                bool(override["fallback_on_timeout"])
                if override.get("fallback_on_timeout") is not None
                else None
            ),
            fallback_provider=(
                str(override["fallback_provider"]).strip()
                if override.get("fallback_provider")
                else None
            ),
            fallback_model=(
                str(override["fallback_model"]).strip()
                if override.get("fallback_model")
                else None
            ),
        )
    else:
        ov = override

    if not ov.has_any_override():
        return base_spec

    provider = canonicalize_provider(ov.provider or base_spec.provider)
    provider_changed = provider != base_spec.provider
    custom_cfg = (
        find_persisted_custom_provider(provider, env=source_env)
        if is_custom_provider_id(provider)
        else None
    )

    if ov.base_url and ov.base_url.strip():
        base_url = ov.base_url.strip().rstrip("/")
    elif custom_cfg and str(custom_cfg.get("base_url") or "").strip():
        base_url = str(custom_cfg["base_url"]).strip().rstrip("/")
    elif provider_changed:
        if provider not in PROVIDER_BASE_URLS:
            raise ModelConfigurationError(
                f"Unknown provider {provider!r} without an explicit base_url. "
                f"Provide base_url or use one of: {', '.join(sorted(PROVIDER_BASE_URLS))}."
            )
        base_url = PROVIDER_BASE_URLS[provider].rstrip("/")
    else:
        base_url = base_spec.base_url

    model = (ov.model or base_spec.model).strip()
    if not model:
        raise ModelConfigurationError(f"{role.upper()} model name must not be empty.")
    if not base_url.startswith(("http://", "https://")):
        raise ModelConfigurationError(
            f"{role.upper()} base_url must be a valid http(s) URL, got {base_url!r}."
        )

    role_env_var = "WORKER_API_KEY" if role == "worker" else "PRIMARY_API_KEY"
    if ov.api_key and ov.api_key.strip():
        api_key: str | None = ov.api_key.strip()
    elif is_custom_provider_id(provider):
        api_key = _resolve_api_key(
            provider, role_env_var, source_env, allow_explicit_fallback=(provider == "custom")
        )
    elif provider_changed:
        api_key = _resolve_api_key(provider, role_env_var, source_env, allow_explicit_fallback=False)
    else:
        api_key = _resolve_api_key(
            provider,
            role_env_var,
            source_env,
            allow_explicit_fallback=True,
        ) or base_spec.api_key

    temperature = ov.temperature if ov.temperature is not None else base_spec.temperature
    if temperature < 0.0:
        raise ModelConfigurationError(f"temperature must be >= 0.0, got {temperature}.")

    max_tokens = ov.max_tokens if ov.max_tokens is not None else base_spec.max_tokens
    if max_tokens < 64:
        raise ModelConfigurationError(f"max_tokens must be >= 64, got {max_tokens}.")

    reasoning_effort = (
        ov.reasoning_effort if ov.reasoning_effort is not None else base_spec.reasoning_effort
    )

    # Check persisted custom provider per-model metadata when explicit values are omitted
    custom_model_ctx: int | None = None
    custom_model_out: int | None = None
    if custom_cfg and isinstance(custom_cfg.get("models"), list):
        for cm in custom_cfg["models"]:
            if isinstance(cm, dict) and str(cm.get("id") or "").strip() == model:
                if isinstance(cm.get("context_window"), (int, float)) and cm["context_window"] >= 512:
                    custom_model_ctx = int(cm["context_window"])
                if (
                    isinstance(cm.get("max_output_tokens"), (int, float))
                    and cm["max_output_tokens"] >= 256
                ):
                    custom_model_out = int(cm["max_output_tokens"])
                break

    manual_context_window = (
        ov.manual_context_window
        if ov.manual_context_window is not None and ov.manual_context_window >= 512
        else base_spec.manual_context_window
    )
    context_window = (
        ov.context_window
        if ov.context_window is not None and ov.context_window >= 512
        else (custom_model_ctx if custom_model_ctx is not None else base_spec.context_window)
    )
    context_window_source = (
        "manual"
        if manual_context_window is not None
        else (
            ov.context_window_source
            or ("custom_provider_config" if custom_model_ctx is not None and ov.context_window is None else base_spec.context_window_source)
        )
    )
    max_output_tokens = (
        ov.max_output_tokens
        if ov.max_output_tokens is not None and ov.max_output_tokens >= 256
        else (custom_model_out if custom_model_out is not None else base_spec.max_output_tokens)
    )
    protocol = (
        ov.protocol
        or (str(custom_cfg.get("protocol") or "").strip() if custom_cfg else "")
        or base_spec.protocol
        or "openai_compatible"
    )

    prof_connect, prof_read = get_provider_timeout_defaults(provider)
    custom_timeout = (
        float(custom_cfg["timeout_seconds"])
        if custom_cfg
        and isinstance(custom_cfg.get("timeout_seconds"), (int, float))
        and float(custom_cfg["timeout_seconds"]) >= 5.0
        else None
    )
    read_timeout = (
        ov.timeout_seconds
        if ov.timeout_seconds is not None and ov.timeout_seconds >= 5.0
        else (
            custom_timeout
            if custom_timeout is not None
            else (prof_read if provider_changed else base_spec.timeout_seconds)
        )
    )
    connect_timeout = (
        ov.connect_timeout_seconds
        if ov.connect_timeout_seconds is not None and ov.connect_timeout_seconds >= 1.0
        else (prof_connect if provider_changed else base_spec.connect_timeout_seconds)
    )
    max_retries = ov.max_retries if ov.max_retries is not None and ov.max_retries >= 0 else base_spec.max_retries

    return ModelSpec(
        role=role,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=read_timeout,
        connect_timeout_seconds=connect_timeout,
        write_timeout_seconds=base_spec.write_timeout_seconds,
        pool_timeout_seconds=base_spec.pool_timeout_seconds,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
        context_window=context_window,
        context_window_source=context_window_source,
        manual_context_window=manual_context_window,
        max_output_tokens=max_output_tokens,
        protocol=protocol,
        fallback_on_timeout=(
            ov.fallback_on_timeout
            if ov.fallback_on_timeout is not None
            else base_spec.fallback_on_timeout
        ),
        fallback_provider=ov.fallback_provider or base_spec.fallback_provider,
        fallback_model=ov.fallback_model or base_spec.fallback_model,
    )


def get_provider_presets() -> dict[str, Any]:
    """Return credential-free provider presets & env availability for the Settings UI."""
    env = dict(os.environ)
    presets: dict[str, dict[str, Any]] = {}
    for name, url in PROVIDER_BASE_URLS.items():
        if name in PROVIDER_ALIASES:
            continue
        env_vars = list(PROVIDER_ENV_KEY_MAP.get(name, ()))
        key_configured = (name in LOCAL_NO_KEY_PROVIDERS) or any(
            bool(env.get(k, "").strip()) for k in env_vars
        )
        presets[name] = {
            "base_url": url,
            "env_keys": env_vars,
            "api_key_configured": key_configured,
            "requires_api_key": name not in LOCAL_NO_KEY_PROVIDERS,
        }
    return {
        "default_primary": load_model_spec("primary", include_persisted=True).safe_summary(),
        "default_worker": load_model_spec("worker", include_persisted=True).safe_summary(),
        "providers": presets,
    }


def classify_provider_exception(exc: Exception, spec: ModelSpec) -> RuntimeError:
    """Translate raw HTTP/SDK errors into typed, credential-safe agent errors."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None and isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code

    raw_msg = str(exc)
    if spec.api_key:
        raw_msg = raw_msg.replace(spec.api_key, "[REDACTED]")

    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return ModelConnectionError(
            f"Model provider '{spec.provider}' timed out after {spec.timeout_seconds}s "
            f"for model '{spec.model}'."
        )
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
        return ModelConnectionError(
            f"Cannot connect to model provider '{spec.provider}' at {spec.base_url}: {raw_msg}"
        )
    if status_code in (401, 403):
        return ModelAuthenticationError(
            f"Authentication failed for provider '{spec.provider}' ({spec.base_url}). "
            "Verify RIVER_AI_API or the configured provider API key."
        )
    if status_code == 429:
        return ModelRateLimitError(
            f"Rate limit exceeded for provider '{spec.provider}' model '{spec.model}'."
        )
    if status_code is not None and status_code >= 400:
        return ModelProviderError(
            f"Provider '{spec.provider}' returned HTTP {status_code} for '{spec.model}': {raw_msg}"
        )
    return ModelProviderError(
        f"Unexpected error from provider '{spec.provider}' ({spec.model}): {raw_msg}"
    )


def is_rate_limit_exception(exc: BaseException, *, _depth: int = 0) -> bool:
    """Return True if an exception (or its wrapped cause) represents an HTTP 429 rate-limit error."""
    if _depth > 4:
        return False
    if isinstance(exc, ModelRateLimitError):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        if exc.response.status_code == 429:
            return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    cls_name = type(exc).__name__.lower()
    if "ratelimit" in cls_name or "toomanyrequests" in cls_name:
        return True
    if _RATE_LIMIT_TEXT_RE.search(str(exc)):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc and is_rate_limit_exception(cause, _depth=_depth + 1):
        return True
    ctx = getattr(exc, "__context__", None)
    if ctx is not None and ctx is not exc and is_rate_limit_exception(ctx, _depth=_depth + 1):
        return True
    return False


def format_rate_limit_error_detail(exc: BaseException) -> str:
    """Extract a clean, compact single-line summary of an HTTP 429 rate-limit error."""
    raw = str(exc).strip() or type(exc).__name__
    compact = " ".join(raw.split())
    return compact[:320]


def _walk_exception_chain(exc: BaseException, max_depth: int = 5) -> list[BaseException]:
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and len(chain) < max_depth and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        cur = nxt if isinstance(nxt, BaseException) else None
    return chain


def classify_model_error_category(exc: BaseException) -> str:
    """Classify an LLM exception into a structured diagnostic category."""
    if is_rate_limit_exception(exc):
        return "rate_limit"
    for item in _walk_exception_chain(exc):
        cls_name = type(item).__name__.lower()
        msg_lower = str(item).lower()
        if isinstance(item, httpx.ConnectTimeout) or "connecttimeout" in cls_name:
            return "connect_timeout"
        if isinstance(item, httpx.ReadTimeout) or "readtimeout" in cls_name:
            return "read_timeout"
        if isinstance(item, httpx.WriteTimeout) or "writetimeout" in cls_name:
            return "write_timeout"
        if isinstance(item, httpx.PoolTimeout) or "pooltimeout" in cls_name:
            return "pool_timeout"
        if (
            isinstance(item, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError))
            or "timeout" in cls_name
            or "timed out" in msg_lower
            or "request timed out" in msg_lower
        ):
            return "read_timeout"
        if (
            isinstance(
                item,
                (
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                    httpx.ReadError,
                    httpx.NetworkError,
                    ConnectionError,
                ),
            )
            or "apiconnectionerror" in cls_name
            or "connection error" in msg_lower
        ):
            return "connection_error"
        status_code = getattr(item, "status_code", None)
        if status_code is None and isinstance(item, httpx.HTTPStatusError) and item.response is not None:
            status_code = item.response.status_code
        if status_code in (401, 403) or isinstance(item, ModelAuthenticationError):
            return "auth_error"
        if status_code == 400 or status_code == 422:
            return "bad_request"
        if status_code in (500, 502, 503, 504, 529) or (isinstance(status_code, int) and status_code >= 500):
            return "server_error"
    return "provider_error"


def is_retryable_transient_model_error(exc: BaseException) -> bool:
    """Return True if an exception is a transient timeout, network drop, or 5xx error safe to retry."""
    cat = classify_model_error_category(exc)
    return cat in {
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "pool_timeout",
        "connection_error",
        "server_error",
    }


def is_timeout_model_error(exc: BaseException) -> bool:
    return classify_model_error_category(exc) in {
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "pool_timeout",
    }


async def invoke_with_resilient_retry(
    call_fn: Callable[[], Awaitable[_T]],
    *,
    label: str = "model",
    provider: str | None = None,
    model: str | None = None,
    stage: str = "planning",
    max_transient_retries: int = 2,
    base_backoff_seconds: float = 1.5,
    max_backoff_seconds: float = 12.0,
    retry_interval_seconds: float | None = None,
    max_wait_seconds: float | None = None,
    turn_count: int | None = None,
    on_retry: Callable[[int, float, float, BaseException], None] | None = None,
) -> _T:
    """Invoke an async LLM call with structured retry handling for 429 rate limits AND transient
    timeouts / connection drops / 5xx server errors.

    - HTTP 429 rate limits retry every ``retry_interval_seconds`` (default 3s) up to ``max_wait_seconds``.
    - Transient timeouts, connection drops, and 5xx errors retry up to ``max_transient_retries`` times
      with bounded exponential backoff + jitter (never retrying auth/400/cancellation errors).
    - Emits live progress heartbeats and structured diagnostic logs without leaking secrets or prompts.
    """
    from agent_core.jobs import (
        DEFAULT_STALL_TIMEOUT_SECONDS,
        emit_progress,
        get_active_timeout_seconds,
    )

    interval = (
        max(0.01, float(retry_interval_seconds))
        if retry_interval_seconds is not None
        else DEFAULT_RATE_LIMIT_RETRY_INTERVAL_SECONDS
    )
    timeout_budget = (
        max(0.05, float(max_wait_seconds))
        if max_wait_seconds is not None
        else get_active_timeout_seconds(DEFAULT_STALL_TIMEOUT_SECONDS)
    )

    first_429_at: float | None = None
    rate_limit_attempt = 0
    transient_attempt = 0
    call_started_at = time.monotonic()

    while True:
        try:
            return await call_fn()
        except (GraphInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001
            category = classify_model_error_category(exc)
            detail = format_rate_limit_error_detail(exc)

            if category == "rate_limit":
                now = time.monotonic()
                if first_429_at is None:
                    first_429_at = now
                rate_limit_attempt += 1
                elapsed = now - first_429_at
                remaining = timeout_budget - elapsed

                if remaining <= 0.0:
                    err_msg = (
                        f"Agent timed out after {elapsed:.1f}s (timeout={timeout_budget:.1f}s) "
                        f"due to persistent HTTP 429 rate limit on {label} after {rate_limit_attempt} attempt(s) "
                        f"(retrying every {interval:g}s): {detail}"
                    )
                    logger.error(err_msg)
                    raise ModelRateLimitError(err_msg) from exc

                sleep_s = min(interval, remaining)
                logger.warning(
                    "HTTP 429 rate limit on %s [stage=%s provider=%s model=%s] "
                    "(attempt %d, %.1fs/%.1fs elapsed) — retrying in %.1fs: %s",
                    label,
                    stage,
                    provider or "-",
                    model or "-",
                    rate_limit_attempt,
                    elapsed,
                    timeout_budget,
                    sleep_s,
                    detail,
                )
                progress_fields: dict[str, Any] = {
                    "detail": (
                        f"Rate limited (HTTP 429) on {label} — retrying in {sleep_s:g}s "
                        f"(attempt {rate_limit_attempt}, {elapsed:.0f}s / {timeout_budget:.0f}s)"
                    ),
                    "retry_attempt": rate_limit_attempt,
                    "rate_limit_error": detail,
                }
                if turn_count is not None:
                    progress_fields["turn_count"] = turn_count
                emit_progress("rate_limited", **progress_fields)

                if on_retry is not None:
                    try:
                        on_retry(rate_limit_attempt, elapsed, sleep_s, exc)
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.debug("on_retry callback error ignored: %s", cb_exc)

                await asyncio.sleep(sleep_s)
                continue

            if is_retryable_transient_model_error(exc) and transient_attempt < max(0, int(max_transient_retries)):
                transient_attempt += 1
                elapsed_total = time.monotonic() - call_started_at
                raw_backoff = min(
                    max_backoff_seconds,
                    max(0.05, base_backoff_seconds) * (2 ** (transient_attempt - 1)),
                )
                jitter = random.uniform(0.0, min(0.5, raw_backoff * 0.2))  # noqa: S311
                sleep_s = round(raw_backoff + jitter, 2)
                logger.warning(
                    "Transient %s during %s on %s [provider=%s model=%s] "
                    "(attempt %d/%d, elapsed=%.1fs) — retrying in %.2fs: %s",
                    category,
                    stage,
                    label,
                    provider or "-",
                    model or "-",
                    transient_attempt,
                    max_transient_retries + 1,
                    elapsed_total,
                    sleep_s,
                    detail,
                )
                progress_fields = {
                    "detail": (
                        f"Model {category.replace('_', ' ')} during {stage} — "
                        f"retrying in {sleep_s:g}s (attempt {transient_attempt + 1}/{max_transient_retries + 1})"
                    ),
                    "retry_attempt": transient_attempt,
                    "error_category": category,
                }
                if turn_count is not None:
                    progress_fields["turn_count"] = turn_count
                emit_progress("thinking", **progress_fields)

                if on_retry is not None:
                    try:
                        on_retry(transient_attempt, elapsed_total, sleep_s, exc)
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.debug("on_retry callback error ignored: %s", cb_exc)

                await asyncio.sleep(sleep_s)
                continue

            raise


async def invoke_with_rate_limit_retry(
    call_fn: Callable[[], Awaitable[_T]],
    *,
    label: str = "model",
    retry_interval_seconds: float | None = None,
    max_wait_seconds: float | None = None,
    turn_count: int | None = None,
    on_retry: Callable[[int, float, float, BaseException], None] | None = None,
) -> _T:
    """Backward-compatible wrapper retrying HTTP 429 rate limits (without retrying other errors)."""
    return await invoke_with_resilient_retry(
        call_fn,
        label=label,
        max_transient_retries=0,
        retry_interval_seconds=retry_interval_seconds,
        max_wait_seconds=max_wait_seconds,
        turn_count=turn_count,
        on_retry=on_retry,
    )


_REASONING_RANK: dict[str, int] = {
    "off": 0,
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
}


def normalize_reasoning_effort_for_model(
    provider: str,
    model: str,
    effort: str | None,
) -> str | None:
    """Adapt a requested thinking / reasoning effort level to what the target model accepts.

    Prevents HTTP 400 errors when switching between models with differing schemas
    (e.g. DeepSeek-V4.1-Flash accepts none/low/high/max, GPT-5.2 accepts none/low/medium/high/xhigh,
    Claude 4 accepts low/medium/high/max, o3-mini accepts low/medium/high, GPT-4o rejects reasoning_effort).
    """
    if not effort:
        return None
    raw = effort.strip().lower()
    if not raw:
        return None

    slug = model.strip().lower().split("/")[-1]
    prov = canonicalize_provider(provider)

    def _pick(allowed: tuple[str, ...], *, send_none_when_off: bool = False) -> str | None:
        if raw in ("off", "none"):
            return "none" if send_none_when_off else None
        if raw in allowed:
            return raw
        if raw == "max" and "xhigh" in allowed:
            return "xhigh"
        if raw == "xhigh" and "max" in allowed:
            return "max"
        target_rank = _REASONING_RANK.get(raw)
        if target_rank is None:
            return allowed[0]
        return min(
            allowed,
            key=lambda c: (abs(_REASONING_RANK[c] - target_rank), -_REASONING_RANK[c]),
        )

    # NVIDIA Nemotron reasoning models (check BEFORE llama-3 exclusion since slugs like
    # llama-3.3-nemotron-super-49b-v1 contain "llama-3")
    if (
        "nemotron-3" in slug
        or "nemotron-super" in slug
        or "nemotron-ultra" in slug
        or "nemotron-nano" in slug
    ):
        return _pick(("low", "medium", "high"))

    # Non-thinking standard chat models: never send reasoning_effort
    if (
        slug.startswith(("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-3.5"))
        or slug == "gpt-4"
        or "claude-3-5-" in slug
        or "claude-3.5-" in slug
        or "claude-3-haiku" in slug
        or "claude-3-opus" in slug
        or "llama-3" in slug
        or "llama3" in slug
        or "qwen2.5" in slug
        or "qwen-2.5" in slug
        or "gemini-2.0" in slug
        or "gemini-1.5" in slug
    ):
        return None

    # River AI / hybrid low-high-max models (DeepSeek V4.1, GLM-5, Kimi-K3)
    if "deepseek-v4" in slug or "glm-5" in slug or "kimi-k3" in slug:
        return _pick(("low", "high", "max"), send_none_when_off=True)

    # OpenAI GPT-5.2+ / o3-pro / codex-max -> supports xhigh (not max)
    if (
        slug.startswith(("gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5"))
        or "gpt-5-pro" in slug
        or "o3-pro" in slug
        or "codex-max" in slug
        or "codex-5" in slug
    ):
        return _pick(("low", "medium", "high", "xhigh"), send_none_when_off=True)

    # OpenAI GPT-5 / mini / nano -> minimal, low, medium, high
    if slug == "gpt-5" or slug.startswith("gpt-5-"):
        if raw in ("off", "none"):
            return "minimal"
        return _pick(("minimal", "low", "medium", "high"))

    # OpenAI o1 / o3 / o4 reasoning series -> low, medium, high
    if slug in ("o1", "o3", "o4") or slug.startswith(("o1-", "o3-", "o4-")):
        if raw in ("off", "none"):
            return "low"
        return _pick(("low", "medium", "high"))

    # Anthropic Claude 4 / 3.7 Sonnet -> low, medium, high, max
    if (
        "claude-opus-4" in slug
        or "claude-sonnet-4" in slug
        or "claude-4" in slug
        or "claude-3-7-sonnet" in slug
        or "claude-3.7-sonnet" in slug
    ):
        return _pick(("low", "medium", "high", "max"))

    # Google Gemini Pro / 3 / 4 -> low, medium, high, max (Gemini OpenAI-compat maps max -> high safely)
    if (
        "gemini-2.5-pro" in slug
        or "gemini-3-pro" in slug
        or "gemini-3.1-pro" in slug
        or "gemini-4" in slug
    ):
        if prov in ("gemini", "google"):
            return _pick(("low", "medium", "high"))
        return _pick(("low", "medium", "high", "max"))

    # Google Gemini Flash -> minimal/low/medium/high
    if "gemini-2.5-flash" in slug or "gemini-3-flash" in slug:
        if prov in ("gemini", "google"):
            return _pick(("low", "medium", "high"))
        return _pick(("minimal", "low", "medium", "high"))

    # Grok 3 Mini -> low, high only
    if "grok-3-mini" in slug:
        if raw in ("off", "none"):
            return "low"
        return _pick(("low", "high"))

    # Grok 4 -> low, medium, high, xhigh
    if "grok-4" in slug or "grok-3-reason" in slug:
        if raw in ("off", "none"):
            return "low"
        return _pick(("low", "medium", "high", "xhigh"))

    # DeepSeek R1 / reasoner
    if "deepseek-r1" in slug or "deepseek-reasoner" in slug:
        if raw in ("off", "none"):
            return None
        return _pick(("low", "medium", "high", "max"))

    # Default adaptive pass-through for unknown/custom models
    if raw in ("off", "none"):
        return None
    return raw


def _extract_reasoning_field(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list):
            parts: list[str] = []
            for item in val:
                if isinstance(item, str) and item:
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text") or item.get("summary") or item.get("content") or item.get("thinking")
                    if isinstance(t, str) and t:
                        parts.append(t)
            if parts:
                return "".join(parts)

    details = payload.get("reasoning_details")
    if isinstance(details, list):
        parts = []
        for item in details:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type and "encrypted" in item_type:
                continue
            t = item.get("text") or item.get("summary") or item.get("content")
            if isinstance(t, str) and t:
                parts.append(t)
        if parts:
            return "".join(parts)
    return None


def _extract_tool_call_extra_content(tool_calls: Any) -> dict[str, Any]:
    """Extract per-tool-call extra_content (e.g. Gemini 3 thought_signature) keyed by call id or index."""
    if not isinstance(tool_calls, list):
        return {}
    captured: dict[str, Any] = {}
    for pos, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        ec = tc.get("extra_content")
        if not isinstance(ec, dict) or not ec:
            continue
        tc_id = str(tc.get("id") or "").strip()
        idx_val = tc.get("index")
        key = tc_id if tc_id else f"idx:{idx_val if idx_val is not None else pos}"
        captured[key] = ec
    return captured


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that preserves reasoning_content and provider tool_call extra_content (e.g. Gemini 3 thought_signature)."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        gen = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen is None:
            return None
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            reasoning = _extract_reasoning_field(delta)
            if reasoning:
                gen.message.additional_kwargs["reasoning_content"] = reasoning
            if isinstance(delta, dict):
                tc_extras = _extract_tool_call_extra_content(delta.get("tool_calls"))
                if tc_extras:
                    gen.message.additional_kwargs["tool_call_extra_content"] = tc_extras
        return gen

    def _create_chat_result(
        self,
        response: dict[str, Any] | Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info=generation_info)
        try:
            response_dict = (
                response
                if isinstance(response, dict)
                else response.model_dump(warnings=False)
            )
        except Exception:  # noqa: BLE001
            return result
        choices = response_dict.get("choices") if isinstance(response_dict, dict) else None
        if isinstance(choices, list):
            for gen, choice in zip(result.generations, choices, strict=False):
                if isinstance(choice, dict):
                    msg_dict = choice.get("message")
                    reasoning = _extract_reasoning_field(msg_dict)
                    if reasoning and not gen.message.additional_kwargs.get("reasoning_content"):
                        gen.message.additional_kwargs["reasoning_content"] = reasoning
                    if isinstance(msg_dict, dict):
                        tc_extras = _extract_tool_call_extra_content(msg_dict.get("tool_calls"))
                        if tc_extras:
                            gen.message.additional_kwargs["tool_call_extra_content"] = tc_extras
        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        try:
            input_messages = self._convert_input(input_).to_messages()
        except Exception:  # noqa: BLE001
            return payload

        extra_by_id: dict[str, Any] = {}
        extra_by_msg_pos: list[dict[str, Any]] = []
        for msg in input_messages:
            ak = getattr(msg, "additional_kwargs", None)
            tc_extras = ak.get("tool_call_extra_content") if isinstance(ak, dict) else None
            if isinstance(tc_extras, dict) and tc_extras:
                extra_by_msg_pos.append(tc_extras)
                for k, v in tc_extras.items():
                    if not k.startswith("idx:"):
                        extra_by_id[k] = v
            elif getattr(msg, "type", None) == "ai":
                extra_by_msg_pos.append({})

        if not extra_by_id and not any(extra_by_msg_pos):
            return payload

        ai_idx = 0
        for pmsg in payload.get("messages", []):
            if not isinstance(pmsg, dict) or pmsg.get("role") != "assistant":
                continue
            pos_map = extra_by_msg_pos[ai_idx] if ai_idx < len(extra_by_msg_pos) else {}
            ai_idx += 1
            p_tcs = pmsg.get("tool_calls")
            if not isinstance(p_tcs, list):
                continue
            for tc_pos, ptc in enumerate(p_tcs):
                if not isinstance(ptc, dict) or "extra_content" in ptc:
                    continue
                tc_id = str(ptc.get("id") or "").strip()
                ec = (
                    (pos_map.get(tc_id) if tc_id else None)
                    or (extra_by_id.get(tc_id) if tc_id else None)
                    or pos_map.get(f"idx:{tc_pos}")
                )
                if isinstance(ec, dict) and ec:
                    ptc["extra_content"] = ec
        return payload


def build_chat_model(
    role: ModelRole = "primary",
    *,
    spec: ModelSpec | None = None,
    require_api_key: bool = True,
    streaming: bool = False,
) -> BaseChatModel:
    """Instantiate a LangChain OpenAI-compatible chat model for the requested role."""
    resolved = spec or load_model_spec(role, include_persisted=True)
    needs_key = require_api_key and resolved.provider not in LOCAL_NO_KEY_PROVIDERS
    if needs_key and not (resolved.api_key and resolved.api_key.strip()):
        expected_vars = ", ".join(
            PROVIDER_ENV_KEY_MAP.get(resolved.provider, ("RIVER_AI_API", "OPENAI_API_KEY"))
        )
        raise ModelConfigurationError(
            f"Missing API key for {resolved.role} model '{resolved.model}' "
            f"(provider='{resolved.provider}'). Set {expected_vars} in your environment."
        )

    extra_body: dict[str, Any] = {}
    wire_effort = normalize_reasoning_effort_for_model(
        resolved.provider, resolved.model, resolved.reasoning_effort
    )
    if wire_effort:
        extra_body["reasoning_effort"] = wire_effort

    slug = resolved.model.strip().lower().split("/")[-1]
    omit_temp = resolved.provider == "openai" and (
        slug in ("o1", "o3", "o4", "gpt-5") or slug.startswith(("o1-", "o3-", "o4-", "gpt-5"))
    )

    from agent_core.context_manager import resolve_model_max_output_tokens

    model_max_out, _ = resolve_model_max_output_tokens(
        resolved.provider, resolved.model, explicit_max_output_tokens=resolved.max_output_tokens
    )
    configured_max = max(64, int(resolved.max_tokens))
    effective_max_tokens = (
        max(64, min(configured_max, int(model_max_out)))
        if isinstance(model_max_out, int) and model_max_out > 0
        else configured_max
    )

    http_timeout = httpx.Timeout(
        connect=max(1.0, float(resolved.connect_timeout_seconds)),
        read=max(5.0, float(resolved.timeout_seconds)),
        write=max(5.0, float(resolved.write_timeout_seconds)),
        pool=max(1.0, float(resolved.pool_timeout_seconds)),
    )

    kwargs: dict[str, Any] = {
        "model": resolved.model,
        "api_key": SecretStr(resolved.api_key or "not-configured"),
        "base_url": resolved.base_url,
        "max_tokens": effective_max_tokens,
        "timeout": http_timeout,
        # Handle HTTP 429 and transient timeout/5xx retries at the agent layer with live
        # progress heartbeats so the OpenAI SDK never hides retries behind opaque sleeps.
        "max_retries": 0,
        "streaming": streaming,
    }
    if resolved.protocol == "anthropic_compatible" and resolved.api_key:
        kwargs["default_headers"] = {
            "x-api-key": resolved.api_key,
            "anthropic-version": "2023-06-01",
        }
    if not omit_temp:
        kwargs["temperature"] = resolved.temperature
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ReasoningChatOpenAI(**kwargs)


def get_primary_model(*, streaming: bool = False) -> BaseChatModel:
    """Build the configured Main Agent (PRIMARY_MODEL)."""
    return build_chat_model("primary", streaming=streaming)


def get_worker_model(*, streaming: bool = False) -> BaseChatModel:
    """Build the configured Sub-Agent (WORKER_MODEL)."""
    return build_chat_model("worker", streaming=streaming)


_NON_CHAT_MODEL_KEYWORDS: tuple[str, ...] = (
    "embed",
    "embedding",
    "rerank",
    "whisper",
    "tts",
    "dall-e",
    "diffusion",
    "stable-diffusion",
    "sdxl",
    "flux",
    "imagen",
    "moderation",
    "omni-moderation",
    "transcribe",
    "audio-preview",
    "realtime-preview",
    "speech",
    "parakeet",
    "canary",
    "riva",
    "nvclip",
    "neva",
    "kosmos",
    "paligemma",
    "parse",
    "ocr",
    "retriever",
    "deplot",
    "fuyu",
    "bge-",
    "e5-",
    "text-similarity",
    "code-search",
    "davinci-002",
    "babbage-002",
    "veo-",
    "lyria",
    "video-detector",
    "content-safety",
    "nemoguard",
    "safety-guard",
    "gliner",
    "calibration",
)


def _is_likely_chat_llm(model_id: str, raw_item: dict[str, Any]) -> bool:
    """Filter out obvious non-chat models (embeddings, TTS, image gen, rerankers)."""
    mtype = str(raw_item.get("type") or "").strip().lower()
    if mtype in ("embedding", "image", "audio", "moderation", "rerank", "video"):
        return False
    arch = raw_item.get("architecture")
    if isinstance(arch, dict):
        out_mods = arch.get("output_modalities")
        if isinstance(out_mods, list) and out_mods and "text" not in [str(m).lower() for m in out_mods]:
            return False
    lowered = model_id.strip().lower()
    return not any(kw in lowered for kw in _NON_CHAT_MODEL_KEYWORDS)


def _build_provider_headers(provider: str, api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    prov = canonicalize_provider(provider)
    if api_key and api_key.strip():
        clean_key = api_key.strip()
        headers["Authorization"] = f"Bearer {clean_key}"
        if prov == "anthropic":
            headers["x-api-key"] = clean_key
    if prov == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _parse_models_payload(provider: str, payload: Any) -> list[dict[str, Any]]:
    """Extract normalized model entries from OpenAI, Together, Anthropic, Gemini, or Ollama responses."""
    prov = canonicalize_provider(provider)
    raw_items: list[dict[str, Any]] = []
    if isinstance(payload, list):
        raw_items = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            raw_items = [x for x in payload["data"] if isinstance(x, dict)]
        elif isinstance(payload.get("models"), list):
            raw_items = [x for x in payload["models"] if isinstance(x, dict)]

    all_parsed: list[dict[str, Any]] = []
    chat_parsed: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in raw_items:
        raw_id = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
        if not raw_id:
            continue
        model_id = raw_id
        if prov == "gemini" and model_id.startswith("models/"):
            model_id = model_id[len("models/") :]
        if model_id in seen:
            continue
        seen.add(model_id)

        display_name = str(item.get("name") or item.get("display_name") or "").strip() or None
        if display_name == model_id:
            display_name = None
        owned_by = str(item.get("owned_by") or item.get("organization") or "").strip() or None

        top_provider = item.get("top_provider") if isinstance(item.get("top_provider"), dict) else {}
        limits = item.get("limits") if isinstance(item.get("limits"), dict) else {}
        ctx_raw = (
            item.get("context_length")
            or item.get("context_window")
            or item.get("contextWindow")
            or item.get("max_context_length")
            or item.get("max_model_len")
            or item.get("context_size")
            or item.get("max_input_tokens")
            or item.get("max_context_tokens")
            or item.get("n_ctx")
            or item.get("input_token_limit")
            or item.get("inputTokenLimit")
            or top_provider.get("context_length")
            or limits.get("context")
            or limits.get("input")
            or limits.get("context_length")
        )
        context_length: int | None = None
        if isinstance(ctx_raw, (int, float)) and ctx_raw > 0:
            context_length = int(ctx_raw)

        out_raw = (
            item.get("max_output_tokens")
            or item.get("maxOutputTokens")
            or item.get("max_completion_tokens")
            or item.get("maxCompletionTokens")
            or item.get("output_token_limit")
            or item.get("outputTokenLimit")
            or top_provider.get("max_completion_tokens")
            or top_provider.get("max_output_tokens")
            or limits.get("output")
            or limits.get("max_output_tokens")
            or limits.get("max_completion_tokens")
        )
        max_output_tokens: int | None = None
        if isinstance(out_raw, (int, float)) and out_raw > 0:
            max_output_tokens = int(out_raw)

        pricing = item.get("pricing")
        is_free = model_id.lower().endswith(":free")
        if not is_free and isinstance(pricing, dict):
            try:
                p_prompt = float(pricing.get("prompt", "1"))
                p_comp = float(pricing.get("completion", "1"))
                is_free = p_prompt == 0.0 and p_comp == 0.0
            except (TypeError, ValueError):
                is_free = False

        input_modalities: list[str] | None = None
        arch = item.get("architecture")
        if isinstance(arch, dict) and isinstance(arch.get("input_modalities"), list):
            mods = [
                str(m).strip().lower()
                for m in arch["input_modalities"]
                if str(m).strip().lower() in {"text", "image", "audio", "video"}
            ]
            if mods:
                input_modalities = mods
        elif isinstance(item.get("input_modalities"), list):
            mods = [
                str(m).strip().lower()
                for m in item["input_modalities"]
                if str(m).strip().lower() in {"text", "image", "audio", "video"}
            ]
            if mods:
                input_modalities = mods

        entry: dict[str, Any] = {
            "id": model_id,
            "name": display_name,
            "owned_by": owned_by,
            "context_length": context_length,
            "max_output_tokens": max_output_tokens,
            "is_free": is_free,
            "input_modalities": input_modalities,
            "source": "live",
        }
        all_parsed.append(entry)
        if _is_likely_chat_llm(model_id, item):
            chat_parsed.append(entry)

    selected = chat_parsed if chat_parsed else all_parsed
    selected.sort(key=lambda m: (not bool(m.get("is_free")), str(m["id"]).lower()))
    return selected


def _is_non_public_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    )


def _validate_outbound_provider_url(url: str, provider: str) -> tuple[bool, str | None]:
    """Validate that a provider base URL is safe for server-side outbound HTTP requests."""
    cleaned = (url or "").strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False, "Provide a valid http(s) Base URL to load models from this provider."
    if parsed.username or parsed.password:
        return False, "Provider Base URL must not contain embedded credentials."

    host = parsed.hostname.lower().rstrip(".")
    allow_loopback = canonicalize_provider(provider) in LOCAL_NO_KEY_PROVIDERS

    if host in ("localhost",) or host.endswith(".localhost"):
        if not allow_loopback:
            return False, f"Loopback Base URL {host!r} is only allowed for local providers (ollama/vllm)."
        return True, None

    if host.endswith((".local", ".internal", ".localdomain")) or host == "metadata.google.internal":
        return False, f"Blocked internal/metadata provider hostname: {host!r}."

    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if literal_ip.is_loopback and allow_loopback:
            return True, None
        if _is_non_public_ip(literal_ip):
            return False, f"Blocked non-public provider IP address: {host!r}."
        return True, None

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return True, None

    for _family, _socktype, _proto, _canon, sockaddr in addr_infos:
        try:
            resolved_ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if resolved_ip.is_loopback and allow_loopback:
            continue
        if _is_non_public_ip(resolved_ip):
            return False, f"Provider hostname {host!r} resolves to a blocked non-public IP address."

    return True, None


def _normalized_origin(url: str) -> tuple[str, str, int] | None:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    return (parsed.scheme.lower(), parsed.hostname.lower().rstrip("."), parsed.port or default_port)


def _is_server_key_allowed_for_url(provider: str, resolved_base: str) -> bool:
    """Only attach stored server-side provider keys when targeting the provider's canonical origin."""
    prov = canonicalize_provider(provider)
    canonical_base = PROVIDER_BASE_URLS.get(prov)
    if not canonical_base:
        return False
    target_origin = _normalized_origin(resolved_base)
    canonical_origin = _normalized_origin(canonical_base)
    return target_origin is not None and target_origin == canonical_origin


async def list_provider_models(
    provider: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    role: ModelRole = "primary",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Query the selected provider's models endpoint and extract available LLM model IDs."""
    source_env = dict(os.environ) if env is None else env
    prov = canonicalize_provider(provider)
    resolved_base = (base_url or PROVIDER_BASE_URLS.get(prov) or "").strip().rstrip("/")
    role_env_var = "WORKER_API_KEY" if role == "worker" else "PRIMARY_API_KEY"
    explicit_key = api_key.strip() if api_key and api_key.strip() else ""
    server_env_key = (
        ""
        if explicit_key
        else _resolve_api_key(
            prov,
            role_env_var,
            source_env,
            allow_explicit_fallback=is_custom_provider_id(prov),
        )
    )

    url_ok, url_err = _validate_outbound_provider_url(resolved_base, prov)
    if not url_ok:
        return {
            "ok": False,
            "provider": prov,
            "base_url": resolved_base,
            "api_key_configured": bool(explicit_key or server_env_key),
            "source": "error",
            "models": [],
            "count": 0,
            "error": url_err,
        }

    # Never forward stored server-side API keys to a non-canonical base URL origin for built-in providers.
    if (
        not explicit_key
        and server_env_key
        and not is_custom_provider_id(prov)
        and not _is_server_key_allowed_for_url(prov, resolved_base)
    ):
        resolved_key = ""
    else:
        resolved_key = explicit_key or server_env_key

    result: dict[str, Any] = {
        "ok": False,
        "provider": prov,
        "base_url": resolved_base,
        "api_key_configured": bool(resolved_key and resolved_key.strip()),
        "source": "error",
        "models": [],
        "count": 0,
        "error": None,
    }

    headers = _build_provider_headers(prov, resolved_key)
    models_url = f"{resolved_base}/models"

    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=False) as client:
            resp = await client.get(models_url, headers=headers)
            result["http_status"] = resp.status_code

            if resp.status_code in (301, 302, 303, 307, 308):
                result["error"] = f"Provider endpoint redirected (HTTP {resp.status_code}); redirects are blocked for security."
                return result

            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    result["error"] = f"Provider returned non-JSON response from {models_url}."
                    return result
                models = _parse_models_payload(prov, payload)
                result["ok"] = True
                result["source"] = "live"
                result["models"] = models
                result["count"] = len(models)
                return result

            # Ollama fallback: try /api/tags if /v1/models is unavailable
            if prov == "ollama" and resp.status_code in (404, 405):
                root_url = resolved_base[:-3] if resolved_base.endswith("/v1") else resolved_base
                tags_resp = await client.get(f"{root_url}/api/tags", headers={"Accept": "application/json"})
                if tags_resp.status_code == 200:
                    models = _parse_models_payload(prov, tags_resp.json())
                    result["ok"] = True
                    result["source"] = "live"
                    result["models"] = models
                    result["count"] = len(models)
                    return result

            if resp.status_code in (401, 403):
                env_hint = ", ".join(PROVIDER_ENV_KEY_MAP.get(prov, (role_env_var,)))
                result["error"] = (
                    f"API key required to fetch live models ({env_hint}). Enter your key above or choose a preset."
                    if not resolved_key
                    else "Provider rejected the API key (HTTP 401/403)."
                )
                return result

            if resp.status_code == 429:
                result["error"] = "Provider rate-limited the model list request (HTTP 429)."
                return result

            result["error"] = f"Provider returned HTTP {resp.status_code} from {models_url}."
            return result
    except httpx.TimeoutException:
        result["error"] = f"Timed out loading models from {models_url}."
        return result
    except httpx.HTTPError as exc:
        result["error"] = f"Could not reach {models_url} ({type(exc).__name__})."
        return result


async def probe_model_connectivity(
    role: ModelRole = "primary",
    *,
    spec: ModelSpec | None = None,
    run_inference: bool = False,
) -> dict[str, Any]:
    """Verify real HTTP reachability, authentication, and optional live completion."""
    resolved = spec or load_model_spec(role, include_persisted=True)
    result: dict[str, Any] = {
        "role": resolved.role,
        "provider": resolved.provider,
        "model": resolved.model,
        "base_url": resolved.base_url,
        "api_key_configured": bool(resolved.api_key and resolved.api_key.strip()),
        "endpoint_reachable": False,
        "authenticated": False,
        "model_listed": None,
        "inference_ok": None,
        "error": None,
    }

    url_ok, url_err = _validate_outbound_provider_url(resolved.base_url, resolved.provider)
    if not url_ok:
        result["error"] = url_err
        return result

    effective_key = resolved.api_key
    if (
        effective_key
        and resolved.provider not in LOCAL_NO_KEY_PROVIDERS
        and not is_custom_provider_id(resolved.provider)
        and not _is_server_key_allowed_for_url(resolved.provider, resolved.base_url)
    ):
        result["error"] = (
            f"Refusing to send stored '{resolved.provider}' API key to non-canonical base URL "
            f"{resolved.base_url!r}. Configure a custom provider for custom endpoints."
        )
        return result

    headers = _build_provider_headers(resolved.provider, effective_key)
    models_url = f"{resolved.base_url}/models"
    needs_key = resolved.provider not in LOCAL_NO_KEY_PROVIDERS
    has_key = bool(effective_key and effective_key.strip())
    env_hint = ", ".join(
        PROVIDER_ENV_KEY_MAP.get(resolved.provider, ("PRIMARY_API_KEY",))
    )

    try:
        async with httpx.AsyncClient(timeout=min(resolved.timeout_seconds, 15.0), follow_redirects=False) as client:
            resp = await client.get(models_url, headers=headers)
            result["endpoint_reachable"] = True
            result["http_status"] = resp.status_code

            if resp.status_code in (301, 302, 303, 307, 308):
                result["error"] = f"Provider endpoint redirected (HTTP {resp.status_code}); redirects are blocked for security."
                return result

            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    parsed_models = _parse_models_payload(resolved.provider, payload)
                    ids = [str(m["id"]) for m in parsed_models]
                    if ids:
                        result["model_listed"] = resolved.model in ids
                        result["available_models_count"] = len(ids)
                    else:
                        result["model_listed"] = None
                except ValueError:
                    result["model_listed"] = None

                # Public /models endpoints (e.g. OpenRouter, NVIDIA NIM) return 200 even with
                # no key or an invalid key. Never report authenticated=True without a verified key.
                if needs_key and not has_key:
                    result["authenticated"] = False
                    result["error"] = (
                        f"Missing API key for '{resolved.provider}'. "
                        f"Enter {env_hint} above and click Save LLM Config."
                    )
                    return result

                if resolved.provider == "openrouter" and has_key:
                    auth_resp = await client.get(
                        f"{resolved.base_url}/auth/key", headers=headers
                    )
                    if auth_resp.status_code in (401, 403):
                        result["http_status"] = auth_resp.status_code
                        result["authenticated"] = False
                        result["error"] = "OpenRouter rejected API key (HTTP 401/403)."
                        return result

                if resolved.provider == "nvidia" and has_key and not run_inference:
                    auth_resp = await client.post(
                        f"{resolved.base_url}/chat/completions",
                        headers={**headers, "Content-Type": "application/json"},
                        json={
                            "model": resolved.model,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                        },
                    )
                    if auth_resp.status_code in (401, 403):
                        result["http_status"] = auth_resp.status_code
                        result["authenticated"] = False
                        result["error"] = "NVIDIA NIM rejected API key (HTTP 401/403)."
                        return result
                    if auth_resp.status_code == 404:
                        result["http_status"] = auth_resp.status_code
                        result["authenticated"] = True
                        result["model_listed"] = False
                        result["error"] = (
                            f"Model '{resolved.model}' was not found on NVIDIA NIM (HTTP 404). "
                            "Pick a model from the dropdown."
                        )
                        return result

                result["authenticated"] = True
            elif resp.status_code in (401, 403):
                result["error"] = (
                    f"Missing API key on server environment (set {env_hint})"
                    if not has_key
                    else "Provider rejected API key (HTTP 401/403)"
                )
                return result
            elif resp.status_code == 429:
                result["error"] = "Provider rate-limited request (HTTP 429)"
                return result
            else:
                result["error"] = f"Unexpected HTTP {resp.status_code} from {models_url}"
                return result
    except httpx.TimeoutException:
        result["error"] = f"Timed out connecting to {models_url}"
        return result
    except httpx.HTTPError as exc:
        result["error"] = f"HTTP error contacting {models_url}: {type(exc).__name__}"
        return result

    if run_inference and result["authenticated"]:
        try:
            chat = build_chat_model(role, spec=resolved, require_api_key=True)
            reply = await chat.ainvoke([HumanMessage(content="Respond with OK only.")])
            result["inference_ok"] = bool(str(reply.content).strip())
        except Exception as exc:  # noqa: BLE001
            classified = classify_provider_exception(exc, resolved)
            result["inference_ok"] = False
            result["error"] = str(classified)

    return result
