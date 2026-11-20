"""Provider-agnostic chat model adapter for Main Agent (Primary) and Sub-Agent (Worker).

Supports OpenAI-compatible endpoints (River AI, OpenRouter, OpenAI, Gemini, DeepSeek,
Groq, Together, Fireworks, Anthropic, local Ollama/vLLM, or custom base URLs)
configurable via environment variables, persisted runtime settings, or per-request overrides.
Credentials are never logged or exposed in reprs or diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_RIVER_AI_BASE_URL = "https://api.river.ai/v1"
DEFAULT_PRIMARY_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_WORKER_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_LLM_CONFIG_PATH = Path(os.getenv("LLM_CONFIG_PATH", "./data/llm_config.json"))

PROVIDER_BASE_URLS: dict[str, str] = {
    "river_ai": DEFAULT_RIVER_AI_BASE_URL,
    "river": DEFAULT_RIVER_AI_BASE_URL,
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
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
}

LOCAL_NO_KEY_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "local"})

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
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: float = 90.0
    max_retries: int = 2
    reasoning_effort: str | None = None

    def __repr__(self) -> str:
        return (
            f"ModelSpec(role={self.role!r}, provider={self.provider!r}, "
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key={mask_secret(self.api_key)!r}, "
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
            "max_retries": self.max_retries,
            "reasoning_effort": self.reasoning_effort,
        }


def _resolve_api_key(
    provider: str,
    explicit_env_var: str | None,
    env: dict[str, str],
) -> str | None:
    if explicit_env_var:
        direct = env.get(explicit_env_var, "").strip()
        if direct:
            return direct
    for key_name in PROVIDER_ENV_KEY_MAP.get(provider, ("RIVER_AI_API", "OPENAI_API_KEY")):
        candidate = env.get(key_name, "").strip()
        if candidate:
            return candidate
    return None


def load_persisted_llm_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load non-secret runtime LLM overrides saved from the Nemofy Agent Settings panel."""
    path = config_path or DEFAULT_LLM_CONFIG_PATH
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
    """Persist non-secret Main Agent & Sub-Agent LLM configuration cleanly to disk."""
    path = config_path or DEFAULT_LLM_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip any accidental secret fields before persisting
    clean: dict[str, Any] = {}
    for key in ("primary", "worker"):
        if isinstance(config.get(key), dict):
            clean[key] = {
                k: v for k, v in config[key].items() if k != "api_key" and v is not None
            }
    if "max_turns" in config and config["max_turns"] is not None:
        clean["max_turns"] = int(config["max_turns"])
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    return path


def load_model_spec(
    role: ModelRole = "primary",
    env: dict[str, str] | None = None,
    *,
    include_persisted: bool = False,
) -> ModelSpec:
    """Resolve Primary or Worker model specification from environment variables."""
    source_env = dict(os.environ) if env is None else env

    timeout_seconds = _parse_float(
        source_env.get("MODEL_TIMEOUT_SECONDS"),
        default=90.0,
        min_val=1.0,
    )
    max_retries = _parse_int(
        source_env.get("MODEL_MAX_RETRIES"),
        default=2,
        min_val=0,
    )

    if role == "primary":
        provider = (source_env.get("PRIMARY_PROVIDER") or "river_ai").strip().lower()
        default_base = PROVIDER_BASE_URLS.get(provider, DEFAULT_RIVER_AI_BASE_URL)
        base_url = (source_env.get("PRIMARY_BASE_URL") or default_base).strip().rstrip("/")
        model = (source_env.get("PRIMARY_MODEL") or DEFAULT_PRIMARY_MODEL).strip()
        api_key = _resolve_api_key(provider, "PRIMARY_API_KEY", source_env)
        temperature = _parse_float(source_env.get("PRIMARY_TEMPERATURE"), 0.2, min_val=0.0)
        max_tokens = _parse_int(source_env.get("PRIMARY_MAX_TOKENS"), 4096, min_val=64)
        reasoning_effort = (source_env.get("PRIMARY_REASONING_EFFORT") or "").strip() or None
    else:
        primary_provider = (source_env.get("PRIMARY_PROVIDER") or "river_ai").strip().lower()
        provider = (source_env.get("WORKER_PROVIDER") or primary_provider).strip().lower()
        default_base = PROVIDER_BASE_URLS.get(provider, DEFAULT_RIVER_AI_BASE_URL)
        base_url = (source_env.get("WORKER_BASE_URL") or default_base).strip().rstrip("/")
        model = (source_env.get("WORKER_MODEL") or DEFAULT_WORKER_MODEL).strip()
        api_key = _resolve_api_key(provider, "WORKER_API_KEY", source_env)
        temperature = _parse_float(source_env.get("WORKER_TEMPERATURE"), 0.1, min_val=0.0)
        max_tokens = _parse_int(source_env.get("WORKER_MAX_TOKENS"), 2048, min_val=64)
        reasoning_effort = (source_env.get("WORKER_REASONING_EFFORT") or "").strip() or None

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
    """Per-request model override selected from UI Settings or API payload."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None

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
        )
    else:
        ov = override

    if not ov.has_any_override():
        return base_spec

    provider = (ov.provider or base_spec.provider).strip().lower()
    provider_changed = provider != base_spec.provider

    if ov.base_url and ov.base_url.strip():
        base_url = ov.base_url.strip().rstrip("/")
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

    if ov.api_key and ov.api_key.strip():
        api_key: str | None = ov.api_key.strip()
    elif provider_changed:
        api_key = _resolve_api_key(provider, None, source_env)
    else:
        api_key = base_spec.api_key

    temperature = ov.temperature if ov.temperature is not None else base_spec.temperature
    if temperature < 0.0:
        raise ModelConfigurationError(f"temperature must be >= 0.0, got {temperature}.")

    max_tokens = ov.max_tokens if ov.max_tokens is not None else base_spec.max_tokens
    if max_tokens < 64:
        raise ModelConfigurationError(f"max_tokens must be >= 64, got {max_tokens}.")

    reasoning_effort = (
        ov.reasoning_effort if ov.reasoning_effort is not None else base_spec.reasoning_effort
    )

    return ModelSpec(
        role=role,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=base_spec.timeout_seconds,
        max_retries=base_spec.max_retries,
        reasoning_effort=reasoning_effort,
    )


def get_provider_presets() -> dict[str, Any]:
    """Return credential-free provider presets & env availability for the Settings UI."""
    env = dict(os.environ)
    presets: dict[str, dict[str, Any]] = {}
    for name, url in PROVIDER_BASE_URLS.items():
        if name in ("river", "google", "local"):
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
    if resolved.reasoning_effort and resolved.reasoning_effort.lower() not in {"off", "none"}:
        extra_body["reasoning_effort"] = resolved.reasoning_effort

    kwargs: dict[str, Any] = {
        "model": resolved.model,
        "api_key": SecretStr(resolved.api_key or "not-configured"),
        "base_url": resolved.base_url,
        "temperature": resolved.temperature,
        "max_tokens": resolved.max_tokens,
        "timeout": resolved.timeout_seconds,
        "max_retries": resolved.max_retries,
        "streaming": streaming,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


def get_primary_model(*, streaming: bool = False) -> BaseChatModel:
    """Build the configured Main Agent (PRIMARY_MODEL)."""
    return build_chat_model("primary", streaming=streaming)


def get_worker_model(*, streaming: bool = False) -> BaseChatModel:
    """Build the configured Sub-Agent (WORKER_MODEL)."""
    return build_chat_model("worker", streaming=streaming)


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

    headers: dict[str, str] = {"Accept": "application/json"}
    if resolved.api_key:
        headers["Authorization"] = f"Bearer {resolved.api_key}"

    models_url = f"{resolved.base_url}/models"
    try:
        async with httpx.AsyncClient(timeout=min(resolved.timeout_seconds, 15.0)) as client:
            resp = await client.get(models_url, headers=headers)
            result["endpoint_reachable"] = True
            result["http_status"] = resp.status_code

            if resp.status_code == 200:
                result["authenticated"] = True
                try:
                    payload = resp.json()
                    ids = [
                        item.get("id")
                        for item in payload.get("data", [])
                        if isinstance(item, dict) and item.get("id")
                    ]
                    result["model_listed"] = resolved.model in ids if ids else None
                except ValueError:
                    result["model_listed"] = None
            elif resp.status_code in (401, 403):
                result["error"] = (
                    "Missing API key on server environment"
                    if not resolved.api_key
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
