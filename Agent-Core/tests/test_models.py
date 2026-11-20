"""Phase 11B: Model integration verification tests (River AI + Worker + Error Handling)."""

from __future__ import annotations

import os

import httpx
import pytest
from agent_core.models import (
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_RIVER_AI_BASE_URL,
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelProviderError,
    ModelRateLimitError,
    ModelSpec,
    build_chat_model,
    classify_provider_exception,
    load_model_spec,
    mask_secret,
    probe_model_connectivity,
)


def test_default_primary_and_worker_specs() -> None:
    spec_primary = load_model_spec("primary", env={"RIVER_AI_API": "sk-river-test-1234567890"})
    spec_worker = load_model_spec("worker", env={"RIVER_AI_API": "sk-river-test-1234567890"})

    assert spec_primary.provider == "river_ai"
    assert spec_primary.base_url == DEFAULT_RIVER_AI_BASE_URL
    assert spec_primary.model == DEFAULT_PRIMARY_MODEL
    assert spec_primary.api_key == "sk-river-test-1234567890"
    assert spec_worker.model == DEFAULT_PRIMARY_MODEL
    assert spec_worker.api_key == "sk-river-test-1234567890"


def test_provider_switchable_purely_via_env() -> None:
    env = {
        "PRIMARY_PROVIDER": "openrouter",
        "PRIMARY_MODEL": "deepseek/deepseek-chat",
        "OPENROUTER_API_KEY": "sk-or-test-9999999999",
        "WORKER_PROVIDER": "openai",
        "WORKER_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-oa-test-8888888888",
        "MODEL_TIMEOUT_SECONDS": "45",
    }
    primary = load_model_spec("primary", env=env)
    worker = load_model_spec("worker", env=env)

    assert primary.provider == "openrouter"
    assert primary.base_url == "https://openrouter.ai/api/v1"
    assert primary.model == "deepseek/deepseek-chat"
    assert primary.timeout_seconds == 45.0

    assert worker.provider == "openai"
    assert worker.base_url == "https://api.openai.com/v1"
    assert worker.model == "gpt-4o-mini"


def test_secret_redaction_in_repr_and_summary() -> None:
    secret = "sk-river-SUPER-SECRET-TOKEN-98765"
    spec = load_model_spec("primary", env={"RIVER_AI_API": secret})

    assert secret not in repr(spec)
    assert secret not in str(spec.safe_summary())
    assert spec.safe_summary()["api_key_configured"] is True
    assert mask_secret(secret) == "sk-...8765"
    assert mask_secret(None) == "<unset>"


def test_missing_api_key_raises_configuration_error() -> None:
    spec = load_model_spec("primary", env={})
    with pytest.raises(ModelConfigurationError, match="RIVER_AI_API"):
        build_chat_model("primary", spec=spec, require_api_key=True)


def test_invalid_url_and_invalid_timeout_raise_configuration_error() -> None:
    with pytest.raises(ModelConfigurationError, match="http"):
        load_model_spec("primary", env={"PRIMARY_BASE_URL": "ftp://invalid.local"})
    with pytest.raises(ModelConfigurationError, match="Invalid float"):
        load_model_spec("primary", env={"MODEL_TIMEOUT_SECONDS": "not-a-number"})


def test_error_classification_covers_auth_rate_limit_timeout_and_redacts_key() -> None:
    secret = "sk-secret-key-abc123456"
    spec = ModelSpec(
        role="primary",
        provider="river_ai",
        model=DEFAULT_PRIMARY_MODEL,
        base_url=DEFAULT_RIVER_AI_BASE_URL,
        api_key=secret,
        timeout_seconds=15.0,
    )

    req = httpx.Request("POST", f"{DEFAULT_RIVER_AI_BASE_URL}/chat/completions")

    auth_err = classify_provider_exception(
        httpx.HTTPStatusError("401", request=req, response=httpx.Response(401, request=req)),
        spec,
    )
    assert isinstance(auth_err, ModelAuthenticationError)

    rl_err = classify_provider_exception(
        httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req)),
        spec,
    )
    assert isinstance(rl_err, ModelRateLimitError)

    to_err = classify_provider_exception(httpx.ReadTimeout("timeout"), spec)
    assert isinstance(to_err, ModelConnectionError)

    conn_err = classify_provider_exception(
        httpx.ConnectError(f"failed with {secret}"),
        spec,
    )
    assert isinstance(conn_err, ModelConnectionError)
    assert secret not in str(conn_err)

    srv_err = classify_provider_exception(
        httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, request=req)),
        spec,
    )
    assert isinstance(srv_err, ModelProviderError)


@pytest.mark.asyncio
async def test_live_river_ai_endpoint_probe_and_invalid_key_rejection() -> None:
    """Real HTTP check against https://api.river.ai/v1/models (no mocking)."""
    from dotenv import load_dotenv

    load_dotenv(".env")

    # 1. Invalid key must be rejected honestly by real River AI endpoint
    bad_spec = load_model_spec("primary", env={"RIVER_AI_API": "sk-invalid-test-key-0000"})
    bad_probe = await probe_model_connectivity("primary", spec=bad_spec, run_inference=False)
    assert bad_probe["endpoint_reachable"] is True
    assert bad_probe["authenticated"] is False
    assert bad_probe["http_status"] in (401, 403)

    # 2. If a real RIVER_AI_API credential is present in environment, verify actual auth & inference
    if os.getenv("RIVER_AI_API", "").strip():
        live_probe = await probe_model_connectivity("primary", run_inference=True)
        assert live_probe["endpoint_reachable"] is True
        assert live_probe["authenticated"] is True
        assert live_probe["model_listed"] is True
        assert live_probe["inference_ok"] is True


def test_dynamic_model_override_supports_any_provider_and_local_ollama() -> None:
    """Verify Brain and Sub-Agent models can shift freely away from River AI per request."""
    from agent_core.models import ModelOverride, get_provider_presets, resolve_model_spec_with_override

    base_env = {
        "RIVER_AI_API": "rv-test-primary-1234",
        "OPENROUTER_API_KEY": "sk-or-test-5678",
        "OPENAI_API_KEY": "sk-oa-test-9012",
    }

    # Shift Brain to OpenRouter Claude while keeping River AI in env
    brain_spec = resolve_model_spec_with_override(
        "primary",
        override=ModelOverride(
            provider="openrouter",
            model="anthropic/claude-3.7-sonnet",
            temperature=0.4,
        ),
        env=base_env,
    )
    assert brain_spec.provider == "openrouter"
    assert brain_spec.base_url == "https://openrouter.ai/api/v1"
    assert brain_spec.model == "anthropic/claude-3.7-sonnet"
    assert brain_spec.api_key == "sk-or-test-5678"
    assert brain_spec.temperature == 0.4

    # Shift Sub-Agent to OpenAI GPT-4o-mini with explicit custom key override
    sub_spec = resolve_model_spec_with_override(
        "worker",
        override={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-explicit-override-key",
        },
        env=base_env,
    )
    assert sub_spec.provider == "openai"
    assert sub_spec.base_url == "https://api.openai.com/v1"
    assert sub_spec.model == "gpt-4o-mini"
    assert sub_spec.api_key == "sk-explicit-override-key"

    # Shift to local Ollama without requiring an API key
    ollama_spec = resolve_model_spec_with_override(
        "worker",
        override={"provider": "ollama", "model": "qwen2.5-coder:14b"},
        env={},
    )
    assert ollama_spec.provider == "ollama"
    assert ollama_spec.base_url == "http://127.0.0.1:11434/v1"
    # build_chat_model must succeed for local no-key providers even with require_api_key=True
    ollama_chat = build_chat_model("worker", spec=ollama_spec, require_api_key=True)
    assert ollama_chat is not None

    # Shift to arbitrary custom OpenAI-compatible endpoint
    custom_spec = resolve_model_spec_with_override(
        "primary",
        override={
            "provider": "custom",
            "model": "my-org/custom-70b",
            "base_url": "https://llm.internal.example/v1",
            "api_key": "custom-token-xyz",
        },
        env={},
    )
    assert custom_spec.base_url == "https://llm.internal.example/v1"
    assert custom_spec.model == "my-org/custom-70b"

    presets = get_provider_presets()
    assert "river_ai" in presets["providers"]
    assert "openrouter" in presets["providers"]
    assert "openai" in presets["providers"]
    assert "gemini" in presets["providers"]
    assert "ollama" in presets["providers"]
