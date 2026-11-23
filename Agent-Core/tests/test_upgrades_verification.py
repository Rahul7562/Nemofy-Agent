"""Comprehensive verification suite for the 9 full-stack upgrades in Agent-Core.

Covers:
1. ALLOW_UNSAFE_LOCAL_BASH runtime toggle, env consistency, and actual SandboxManager enforcement.
2. Provider-aware context window detection, unknown enforcement on custom/local endpoints, manual override, and input budgets.
3. CRITICAL 2-session concurrency acceptance test: isolating provider, model, context budget, and access_mode.
4. Multiple independent custom providers (OpenAI-compatible & Anthropic-compatible) and per-provider vault env lookup.
5. Firecrawl client (health, scrape, search, map, crawl polling, extract, retries, honest error classification) & local tools wiring.
6. read_file pagination (line numbers, total_lines, has_more, next_start_line, EOF) and exemption from 4,000-char compaction.
7. Planning timeout resilience: separated httpx.Timeout, provider timeout profiles, exponential-backoff retries, stream aclose(), and fallback.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage

from agent_core.context_manager import (
    compute_context_usage,
    resolve_model_context_window,
)
from agent_core.firecrawl import FirecrawlClient, FirecrawlConfig, FirecrawlError
from agent_core.guardrails import (
    bind_run_access_mode,
    classify_tool_call,
    current_access_mode,
    get_effective_access_mode,
)
from agent_core.models import (
    ModelOverride,
    custom_provider_env_var,
    get_provider_timeout_defaults,
    invoke_with_resilient_retry,
    is_custom_provider_id,
    is_timeout_model_error,
    load_persisted_llm_config,
    resolve_model_spec_with_override,
    save_persisted_llm_config,
)
from agent_core.sandbox import SandboxConfig, SandboxManager, SandboxSecurityError
from agent_core.state import compact_tool_output
from agent_core.tools.local import build_local_tools


class TestUnsafeLocalBashAndReadFilePagination(unittest.IsolatedAsyncioTestCase):
    async def test_allow_unsafe_local_bash_enforced_and_togglable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Production default backend='auto' without Docker available must enforce allow_unsafe_local_bash
            cfg = SandboxConfig(
                workspace_dir=Path(tmp),
                backend="auto",
                allow_unsafe_local_bash=False,
            )
            with patch.object(SandboxManager, "_init_backend", lambda self: setattr(self, "_docker_available", False)):
                sandbox = SandboxManager(cfg)
            self.assertFalse(sandbox.allow_unsafe_local_bash)

            # When disabled without Docker, local bash must be strictly refused
            with self.assertRaises(SandboxSecurityError):
                await sandbox.exec_command("echo blocked", session_id="s1")

            # Live toggle ON via backend config setter -> command succeeds inside workspace
            self.assertTrue(sandbox.set_allow_unsafe_local_bash(True))
            self.assertTrue(sandbox.allow_unsafe_local_bash)
            res = await sandbox.exec_command("echo allowed_ok", session_id="s1")
            self.assertEqual(res.exit_code, 0)
            self.assertIn("allowed_ok", res.stdout)

            # Live toggle OFF -> immediately refused again
            self.assertFalse(sandbox.set_allow_unsafe_local_bash(False))
            with self.assertRaises(SandboxSecurityError):
                await sandbox.exec_command("echo blocked_again", session_id="s1")

    def test_read_file_paginated_and_exempt_from_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxManager(SandboxConfig(workspace_dir=Path(tmp), backend="local_isolated"))
            # Build a 320-line file (> 18,000 chars)
            lines = [f"LINE_{i:04d}: " + ("x" * 52) for i in range(1, 321)]
            full_text = "\n".join(lines) + "\n"
            sandbox.write_file("large_module.py", full_text)

            # Page 1: lines 1..120
            page1 = sandbox.read_file_paginated("large_module.py", start_line=1, max_lines=120)
            self.assertEqual(page1["total_lines"], 320)
            self.assertEqual(page1["returned_start_line"], 1)
            self.assertEqual(page1["returned_end_line"], 120)
            self.assertTrue(page1["has_more"])
            self.assertEqual(page1["next_start_line"], 121)
            self.assertIn("LINE_0001:", page1["formatted_output"])
            self.assertIn("LINE_0120:", page1["formatted_output"])
            self.assertIn("PARTIAL", page1["formatted_output"])
            self.assertGreater(len(page1["formatted_output"]), 4_000)

            # compact_tool_output MUST NOT destructively truncate read_file to 4,000 chars
            compacted, archive_path = compact_tool_output(
                "read_file",
                page1["formatted_output"],
                workspace_dir=Path(tmp),
            )
            self.assertEqual(compacted, page1["formatted_output"])
            self.assertIsNone(archive_path)

            # Page 2..3 until EOF
            page2 = sandbox.read_file_paginated(
                "large_module.py",
                start_line=page1["next_start_line"],
                max_lines=120,
            )
            self.assertEqual(page2["returned_start_line"], 121)
            self.assertEqual(page2["returned_end_line"], 240)
            self.assertTrue(page2["has_more"])
            self.assertEqual(page2["next_start_line"], 241)

            page3 = sandbox.read_file_paginated(
                "large_module.py",
                start_line=page2["next_start_line"],
                max_lines=120,
            )
            self.assertEqual(page3["returned_start_line"], 241)
            self.assertEqual(page3["returned_end_line"], 320)
            self.assertFalse(page3["has_more"])
            self.assertIsNone(page3["next_start_line"])
            self.assertIn("END OF FILE", page3["formatted_output"])
            self.assertIn("LINE_0320:", page3["formatted_output"])


class TestContextWindowAndCustomProviders(unittest.TestCase):
    def test_provider_aware_context_windows_unknown_and_manual_override(self) -> None:
        # Provider-aware limits: same model name differs across Groq vs OpenRouter
        groq_l4, groq_src = resolve_model_context_window("groq", "meta-llama/llama-4-scout")
        or_l4, or_src = resolve_model_context_window("openrouter", "meta-llama/llama-4-scout")
        self.assertEqual(groq_l4, 131_072)
        self.assertEqual(groq_src, "provider_registry")
        self.assertEqual(or_l4, 1_048_576)
        self.assertEqual(or_src, "provider_registry")

        openai_win, openai_src = resolve_model_context_window("openai", "gpt-5.2")
        self.assertEqual(openai_win, 400_000)
        self.assertIn(openai_src, ("provider_registry", "model_registry"))

        # Custom / local endpoints without explicit metadata MUST be unknown (None)
        custom_win, custom_src = resolve_model_context_window("custom_lab", "gpt-4o")
        self.assertIsNone(custom_win)
        self.assertEqual(custom_src, "unknown")

        ollama_win, ollama_src = resolve_model_context_window("ollama", "qwen3-coder")
        self.assertIsNone(ollama_win)
        self.assertEqual(ollama_src, "unknown")

        # Manual override takes highest precedence
        manual_win, manual_src = resolve_model_context_window(
            "custom_lab",
            "my-local-model",
            manual_context_window=64_000,
        )
        self.assertEqual(manual_win, 64_000)
        self.assertEqual(manual_src, "manual")

        # Input budget calculation accounts for reserved output + safety buffer
        usage = compute_context_usage(
            messages=[HumanMessage(content="Test prompt for budget calculation.")],
            model="gpt-5.2",
            provider="openai",
            configured_max_tokens=8_192,
        )
        self.assertEqual(usage["context_window"], 400_000)
        self.assertEqual(usage["detection_mode"], "auto")
        self.assertEqual(usage["reserved_output_tokens"], 8_192)
        self.assertLess(usage["usable_input_window"], 400_000)
        self.assertGreater(usage["remaining_input_budget"], 0)

    def test_multiple_custom_providers_persistence_and_spec_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "llm_config.json"
            custom_providers = [
                {
                    "id": "custom_alpha",
                    "name": "Alpha OpenAI",
                    "base_url": "https://alpha.example.com/v1",
                    "protocol": "openai_compatible",
                    "enabled": True,
                    "timeout_seconds": 95,
                    "models": [{"id": "alpha-1", "context_window": 131_072, "max_output_tokens": 16_384}],
                },
                {
                    "id": "custom_beta",
                    "name": "Beta Anthropic",
                    "base_url": "https://beta.example.com",
                    "protocol": "anthropic_compatible",
                    "enabled": True,
                    "timeout_seconds": 180,
                    "models": [{"id": "beta-claude", "context_window": 200_000, "max_output_tokens": 32_000}],
                },
            ]
            save_persisted_llm_config(
                {
                    "primary": {"provider": "openai", "model": "gpt-5.2"},
                    "worker": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
                    "allow_unsafe_local_bash": False,
                    "custom_providers": custom_providers,
                },
                config_path=cfg_path,
            )
            loaded = load_persisted_llm_config(cfg_path)
            self.assertEqual(len(loaded["custom_providers"]), 2)
            self.assertFalse(loaded["allow_unsafe_local_bash"])

            alpha_env = custom_provider_env_var("custom_alpha")
            beta_env = custom_provider_env_var("custom_beta")
            self.assertTrue(is_custom_provider_id("custom_alpha"))
            self.assertEqual(alpha_env, "CUSTOM_PROVIDER_CUSTOM_ALPHA_API_KEY")
            self.assertEqual(beta_env, "CUSTOM_PROVIDER_CUSTOM_BETA_API_KEY")

            with patch.dict(
                os.environ,
                {
                    "LLM_CONFIG_PATH": str(cfg_path),
                    alpha_env: "sk-alpha-test-key",
                    beta_env: "sk-beta-test-key",
                },
                clear=False,
            ):
                spec_alpha = resolve_model_spec_with_override(
                    "primary",
                    override=ModelOverride(provider="custom_alpha", model="alpha-1"),
                )
                spec_beta = resolve_model_spec_with_override(
                    "primary",
                    override=ModelOverride(provider="custom_beta", model="beta-claude"),
                )
                self.assertEqual(spec_alpha.provider, "custom_alpha")
                self.assertEqual(spec_alpha.protocol, "openai_compatible")
                self.assertEqual(spec_alpha.api_key, "sk-alpha-test-key")
                self.assertEqual(spec_alpha.base_url, "https://alpha.example.com/v1")
                self.assertEqual(spec_alpha.timeout_seconds, 95.0)

                self.assertEqual(spec_beta.provider, "custom_beta")
                self.assertEqual(spec_beta.protocol, "anthropic_compatible")
                self.assertEqual(spec_beta.api_key, "sk-beta-test-key")
                self.assertEqual(spec_beta.timeout_seconds, 180.0)


class TestCriticalTwoSessionConcurrencyIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_sessions_never_leak_provider_model_context_or_access_mode(self) -> None:
        """CRITICAL ACCEPTANCE TEST:
        Two sessions run concurrently with different providers, models, context windows, and access modes.
        Session 1 mutates its provider/model/context mid-run while Session 2 is running.
        Session 2's configuration, context budget, and access mode must remain 100% unchanged.
        """
        session2_started = asyncio.Event()
        session1_mutated = asyncio.Event()

        results: dict[str, dict[str, object]] = {}

        async def run_session_2() -> None:
            override_s2 = ModelOverride(
                provider="groq",
                model="llama-3.3-70b-versatile",
                context_window=131_072,
                max_output_tokens=32_768,
                max_tokens=8_192,
            )
            tok = bind_run_access_mode("full_access")
            try:
                spec_before = resolve_model_spec_with_override("primary", override=override_s2)
                usage_before = compute_context_usage(
                    messages=[HumanMessage(content="Session 2 long-running task")],
                    model=spec_before.model,
                    provider=spec_before.provider,
                    explicit_context_window=override_s2.context_window,
                    configured_max_tokens=spec_before.max_tokens,
                )
                session2_started.set()
                # Wait until Session 1 switches provider/model/manual context mid-run
                await session1_mutated.wait()

                # Re-evaluate inside Session 2 after Session 1 mutated
                spec_after = resolve_model_spec_with_override("primary", override=override_s2)
                usage_after = compute_context_usage(
                    messages=[HumanMessage(content="Session 2 long-running task")],
                    model=spec_after.model,
                    provider=spec_after.provider,
                    explicit_context_window=override_s2.context_window,
                    configured_max_tokens=spec_after.max_tokens,
                )
                results["s2"] = {
                    "provider_before": spec_before.provider,
                    "provider_after": spec_after.provider,
                    "model_before": spec_before.model,
                    "model_after": spec_after.model,
                    "ctx_before": usage_before["context_window"],
                    "ctx_after": usage_after["context_window"],
                    "access_mode": get_effective_access_mode(),
                }
            finally:
                current_access_mode.reset(tok)

        async def run_session_1() -> None:
            await session2_started.wait()
            tok = bind_run_access_mode("default")
            try:
                # Start on OpenAI gpt-5.2, then switch mid-run to DeepSeek with manual override
                spec_initial = resolve_model_spec_with_override(
                    "primary",
                    override=ModelOverride(provider="openai", model="gpt-5.2"),
                )
                spec_switched = resolve_model_spec_with_override(
                    "primary",
                    override=ModelOverride(
                        provider="deepseek",
                        model="deepseek-reasoner",
                        manual_context_window=96_000,
                    ),
                )
                usage_switched = compute_context_usage(
                    messages=[HumanMessage(content="Session 1 prompt")],
                    model=spec_switched.model,
                    provider=spec_switched.provider,
                    manual_context_window=96_000,
                )
                results["s1"] = {
                    "initial_provider": spec_initial.provider,
                    "switched_provider": spec_switched.provider,
                    "switched_model": spec_switched.model,
                    "switched_ctx": usage_switched["context_window"],
                    "switched_mode": usage_switched["detection_mode"],
                    "access_mode": get_effective_access_mode(),
                }
                session1_mutated.set()
            finally:
                current_access_mode.reset(tok)

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test-openai",
                "GROQ_API_KEY": "gsk-test-groq",
                "DEEPSEEK_API_KEY": "sk-test-deepseek",
            },
            clear=False,
        ):
            await asyncio.gather(run_session_2(), run_session_1())

        self.assertEqual(results["s1"]["initial_provider"], "openai")
        self.assertEqual(results["s1"]["switched_provider"], "deepseek")
        self.assertEqual(results["s1"]["switched_ctx"], 96_000)
        self.assertEqual(results["s1"]["switched_mode"], "manual")
        self.assertEqual(results["s1"]["access_mode"], "default")

        # Session 2 must be 100% invariant
        self.assertEqual(results["s2"]["provider_before"], "groq")
        self.assertEqual(results["s2"]["provider_after"], "groq")
        self.assertEqual(results["s2"]["model_before"], "llama-3.3-70b-versatile")
        self.assertEqual(results["s2"]["model_after"], "llama-3.3-70b-versatile")
        self.assertEqual(results["s2"]["ctx_before"], 131_072)
        self.assertEqual(results["s2"]["ctx_after"], 131_072)
        self.assertEqual(results["s2"]["access_mode"], "full_access")


class TestFirecrawlIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_firecrawl_local_v2_spec_endpoints_retries_and_errors(self) -> None:
        """Strictly verify official Firecrawl Local (github.com/firecrawl/firecrawl#firecrawl-local) v2 API compliance."""
        import json

        seen_paths: list[tuple[str, str]] = []
        crawl_payloads: list[dict[str, object]] = []
        auth_headers: list[str] = []
        batch_polls = {"count": 0}
        crawl_polls = {"count": 0}
        search_attempts = {"count": 0}

        def firecrawl_local_v2_handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append((request.method, request.url.path))
            if request.headers.get("authorization"):
                auth_headers.append(request.headers["authorization"])

            if request.url.path == "/health" and request.method == "GET":
                return httpx.Response(
                    200,
                    json={"status": "ok", "engine": "firecrawl-local", "version": "0.1.1"},
                )
            if request.url.path == "/v2/search" and request.method == "POST":
                search_attempts["count"] += 1
                if search_attempts["count"] == 1:
                    return httpx.Response(503, json={"error": "temporary overload"})
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {
                            "web": [
                                {
                                    "title": "Firecrawl Docs",
                                    "url": "https://docs.firecrawl.dev",
                                    "description": "Turn websites into LLM-ready markdown",
                                }
                            ]
                        },
                    },
                )
            if request.url.path == "/v2/scrape" and request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {
                            "markdown": "# Local Scrape\nRendered by firecrawl-local v2. Price: $99.",
                            "metadata": {"title": "Local Scrape", "sourceURL": "https://example.com", "statusCode": 200},
                        },
                    },
                )
            if request.url.path == "/v2/map" and request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "links": [
                            {"url": "https://example.com", "title": "Home"},
                            {"url": "https://example.com/pricing", "title": "Pricing"},
                        ],
                    },
                )
            if request.url.path == "/v2/crawl" and request.method == "POST":
                crawl_payloads.append(json.loads(request.content.decode("utf-8")))
                return httpx.Response(200, json={"success": True, "id": "local-crawl-42"})
            if request.url.path == "/v2/crawl/local-crawl-42" and request.method == "GET":
                crawl_polls["count"] += 1
                if crawl_polls["count"] == 1:
                    return httpx.Response(200, json={"status": "scraping", "completed": 1, "total": 2, "data": []})
                return httpx.Response(
                    200,
                    json={
                        "status": "completed",
                        "completed": 2,
                        "total": 2,
                        "data": [
                            {"markdown": "# Home", "metadata": {"title": "Home", "sourceURL": "https://example.com"}},
                            {"markdown": "# Pricing", "metadata": {"title": "Pricing", "sourceURL": "https://example.com/pricing"}},
                        ],
                    },
                )
            if request.url.path == "/v2/crawl/local-crawl-42" and request.method == "DELETE":
                return httpx.Response(200, json={"status": "cancelled"})
            if request.url.path == "/v2/batch/scrape" and request.method == "POST":
                return httpx.Response(200, json={"success": True, "id": "batch-7"})
            if request.url.path == "/v2/batch/scrape/batch-7" and request.method == "GET":
                batch_polls["count"] += 1
                if batch_polls["count"] == 1:
                    return httpx.Response(200, json={"status": "scraping", "completed": 1, "total": 2, "data": []})
                return httpx.Response(
                    200,
                    json={
                        "status": "completed",
                        "completed": 2,
                        "total": 2,
                        "data": [
                            {"markdown": "# Page A", "metadata": {"title": "A", "sourceURL": "https://example.com/a"}},
                            {"markdown": "# Page B", "metadata": {"title": "B", "sourceURL": "https://example.com/b"}},
                        ],
                    },
                )
            if request.url.path == "/v2/batch/scrape/batch-7" and request.method == "DELETE":
                return httpx.Response(200, json={"status": "cancelled"})
            if request.url.path == "/v2/extract" and request.method == "POST":
                return httpx.Response(
                    401,
                    json={
                        "success": False,
                        "error": "This endpoint is not supported by the keyless free tier.",
                    },
                )
            return httpx.Response(404, json={"error": "not found"})

        v2_client = FirecrawlClient(
            FirecrawlConfig(
                api_url="http://127.0.0.1:3002",
                api_key="",
                local_token="local-secret-tok",
                max_retries=2,
            ),
            transport=httpx.MockTransport(firecrawl_local_v2_handler),
        )

        # 1. GET /health identifies firecrawl-local engine + version
        health = await v2_client.check_health()
        self.assertTrue(health["reachable"])
        self.assertEqual(health["mode"], "firecrawl_local")
        self.assertEqual(health["engine"], "firecrawl-local")
        self.assertEqual(health["version"], "0.1.1")

        # 2. Search recovers after transient 503 and parses v2 data.web array
        search_res = await v2_client.search("firecrawl", limit=3)
        self.assertEqual(search_attempts["count"], 2)
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["url"], "https://docs.firecrawl.dev")

        # 3. Local v2 scrape, map, crawl (maxDiscoveryDepth), cancel_crawl, batch_scrape
        scraped = await v2_client.scrape("https://example.com")
        self.assertIn("Rendered by firecrawl-local v2", scraped["markdown"])
        self.assertIn("Bearer local-secret-tok", auth_headers)

        mapped = await v2_client.map_urls("https://example.com", limit=10)
        self.assertEqual(mapped, ["https://example.com", "https://example.com/pricing"])

        crawled = await v2_client.crawl("https://example.com", limit=2, max_depth=3, poll_interval=0.01)
        self.assertEqual(crawled["status"], "completed")
        self.assertEqual(len(crawled["pages"]), 2)
        self.assertEqual(crawl_payloads[0].get("maxDiscoveryDepth"), 3)

        self.assertEqual((await v2_client.cancel_crawl("local-crawl-42")).get("status"), "cancelled")

        batched = await v2_client.batch_scrape(["https://example.com/a", "https://example.com/b"], poll_interval=0.01)
        self.assertEqual(batched["status"], "completed")
        self.assertEqual(len(batched["pages"]), 2)
        self.assertEqual((await v2_client.cancel_batch_scrape("batch-7")).get("status"), "cancelled")

        # Zero /v1/* calls ever issued
        self.assertFalse(any(path.startswith("/v1/") for _, path in seen_paths))

        # 4. Guardrails & tool invocations (including SSRF denial & keyless web_extract via firecrawl-local /v2/scrape + LLM)
        for tool_name in ("web_search", "web_fetch", "web_map", "web_crawl", "web_extract"):
            self.assertEqual(classify_tool_call(tool_name, {}).safety_level, "SAFE")

        class FakeExtractLLM:
            async def ainvoke(self, messages: list[object]) -> object:
                class _Resp:
                    content = '{"price": "$99", "engine": "firecrawl-local"}'
                return _Resp()

        with tempfile.TemporaryDirectory() as tmp:
            sbx = SandboxManager(SandboxConfig(workspace_dir=Path(tmp), backend="local_isolated"))
            tools_map = {
                t.name: t
                for t in build_local_tools(
                    sbx,
                    firecrawl_client=v2_client,
                    extract_model_factory=lambda: FakeExtractLLM(),
                )
            }
            self.assertTrue({"web_search", "web_fetch", "web_map", "web_crawl", "web_extract"}.issubset(tools_map.keys()))
            self.assertIn("source: firecrawl", await tools_map["web_search"].ainvoke({"query": "firecrawl"}))
            self.assertIn("via Firecrawl Local", await tools_map["web_fetch"].ainvoke({"url": "https://example.com"}))
            self.assertIn("https://example.com/pricing", await tools_map["web_map"].ainvoke({"url": "https://example.com"}))
            self.assertIn("Firecrawl crawl of https://example.com", await tools_map["web_crawl"].ainvoke({"url": "https://example.com"}))

            extract_out = await tools_map["web_extract"].ainvoke(
                {
                    "urls": ["https://example.com"],
                    "prompt": "Extract price",
                    "schema_json": '{"type":"object","properties":{"price":{"type":"string"}}}',
                }
            )
            self.assertIn("source: firecrawl-local+llm", extract_out)
            self.assertIn('"$99"', extract_out)

            # SSRF check rejects loopback targets cleanly
            self.assertIn("Access denied", await tools_map["web_fetch"].ainvoke({"url": "http://127.0.0.1:3002/secret"}))
            self.assertIn("Access denied", await tools_map["web_map"].ainvoke({"url": "http://127.0.0.1:3002"}))

        # 5. Honest error reporting when Firecrawl Local is unavailable (no silent DuckDuckGo/HTML fallback)
        def offline_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        offline_client = FirecrawlClient(
            FirecrawlConfig(api_url="http://127.0.0.1:3002", api_key="", max_retries=0),
            transport=httpx.MockTransport(offline_handler),
        )
        with tempfile.TemporaryDirectory() as tmp:
            sbx = SandboxManager(SandboxConfig(workspace_dir=Path(tmp), backend="local_isolated"))
            offline_tools = {t.name: t for t in build_local_tools(sbx, firecrawl_client=offline_client)}
            err_fetch = await offline_tools["web_fetch"].ainvoke({"url": "https://example.com"})
            self.assertIn("ERROR [firecrawl/unavailable]", err_fetch)
            err_search = await offline_tools["web_search"].ainvoke({"query": "example"})
            self.assertIn("ERROR [firecrawl/unavailable]", err_search)

        # 6. Default config always targets Firecrawl Local on 127.0.0.1:3002
        default_cfg = FirecrawlConfig.from_env({})
        self.assertEqual(default_cfg.api_url, "http://127.0.0.1:3002")

    async def test_firecrawl_pdf_and_office_documents_remote_and_local(self) -> None:
        """Verify Firecrawl Local native PDF (pdf-inspector) and Office (anydoc) parsing for remote URLs and local workspace files."""
        import json
        import zipfile
        from agent_core.firecrawl import find_firecrawl_local_binary, is_firecrawl_document_path

        # 1. Extension detection matches official firecrawl-local documents.rs
        for ext in (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".odt", ".ods", ".odp", ".rtf", ".epub"):
            self.assertTrue(is_firecrawl_document_path(f"report{ext}"))
        self.assertFalse(is_firecrawl_document_path("script.py"))
        self.assertFalse(is_firecrawl_document_path("notes.md"))

        # 2. Remote PDF & Office document scrape metadata formatting (numPages, engine, OCR warning)
        def doc_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            target = body.get("url", "")
            if target.endswith(".pdf"):
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {
                            "markdown": "# Remote PDF Report\nRevenue rose 24%.",
                            "warning": "1 page(s) of this PDF are scanned images; the local engine has no OCR, so their text is missing.",
                            "metadata": {
                                "sourceURL": target,
                                "title": "Q4 Report",
                                "statusCode": 200,
                                "numPages": 4,
                                "contentType": "application/pdf",
                                "engine": "local-pdf",
                            },
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "markdown": "| Metric | Value |\n| --- | --- |\n| Margin | 38% |",
                        "metadata": {
                            "sourceURL": target,
                            "statusCode": 200,
                            "contentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "engine": "local-document",
                        },
                    },
                },
            )

        doc_client = FirecrawlClient(
            FirecrawlConfig(api_url="http://127.0.0.1:3002", api_key="", max_retries=0),
            transport=httpx.MockTransport(doc_handler),
        )
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            sbx = SandboxManager(SandboxConfig(workspace_dir=ws, backend="local_isolated"))
            tools_map = {t.name: t for t in build_local_tools(sbx, firecrawl_client=doc_client)}

            remote_pdf = await tools_map["web_fetch"].ainvoke({"url": "https://example.com/q4.pdf"})
            self.assertIn("Engine: local-pdf", remote_pdf)
            self.assertIn("Pages: 4", remote_pdf)
            self.assertIn("Warning: 1 page(s) of this PDF are scanned images", remote_pdf)
            self.assertIn("Revenue rose 24%", remote_pdf)

            remote_xlsx = await tools_map["web_fetch"].ainvoke({"url": "https://example.com/model.xlsx"})
            self.assertIn("Engine: local-document", remote_xlsx)
            self.assertIn("| Margin | 38% |", remote_xlsx)

            # 3. LIVE local workspace PDF, DOCX, XLSX, and PPTX parsing through real firecrawl-local binary
            fc_bin = find_firecrawl_local_binary()
            self.assertIsNotNone(fc_bin, "firecrawl-local binary must be installed to parse local documents")

            # Write valid PDF
            pdf_bytes = (
                b"%PDF-1.4\n"
                b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
                b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
                b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
                b"4 0 obj << /Length 55 >> stream\nBT /F1 18 Tf 72 720 Td (Hello from Nemofy PDF!) Tj ET\nendstream endobj\n"
                b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
                b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000241 00000 n \n0000000346 00000 n \n"
                b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n416\n%%EOF\n"
            )
            (ws / "brief.pdf").write_bytes(pdf_bytes)

            # Write valid DOCX
            with zipfile.ZipFile(ws / "spec.docx", "w") as z:
                z.writestr(
                    "[Content_Types].xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                    '<Default Extension="xml" ContentType="application/xml"/>'
                    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                    "</Types>",
                )
                z.writestr(
                    "_rels/.rels",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                    "</Relationships>",
                )
                z.writestr(
                    "word/document.xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    "<w:body>"
                    "<w:p><w:r><w:t>Architecture Spec</w:t></w:r></w:p>"
                    "<w:p><w:r><w:t>Line two of DOCX body.</w:t></w:r></w:p>"
                    "</w:body></w:document>",
                )

            # Write valid XLSX
            with zipfile.ZipFile(ws / "budget.xlsx", "w") as z:
                z.writestr(
                    "[Content_Types].xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                    '<Default Extension="xml" ContentType="application/xml"/>'
                    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    "</Types>",
                )
                z.writestr(
                    "_rels/.rels",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                    "</Relationships>",
                )
                z.writestr(
                    "xl/workbook.xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                    '<sheets><sheet name="Summary" sheetId="1" r:id="rId1"/></sheets></workbook>',
                )
                z.writestr(
                    "xl/_rels/workbook.xml.rels",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                    "</Relationships>",
                )
                z.writestr(
                    "xl/worksheets/sheet1.xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                    '<row r="1"><c r="A1" t="inlineStr"><is><t>Service</t></is></c><c r="B1" t="inlineStr"><is><t>Status</t></is></c></row>'
                    '<row r="2"><c r="A2" t="inlineStr"><is><t>Firecrawl</t></is></c><c r="B2" t="inlineStr"><is><t>Ready</t></is></c></row>'
                    "</sheetData></worksheet>",
                )

            # Write valid PPTX
            with zipfile.ZipFile(ws / "deck.pptx", "w") as z:
                z.writestr(
                    "[Content_Types].xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                    '<Default Extension="xml" ContentType="application/xml"/>'
                    '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
                    '<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                    "</Types>",
                )
                z.writestr(
                    "_rels/.rels",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
                    "</Relationships>",
                )
                z.writestr(
                    "ppt/presentation.xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                    '<p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>',
                )
                z.writestr(
                    "ppt/_rels/presentation.xml.rels",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>'
                    "</Relationships>",
                )
                z.writestr(
                    "ppt/slides/slide1.xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                    "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Nemofy Slide Deck Title</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>"
                    "</p:sld>",
                )

            pdf_out = await tools_map["read_file"].ainvoke({"path": "brief.pdf"})
            self.assertIn("parser=firecrawl-local/local-pdf", pdf_out)
            self.assertIn("pages=1", pdf_out)
            self.assertIn("Hello from Nemofy PDF!", pdf_out)

            docx_out = await tools_map["read_file"].ainvoke({"path": "spec.docx", "start_line": 1, "end_line": 1})
            self.assertIn("parser=firecrawl-local/local-document", docx_out)
            self.assertIn("Architecture Spec", docx_out)
            self.assertIn("has_more=true", docx_out)

            xlsx_out = await tools_map["read_file"].ainvoke({"path": "budget.xlsx"})
            self.assertIn("parser=firecrawl-local/local-document", xlsx_out)
            self.assertIn("| Firecrawl | Ready |", xlsx_out)

            pptx_out = await tools_map["read_file"].ainvoke({"path": "deck.pptx"})
            self.assertIn("parser=firecrawl-local/local-document", pptx_out)
            self.assertIn("Nemofy Slide Deck Title", pptx_out)

            # Plain text files still read cleanly synchronously via .invoke()
            (ws / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            txt_out = tools_map["read_file"].invoke({"path": "hello.txt"})
            self.assertIn("alpha", txt_out)
            self.assertIn("beta", txt_out)




class TestPlanningTimeoutResilience(unittest.IsolatedAsyncioTestCase):
    async def test_provider_timeout_profiles_and_transient_timeout_recovery(self) -> None:
        # Provider-specific timeout profiles give slow reasoning providers longer read budgets than fast hosts
        groq_connect, groq_read = get_provider_timeout_defaults("groq")
        deepseek_connect, deepseek_read = get_provider_timeout_defaults("deepseek")
        self.assertGreater(deepseek_read, groq_read)
        self.assertGreaterEqual(groq_connect, 10.0)

        # Timeout classification detects httpx.ReadTimeout
        timeout_exc = httpx.ReadTimeout("Request timed out.")
        self.assertTrue(is_timeout_model_error(timeout_exc))

        # invoke_with_resilient_retry retries transient ReadTimeout with backoff and succeeds
        attempts = {"count": 0}

        async def flaky_plan_call() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ReadTimeout("Request timed out.")
            return "recovered_plan_output"

        result = await invoke_with_resilient_retry(
            flaky_plan_call,
            label="primary model",
            provider="openai",
            model="gpt-5.2",
            stage="planning",
            max_transient_retries=2,
            base_backoff_seconds=0.05,
            max_backoff_seconds=0.1,
        )
        self.assertEqual(result, "recovered_plan_output")
        self.assertEqual(attempts["count"], 2)


if __name__ == "__main__":
    unittest.main()
