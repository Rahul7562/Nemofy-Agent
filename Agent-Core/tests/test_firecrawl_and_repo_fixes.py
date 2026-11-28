"""Regression tests for Firecrawl self-forwarding fix, web_extract metadata/fallback fixes,
and custom provider / accessory model resolution fixes.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent_core.firecrawl import build_firecrawl_cli_subprocess_env, is_loopback_firecrawl_url
from agent_core.models import (
    _build_provider_headers,
    _validate_outbound_provider_url,
    build_chat_model,
    resolve_accessory_model_spec,
)
from agent_core.sandbox import SandboxConfig, SandboxManager
from agent_core.tools.local import build_local_tools, extract_structured_with_llm


class FirecrawlAndRepoFixesTest(unittest.TestCase):
    def test_is_loopback_firecrawl_url(self) -> None:
        self.assertTrue(is_loopback_firecrawl_url("http://127.0.0.1:3002"))
        self.assertTrue(is_loopback_firecrawl_url("http://localhost:3002"))
        self.assertTrue(is_loopback_firecrawl_url("http://[::1]:3002"))
        self.assertTrue(is_loopback_firecrawl_url("http://0.0.0.0:3002"))
        self.assertFalse(is_loopback_firecrawl_url("https://api.firecrawl.dev"))
        self.assertFalse(is_loopback_firecrawl_url(""))

    def test_build_firecrawl_cli_subprocess_env_strips_loopback_and_secrets(self) -> None:
        sanitized = build_firecrawl_cli_subprocess_env(
            {
                "PATH": "/usr/local/bin:/usr/bin",
                "HOME": "/home/tester",
                "FIRECRAWL_API_URL": "http://127.0.0.1:3002",
                "FIRECRAWL_CLOUD": "auto",
                "OPENAI_API_KEY": "sk-secret-should-not-leak",
                "RIVER_AI_API": "sk-openrouter-secret",
                "FIRECRAWL_LOCAL_HOME": "/tmp/fc-home",
            }
        )
        self.assertNotIn("FIRECRAWL_API_URL", sanitized)
        self.assertNotIn("OPENAI_API_KEY", sanitized)
        self.assertNotIn("RIVER_AI_API", sanitized)
        self.assertEqual(sanitized["FIRECRAWL_CLOUD"], "off")
        self.assertEqual(sanitized["FIRECRAWL_LOCAL_HOME"], "/tmp/fc-home")
        self.assertEqual(sanitized["PATH"], "/usr/local/bin:/usr/bin")

    def test_extract_structured_with_llm_handles_list_content_blocks_on_repair(self) -> None:
        first_reply = MagicMock()
        first_reply.content = [{"type": "text", "text": "not valid json"}]

        repair_reply = MagicMock()
        repair_reply.content = [
            {"type": "text", "text": '```json\n{"title": "Example Domain"}\n```'}
        ]

        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(side_effect=[first_reply, repair_reply])

        parsed = asyncio.run(
            extract_structured_with_llm(
                [("https://example.com", "Page Title: Example Domain\n\nBody text")],
                prompt="Extract title",
                schema={"type": "object", "properties": {"title": {"type": "string"}}},
                model=fake_llm,
            )
        )

        self.assertEqual(parsed, {"title": "Example Domain"})
        self.assertEqual(fake_llm.ainvoke.await_count, 2)

    def test_web_extract_fallback_includes_title_metadata_and_retries_without_main_content(self) -> None:
        captured: dict[str, str] = {}

        async def fake_extract_structured(fetched_pages, *, prompt: str, schema=None, model=None):
            captured["content"] = "\n".join(md for _, md in fetched_pages)
            return {"title": "Example Domain"}

        from agent_core.firecrawl import FirecrawlError

        fake_fc_client = MagicMock()
        fake_fc_client.extract = AsyncMock(
            side_effect=FirecrawlError(
                "Keyless extraction requires local scrape fallback",
                category="cloud_only",
            )
        )
        # First call (only_main_content=True) returns empty markdown; retry returns full page + metadata
        fake_fc_client.scrape = AsyncMock(
            side_effect=[
                {"url": "https://example.com", "markdown": "", "title": "Example Domain"},
                {
                    "url": "https://example.com",
                    "markdown": "This domain is for use in documentation examples.",
                    "title": "Example Domain",
                    "description": "IANA example domain page",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxManager(
                SandboxConfig(workspace_dir=Path(tmp), backend="local_isolated"),
                docker_client=None,
            )
            tools = {
                t.name: t
                for t in build_local_tools(
                    sandbox,
                    firecrawl_client=fake_fc_client,
                    extract_model_factory=lambda: MagicMock(),
                )
            }
            web_extract = tools["web_extract"]

            with patch(
                "agent_core.tools.local.extract_structured_with_llm",
                side_effect=fake_extract_structured,
            ):
                raw_result = asyncio.run(
                    web_extract.ainvoke(
                        {
                            "urls": ["https://example.com"],
                            "prompt": "Extract the page title",
                            "schema_json": json.dumps(
                                {"type": "object", "properties": {"title": {"type": "string"}}}
                            ),
                        }
                    )
                )

        self.assertIn("source: firecrawl-local+llm", raw_result)
        self.assertIn('"title": "Example Domain"', raw_result)
        self.assertEqual(fake_fc_client.scrape.await_count, 2)
        self.assertIn("Page Title: Example Domain", captured["content"])
        self.assertIn("Page Description: IANA example domain page", captured["content"])

    def test_resolve_accessory_model_spec_and_custom_provider_protocol_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "llm_config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "custom_providers": [
                            {
                                "id": "custom_local_claude",
                                "name": "Local Claude Proxy",
                                "base_url": "http://127.0.0.1:4000",
                                "protocol": "anthropic_compatible",
                                "enabled": True,
                                "models": [{"id": "claude-local-vision"}],
                            }
                        ],
                        "accessories": {
                            "image": {
                                "provider": "custom_local_claude",
                                "model": "claude-local-vision",
                                "enabled": True,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            env = {"LLM_CONFIG_PATH": str(cfg_path)}
            spec, enabled = resolve_accessory_model_spec("image", env=env)
            self.assertTrue(enabled)
            self.assertEqual(spec.provider, "custom_local_claude")
            self.assertEqual(spec.base_url, "http://127.0.0.1:4000")
            self.assertEqual(spec.protocol, "anthropic_compatible")

            # Keyless persisted custom_* provider should build without raising Missing API key
            llm = build_chat_model("primary", spec=spec, require_api_key=True)
            self.assertIsNotNone(llm)

        # Anthropic-compatible custom provider headers
        headers = _build_provider_headers(
            "custom_local_claude",
            "sk-ant-test",
            protocol="anthropic_compatible",
        )
        self.assertEqual(headers["x-api-key"], "sk-ant-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["Authorization"], "Bearer sk-ant-test")

        # Loopback override allowed only when explicitly requested
        ok_blocked, err_blocked = _validate_outbound_provider_url(
            "http://127.0.0.1:4000/v1", "custom_local_claude"
        )
        self.assertFalse(ok_blocked)
        self.assertIsNotNone(err_blocked)

        ok_allowed, err_allowed = _validate_outbound_provider_url(
            "http://127.0.0.1:4000/v1",
            "custom_local_claude",
            allow_loopback_override=True,
        )
        self.assertTrue(ok_allowed)
        self.assertIsNone(err_allowed)


if __name__ == "__main__":
    unittest.main()
