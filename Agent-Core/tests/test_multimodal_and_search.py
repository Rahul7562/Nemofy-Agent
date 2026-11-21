"""Unit & integration tests for web_search tool, modality detection, and Accessory subagent routing."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_core.graph import create_agent_runtime
from agent_core.guardrails import classify_tool_call, normalize_tool_arguments
from agent_core.multimodal import (
    MediaAttachment,
    build_multimodal_content_blocks,
    detect_model_input_modalities,
    load_attachment_base64,
    partition_attachments_by_capability,
)
from agent_core.sandbox import SandboxConfig, SandboxManager
from agent_core.subagents import ROLE_DEFAULT_TOOLS
from agent_core.tools.local import (
    _parse_ddg_html_results,
    _parse_ddg_lite_results,
    _unwrap_search_url,
    build_local_tools,
    format_search_results_markdown,
    search_web_results,
)


class _RecordingChatModel(BaseChatModel):
    """Fake BaseChatModel recording received messages and returning scripted responses."""

    model_name: str = "fake-model"
    reply_text: str = "OK"
    fail_on_multimodal: bool = False
    recorded_calls: list[list[BaseMessage]] = []

    def model_post_init(self, __context: Any) -> None:
        self.recorded_calls = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _RecordingChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.recorded_calls.append(list(messages))
        if self.fail_on_multimodal:
            for m in messages:
                if isinstance(m, HumanMessage) and isinstance(m.content, list):
                    raise ValueError("HTTP 400: model does not support image_url content type")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply_text))])


# ---------------------------------------------------------------------------
# 1. web_search Tests
# ---------------------------------------------------------------------------
def test_ddg_url_unwrapping_and_html_parsing() -> None:
    raw_link = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&rut=abc"
    assert _unwrap_search_url(raw_link) == "https://docs.python.org/3/"
    assert _unwrap_search_url("javascript:alert(1)") == ""

    sample_html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide">Example Guide</a>
        <a class="result__snippet">A concise guide to testing web search.</a>
      </div>
      <div class="result">
        <a class="result__a" href="https://python.org">Python.org</a>
        <div class="result__snippet">Official Python website.</div>
      </div>
    </body></html>
    """
    parsed = _parse_ddg_html_results(sample_html, max_results=5)
    assert len(parsed) == 2
    assert parsed[0]["title"] == "Example Guide"
    assert parsed[0]["url"] == "https://example.com/guide"
    assert "concise guide" in parsed[0]["snippet"]
    assert parsed[1]["url"] == "https://python.org"

    md = format_search_results_markdown("python guide", parsed, "duckduckgo")
    assert "**Example Guide**" in md
    assert "https://example.com/guide" in md
    assert "web_fetch" in md


def test_ddg_lite_fallback_parsing() -> None:
    lite_html = """
    <table>
      <tr><td><a class="result-link" href="https://fastapi.tiangolo.com/">FastAPI</a></td></tr>
      <tr><td class="result-snippet">Modern fast web framework for Python.</td></tr>
    </table>
    """
    parsed = _parse_ddg_lite_results(lite_html, max_results=3)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "FastAPI"
    assert parsed[0]["url"] == "https://fastapi.tiangolo.com/"


@pytest.mark.asyncio
async def test_search_web_results_mocked_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    def _handler(request: httpx.Request) -> httpx.Response:
        if "html.duckduckgo.com" in str(request.url):
            body = (
                '<div class="result">'
                '<a class="result__a" href="https://langgraph.dev/docs">LangGraph Docs</a>'
                '<a class="result__snippet">Stateful agent graphs.</a>'
                "</div>"
            )
            return httpx.Response(200, text=body)
        return httpx.Response(404)

    transport = httpx.MockTransport(_handler)
    results, provider = await search_web_results("langgraph", max_results=3, transport=transport)
    assert provider == "duckduckgo"
    assert len(results) == 1
    assert results[0]["url"] == "https://langgraph.dev/docs"


def test_web_search_registered_in_tools_guardrails_and_subagents(tmp_path: Path) -> None:
    sandbox = SandboxManager(SandboxConfig(workspace_dir=tmp_path, backend="local_isolated"))
    tool_names = {t.name for t in build_local_tools(sandbox)}
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names

    cls = classify_tool_call("web_search", {"query": "nemofy agent"})
    assert cls.safety_level == "SAFE"

    norm = normalize_tool_arguments("web_search", {"q": "python", "limit": 3})
    assert norm == {"query": "python", "max_results": 3}

    assert "web_search" in ROLE_DEFAULT_TOOLS["deep_research"]
    assert "web_search" in ROLE_DEFAULT_TOOLS["general"]


# ---------------------------------------------------------------------------
# 2. Multimodal Capability Detection & Security
# ---------------------------------------------------------------------------
def test_modality_detection_and_partitioning() -> None:
    # Exact user scenario: DeepSeek-V4.1-Flash via River AI supports text + image (not video/audio)
    ds_mods = detect_model_input_modalities("river_ai", "deepseek-ai/DeepSeek-V4.1-Flash")
    assert ds_mods == {"text", "image"}

    # Exact user scenario: Nemotron Omni via OpenRouter supports text + image + audio + video
    omni_mods = detect_model_input_modalities(
        "openrouter", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    )
    assert omni_mods == {"text", "image", "audio", "video"}

    # Text-only model
    text_mods = detect_model_input_modalities("deepseek", "deepseek-r1")
    assert text_mods == {"text"}

    atts = [
        MediaAttachment(name="chart.png", kind="image", mime_type="image/png", data_b64="AAAA"),
        MediaAttachment(name="voice.wav", kind="audio", mime_type="audio/wav", data_b64="BBBB"),
        MediaAttachment(name="demo.mp4", kind="video", mime_type="video/mp4", data_b64="CCCC"),
    ]
    native, delegated = partition_attachments_by_capability(atts, ds_mods)
    assert [a.name for a in native] == ["chart.png"]
    assert set(delegated.keys()) == {"audio", "video"}
    assert delegated["audio"][0].name == "voice.wav"
    assert delegated["video"][0].name == "demo.mp4"


def test_attachment_path_confinement_blocks_traversal(tmp_path: Path) -> None:
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")

    att = MediaAttachment(
        name="secret.png",
        kind="image",
        mime_type="image/png",
        local_path=str(outside),
    )
    with pytest.raises(ValueError, match="outside the permitted uploads directory"):
        load_attachment_base64(att, allowed_uploads_dir=uploads_dir)

    inside = uploads_dir / "ok.png"
    inside.write_bytes(b"hello-png")
    att_ok = MediaAttachment(
        name="ok.png",
        kind="image",
        mime_type="image/png",
        local_path=str(inside),
    )
    b64 = load_attachment_base64(att_ok, allowed_uploads_dir=uploads_dir)
    assert base64.b64decode(b64) == b"hello-png"


def test_multimodal_content_blocks_format() -> None:
    atts = [
        MediaAttachment(name="pic.png", kind="image", mime_type="image/png", data_b64="aW1n"),
        MediaAttachment(name="clip.mp3", kind="audio", mime_type="audio/mpeg", data_b64="YXVk"),
        MediaAttachment(name="vid.mp4", kind="video", mime_type="video/mp4", data_b64="dmlk"),
    ]
    blocks = build_multimodal_content_blocks("Inspect all media", atts)
    assert blocks[0] == {"type": "text", "text": "Inspect all media"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"] == "data:image/png;base64,aW1n"
    assert blocks[2] == {"type": "input_audio", "input_audio": {"data": "YXVk", "format": "mp3"}}
    assert blocks[3] == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,dmlk"}}


# ---------------------------------------------------------------------------
# 3. End-to-End Graph Routing Tests (Native vs Accessory Subagent)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mixed_media_routes_image_native_and_audio_to_accessory(tmp_path: Path) -> None:
    primary = _RecordingChatModel(
        model_name="deepseek-ai/DeepSeek-V4.1-Flash",
        reply_text="Combined answer using native image and audio transcript.",
    )
    audio_accessory = _RecordingChatModel(
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        reply_text="Audio transcript: 'Launch sequence approved for Friday.'",
    )
    sandbox = SandboxManager(SandboxConfig(workspace_dir=tmp_path / "ws", backend="local_isolated"))

    async with create_agent_runtime(
        primary_model=primary,
        worker_model=primary,
        checkpoint_db_path=tmp_path / "ckpt.db",
        sandbox=sandbox,
        connect_mcp=False,
    ) as runtime:
        result = await runtime.run_task(
            "Summarize both the chart and the voice memo.",
            thread_id="mixed-media-thread",
            attachments=[
                {
                    "name": "chart.png",
                    "kind": "image",
                    "mime_type": "image/png",
                    "data_b64": base64.b64encode(b"fake-png").decode(),
                },
                {
                    "name": "memo.wav",
                    "kind": "audio",
                    "mime_type": "audio/wav",
                    "data_b64": base64.b64encode(b"fake-wav").decode(),
                },
            ],
            accessory_models={"audio": audio_accessory},
        )

    assert result["status"] == "completed"
    # Audio accessory subagent should have been invoked with input_audio
    assert len(audio_accessory.recorded_calls) == 1
    acc_human = audio_accessory.recorded_calls[0][-1]
    assert isinstance(acc_human, HumanMessage)
    assert isinstance(acc_human.content, list)
    assert any(
        b.get("type") in ("input_audio", "audio")
        for b in acc_human.content
        if isinstance(b, dict)
    )

    # Primary (DeepSeek-V4.1-Flash) should receive native image_url AND the audio subagent report text
    assert len(primary.recorded_calls) >= 1
    prim_human = [m for m in primary.recorded_calls[0] if isinstance(m, HumanMessage)][-1]
    assert isinstance(prim_human.content, list)
    assert any(b.get("type") == "image_url" for b in prim_human.content if isinstance(b, dict))
    assert not any(b.get("type") == "input_audio" for b in prim_human.content if isinstance(b, dict))
    text_block = prim_human.content[0]["text"]
    assert "Launch sequence approved for Friday" in text_block

    # Report and tool_event recorded
    assert len(result["subagent_reports"]) == 1
    assert result["subagent_reports"][0]["modality"] == "audio"
    assert any(ev["tool"] == "accessory_audio_subagent" for ev in result["tool_events"])


@pytest.mark.asyncio
async def test_omni_primary_handles_video_and_audio_natively_without_accessory(tmp_path: Path) -> None:
    primary = _RecordingChatModel(
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        reply_text="I watched the video and listened to the audio directly.",
    )
    unused_accessory = _RecordingChatModel(
        model_name="should-not-run",
        reply_text="Should never be called",
    )
    sandbox = SandboxManager(SandboxConfig(workspace_dir=tmp_path / "ws", backend="local_isolated"))

    async with create_agent_runtime(
        primary_model=primary,
        worker_model=primary,
        checkpoint_db_path=tmp_path / "ckpt.db",
        sandbox=sandbox,
        connect_mcp=False,
    ) as runtime:
        result = await runtime.run_task(
            "What happens in this clip?",
            thread_id="omni-native-thread",
            attachments=[
                {
                    "name": "clip.mp4",
                    "kind": "video",
                    "mime_type": "video/mp4",
                    "data_b64": base64.b64encode(b"fake-mp4").decode(),
                }
            ],
            accessory_models={"video": unused_accessory},
        )

    assert result["status"] == "completed"
    assert len(unused_accessory.recorded_calls) == 0
    prim_human = [m for m in primary.recorded_calls[0] if isinstance(m, HumanMessage)][-1]
    assert isinstance(prim_human.content, list)
    assert any(b.get("type") == "video_url" for b in prim_human.content if isinstance(b, dict))
    assert result["subagent_reports"] == []


@pytest.mark.asyncio
async def test_runtime_fallback_to_accessory_when_primary_rejects_native_media(tmp_path: Path) -> None:
    primary = _RecordingChatModel(
        model_name="gpt-4o",
        reply_text="Recovered via image accessory report.",
        fail_on_multimodal=True,
    )
    image_accessory = _RecordingChatModel(
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        reply_text="Image shows a blue triangle with label 42.",
    )
    sandbox = SandboxManager(SandboxConfig(workspace_dir=tmp_path / "ws", backend="local_isolated"))

    async with create_agent_runtime(
        primary_model=primary,
        worker_model=primary,
        checkpoint_db_path=tmp_path / "ckpt.db",
        sandbox=sandbox,
        connect_mcp=False,
    ) as runtime:
        result = await runtime.run_task(
            "What number is on the shape?",
            thread_id="fallback-thread",
            attachments=[
                {
                    "name": "shape.png",
                    "kind": "image",
                    "mime_type": "image/png",
                    "data_b64": base64.b64encode(b"fake-png").decode(),
                }
            ],
            accessory_models={"image": image_accessory},
        )

    assert result["status"] == "completed"
    assert "Recovered via image accessory report" in result["final_response"]
    assert len(image_accessory.recorded_calls) == 1
    assert len(primary.recorded_calls) == 2
    # Second call to primary should be plain text containing the accessory report
    retry_human = [m for m in primary.recorded_calls[1] if isinstance(m, HumanMessage)][-1]
    assert isinstance(retry_human.content, str)
    assert "blue triangle with label 42" in retry_human.content
