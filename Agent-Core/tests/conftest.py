"""Shared test utilities and fixtures for the Phase 11 verification suite."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr
from agent_core.sandbox import SandboxConfig, SandboxManager
from agent_core.skills import SkillRegistry


class ScriptedChatModel(BaseChatModel):
    """Deterministic BaseChatModel for testing LangGraph routing, retries, HITL, and sub-agents.

    NOTE: Never used as proof of live River AI connectivity (which is tested separately in
    ``tests/test_models.py`` against the real endpoint).
    """

    model_name: str = "scripted-test-model"
    _responses: list[Any] = PrivateAttr(default_factory=list)
    _handler: Callable[[list[BaseMessage], int], Any] | None = PrivateAttr(default=None)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        *,
        handler: Callable[[list[BaseMessage], int], Any] | None = None,
        model_name: str = "scripted-test-model",
    ) -> None:
        super().__init__(model_name=model_name)
        self._responses = list(responses or [])
        self._handler = handler
        self._calls = []
        self._bound_tool_names = []

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def bound_tool_names(self) -> list[str]:
        return list(self._bound_tool_names)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> ScriptedChatModel:
        names: list[str] = []
        for item in tools:
            if hasattr(item, "name"):
                names.append(str(item.name))
            elif isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
        self._bound_tool_names = names
        return self

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        call_idx = len(self._calls)
        self._calls.append(list(messages))

        if self._handler is not None:
            outcome = self._handler(list(messages), call_idx)
        elif call_idx < len(self._responses):
            outcome = self._responses[call_idx]
        else:
            outcome = AIMessage(content="Completed.")

        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, AIMessage):
            return outcome
        return AIMessage(content=str(outcome))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture()
def tmp_checkpoint_db(tmp_path: Path) -> Path:
    return tmp_path / "data" / "checkpoints.db"


@pytest.fixture()
def isolated_sandbox(tmp_workspace: Path) -> SandboxManager:
    cfg = SandboxConfig(
        workspace_dir=tmp_workspace,
        backend="local_isolated",
        timeout_seconds=10,
    )
    mgr = SandboxManager(cfg)
    yield mgr
    mgr.cleanup_all()


@pytest.fixture()
def repo_skills() -> SkillRegistry:
    return SkillRegistry(skills_dir=Path.cwd() / "Skills")
