"""Phase 8B: Authenticated FastAPI Service for the LangGraph Autonomous AI Agent.

Endpoints:
- GET  /health                   -> Service & dependency status (no secrets leaked)
- POST /agent/run                -> Execute or continue an autonomous task (Bearer auth required)
- POST /agent/resume             -> Resume a HITL-interrupted session (Bearer auth required)
- GET  /agent/threads/{thread_id}-> Inspect session state & plan (Bearer auth required)

Designed for reliable communication from the n8n Gateway Workflow and external clients.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from src.graph import (
    DEFAULT_CHECKPOINT_DB_PATH,
    AgentRuntime,
    ModelOverrideResolver,
    create_agent_runtime,
)
from src.jobs import (
    AgentJob,
    BridgeToolSpec,
    CanvasSkillSpec,
    ExternalMCPServerConfig,
    JobManager,
    format_bridge_tool_output,
    load_external_mcp_tools,
)
from src.models import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelOverride,
    ModelProviderError,
    ModelRateLimitError,
    get_primary_model,
    get_provider_presets,
    get_worker_model,
    load_model_spec,
)
from src.sandbox import SandboxManager
from src.skills import SkillRegistry, get_skills_health
from src.tools.mcp_n8n import N8nMCPManager

load_dotenv()

logger = logging.getLogger(__name__)


class ModelConfigOverride(BaseModel):
    """Optional per-request LLM selection for Brain (primary_model) or Sub-Agents (worker_model)."""

    provider: str | None = Field(
        default=None,
        max_length=64,
        description="Provider name (river_ai, openrouter, openai, gemini, groq, deepseek, together, fireworks, ollama, vllm, custom)",
    )
    model: str | None = Field(
        default=None,
        max_length=256,
        description="Model identifier supported by the selected provider",
    )
    base_url: str | None = Field(
        default=None,
        max_length=512,
        description="Optional custom OpenAI-compatible base URL",
    )
    api_key: str | None = Field(
        default=None,
        max_length=512,
        description="Optional provider API key override (never logged or returned)",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=64, le=128_000)
    reasoning_effort: str | None = Field(default=None, max_length=32)

    def to_model_override(self) -> ModelOverride | None:
        ov = ModelOverride(
            provider=self.provider.strip() if self.provider else None,
            model=self.model.strip() if self.model else None,
            base_url=self.base_url.strip() if self.base_url else None,
            api_key=self.api_key.strip() if self.api_key else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort.strip() if self.reasoning_effort else None,
        )
        return ov if ov.has_any_override() else None


class AgentRunRequest(BaseModel):
    """Validated request body for POST /agent/run."""

    task: str = Field(..., min_length=1, max_length=32_000, description="User task prompt")
    thread_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional conversation/session thread ID for checkpoint continuity",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Optional override for MAX_TURNS loop limit",
    )
    primary_model: ModelConfigOverride | None = Field(
        default=None,
        description="Optional Brain (supervisor) LLM override selected via n8n or API",
    )
    worker_model: ModelConfigOverride | None = Field(
        default=None,
        description="Optional Sub-Agent (worker) LLM override selected via n8n or API",
    )
    canvas_skills: list[CanvasSkillSpec] | None = Field(
        default=None,
        description="Optional dynamic Markdown skills wired from n8n canvas Skill slots",
    )
    mcp_servers: list[ExternalMCPServerConfig] | None = Field(
        default=None,
        description="Optional external MCP servers to connect dynamically for this request",
    )
    stall_timeout_seconds: float | None = Field(
        default=None,
        ge=0.2,
        le=3600.0,
        description="Optional stall watchdog limit (seconds with zero forward progress before aborting)",
    )

    @field_validator("task")
    @classmethod
    def _strip_task(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("task must not be blank.")
        return cleaned

    @field_validator("thread_id")
    @classmethod
    def _clean_thread_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned or None


class AgentResumeRequest(BaseModel):
    """Validated request body for POST /agent/resume (HITL approve/reject/edit)."""

    thread_id: str = Field(..., min_length=1, max_length=128)
    decision: Literal["approve", "reject", "edit"]
    arguments: dict[str, Any] | None = None
    reason: str | None = Field(default=None, max_length=1000)
    primary_model: ModelConfigOverride | None = Field(
        default=None,
        description="Optional Brain (supervisor) LLM override on resume",
    )
    worker_model: ModelConfigOverride | None = Field(
        default=None,
        description="Optional Sub-Agent (worker) LLM override on resume",
    )
    canvas_skills: list[CanvasSkillSpec] | None = Field(
        default=None,
        description="Optional dynamic Markdown skills wired from n8n canvas Skill slots",
    )
    mcp_servers: list[ExternalMCPServerConfig] | None = Field(
        default=None,
        description="Optional external MCP servers to connect dynamically on resume",
    )
    stall_timeout_seconds: float | None = Field(default=None, ge=0.2, le=3600.0)


class AgentJobCreateRequest(BaseModel):
    """Unified request body for POST /agent/jobs/create (stages a run or resume job)."""

    endpoint: Literal["run", "resume"] = "run"
    task: str | None = Field(default=None, max_length=32_000)
    thread_id: str | None = Field(default=None, max_length=128)
    max_turns: int | None = Field(default=None, ge=1, le=100)
    decision: Literal["approve", "reject", "edit"] | None = None
    arguments: dict[str, Any] | None = None
    reason: str | None = Field(default=None, max_length=1000)
    primary_model: ModelConfigOverride | None = None
    worker_model: ModelConfigOverride | None = None
    canvas_skills: list[CanvasSkillSpec] | None = None
    mcp_servers: list[ExternalMCPServerConfig] | None = None
    stall_timeout_seconds: float | None = Field(default=None, ge=0.2, le=3600.0)
    bridge_tool_timeout_seconds: float | None = Field(default=None, ge=1.0, le=300.0)


class AgentJobStartRequest(BaseModel):
    """Payload posted by the n8n Agent Heartbeat & MCP Bridge to start a staged job."""

    n8n_bridge_tools: list[BridgeToolSpec] = Field(default_factory=list)
    canvas_skills: list[CanvasSkillSpec] | None = None
    mcp_servers: list[ExternalMCPServerConfig] | None = None


class BridgeToolResultRequest(BaseModel):
    """Result returned by the n8n canvas bridge after executing a connected MCP / AI Tool."""

    call_id: str = Field(..., min_length=1, max_length=128)
    ok: bool = True
    output: Any = ""
    error: str | None = None


class AgentRunResponse(BaseModel):
    """Structured response returned by /agent/run and /agent/resume."""

    thread_id: str
    status: Literal["completed", "awaiting_approval", "error"]
    final_response: str
    plan: list[dict[str, Any]]
    active_skills: list[str]
    turn_count: int
    pending_approval: dict[str, Any] | None = None
    subagent_reports: list[dict[str, Any]] = Field(default_factory=list)
    active_models: dict[str, Any] | None = None
    error: str | None = None


def _get_configured_api_token(explicit_token: str | None) -> str | None:
    if explicit_token is not None:
        cleaned = explicit_token.strip()
        return cleaned or None
    env_tok = os.getenv("AGENT_API_TOKEN", "").strip()
    return env_tok or None


def create_app(
    *,
    api_token: str | None = None,
    primary_model: BaseChatModel | None = None,
    worker_model: BaseChatModel | None = None,
    checkpoint_db_path: Path | str | None = None,
    sandbox: SandboxManager | None = None,
    skills: SkillRegistry | None = None,
    mcp_manager: N8nMCPManager | None = None,
    connect_mcp: bool = True,
    extra_tools: list[BaseTool] | None = None,
    max_turns: int | None = None,
    allow_private_web_fetch: bool = False,
    model_override_resolver: ModelOverrideResolver | None = None,
) -> FastAPI:
    """Create the FastAPI application with lifespan-managed AgentRuntime."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.api_token = _get_configured_api_token(api_token)
        app.state.model_error = None
        app.state.runtime = None

        resolved_sandbox = sandbox or SandboxManager()
        resolved_skills = skills or SkillRegistry()
        resolved_mcp = mcp_manager or N8nMCPManager()
        job_manager = JobManager()
        app.state.sandbox = resolved_sandbox
        app.state.skills = resolved_skills
        app.state.mcp_manager = resolved_mcp
        app.state.job_manager = job_manager
        raw_db_path = (
            checkpoint_db_path if checkpoint_db_path is not None else DEFAULT_CHECKPOINT_DB_PATH
        )
        app.state.checkpoint_db_path = await asyncio.to_thread(lambda: Path(raw_db_path).resolve())

        # Resolve models gracefully: if RIVER_AI_API is missing, keep /health online and fail /agent/run cleanly
        resolved_primary = primary_model
        resolved_worker = worker_model
        if resolved_primary is None:
            try:
                resolved_primary = get_primary_model()
            except ModelConfigurationError as exc:
                app.state.model_error = str(exc)
                logger.warning("Primary model not ready at startup: %s", exc)

        if resolved_worker is None and resolved_primary is not None:
            try:
                resolved_worker = get_worker_model()
            except ModelConfigurationError as exc:
                logger.warning("Worker model config error; falling back to primary model: %s", exc)
                resolved_worker = resolved_primary

        if resolved_primary is not None:
            async with create_agent_runtime(
                primary_model=resolved_primary,
                worker_model=resolved_worker,
                checkpoint_db_path=app.state.checkpoint_db_path,
                sandbox=resolved_sandbox,
                skills=resolved_skills,
                mcp_manager=resolved_mcp,
                connect_mcp=connect_mcp,
                extra_tools=extra_tools,
                max_turns=max_turns,
                allow_private_web_fetch=allow_private_web_fetch,
                model_override_resolver=model_override_resolver,
            ) as runtime:
                app.state.runtime = runtime
                try:
                    yield
                finally:
                    await job_manager.shutdown()
                    resolved_sandbox.cleanup_all()
        else:
            # Still connect MCP so /health reflects real n8n MCP discovery status
            if connect_mcp:
                await resolved_mcp.connect()
            try:
                yield
            finally:
                await job_manager.shutdown()
                resolved_sandbox.cleanup_all()

    app = FastAPI(
        title="Autonomous LangGraph AI Agent API",
        version="1.0.0",
        description="Authenticated HTTP API for the LangGraph + Docker Sandbox + n8n MCP Autonomous Agent.",
        lifespan=lifespan,
    )

    def require_auth(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        expected_token: str | None = getattr(request.app.state, "api_token", None)
        if not expected_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server authentication is not configured (set AGENT_API_TOKEN).",
            )

        provided: str | None = None
        if authorization:
            parts = authorization.strip().split(" ", 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                provided = parts[1].strip()
        elif x_api_key:
            provided = x_api_key.strip()

        if not provided or not secrets.compare_digest(provided, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        primary_summary: dict[str, Any]
        worker_summary: dict[str, Any]
        try:
            primary_summary = load_model_spec("primary").safe_summary()
        except ModelConfigurationError as exc:
            primary_summary = {"error": str(exc), "api_key_configured": False}
        try:
            worker_summary = load_model_spec("worker").safe_summary()
        except ModelConfigurationError as exc:
            worker_summary = {"error": str(exc), "api_key_configured": False}

        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        sandbox_mgr: SandboxManager = request.app.state.sandbox
        skills_reg: SkillRegistry = request.app.state.skills
        mcp_mgr: N8nMCPManager = request.app.state.mcp_manager
        job_mgr: JobManager = request.app.state.job_manager
        db_path: Path = request.app.state.checkpoint_db_path

        overall = "ok" if runtime is not None else "degraded"
        return {
            "status": overall,
            "agent_ready": runtime is not None,
            "auth_configured": bool(getattr(request.app.state, "api_token", None)),
            "model_error": getattr(request.app.state, "model_error", None),
            "models": {
                "primary": primary_summary,
                "worker": worker_summary,
            },
            "checkpoint": {
                "path": str(db_path),
                "exists": await asyncio.to_thread(db_path.exists),
            },
            "sandbox": sandbox_mgr.health_status(),
            "skills": get_skills_health(skills_reg),
            "n8n_mcp": mcp_mgr.health_status(),
            "jobs": job_mgr.health_summary(),
        }

    def _require_job_or_master_auth(
        job_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_job_token: str | None = Header(default=None, alias="X-Job-Token"),
    ) -> AgentJob:
        job_mgr: JobManager = request.app.state.job_manager
        job = job_mgr.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        if x_job_token and job_mgr.verify_job_token(job_id, x_job_token):
            return job
        require_auth(request, authorization=authorization, x_api_key=x_api_key)
        return job

    def _build_job_runner(
        runtime: AgentRuntime,
        *,
        bridge_tools: list[BridgeToolSpec] | None = None,
        extra_skills: list[CanvasSkillSpec] | None = None,
        extra_mcp_servers: list[ExternalMCPServerConfig] | None = None,
    ) -> Any:
        async def _runner(job: AgentJob) -> dict[str, Any]:
            req = job.request_payload
            prim_raw = req.get("primary_model")
            work_raw = req.get("worker_model")
            prim_ov = (
                ModelConfigOverride.model_validate(prim_raw).to_model_override()
                if isinstance(prim_raw, dict)
                else None
            )
            work_ov = (
                ModelConfigOverride.model_validate(work_raw).to_model_override()
                if isinstance(work_raw, dict)
                else None
            )

            # 1. Build n8n canvas bridge tools (native MCP Client Tool / n8n AI Tool nodes)
            dyn_tools: dict[str, BaseTool] = {}
            dyn_mcp_names: set[str] = set()
            if bridge_tools:
                b_tools, b_mcp = job.build_bridge_tools(bridge_tools)
                dyn_tools.update(b_tools)
                dyn_mcp_names.update(b_mcp)

            # 2. Connect any direct external MCP server configs
            raw_servers: list[ExternalMCPServerConfig] = []
            for item in req.get("mcp_servers") or []:
                raw_servers.append(
                    item
                    if isinstance(item, ExternalMCPServerConfig)
                    else ExternalMCPServerConfig.model_validate(item)
                )
            if extra_mcp_servers:
                raw_servers.extend(extra_mcp_servers)
            if raw_servers:
                job.record_progress(
                    {
                        "phase": "connecting_mcp_servers",
                        "detail": f"Connecting {len(raw_servers)} external MCP server(s)",
                    }
                )
                ext_tools, ext_mcp, ext_warnings = await load_external_mcp_tools(raw_servers)
                dyn_tools.update(ext_tools)
                dyn_mcp_names.update(ext_mcp)
                job.warnings.extend(ext_warnings)

            job.canvas_mcp_tools = sorted(dyn_tools.keys())

            # 3. Merge canvas skills from create + start payloads
            skill_dicts: list[dict[str, Any]] = []
            for sk in req.get("canvas_skills") or []:
                skill_dicts.append(sk.model_dump() if isinstance(sk, BaseModel) else dict(sk))
            if extra_skills:
                for sk in extra_skills:
                    skill_dicts.append(sk.model_dump())
            job.canvas_skills = [
                str(s.get("name") or f"canvas_skill_{i + 1}") for i, s in enumerate(skill_dicts)
            ]

            if job.endpoint == "resume":
                current = await runtime.inspect_thread(job.thread_id)
                if current["status"] != "awaiting_approval":
                    raise ValueError(f"Thread '{job.thread_id}' is not waiting for human approval.")
                resume_body: dict[str, Any] = {"decision": req.get("decision") or "approve"}
                if req.get("arguments") is not None:
                    resume_body["arguments"] = req["arguments"]
                if req.get("reason") is not None:
                    resume_body["reason"] = req["reason"]
                return await runtime.resume_approval(
                    thread_id=job.thread_id,
                    decision=resume_body,
                    primary_model_override=prim_ov,
                    worker_model_override=work_ov,
                    extra_tools=dyn_tools or None,
                    extra_mcp_tool_names=dyn_mcp_names or None,
                    canvas_skills=skill_dicts or None,
                    progress_callback=job.record_progress,
                )

            return await runtime.run_task(
                str(req.get("task") or ""),
                thread_id=job.thread_id,
                max_turns=req.get("max_turns"),
                primary_model_override=prim_ov,
                worker_model_override=work_ov,
                extra_tools=dyn_tools or None,
                extra_mcp_tool_names=dyn_mcp_names or None,
                canvas_skills=skill_dicts or None,
                progress_callback=job.record_progress,
            )

        return _runner

    @app.get("/models/presets")
    async def model_presets() -> dict[str, Any]:
        """List supported OpenAI-compatible provider presets and current default models (no secrets)."""
        return get_provider_presets()

    @app.post(
        "/agent/jobs/create",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
    )
    async def create_agent_job(
        payload: AgentJobCreateRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Stage an async job and return an ephemeral job_token for the n8n Heartbeat & MCP Bridge."""
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            detail = getattr(request.app.state, "model_error", None) or (
                "Agent runtime is not initialized. Check model credentials (RIVER_AI_API)."
            )
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

        if payload.endpoint == "run":
            if not payload.task or not payload.task.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="task must not be blank.",
                )
        elif payload.endpoint == "resume":
            if not payload.thread_id or not payload.thread_id.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="thread_id is required when endpoint is 'resume'.",
                )
            if payload.decision not in ("approve", "reject", "edit"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="decision must be 'approve', 'reject', or 'edit'.",
                )

        thread_id = (payload.thread_id or "").strip() or f"thread-{uuid.uuid4().hex[:12]}"
        job_mgr: JobManager = request.app.state.job_manager
        job = job_mgr.create_job(
            thread_id=thread_id,
            endpoint=payload.endpoint,
            request_payload=payload.model_dump(exclude_none=True),
            stall_timeout_seconds=payload.stall_timeout_seconds,
            bridge_tool_timeout_seconds=payload.bridge_tool_timeout_seconds,
        )
        return {
            **job.to_status_payload(),
            "job_token": job.job_token,
        }

    @app.post(
        "/agent/jobs/run",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
    )
    async def start_async_run_job(
        payload: AgentRunRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Immediately start an async run job with progress heartbeat and stall watchdog."""
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            detail = getattr(request.app.state, "model_error", None) or (
                "Agent runtime is not initialized. Check model credentials (RIVER_AI_API)."
            )
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

        thread_id = payload.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
        job_mgr: JobManager = request.app.state.job_manager
        job = job_mgr.create_job(
            thread_id=thread_id,
            endpoint="run",
            request_payload=payload.model_dump(exclude_none=True),
            stall_timeout_seconds=payload.stall_timeout_seconds,
        )
        job_mgr.launch_job(job, _build_job_runner(runtime))
        return {
            **job.to_status_payload(),
            "job_token": job.job_token,
        }

    @app.post(
        "/agent/jobs/resume",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
    )
    async def start_async_resume_job(
        payload: AgentResumeRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Immediately start an async HITL resume job with progress heartbeat and stall watchdog."""
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent runtime is not initialized.",
            )
        job_mgr: JobManager = request.app.state.job_manager
        job = job_mgr.create_job(
            thread_id=payload.thread_id,
            endpoint="resume",
            request_payload=payload.model_dump(exclude_none=True),
            stall_timeout_seconds=payload.stall_timeout_seconds,
        )
        job_mgr.launch_job(job, _build_job_runner(runtime))
        return {
            **job.to_status_payload(),
            "job_token": job.job_token,
        }

    @app.post("/agent/jobs/{job_id}/start")
    async def launch_staged_job(
        job_id: str,
        payload: AgentJobStartRequest,
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_job_token: str | None = Header(default=None, alias="X-Job-Token"),
    ) -> dict[str, Any]:
        """Register connected n8n canvas MCP tools & skills and launch the staged job."""
        job = _require_job_or_master_auth(
            job_id,
            request,
            authorization=authorization,
            x_api_key=x_api_key,
            x_job_token=x_job_token,
        )
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent runtime is not initialized.",
            )
        job_mgr: JobManager = request.app.state.job_manager
        if job.status == "staged":
            job_mgr.launch_job(
                job,
                _build_job_runner(
                    runtime,
                    bridge_tools=payload.n8n_bridge_tools,
                    extra_skills=payload.canvas_skills,
                    extra_mcp_servers=payload.mcp_servers,
                ),
            )
        return job.to_status_payload()

    @app.get("/agent/jobs/{job_id}")
    async def poll_agent_job(
        job_id: str,
        request: Request,
        wait_ms: int = Query(default=0, ge=0, le=15_000),
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_job_token: str | None = Header(default=None, alias="X-Job-Token"),
    ) -> dict[str, Any]:
        """Heartbeat ping & long-poll endpoint returning live progress, pending canvas tool calls, or final result."""
        _require_job_or_master_auth(
            job_id,
            request,
            authorization=authorization,
            x_api_key=x_api_key,
            x_job_token=x_job_token,
        )
        job_mgr: JobManager = request.app.state.job_manager
        status_payload = await job_mgr.poll_job(job_id, wait_ms=wait_ms)
        if status_payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        return status_payload

    @app.post("/agent/jobs/{job_id}/tool-result")
    async def submit_bridge_tool_result(
        job_id: str,
        payload: BridgeToolResultRequest,
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_job_token: str | None = Header(default=None, alias="X-Job-Token"),
    ) -> dict[str, Any]:
        """Receive executed output from a connected n8n MCP Client Tool / AI Tool node."""
        job = _require_job_or_master_auth(
            job_id,
            request,
            authorization=authorization,
            x_api_key=x_api_key,
            x_job_token=x_job_token,
        )
        text_output = (
            str(payload.error)
            if not payload.ok and payload.error
            else format_bridge_tool_output(payload.output)
        )
        resolved = job.resolve_tool_call(payload.call_id, ok=payload.ok, output=text_output)
        if not resolved:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Tool call '{payload.call_id}' is not pending for job '{job_id}'.",
            )
        return {"ok": True, "call_id": payload.call_id, "job_id": job_id}

    @app.post("/agent/jobs/{job_id}/cancel")
    async def cancel_agent_job(
        job_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_job_token: str | None = Header(default=None, alias="X-Job-Token"),
    ) -> dict[str, Any]:
        """Cancel a running async agent job cleanly."""
        _require_job_or_master_auth(
            job_id,
            request,
            authorization=authorization,
            x_api_key=x_api_key,
            x_job_token=x_job_token,
        )
        job_mgr: JobManager = request.app.state.job_manager
        job_mgr.cancel_job(job_id)
        job = job_mgr.get_job(job_id)
        return job.to_status_payload() if job else {"job_id": job_id, "status": "cancelled"}

    @app.post("/agent/run", response_model=AgentRunResponse, dependencies=[Depends(require_auth)])
    async def run_agent(payload: AgentRunRequest, request: Request) -> AgentRunResponse:
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            detail = getattr(request.app.state, "model_error", None) or (
                "Agent runtime is not initialized. Check model credentials (RIVER_AI_API)."
            )
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

        thread_id = payload.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
        prim_ov = payload.primary_model.to_model_override() if payload.primary_model else None
        work_ov = payload.worker_model.to_model_override() if payload.worker_model else None
        skill_dicts = (
            [s.model_dump() for s in payload.canvas_skills] if payload.canvas_skills else None
        )
        ext_tools, ext_mcp, _warnings = await load_external_mcp_tools(payload.mcp_servers)
        try:
            result = await runtime.run_task(
                payload.task,
                thread_id=thread_id,
                max_turns=payload.max_turns,
                primary_model_override=prim_ov,
                worker_model_override=work_ov,
                extra_tools=ext_tools or None,
                extra_mcp_tool_names=ext_mcp or None,
                canvas_skills=skill_dicts,
            )
            return AgentRunResponse(
                **{k: result[k] for k in AgentRunResponse.model_fields if k in result}
            )
        except (ValueError, ModelConfigurationError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except (
            ModelAuthenticationError,
            ModelRateLimitError,
            ModelConnectionError,
            ModelProviderError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error during /agent/run for thread=%s", thread_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution failed: {type(exc).__name__}: {exc}",
            ) from exc

    @app.post(
        "/agent/resume", response_model=AgentRunResponse, dependencies=[Depends(require_auth)]
    )
    async def resume_agent(payload: AgentResumeRequest, request: Request) -> AgentRunResponse:
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent runtime is not initialized.",
            )

        current = await runtime.inspect_thread(payload.thread_id)
        if current["status"] != "awaiting_approval":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Thread '{payload.thread_id}' is not waiting for human approval.",
            )

        resume_body: dict[str, Any] = {"decision": payload.decision}
        if payload.arguments is not None:
            resume_body["arguments"] = payload.arguments
        if payload.reason is not None:
            resume_body["reason"] = payload.reason

        prim_ov = payload.primary_model.to_model_override() if payload.primary_model else None
        work_ov = payload.worker_model.to_model_override() if payload.worker_model else None
        skill_dicts = (
            [s.model_dump() for s in payload.canvas_skills] if payload.canvas_skills else None
        )
        ext_tools, ext_mcp, _warnings = await load_external_mcp_tools(payload.mcp_servers)
        try:
            result = await runtime.resume_approval(
                thread_id=payload.thread_id,
                decision=resume_body,
                primary_model_override=prim_ov,
                worker_model_override=work_ov,
                extra_tools=ext_tools or None,
                extra_mcp_tool_names=ext_mcp or None,
                canvas_skills=skill_dicts,
            )
            return AgentRunResponse(
                **{k: result[k] for k in AgentRunResponse.model_fields if k in result}
            )
        except (ValueError, ModelConfigurationError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unhandled error during /agent/resume for thread=%s", payload.thread_id
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Resume failed: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get(
        "/agent/threads/{thread_id}",
        response_model=AgentRunResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_thread(thread_id: str, request: Request) -> AgentRunResponse:
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent runtime is not initialized.",
            )
        result = await runtime.inspect_thread(thread_id)
        return AgentRunResponse(
            **{k: result[k] for k in AgentRunResponse.model_fields if k in result}
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("AGENT_API_PORT", "8100"))
    uvicorn.run("src.api:app", host=host, port=port, reload=False)
