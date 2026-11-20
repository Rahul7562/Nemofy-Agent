"""Pure-Python Authenticated FastAPI Service for Nemofy Agent-Core.

Exposes:
- GET  /health                   -> Service & dependency status (no secrets leaked)
- GET  /models/presets           -> Provider presets & active Main/Sub-Agent defaults
- POST /models/probe             -> Connectivity check for Main or Sub-Agent model
- GET  /config/llm               -> Current Main Agent & Sub-Agent LLM configuration
- PUT  /config/llm               -> Persist Main Agent & Sub-Agent LLM configuration
- POST /agent/run                -> Execute or continue an autonomous task
- POST /agent/resume             -> Resume a HITL-interrupted session
- GET  /agent/threads/{thread_id}-> Inspect session state & plan
- POST /agent/jobs/run           -> Launch an async job with progress heartbeat
- GET  /agent/jobs/{job_id}      -> Long-poll job progress / final result
- POST /agent/jobs/{job_id}/cancel -> Cancel an active async job
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
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from agent_core.graph import (
    DEFAULT_CHECKPOINT_DB_PATH,
    AgentRuntime,
    ModelOverrideResolver,
    create_agent_runtime,
)
from agent_core.jobs import AgentJob, DynamicSkillSpec, JobManager
from agent_core.models import (
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
    load_persisted_llm_config,
    probe_model_connectivity,
    resolve_model_spec_with_override,
    save_persisted_llm_config,
)
from agent_core.sandbox import SandboxManager
from agent_core.skills import SkillRegistry, get_skills_health
from MCPs.manager import MCPRegistryManager, MCPServerConfig, load_mcp_tools_from_configs

load_dotenv()

logger = logging.getLogger(__name__)


class ModelConfigOverride(BaseModel):
    """Per-request or persisted LLM selection for Main Agent (primary) or Sub-Agent (worker)."""

    provider: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=256)
    base_url: str | None = Field(default=None, max_length=512)
    api_key: str | None = Field(default=None, max_length=512)
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


class LLMConfigUpdateRequest(BaseModel):
    primary: ModelConfigOverride | None = None
    worker: ModelConfigOverride | None = None
    max_turns: int | None = Field(default=None, ge=1, le=100)


class ModelProbeRequest(BaseModel):
    role: Literal["primary", "worker"] = "primary"
    override: ModelConfigOverride | None = None
    run_inference: bool = False


class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=32_000)
    thread_id: str | None = Field(default=None, max_length=128)
    max_turns: int | None = Field(default=None, ge=1, le=100)
    primary_model: ModelConfigOverride | None = None
    worker_model: ModelConfigOverride | None = None
    dynamic_skills: list[DynamicSkillSpec] | None = None
    mcp_servers: list[MCPServerConfig] | None = None
    stall_timeout_seconds: float | None = Field(default=None, ge=0.2, le=3600.0)

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
    thread_id: str = Field(..., min_length=1, max_length=128)
    decision: Literal["approve", "reject", "edit"]
    arguments: dict[str, Any] | None = None
    reason: str | None = Field(default=None, max_length=1000)
    primary_model: ModelConfigOverride | None = None
    worker_model: ModelConfigOverride | None = None
    dynamic_skills: list[DynamicSkillSpec] | None = None
    mcp_servers: list[MCPServerConfig] | None = None
    stall_timeout_seconds: float | None = Field(default=None, ge=0.2, le=3600.0)


class AgentRunResponse(BaseModel):
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
    mcp_manager: MCPRegistryManager | None = None,
    connect_mcp: bool = True,
    extra_tools: list[BaseTool] | None = None,
    max_turns: int | None = None,
    allow_private_web_fetch: bool = False,
    model_override_resolver: ModelOverrideResolver | None = None,
) -> FastAPI:
    """Create the Pure-Python Agent-Core FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.api_token = _get_configured_api_token(api_token)
        app.state.model_error = None
        app.state.runtime = None

        resolved_sandbox = sandbox or SandboxManager()
        resolved_skills = skills or SkillRegistry()
        resolved_mcp = mcp_manager or MCPRegistryManager()
        job_manager = JobManager()
        app.state.sandbox = resolved_sandbox
        app.state.skills = resolved_skills
        app.state.mcp_manager = resolved_mcp
        app.state.job_manager = job_manager
        raw_db_path = (
            checkpoint_db_path if checkpoint_db_path is not None else DEFAULT_CHECKPOINT_DB_PATH
        )
        app.state.checkpoint_db_path = await asyncio.to_thread(lambda: Path(raw_db_path).resolve())

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
            if connect_mcp:
                await resolved_mcp.connect()
            try:
                yield
            finally:
                await job_manager.shutdown()
                resolved_sandbox.cleanup_all()

    app = FastAPI(
        title="Nemofy Agent-Core API",
        version="2.0.0",
        description="Pure-Python LangGraph Autonomous AI Agent API (Sandbox + Skills + MCPs).",
        lifespan=lifespan,
    )

    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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

    def _effective_override(
        explicit: ModelConfigOverride | None,
        role: Literal["primary", "worker"],
    ) -> ModelOverride | None:
        if explicit is not None:
            ov = explicit.to_model_override()
            if ov is not None:
                return ov
        persisted = load_persisted_llm_config().get(role)
        if isinstance(persisted, dict) and persisted:
            return ModelConfigOverride.model_validate(persisted).to_model_override()
        return None

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        try:
            primary_summary = load_model_spec("primary", include_persisted=True).safe_summary()
        except ModelConfigurationError as exc:
            primary_summary = {"error": str(exc), "api_key_configured": False}
        try:
            worker_summary = load_model_spec("worker", include_persisted=True).safe_summary()
        except ModelConfigurationError as exc:
            worker_summary = {"error": str(exc), "api_key_configured": False}

        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        sandbox_mgr: SandboxManager = request.app.state.sandbox
        skills_reg: SkillRegistry = request.app.state.skills
        mcp_mgr: MCPRegistryManager = request.app.state.mcp_manager
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
            "mcps": mcp_mgr.health_status(),
            "jobs": job_mgr.health_summary(),
        }

    @app.get("/models/presets")
    async def model_presets() -> dict[str, Any]:
        return get_provider_presets()

    @app.get("/config/llm", dependencies=[Depends(require_auth)])
    async def get_llm_config() -> dict[str, Any]:
        persisted = load_persisted_llm_config()
        return {
            "primary": load_model_spec("primary", include_persisted=True).safe_summary(),
            "worker": load_model_spec("worker", include_persisted=True).safe_summary(),
            "max_turns": int(persisted.get("max_turns") or os.getenv("MAX_TURNS", "30")),
            "providers": get_provider_presets()["providers"],
        }

    @app.put("/config/llm", dependencies=[Depends(require_auth)])
    async def update_llm_config(payload: LLMConfigUpdateRequest) -> dict[str, Any]:
        current = load_persisted_llm_config()
        if payload.primary is not None:
            prim_dict = payload.primary.model_dump(exclude_none=True, exclude={"api_key"})
            resolve_model_spec_with_override("primary", override=prim_dict)
            current["primary"] = prim_dict
        if payload.worker is not None:
            work_dict = payload.worker.model_dump(exclude_none=True, exclude={"api_key"})
            resolve_model_spec_with_override("worker", override=work_dict)
            current["worker"] = work_dict
        if payload.max_turns is not None:
            current["max_turns"] = payload.max_turns
        save_persisted_llm_config(current)
        return {
            "ok": True,
            "primary": load_model_spec("primary", include_persisted=True).safe_summary(),
            "worker": load_model_spec("worker", include_persisted=True).safe_summary(),
            "max_turns": int(current.get("max_turns") or os.getenv("MAX_TURNS", "30")),
        }

    @app.post("/models/probe", dependencies=[Depends(require_auth)])
    async def probe_model(payload: ModelProbeRequest) -> dict[str, Any]:
        ov = payload.override.to_model_override() if payload.override else None
        spec = (
            resolve_model_spec_with_override(payload.role, override=ov)
            if ov is not None
            else load_model_spec(payload.role, include_persisted=True)
        )
        return await probe_model_connectivity(
            payload.role,
            spec=spec,
            run_inference=payload.run_inference,
        )

    def _build_job_runner(runtime: AgentRuntime) -> Any:
        async def _runner(job: AgentJob) -> dict[str, Any]:
            req = job.request_payload
            prim_raw = req.get("primary_model")
            work_raw = req.get("worker_model")
            prim_ov = _effective_override(
                ModelConfigOverride.model_validate(prim_raw)
                if isinstance(prim_raw, dict)
                else None,
                "primary",
            )
            work_ov = _effective_override(
                ModelConfigOverride.model_validate(work_raw)
                if isinstance(work_raw, dict)
                else None,
                "worker",
            )

            dyn_tools: dict[str, BaseTool] = {}
            dyn_mcp_names: set[str] = set()
            raw_servers: list[MCPServerConfig] = [
                MCPServerConfig.model_validate(item)
                for item in (req.get("mcp_servers") or [])
                if isinstance(item, dict)
            ]
            if raw_servers:
                job.record_progress(
                    {
                        "phase": "connecting_mcp_servers",
                        "detail": f"Connecting {len(raw_servers)} MCP server(s)",
                    }
                )
                ext_tools, ext_mcp, ext_warnings = await load_mcp_tools_from_configs(raw_servers)
                dyn_tools.update(ext_tools)
                dyn_mcp_names.update(ext_mcp)
                job.warnings.extend(ext_warnings)
            job.mcp_tools = sorted(dyn_tools.keys())

            skill_dicts: list[dict[str, Any]] = [
                sk.model_dump() if isinstance(sk, BaseModel) else dict(sk)
                for sk in (req.get("dynamic_skills") or [])
            ]
            job.dynamic_skills = [
                str(s.get("name") or f"skill_{i + 1}") for i, s in enumerate(skill_dicts)
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
                    dynamic_skills=skill_dicts or None,
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
                dynamic_skills=skill_dicts or None,
                progress_callback=job.record_progress,
            )

        return _runner

    @app.post(
        "/agent/jobs/run",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
    )
    async def start_async_run_job(payload: AgentRunRequest, request: Request) -> dict[str, Any]:
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            detail = getattr(request.app.state, "model_error", None) or (
                "Agent runtime is not initialized. Check model credentials on the server."
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
        return {**job.to_status_payload(), "job_token": job.job_token}

    @app.post(
        "/agent/jobs/resume",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
    )
    async def start_async_resume_job(
        payload: AgentResumeRequest, request: Request
    ) -> dict[str, Any]:
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
        return {**job.to_status_payload(), "job_token": job.job_token}

    @app.get("/agent/jobs/{job_id}", dependencies=[Depends(require_auth)])
    async def poll_agent_job(
        job_id: str,
        request: Request,
        wait_ms: int = Query(default=0, ge=0, le=15_000),
    ) -> dict[str, Any]:
        job_mgr: JobManager = request.app.state.job_manager
        status_payload = await job_mgr.poll_job(job_id, wait_ms=wait_ms)
        if status_payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        return status_payload

    @app.post("/agent/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
    async def cancel_agent_job(job_id: str, request: Request) -> dict[str, Any]:
        job_mgr: JobManager = request.app.state.job_manager
        job_mgr.cancel_job(job_id)
        job = job_mgr.get_job(job_id)
        return job.to_status_payload() if job else {"job_id": job_id, "status": "cancelled"}

    @app.post("/agent/run", response_model=AgentRunResponse, dependencies=[Depends(require_auth)])
    async def run_agent(payload: AgentRunRequest, request: Request) -> AgentRunResponse:
        runtime: AgentRuntime | None = getattr(request.app.state, "runtime", None)
        if runtime is None:
            detail = getattr(request.app.state, "model_error", None) or (
                "Agent runtime is not initialized. Check model credentials on the server."
            )
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

        thread_id = payload.thread_id or f"thread-{uuid.uuid4().hex[:12]}"
        prim_ov = _effective_override(payload.primary_model, "primary")
        work_ov = _effective_override(payload.worker_model, "worker")
        skill_dicts = (
            [s.model_dump() for s in payload.dynamic_skills] if payload.dynamic_skills else None
        )
        ext_tools, ext_mcp, _warnings = await load_mcp_tools_from_configs(payload.mcp_servers)
        try:
            result = await runtime.run_task(
                payload.task,
                thread_id=thread_id,
                max_turns=payload.max_turns,
                primary_model_override=prim_ov,
                worker_model_override=work_ov,
                extra_tools=ext_tools or None,
                extra_mcp_tool_names=ext_mcp or None,
                dynamic_skills=skill_dicts,
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

        prim_ov = _effective_override(payload.primary_model, "primary")
        work_ov = _effective_override(payload.worker_model, "worker")
        skill_dicts = (
            [s.model_dump() for s in payload.dynamic_skills] if payload.dynamic_skills else None
        )
        ext_tools, ext_mcp, _warnings = await load_mcp_tools_from_configs(payload.mcp_servers)
        try:
            result = await runtime.resume_approval(
                thread_id=payload.thread_id,
                decision=resume_body,
                primary_model_override=prim_ov,
                worker_model_override=work_ov,
                extra_tools=ext_tools or None,
                extra_mcp_tool_names=ext_mcp or None,
                dynamic_skills=skill_dicts,
            )
            return AgentRunResponse(
                **{k: result[k] for k in AgentRunResponse.model_fields if k in result}
            )
        except (ValueError, ModelConfigurationError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error during /agent/resume for thread=%s", payload.thread_id)
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
    uvicorn.run("agent_core.api:app", host=host, port=port, reload=False)
