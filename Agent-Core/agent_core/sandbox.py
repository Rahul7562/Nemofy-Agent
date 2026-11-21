"""Docker Sandbox Manager and Hardened Workspace Isolation."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import resource
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SandboxBackendMode = Literal["auto", "docker", "local_isolated"]

SENSITIVE_ENV_PREFIXES = (
    "RIVER_AI",
    "OPENAI_",
    "OPENROUTER_",
    "NVIDIA_",
    "ANTHROPIC_",
    "GEMINI_",
    "DEEPSEEK_",
    "GROQ_",
    "TOGETHER_",
    "FIREWORKS_",
    "MISTRAL_",
    "XAI_",
    "PERPLEXITY_",
    "VAULT_",
    "PRIMARY_API",
    "WORKER_API",
    "AGENT_API_TOKEN",
    "AWS_",
    "GCP_",
    "GOOGLE_API",
)

_LOCAL_ESCAPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/var/run/docker\.sock", re.IGNORECASE),
    re.compile(r"/etc/(?:shadow|passwd|sudoers)", re.IGNORECASE),
    re.compile(r"/proc/\d+/(?:environ|mem|root)", re.IGNORECASE),
    re.compile(r"(?:^|[\s;|&`\"'/])\.env(?=$|[\s;|&`\"'])", re.IGNORECASE),
)

_WORKSPACE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])/workspace(?=/|[\s\"'`;|&<>):]|$)"
)


def rewrite_workspace_paths_in_command(command: str, workspace_root: Path) -> str:
    """Rewrite standalone /workspace path tokens to the host workspace path without mangling URLs or already-resolved paths."""
    ws_str = str(workspace_root.resolve())
    return _WORKSPACE_TOKEN_RE.sub(lambda _m: ws_str, command)


class SandboxSecurityError(PermissionError):
    """Raised when a file or command attempts to escape /workspace or violate sandbox policy."""


class SandboxExecutionError(RuntimeError):
    """Raised when the sandbox container or execution backend fails unexpectedly."""


class SandboxTimeoutError(TimeoutError):
    """Raised when a sandbox command exceeds its configured timeout."""


@dataclass(frozen=True)
class SandboxConfig:
    """Runtime configuration for the execution sandbox."""

    workspace_dir: Path
    backend: SandboxBackendMode = "auto"
    image: str = "nemofy-agent-sandbox:latest"
    memory_limit: str = "512m"
    cpu_quota: int = 100_000
    cpu_period: int = 100_000
    pids_limit: int = 128
    timeout_seconds: int = 60
    network_mode: str = "bridge"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> SandboxConfig:
        src = dict(os.environ) if env is None else env
        raw_backend = (src.get("SANDBOX_BACKEND") or "auto").strip().lower()
        backend: SandboxBackendMode
        if raw_backend == "docker":
            backend = "docker"
        elif raw_backend == "local_isolated":
            backend = "local_isolated"
        else:
            backend = "auto"
        return cls(
            workspace_dir=Path(src.get("WORKSPACE_DIR") or "./workspace").resolve(),
            backend=backend,
            image=(src.get("SANDBOX_IMAGE") or "nemofy-agent-sandbox:latest").strip(),
            memory_limit=(src.get("SANDBOX_MEMORY_LIMIT") or "512m").strip(),
            cpu_quota=max(10_000, int(src.get("SANDBOX_CPU_QUOTA") or "100000")),
            cpu_period=max(10_000, int(src.get("SANDBOX_CPU_PERIOD") or "100000")),
            pids_limit=max(16, int(src.get("SANDBOX_PIDS_LIMIT") or "128")),
            timeout_seconds=max(1, int(src.get("SANDBOX_TIMEOUT_SECONDS") or "60")),
            network_mode=(src.get("SANDBOX_NETWORK_MODE") or "bridge").strip(),
        )


@dataclass(frozen=True)
class ExecResult:
    """Structured result from executing a shell command inside the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    backend: str = "local_isolated"
    session_id: str = "default"
    recovered: bool = False

    def format_output(self) -> str:
        parts: list[str] = [f"exit_code: {self.exit_code}"]
        if self.timed_out:
            parts.append("status: TIMED_OUT")
        if self.recovered:
            parts.append("note: sandbox container recovered automatically")
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


def resolve_workspace_path(
    requested_path: str,
    workspace_root: Path,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve and strictly confine a requested path inside ``workspace_root``."""
    ws_real = workspace_root.resolve()
    ws_real.mkdir(parents=True, exist_ok=True)
    ws_prefix = str(ws_real)

    if "\x00" in requested_path:
        raise SandboxSecurityError("Null byte detected in path.")

    raw = requested_path.strip()
    if not raw or raw in (".", "./", "/workspace", "/workspace/", ws_prefix, ws_prefix + "/"):
        if allow_root:
            return ws_real
        raise SandboxSecurityError("A specific file path inside /workspace is required.")

    if raw.startswith("/workspace/"):
        rel_str = raw[len("/workspace/") :]
    elif raw.startswith(ws_prefix + "/"):
        rel_str = raw[len(ws_prefix) + 1 :]
    elif raw.startswith("/"):
        raise SandboxSecurityError(f"Access denied: absolute path {raw!r} is outside /workspace.")
    else:
        rel_str = raw

    parts = Path(rel_str).parts
    if any(part == ".." for part in parts):
        raise SandboxSecurityError(
            f"Access denied: path traversal ('..') rejected in {requested_path!r}."
        )

    candidate = ws_real / rel_str

    current = ws_real
    for part in Path(rel_str).parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            target_real = current.resolve()
            try:
                target_real.relative_to(ws_real)
            except ValueError as exc:
                raise SandboxSecurityError(
                    f"Access denied: symbolic link {requested_path!r} escapes /workspace."
                ) from exc

    resolved = candidate.resolve()
    try:
        resolved.relative_to(ws_real)
    except ValueError as exc:
        raise SandboxSecurityError(
            f"Access denied: resolved path {requested_path!r} escapes /workspace."
        ) from exc

    return resolved


def to_virtual_workspace_path(real_path: Path, workspace_root: Path) -> str:
    """Convert a host path inside ``workspace_root`` into ``/workspace/...`` display form."""
    ws_real = workspace_root.resolve()
    rel = real_path.resolve().relative_to(ws_real)
    rel_posix = rel.as_posix()
    return "/workspace" if rel_posix == "." else f"/workspace/{rel_posix}"


def _parse_memory_bytes(spec: str) -> int:
    cleaned = spec.strip().lower()
    match = re.match(r"^(\d+)\s*([kmg]?)b?$", cleaned)
    if not match:
        return 512 * 1024 * 1024
    value = int(match.group(1))
    unit = match.group(2)
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[unit]
    return value * mult


def _sanitized_sandbox_env(workspace_root: Path) -> dict[str, str]:
    """Return a minimal subprocess environment stripped of host secrets while keeping active Python venv on PATH."""
    safe: dict[str, str] = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(upper.startswith(prefix) for prefix in SENSITIVE_ENV_PREFIXES):
            continue
        if any(
            tok in upper
            for tok in ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "MASTER_KEY", "PRIVATE_KEY")
        ):
            continue
        safe[k] = v

    # Ensure the active Python interpreter's bin dir (e.g. .venv/bin) is first on PATH.
    # Avoid .resolve() on sys.executable because venv binaries symlink to /usr/bin/python3.
    py_bin = os.path.abspath(os.path.dirname(sys.executable))
    existing_path = safe.get("PATH") or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    path_parts = [p for p in existing_path.split(os.pathsep) if p]
    if py_bin not in path_parts:
        safe["PATH"] = os.pathsep.join([py_bin, *path_parts])
    else:
        safe["PATH"] = existing_path
    if sys.prefix != sys.base_prefix:
        safe["VIRTUAL_ENV"] = os.path.abspath(sys.prefix)

    safe["HOME"] = str(workspace_root)
    safe["WORKSPACE"] = str(workspace_root)
    safe["PWD"] = str(workspace_root)
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe


def _check_local_command_boundaries(command: str) -> None:
    """Block obvious host escape reads when running in local_isolated mode."""
    for pattern in _LOCAL_ESCAPE_PATTERNS:
        if pattern.search(command):
            raise SandboxSecurityError(
                f"Command rejected by sandbox policy (matched restricted host path: "
                f"{pattern.pattern})."
            )


class SandboxManager:
    """Manages persistent per-session sandbox execution (Docker primary, local_isolated fallback)."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        docker_client: Any | None = None,
        isolate_session_dirs: bool = False,
    ) -> None:
        self.config = config or SandboxConfig.from_env()
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.config.workspace_dir / ".logs").mkdir(parents=True, exist_ok=True)
        self.isolate_session_dirs = isolate_session_dirs

        self._docker_client: Any | None = docker_client
        self._docker_available: bool = False
        self._docker_error: str | None = None
        self._containers: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        self._init_backend()

    def _init_backend(self) -> None:
        if self.config.backend == "local_isolated":
            self._docker_available = False
            self._docker_error = "Disabled by SANDBOX_BACKEND=local_isolated"
            return

        if self._docker_client is not None:
            try:
                self._docker_client.ping()
                self._docker_available = True
                return
            except Exception as exc:  # noqa: BLE001
                self._docker_error = f"Injected Docker client ping failed: {exc}"
                if self.config.backend == "docker":
                    raise SandboxExecutionError(self._docker_error) from exc
                self._docker_available = False
                return

        try:
            import docker

            client = docker.from_env()
            client.ping()
            self._docker_client = client
            self._docker_available = True
        except Exception as exc:  # noqa: BLE001
            self._docker_available = False
            self._docker_error = f"Docker daemon unavailable: {exc}"
            if self.config.backend == "docker":
                raise SandboxExecutionError(self._docker_error) from exc

    @property
    def active_backend(self) -> Literal["docker", "local_isolated"]:
        return "docker" if self._docker_available else "local_isolated"

    def get_session_workspace(self, session_id: str = "default") -> Path:
        if not self.isolate_session_dirs or session_id == "default":
            ws = self.config.workspace_dir.resolve()
        else:
            safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:48] or "session"
            ws = (self.config.workspace_dir / "sessions" / safe_id).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        (ws / ".logs").mkdir(parents=True, exist_ok=True)
        return ws

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def build_docker_run_kwargs(self, session_id: str) -> dict[str, Any]:
        ws = self.get_session_workspace(session_id)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", session_id)[:40] or "default"
        return {
            "image": self.config.image,
            "name": f"nemofy-agent-sbx-{safe_id}",
            "command": ["sleep", "infinity"],
            "detach": True,
            "working_dir": "/workspace",
            "volumes": {str(ws): {"bind": "/workspace", "mode": "rw"}},
            "mem_limit": self.config.memory_limit,
            "memswap_limit": self.config.memory_limit,
            "cpu_period": self.config.cpu_period,
            "cpu_quota": self.config.cpu_quota,
            "pids_limit": self.config.pids_limit,
            "network_mode": self.config.network_mode,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "labels": {
                "managed_by": "nemofy-agent-core",
                "session_id": session_id,
            },
        }

    def _ensure_docker_container(self, session_id: str) -> tuple[Any, bool]:
        assert self._docker_client is not None
        recovered = False
        existing = self._containers.get(session_id)
        if existing is not None:
            try:
                existing.reload()
                if getattr(existing, "status", None) == "running":
                    return existing, False
                try:
                    existing.remove(force=True)
                except Exception:  # noqa: BLE001
                    pass
                recovered = True
            except Exception:  # noqa: BLE001
                recovered = True
            self._containers.pop(session_id, None)

        run_kwargs = self.build_docker_run_kwargs(session_id)
        for host_path in run_kwargs["volumes"]:
            if "docker.sock" in str(host_path):
                raise SandboxSecurityError("Mounting docker.sock into sandbox is forbidden.")

        container = self._docker_client.containers.run(**run_kwargs)
        self._containers[session_id] = container
        return container, recovered

    async def exec_command(
        self,
        command: str,
        *,
        session_id: str = "default",
        timeout: int | None = None,
    ) -> ExecResult:
        if not command or not command.strip():
            return ExecResult(
                exit_code=2,
                stdout="",
                stderr="Empty command provided.",
                backend=self.active_backend,
                session_id=session_id,
            )

        effective_timeout = (
            min(timeout, self.config.timeout_seconds)
            if timeout is not None and timeout > 0
            else self.config.timeout_seconds
        )

        async with self._get_lock(session_id):
            if self._docker_available:
                return await self._exec_docker(command, session_id, effective_timeout)
            return await self._exec_local_isolated(command, session_id, effective_timeout)

    async def _exec_docker(
        self,
        command: str,
        session_id: str,
        timeout: int,
    ) -> ExecResult:
        def _run_once() -> tuple[int, str, str, bool]:
            container, recovered = self._ensure_docker_container(session_id)
            try:
                exec_res = container.exec_run(
                    cmd=["bash", "-lc", command],
                    workdir="/workspace",
                    demux=True,
                )
            except Exception:
                self._containers.pop(session_id, None)
                container, _ = self._ensure_docker_container(session_id)
                recovered = True
                exec_res = container.exec_run(
                    cmd=["bash", "-lc", command],
                    workdir="/workspace",
                    demux=True,
                )
            exit_code = int(getattr(exec_res, "exit_code", 1))
            output = getattr(exec_res, "output", (b"", b""))
            if isinstance(output, tuple):
                out_b, err_b = output
            else:
                out_b, err_b = output, b""
            stdout = (out_b or b"").decode("utf-8", errors="replace")
            stderr = (err_b or b"").decode("utf-8", errors="replace")
            return exit_code, stdout, stderr, recovered

        try:
            exit_code, stdout, stderr, recovered = await asyncio.wait_for(
                asyncio.to_thread(_run_once),
                timeout=float(timeout),
            )
            return ExecResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                backend="docker",
                session_id=session_id,
                recovered=recovered,
            )
        except TimeoutError:
            return ExecResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {timeout}s inside Docker sandbox.",
                timed_out=True,
                backend="docker",
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            raise SandboxExecutionError(f"Docker sandbox execution error: {exc}") from exc

    async def _exec_local_isolated(
        self,
        command: str,
        session_id: str,
        timeout: int,
    ) -> ExecResult:
        _check_local_command_boundaries(command)
        ws = self.get_session_workspace(session_id)
        env = _sanitized_sandbox_env(ws)
        mem_bytes = _parse_memory_bytes(self.config.memory_limit)
        cpu_seconds = max(timeout + 2, 5)

        def _preexec() -> None:
            os.setsid()
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
            except (ValueError, OSError):
                pass
            # Use RLIMIT_DATA (heap/data segment) instead of RLIMIT_AS (virtual address space)
            # because V8/Node.js, Go, OpenBLAS, and modern runtimes reserve multi-GB virtual
            # address ranges at startup even while using minimal physical RAM.
            try:
                _soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
                target = max(mem_bytes, 512 * 1024 * 1024)
                if hard == resource.RLIM_INFINITY or target <= hard:
                    resource.setrlimit(resource.RLIMIT_DATA, (target, hard))
            except (ValueError, OSError):
                pass

        rewritten_cmd = rewrite_workspace_paths_in_command(command, ws)

        # Use non-login `bash -c` so /etc/profile does not overwrite our curated PATH (.venv/bin).
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            rewritten_cmd,
            cwd=str(ws),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_preexec,
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=float(timeout))
            exit_code = int(proc.returncode or 0)
            stdout_str = stdout_b.decode("utf-8", errors="replace")
            stderr_str = stderr_b.decode("utf-8", errors="replace")
            if (
                exit_code != 0
                and "/workspace" in stderr_str
                and any(
                    marker in stderr_str
                    for marker in (
                        "FileNotFoundError",
                        "No such file or directory",
                        "ENOENT",
                        "cannot open",
                        "does not exist",
                    )
                )
            ):
                hint = (
                    "\n[Sandbox hint: In local-isolated mode, commands already run with the workspace "
                    "as the current working directory (CWD), while literal '/workspace/...' paths inside "
                    "script source code are not mounted at root. Use relative paths (e.g. 'data.txt') "
                    "or os.environ.get('WORKSPACE', '.') / $WORKSPACE inside scripts.]"
                )
                stderr_str = stderr_str.rstrip() + hint
            return ExecResult(
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                timed_out=False,
                backend="local_isolated",
                session_id=session_id,
            )
        except TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                pass
            return ExecResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {timeout}s and process group was terminated.",
                timed_out=True,
                backend="local_isolated",
                session_id=session_id,
            )

    def read_file(
        self,
        path: str,
        *,
        session_id: str = "default",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        ws = self.get_session_workspace(session_id)
        resolved = resolve_workspace_path(path, ws)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if resolved.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        if start_line is None and end_line is None:
            return text
        lines = text.splitlines(keepends=True)
        total = len(lines)
        s = max(1, int(start_line)) if start_line is not None else 1
        e = min(total, int(end_line)) if end_line is not None else total
        if s > total:
            return ""
        return "".join(lines[s - 1 : e])

    def write_file(self, path: str, content: str, *, session_id: str = "default") -> str:
        ws = self.get_session_workspace(session_id)
        resolved = resolve_workspace_path(path, ws)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolve_workspace_path(path, ws)
        resolved.write_text(content, encoding="utf-8")
        vpath = to_virtual_workspace_path(resolved, ws)
        return f"Wrote {len(content)} characters to {vpath}"

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        session_id: str = "default",
    ) -> str:
        ws = self.get_session_workspace(session_id)
        resolved = resolve_workspace_path(path, ws)
        if not resolved.exists():
            raise FileNotFoundError(f"Cannot edit non-existent file: {path}")
        if not old_text:
            raise ValueError("old_text must not be empty.")
        current = resolved.read_bytes().decode("utf-8", errors="replace")
        count = current.count(old_text)
        updated: str | None = None
        if count == 1:
            updated = current.replace(old_text, new_text, 1)
        elif count > 1:
            raise ValueError(
                f"old_text matched {count} locations in {path}; provide more surrounding context."
            )
        else:
            # Handle CRLF vs LF line-ending mismatches gracefully when unique
            norm_current = current.replace("\r\n", "\n")
            norm_old = old_text.replace("\r\n", "\n")
            norm_count = norm_current.count(norm_old)
            if norm_count == 1:
                uses_crlf = "\r\n" in current
                norm_new = new_text.replace("\r\n", "\n")
                replaced = norm_current.replace(norm_old, norm_new, 1)
                updated = replaced.replace("\n", "\r\n") if uses_crlf else replaced
            elif norm_count > 1:
                raise ValueError(
                    f"old_text matched {norm_count} locations in {path}; provide more surrounding context."
                )
            else:
                line_count = len(current.splitlines())
                raise ValueError(
                    f"old_text was not found in {path} (file has {line_count} lines). "
                    "Call read_file first to inspect the exact current text."
                )
        resolved.write_bytes(updated.encode("utf-8"))
        vpath = to_virtual_workspace_path(resolved, ws)
        return f"Updated {vpath} (replaced 1 occurrence)."

    def list_dir(
        self,
        path: str = "/workspace",
        *,
        session_id: str = "default",
        include_internal: bool = False,
    ) -> list[str]:
        ws = self.get_session_workspace(session_id)
        resolved = resolve_workspace_path(path, ws, allow_root=True)
        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        is_ws_root = resolved == ws.resolve()
        entries: list[str] = []
        for entry in sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            if is_ws_root and not include_internal and entry.name == ".logs":
                continue
            suffix = "/" if entry.is_dir() else ""
            entries.append(f"{entry.name}{suffix}")
        return entries

    def stop_session(self, session_id: str) -> None:
        container = self._containers.pop(session_id, None)
        if container is not None:
            try:
                container.stop(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            try:
                container.remove(force=True)
            except Exception:  # noqa: BLE001
                pass

    def cleanup_all(self) -> None:
        for sid in list(self._containers.keys()):
            self.stop_session(sid)

    def health_status(self) -> dict[str, Any]:
        return {
            "backend": self.active_backend,
            "configured_mode": self.config.backend,
            "docker_available": self._docker_available,
            "docker_error": self._docker_error,
            "image": self.config.image,
            "workspace_dir": str(self.config.workspace_dir),
            "memory_limit": self.config.memory_limit,
            "cpu_quota": self.config.cpu_quota,
            "cpu_period": self.config.cpu_period,
            "pids_limit": self.config.pids_limit,
            "timeout_seconds": self.config.timeout_seconds,
            "active_containers": len(self._containers),
        }
