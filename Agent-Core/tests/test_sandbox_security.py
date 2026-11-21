"""Phase 11D: Docker Sandbox Security, Workspace Confinement, Timeouts, Recovery & Web Fetch."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from agent_core.sandbox import (
    SandboxConfig,
    SandboxManager,
    SandboxSecurityError,
    resolve_workspace_path,
)
from agent_core.tools.local import (
    build_local_tools,
    fetch_webpage_markdown,
    html_to_clean_markdown,
)


def test_dockerfile_sandbox_security_and_requirements() -> None:
    dockerfile = (Path.cwd() / "Agent-Core" / "Dockerfile.sandbox").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "playwright install chromium" in dockerfile
    assert "USER agent" in dockerfile
    assert "WORKDIR /workspace" in dockerfile
    assert "docker.sock" not in dockerfile


@pytest.mark.asyncio
async def test_workspace_crud_python_and_bash_execution(
    isolated_sandbox: SandboxManager,
) -> None:
    tools = {t.name: t for t in build_local_tools(isolated_sandbox)}

    write_res = tools["write_file"].invoke(
        {"path": "/workspace/sample.py", "content": "x = 10\nprint(x * 3)\n"}
    )
    assert "/workspace/sample.py" in write_res

    edit_res = tools["edit_file"].invoke(
        {"path": "/workspace/sample.py", "old_text": "x = 10", "new_text": "x = 15"}
    )
    assert "Updated" in edit_res

    read_res = tools["read_file"].invoke({"path": "/workspace/sample.py"})
    assert "x = 15" in read_res

    listing = tools["list_dir"].invoke({"path": "/workspace"})
    assert "sample.py" in listing

    exec_res = await tools["bash_exec"].ainvoke({"command": "python3 /workspace/sample.py"})
    assert "exit_code: 0" in exec_res
    assert "45" in exec_res


def test_path_traversal_absolute_path_and_symlink_escapes_rejected(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside_secret = tmp_path / "host_secret.txt"
    outside_secret.write_text("TOP_SECRET", encoding="utf-8")

    # 1. Path traversal attacks
    for bad_path in [
        "../host_secret.txt",
        "/workspace/../host_secret.txt",
        "subdir/../../host_secret.txt",
        "foo\x00bar",
    ]:
        with pytest.raises(SandboxSecurityError):
            resolve_workspace_path(bad_path, ws)

    # 2. Absolute host path attacks
    for abs_bad in ["/etc/passwd", "/root/.ssh/id_rsa", str(outside_secret)]:
        with pytest.raises(SandboxSecurityError):
            resolve_workspace_path(abs_bad, ws)

    # 3. Direct file symlink escape attack
    escape_link = ws / "escape.txt"
    escape_link.symlink_to(outside_secret)
    with pytest.raises(SandboxSecurityError):
        resolve_workspace_path("/workspace/escape.txt", ws)

    # 4. Parent directory symlink escape attack
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    dir_link = ws / "linked_dir"
    dir_link.symlink_to(outside_dir)
    with pytest.raises(SandboxSecurityError):
        resolve_workspace_path("/workspace/linked_dir/pwned.txt", ws)


@pytest.mark.asyncio
async def test_bash_command_timeout_and_host_escape_blocking(
    tmp_workspace: Path,
) -> None:
    cfg = SandboxConfig(
        workspace_dir=tmp_workspace,
        backend="local_isolated",
        timeout_seconds=2,
    )
    mgr = SandboxManager(cfg)

    # Timeout enforcement
    res = await mgr.exec_command("sleep 10", timeout=1)
    assert res.timed_out is True
    assert res.exit_code == 124

    # Host escape command blocked
    with pytest.raises(SandboxSecurityError):
        await mgr.exec_command("cat /etc/passwd")
    with pytest.raises(SandboxSecurityError):
        await mgr.exec_command("ls /var/run/docker.sock")

    # Secrets stripped from subprocess environment
    env_res = await mgr.exec_command("env")
    assert "RIVER_AI_API" not in env_res.stdout
    assert "AGENT_API_TOKEN" not in env_res.stdout
    assert "OPENAI_API_KEY" not in env_res.stdout


@pytest.mark.asyncio
async def test_docker_sandbox_resource_limits_no_socket_and_crash_recovery(
    tmp_workspace: Path,
) -> None:
    """Verify Docker backend container limits, no docker.sock mount, and crash auto-recovery."""

    class FakeExecResult:
        def __init__(self, exit_code: int, out: bytes, err: bytes = b"") -> None:
            self.exit_code = exit_code
            self.output = (out, err)

    class FakeContainer:
        def __init__(self, fail_once: bool = False) -> None:
            self.status = "running"
            self.fail_once = fail_once
            self.removed = False
            self.stopped = False

        def reload(self) -> None:
            pass

        def exec_run(self, **_kwargs: Any) -> FakeExecResult:
            if self.fail_once:
                self.fail_once = False
                self.status = "exited"
                raise RuntimeError("Container terminated unexpectedly")
            return FakeExecResult(0, b"hello from docker\n")

        def stop(self, timeout: int = 2) -> None:
            self.stopped = True
            self.status = "exited"

        def remove(self, force: bool = True) -> None:
            self.removed = True

    class FakeContainersAPI:
        def __init__(self) -> None:
            self.run_calls: list[dict[str, Any]] = []

        def run(self, **kwargs: Any) -> FakeContainer:
            self.run_calls.append(kwargs)
            # First container crashes on exec to test automatic recovery
            return FakeContainer(fail_once=(len(self.run_calls) == 1))

    class FakeDockerClient:
        def __init__(self) -> None:
            self.containers = FakeContainersAPI()

        def ping(self) -> bool:
            return True

    fake_docker = FakeDockerClient()
    cfg = SandboxConfig(
        workspace_dir=tmp_workspace,
        backend="docker",
        memory_limit="512m",
        cpu_quota=100_000,
        cpu_period=100_000,
        pids_limit=128,
    )
    mgr = SandboxManager(cfg, docker_client=fake_docker)
    assert mgr.active_backend == "docker"

    # First exec crashes container #1 -> must recover automatically into container #2!
    res = await mgr.exec_command("echo hi", session_id="sess-docker-1")
    assert res.exit_code == 0
    assert res.recovered is True
    assert "hello from docker" in res.stdout
    assert len(fake_docker.containers.run_calls) == 2

    # Verify strict security & resource parameters on container creation
    run_kwargs = fake_docker.containers.run_calls[0]
    assert run_kwargs["mem_limit"] == "512m"
    assert run_kwargs["cpu_quota"] == 100_000
    assert run_kwargs["pids_limit"] == 128
    assert run_kwargs["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in run_kwargs["security_opt"]
    for host_vol in run_kwargs["volumes"]:
        assert "docker.sock" not in host_vol

    mgr.cleanup_all()
    assert mgr.health_status()["active_containers"] == 0


@pytest.mark.asyncio
async def test_concurrent_session_isolation(tmp_workspace: Path) -> None:
    cfg = SandboxConfig(workspace_dir=tmp_workspace, backend="local_isolated", timeout_seconds=5)
    mgr = SandboxManager(cfg, isolate_session_dirs=True)

    mgr.write_file("/workspace/secret.txt", "alpha-only", session_id="thread-a")
    mgr.write_file("/workspace/secret.txt", "beta-only", session_id="thread-b")

    res_a, res_b = await asyncio.gather(
        mgr.exec_command("cat /workspace/secret.txt", session_id="thread-a"),
        mgr.exec_command("cat /workspace/secret.txt", session_id="thread-b"),
    )
    assert res_a.stdout.strip() == "alpha-only"
    assert res_b.stdout.strip() == "beta-only"


@pytest.mark.asyncio
async def test_web_fetch_clean_markdown_and_ssrf_protection() -> None:
    sample_html = """
    <html>
      <head><title>LangGraph Spec</title><script>alert('xss')</script><style>body{color:red}</style></head>
      <body>
        <main>
          <h1>Architecture Overview</h1>
          <p>Strict separation between Brain, Hands, and SaaS Arm.</p>
        </main>
      </body>
    </html>
    """
    md = html_to_clean_markdown(sample_html, base_url="https://example.org/spec")
    assert "# LangGraph Spec" in md
    assert "Architecture Overview" in md
    assert "alert('xss')" not in md
    assert "body{color:red}" not in md

    # Mock transport for deterministic HTTP test
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=sample_html,
        )
    )
    fetched = await fetch_webpage_markdown("https://example.org/spec", transport=transport)
    assert "Strict separation" in fetched

    # Loopback SSRF rejected by default
    with pytest.raises(SandboxSecurityError):
        await fetch_webpage_markdown("http://127.0.0.1:5678/healthz")


@pytest.mark.asyncio
async def test_local_sandbox_node_venv_path_rewrite_and_env_scrubbing(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Node/V8 does not OOM under sandbox limits, venv Python stays on PATH, and /workspace rewriting is boundary-safe."""
    import shutil
    from agent_core.sandbox import rewrite_workspace_paths_in_command

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret-test-999")
    monkeypatch.setenv("VAULT_MASTER_KEY", "master-secret-test-999")

    cfg = SandboxConfig(workspace_dir=tmp_workspace, backend="local_isolated", timeout_seconds=10)
    mgr = SandboxManager(cfg)

    # 1. Node.js (V8) runs without RLIMIT_AS CodeRange OOM crash if installed
    if shutil.which("node"):
        node_res = await mgr.exec_command("node -e 'console.log(21 * 2)'")
        assert node_res.exit_code == 0, f"node failed: {node_res.stderr}"
        assert node_res.stdout.strip() == "42"

    # 2. Active virtualenv Python (with pytest installed) stays on PATH under non-login bash -c
    py_res = await mgr.exec_command("python3 -c 'import pytest; print(pytest.__version__)'")
    assert py_res.exit_code == 0, f"venv python failed: {py_res.stderr}"
    assert py_res.stdout.strip()

    # 3. Boundary-aware /workspace rewriting never corrupts URLs, /workspaces, or already-resolved paths
    ws_str = str(tmp_workspace.resolve())
    assert rewrite_workspace_paths_in_command("ls /workspace/app.py", tmp_workspace) == f"ls {ws_str}/app.py"
    assert (
        rewrite_workspace_paths_in_command(f"cat {ws_str}/app.py", tmp_workspace)
        == f"cat {ws_str}/app.py"
    )
    assert (
        rewrite_workspace_paths_in_command("curl https://example.com/workspace/v1", tmp_workspace)
        == "curl https://example.com/workspace/v1"
    )
    assert rewrite_workspace_paths_in_command("echo /workspaces/demo", tmp_workspace) == "echo /workspaces/demo"

    # 4. Already-resolved host path inside workspace is accepted by resolve_workspace_path
    mgr.write_file("/workspace/lines.txt", "line1\nline2\nline3\nline4\n")
    assert mgr.read_file(f"{ws_str}/lines.txt", start_line=2, end_line=3) == "line2\nline3\n"

    # 5. Internal .logs/ is hidden at workspace root by default
    entries = mgr.list_dir("/workspace")
    assert "lines.txt" in entries
    assert ".logs/" not in entries

    # 6. Sensitive provider & vault secrets are stripped from subprocess env, and ../.env is blocked
    env_res = await mgr.exec_command("env")
    assert "NVIDIA_API_KEY" not in env_res.stdout
    assert "VAULT_MASTER_KEY" not in env_res.stdout
    with pytest.raises(SandboxSecurityError):
        await mgr.exec_command("cat ../.env")
    # Non-secret .env.example filename should not false-positive
    ex_res = await mgr.exec_command("echo .env.example")
    assert ex_res.exit_code == 0
    assert ".env.example" in ex_res.stdout


@pytest.mark.asyncio
async def test_sandbox_path_env_rewrite_script_hint_crlf_edit_and_malformed_args(
    tmp_workspace: Path,
) -> None:
    """Verify PATH/PYTHONPATH /workspace rewriting, script error hint, CRLF edit_file, and malformed JSON arg errors."""
    from agent_core.guardrails import normalize_tool_arguments, validate_tool_arguments
    from agent_core.sandbox import rewrite_workspace_paths_in_command
    from agent_core.tools.local import build_local_tools

    ws_str = str(tmp_workspace.resolve())

    # 1. PATH and PYTHONPATH containing :/workspace and /workspace: rewrite cleanly while URLs stay intact
    assert (
        rewrite_workspace_paths_in_command("PATH=/usr/bin:/workspace/bin python3 app.py", tmp_workspace)
        == f"PATH=/usr/bin:{ws_str}/bin python3 app.py"
    )
    assert (
        rewrite_workspace_paths_in_command("PYTHONPATH=/workspace:/tmp/lib python3 app.py", tmp_workspace)
        == f"PYTHONPATH={ws_str}:/tmp/lib python3 app.py"
    )
    assert (
        rewrite_workspace_paths_in_command("curl http://127.0.0.1:8080/workspace/api", tmp_workspace)
        == "curl http://127.0.0.1:8080/workspace/api"
    )

    cfg = SandboxConfig(workspace_dir=tmp_workspace, backend="local_isolated", timeout_seconds=10)
    mgr = SandboxManager(cfg)

    # 2. Script crashing with FileNotFoundError on literal /workspace/... receives actionable hint
    mgr.write_file("/workspace/bad_path.py", "open('/workspace/missing_data.txt').read()\n")
    res = await mgr.exec_command("python3 bad_path.py")
    assert res.exit_code != 0
    assert "Sandbox hint" in res.stderr
    assert "relative paths" in res.stderr

    # 3. edit_file handles CRLF file with LF old_text and gives line-count hint when not found
    crlf_path = tmp_workspace / "crlf.txt"
    crlf_path.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    edit_msg = mgr.edit_file("/workspace/crlf.txt", "alpha\nbeta", "alpha\nBETA")
    assert "Updated" in edit_msg
    assert crlf_path.read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"

    with pytest.raises(ValueError, match="file has 3 lines"):
        mgr.edit_file("/workspace/crlf.txt", "does_not_exist", "x")

    # 4. Expanded aliases + malformed JSON string surfaces explicit validation error
    assert normalize_tool_arguments("write_file", {"file_path": "a.py", "code": "x=1"}) == {
        "path": "a.py",
        "content": "x=1",
    }
    assert normalize_tool_arguments("edit_file", {"path": "a.py", "search": "1", "replace": "2"}) == {
        "path": "a.py",
        "old_text": "1",
        "new_text": "2",
    }
    bad_norm = normalize_tool_arguments("write_file", '{"path": "a.py", "content": "unterminated')
    assert "__invalid_json__" in bad_norm
    tools = {t.name: t for t in build_local_tools(mgr)}
    validated, err = validate_tool_arguments(tools["write_file"], bad_norm)
    assert validated is None
    assert err is not None and "Malformed JSON arguments" in err


