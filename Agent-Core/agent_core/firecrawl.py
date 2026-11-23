"""Official Firecrawl Local & v2 API Client.

Strictly implements the official Firecrawl Local v2 HTTP API contract
(https://github.com/firecrawl/firecrawl#firecrawl-local):
- Health / readiness:     GET  /health -> {"status":"ok","engine":"firecrawl-local","version":"..."}
- Single-page scrape:     POST /v2/scrape
- Batch scrape:           POST /v2/batch/scrape, GET /v2/batch/scrape/{id}, DELETE /v2/batch/scrape/{id}
- Site URL mapping:       POST /v2/map
- Multi-page crawling:    POST /v2/crawl (maxDiscoveryDepth), GET /v2/crawl/{id}, DELETE /v2/crawl/{id}
- Web search:             POST /v2/search (forwarded by firecrawl-local unless FIRECRAWL_CLOUD=off)
- Structured extraction:  POST /v2/extract, GET /v2/extract/{id}

No legacy v0/v1 routes or non-Firecrawl fallbacks. All errors are classified honestly:
unavailable, auth_error, payment_required, cloud_only, rate_limited, timeout,
invalid_request, upstream_error.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import mimetypes
import os
import random
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_FIRECRAWL_API_URL = "http://127.0.0.1:3002"
CLOUD_FIRECRAWL_API_URL = "https://api.firecrawl.dev"
DEFAULT_FIRECRAWL_TIMEOUT_SECONDS = 45.0
DEFAULT_FIRECRAWL_CONNECT_TIMEOUT_SECONDS = 8.0
DEFAULT_FIRECRAWL_MAX_RETRIES = 2
DEFAULT_FIRECRAWL_CRAWL_MAX_WAIT_SECONDS = 60.0
MAX_LOCAL_DOCUMENT_BYTES = 50 * 1024 * 1024  # Matches anydoc's 50 MB guard

# Matches official firecrawl-local `src/documents.rs` supported document formats:
# - PDF via `pdf-inspector` (`engine: "local-pdf"`)
# - Office/ebook formats via `anydoc` (`engine: "local-document"`)
FIRECRAWL_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".odt",
        ".ods",
        ".odp",
        ".rtf",
        ".epub",
    }
)

FIRECRAWL_DOCUMENT_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".rtf": "application/rtf",
    ".epub": "application/epub+zip",
}


def is_firecrawl_document_path(path: str | Path) -> bool:
    """Return True if path has a PDF or Office document extension supported natively by firecrawl-local."""
    return Path(path).suffix.lower() in FIRECRAWL_DOCUMENT_EXTENSIONS


def find_firecrawl_local_binary(cwd: Path | None = None) -> str | None:
    """Locate the installed official `firecrawl-local` binary."""
    root = (cwd or Path(__file__).resolve().parents[2]).resolve()
    candidates = [
        (os.environ.get("FIRECRAWL_LOCAL_BIN") or "").strip(),
        str(root / ".firecrawl" / "bin" / "firecrawl-local"),
        str(Path.home() / ".cargo" / "bin" / "firecrawl-local"),
        "/usr/local/cargo/bin/firecrawl-local",
        "/usr/local/bin/firecrawl-local",
        "/usr/bin/firecrawl-local",
        shutil.which("firecrawl-local") or "",
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None

FirecrawlErrorCategory = Literal[
    "unavailable",
    "auth_error",
    "payment_required",
    "cloud_only",
    "rate_limited",
    "timeout",
    "invalid_request",
    "upstream_error",
]


class FirecrawlError(RuntimeError):
    """Structured error raised when Firecrawl Local fails or is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        category: FirecrawlErrorCategory = "upstream_error",
        status_code: int | None = None,
        retryable: bool = False,
        endpoint: str = "",
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category: FirecrawlErrorCategory = category
        self.status_code: int | None = status_code
        self.retryable: bool = retryable
        self.endpoint: str = endpoint
        self.code: str | None = code


def _is_loopback_url(raw_url: str) -> bool:
    try:
        host = (urlparse(raw_url).hostname or "").strip().lower()
        return host in ("127.0.0.1", "localhost", "::1")
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class FirecrawlConfig:
    """Resolved runtime configuration for the official Firecrawl Local v2 client."""

    api_url: str
    api_key: str
    local_token: str = ""
    connect_timeout_seconds: float = DEFAULT_FIRECRAWL_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_FIRECRAWL_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_FIRECRAWL_MAX_RETRIES
    crawl_max_wait_seconds: float = DEFAULT_FIRECRAWL_CRAWL_MAX_WAIT_SECONDS

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> FirecrawlConfig:
        src = dict(os.environ) if env is None else env
        explicit_url = (src.get("FIRECRAWL_API_URL") or "").strip().rstrip("/")
        api_key = (src.get("FIRECRAWL_API_KEY") or "").strip()
        local_token = (src.get("FIRECRAWL_LOCAL_TOKEN") or "").strip()
        if explicit_url:
            raw_url = explicit_url
        else:
            raw_url = DEFAULT_FIRECRAWL_API_URL

        def _float_env(name: str, default: float, lo: float, hi: float) -> float:
            raw = (src.get(name) or "").strip()
            if not raw:
                return default
            try:
                return max(lo, min(hi, float(raw)))
            except ValueError:
                return default

        def _int_env(name: str, default: int, lo: int, hi: int) -> int:
            raw = (src.get(name) or "").strip()
            if not raw:
                return default
            try:
                return max(lo, min(hi, int(raw)))
            except ValueError:
                return default

        return cls(
            api_url=raw_url,
            api_key=api_key,
            local_token=local_token,
            connect_timeout_seconds=_float_env(
                "FIRECRAWL_CONNECT_TIMEOUT_SECONDS", DEFAULT_FIRECRAWL_CONNECT_TIMEOUT_SECONDS, 1.0, 60.0
            ),
            read_timeout_seconds=_float_env(
                "FIRECRAWL_TIMEOUT_SECONDS", DEFAULT_FIRECRAWL_TIMEOUT_SECONDS, 5.0, 300.0
            ),
            max_retries=_int_env("FIRECRAWL_MAX_RETRIES", DEFAULT_FIRECRAWL_MAX_RETRIES, 0, 5),
            crawl_max_wait_seconds=_float_env(
                "FIRECRAWL_CRAWL_MAX_WAIT_SECONDS", DEFAULT_FIRECRAWL_CRAWL_MAX_WAIT_SECONDS, 5.0, 300.0
            ),
        )


def _extract_error_details(resp: httpx.Response) -> tuple[str, str | None]:
    code: str | None = None
    try:
        data = resp.json()
        if isinstance(data, dict):
            raw_code = data.get("code")
            if isinstance(raw_code, str) and raw_code.strip():
                code = raw_code.strip()
            for key in ("error", "message", "detail"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip(), code
    except Exception:  # noqa: BLE001
        pass
    body = (resp.text or "").strip()
    return (body[:300] if body else f"HTTP {resp.status_code}"), code


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    hdr = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if not hdr:
        return None
    try:
        val = float(hdr.strip())
        return max(0.2, min(30.0, val))
    except ValueError:
        return None


class FirecrawlClient:
    """Async REST client for official Firecrawl Local (v2 API)."""

    def __init__(
        self,
        config: FirecrawlConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._explicit_config = config
        self._transport = transport

    @property
    def config(self) -> FirecrawlConfig:
        return self._explicit_config or FirecrawlConfig.from_env()

    def _headers(self, cfg: FirecrawlConfig) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Nemofy-Agent-Core/1.0",
        }
        # Per firecrawl-local serve.rs: when --token / FIRECRAWL_LOCAL_TOKEN is set on the local
        # server, send that bearer token; otherwise pass FIRECRAWL_API_KEY so serve can forward
        # client auth to Firecrawl Cloud on /v2/search or /v2/extract.
        if cfg.local_token and _is_loopback_url(cfg.api_url):
            headers["Authorization"] = f"Bearer {cfg.local_token}"
        elif cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def _timeout(self, cfg: FirecrawlConfig, read_override: float | None = None) -> httpx.Timeout:
        read_s = read_override if read_override is not None else cfg.read_timeout_seconds
        return httpx.Timeout(
            timeout=read_s,
            connect=cfg.connect_timeout_seconds,
            read=read_s,
            write=15.0,
            pool=10.0,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        read_timeout: float | None = None,
        retries: int | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        url = f"{cfg.api_url}{path}"
        max_attempts = 1 + (cfg.max_retries if retries is None else max(0, retries))
        last_err: FirecrawlError | None = None

        for attempt in range(1, max_attempts + 1):
            t0 = time.monotonic()
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout(cfg, read_override=read_timeout),
                    transport=self._transport,
                    follow_redirects=False,
                ) as client:
                    resp = await client.request(
                        method,
                        url,
                        headers=self._headers(cfg),
                        json=payload,
                    )
            except asyncio.CancelledError:
                raise
            except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                last_err = FirecrawlError(
                    f"Firecrawl Local unavailable at {cfg.api_url} ({type(exc).__name__}): {exc}",
                    category="unavailable",
                    retryable=True,
                    endpoint=path,
                )
            except httpx.TimeoutException as exc:
                last_err = FirecrawlError(
                    f"Firecrawl request to {path} timed out after {read_timeout or cfg.read_timeout_seconds:.0f}s: {exc}",
                    category="timeout",
                    retryable=True,
                    endpoint=path,
                )
            except httpx.HTTPError as exc:
                last_err = FirecrawlError(
                    f"Firecrawl network error on {path}: {exc}",
                    category="unavailable",
                    retryable=True,
                    endpoint=path,
                )
            else:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if resp.status_code in (401, 403):
                    detail, code = _extract_error_details(resp)
                    raise FirecrawlError(
                        f"Firecrawl authentication required or failed (HTTP {resp.status_code}): {detail}",
                        category="auth_error",
                        status_code=resp.status_code,
                        retryable=False,
                        endpoint=path,
                        code=code,
                    )
                if resp.status_code == 402:
                    detail, code = _extract_error_details(resp)
                    raise FirecrawlError(
                        f"Firecrawl Cloud credits/plan required (HTTP 402): {detail}",
                        category="payment_required",
                        status_code=402,
                        retryable=False,
                        endpoint=path,
                        code=code,
                    )
                if resp.status_code == 501:
                    detail, code = _extract_error_details(resp)
                    raise FirecrawlError(
                        f"Firecrawl route {path} requires Firecrawl Cloud (HTTP 501): {detail}",
                        category="cloud_only",
                        status_code=501,
                        retryable=False,
                        endpoint=path,
                        code=code or "CLOUD_ONLY",
                    )
                if resp.status_code == 429:
                    detail, code = _extract_error_details(resp)
                    retry_after = _retry_after_seconds(resp)
                    last_err = FirecrawlError(
                        f"Firecrawl rate limit exceeded (HTTP 429): {detail}",
                        category="rate_limited",
                        status_code=429,
                        retryable=True,
                        endpoint=path,
                        code=code,
                    )
                    if attempt < max_attempts:
                        delay = (
                            retry_after
                            if retry_after is not None
                            else min(8.0, (1.0 * (2 ** (attempt - 1))) + random.uniform(0.1, 0.4))
                        )
                        logger.warning(
                            "firecrawl_retry endpoint=%s status=429 attempt=%d/%d delay_s=%.2f elapsed_ms=%d",
                            path,
                            attempt,
                            max_attempts,
                            delay,
                            elapsed_ms,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise last_err
                if resp.status_code in (500, 502, 503, 504):
                    detail, code = _extract_error_details(resp)
                    last_err = FirecrawlError(
                        f"Firecrawl upstream error (HTTP {resp.status_code}) on {path}: {detail}",
                        category="upstream_error",
                        status_code=resp.status_code,
                        retryable=True,
                        endpoint=path,
                        code=code,
                    )
                elif resp.status_code >= 400:
                    detail, code = _extract_error_details(resp)
                    raise FirecrawlError(
                        f"Firecrawl rejected request to {path} (HTTP {resp.status_code}): {detail}",
                        category="invalid_request",
                        status_code=resp.status_code,
                        retryable=False,
                        endpoint=path,
                        code=code,
                    )
                else:
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        raise FirecrawlError(
                            f"Firecrawl returned non-JSON payload on {path}: {exc}",
                            category="upstream_error",
                            status_code=resp.status_code,
                            retryable=False,
                            endpoint=path,
                        ) from exc
                    if isinstance(data, dict) and data.get("success") is False:
                        err_text = str(data.get("error") or data.get("message") or "Unknown Firecrawl failure")
                        raw_code = data.get("code")
                        raise FirecrawlError(
                            f"Firecrawl reported failure on {path}: {err_text}",
                            category="upstream_error",
                            status_code=resp.status_code,
                            retryable=False,
                            endpoint=path,
                            code=str(raw_code) if raw_code else None,
                        )
                    return data if isinstance(data, dict) else {"success": True, "data": data}

            if last_err is not None and last_err.retryable and attempt < max_attempts:
                delay = min(6.0, (0.8 * (2 ** (attempt - 1))) + random.uniform(0.05, 0.35))
                logger.warning(
                    "firecrawl_retry endpoint=%s category=%s attempt=%d/%d delay_s=%.2f",
                    path,
                    last_err.category,
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if last_err is not None:
                raise last_err

        raise last_err or FirecrawlError("Firecrawl request failed unexpectedly", endpoint=path)

    async def check_health(self, *, timeout_seconds: float = 4.0) -> dict[str, Any]:
        """Probe official Firecrawl Local GET /health endpoint.

        Official `firecrawl-local serve` responds with:
        {"status": "ok", "engine": "firecrawl-local", "version": "..."}
        """
        cfg = self.config
        t0 = time.monotonic()
        is_cloud = "api.firecrawl.dev" in cfg.api_url.lower()
        default_mode = "cloud" if is_cloud else "firecrawl_local"

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=timeout_seconds,
                    connect=min(timeout_seconds, cfg.connect_timeout_seconds),
                    read=timeout_seconds,
                    write=timeout_seconds,
                    pool=timeout_seconds,
                ),
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                resp = await client.get(f"{cfg.api_url}/health", headers=self._headers(cfg))
            latency_ms = int((time.monotonic() - t0) * 1000)
            if resp.status_code < 400:
                engine: str | None = None
                version: str | None = None
                try:
                    body_json = resp.json()
                    if isinstance(body_json, dict):
                        if isinstance(body_json.get("engine"), str):
                            engine = body_json["engine"].strip()
                        if isinstance(body_json.get("version"), str):
                            version = body_json["version"].strip()
                except Exception:  # noqa: BLE001
                    pass

                mode = "firecrawl_local" if engine == "firecrawl-local" else default_mode
                return {
                    "healthy": True,
                    "reachable": True,
                    "mode": mode,
                    "engine": engine or mode,
                    "version": version,
                    "status": "ready",
                    "api_url": cfg.api_url,
                    "api_key_configured": bool(cfg.api_key),
                    "http_status": resp.status_code,
                    "probe_path": "/health",
                    "latency_ms": latency_ms,
                    "error": None,
                }
            last_error = f"HTTP {resp.status_code} on /health"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

        return {
            "healthy": False,
            "reachable": False,
            "mode": default_mode,
            "engine": None,
            "version": None,
            "status": "unavailable",
            "api_url": cfg.api_url,
            "api_key_configured": bool(cfg.api_key),
            "http_status": None,
            "probe_path": "/health",
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "error": last_error,
        }

    async def scrape(
        self,
        url: str,
        *,
        formats: list[str] | None = None,
        only_main_content: bool = True,
        wait_for: int | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Scrape a single URL via Firecrawl Local POST /v2/scrape."""
        payload: dict[str, Any] = {
            "url": url,
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
        }
        if wait_for is not None and wait_for > 0:
            payload["waitFor"] = int(wait_for)
        if timeout_ms is not None and timeout_ms > 0:
            payload["timeout"] = int(timeout_ms)

        resp = await self._request("POST", "/v2/scrape", payload=payload)
        return self._normalize_scrape_response(resp, url)

    @staticmethod
    def _normalize_scrape_response(resp: dict[str, Any], fallback_url: str) -> dict[str, Any]:
        data = resp.get("data") if isinstance(resp.get("data"), dict) else resp
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        engine = str(metadata.get("engine") or "").strip() or None
        raw_num_pages = metadata.get("numPages")
        num_pages = int(raw_num_pages) if isinstance(raw_num_pages, int) else None
        content_type = str(metadata.get("contentType") or "").strip() or None
        return {
            "url": str(metadata.get("sourceURL") or metadata.get("url") or fallback_url),
            "title": str(metadata.get("title") or "").strip(),
            "description": str(metadata.get("description") or "").strip(),
            "status_code": metadata.get("statusCode"),
            "markdown": str(data.get("markdown") or data.get("content") or "").strip(),
            "links": data.get("links") if isinstance(data.get("links"), list) else [],
            "warning": data.get("warning") or metadata.get("warning"),
            "engine": engine,
            "num_pages": num_pages,
            "content_type": content_type,
            "source": engine or data.get("source") or "firecrawl-local",
            "metadata": metadata,
        }

    async def parse_local_document(
        self,
        file_path: str | Path,
        *,
        binary_path: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Parse a local PDF or Office document using official `firecrawl-local` (pdf-inspector + anydoc).

        Serves ONLY the single validated file on an ephemeral 127.0.0.1:0 one-shot HTTP handler
        protected by a 256-bit random URL token, invokes `firecrawl-local scrape --browser never`,
        and immediately shuts down the ephemeral listener. Keeps `firecrawl-local serve` locked
        against private/loopback SSRF while reusing the exact official Firecrawl document parsers.
        """
        resolved = Path(file_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Path is not a file: {file_path}")

        raw_bytes = resolved.read_bytes()
        if len(raw_bytes) > MAX_LOCAL_DOCUMENT_BYTES:
            raise ValueError(
                f"Document exceeds maximum supported size ({len(raw_bytes):,} > {MAX_LOCAL_DOCUMENT_BYTES:,} bytes)."
            )

        bin_path = binary_path or find_firecrawl_local_binary()
        if not bin_path:
            raise FirecrawlError(
                "firecrawl-local binary is not installed; cannot parse local PDF/Office document. "
                "Run `npm run firecrawl:install` or `npm run build`.",
                category="unavailable",
                endpoint="local-document",
            )

        ext = resolved.suffix.lower()
        mime_type = (
            FIRECRAWL_DOCUMENT_MIME_TYPES.get(ext)
            or mimetypes.guess_type(resolved.name)[0]
            or "application/octet-stream"
        )
        token = secrets.token_urlsafe(24)
        safe_name = quote(resolved.name or f"document{ext}")
        expected_path = f"/{token}/{safe_name}"

        class _SingleFileHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != expected_path:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(raw_bytes)))
                self.end_headers()
                self.wfile.write(raw_bytes)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), _SingleFileHandler)
        port = int(server.server_address[1])
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()

        effective_timeout = float(
            timeout_seconds if timeout_seconds is not None else self.config.read_timeout_seconds
        )
        timeout_ms = max(5_000, min(180_000, int(effective_timeout * 1000)))
        ephemeral_url = f"http://127.0.0.1:{port}{expected_path}"

        try:
            proc = await asyncio.create_subprocess_exec(
                bin_path,
                "scrape",
                "--browser",
                "never",
                "--timeout",
                str(timeout_ms),
                ephemeral_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout + 5.0,
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.communicate()
                raise FirecrawlError(
                    f"Firecrawl Local timed out parsing document '{resolved.name}' after {effective_timeout:.0f}s.",
                    category="timeout",
                    endpoint="local-document",
                ) from exc
        finally:
            server.shutdown()
            server.server_close()
            srv_thread.join(timeout=2.0)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            detail = stderr_text or stdout_text or f"exit code {proc.returncode}"
            raise FirecrawlError(
                f"Firecrawl Local could not parse '{resolved.name}': {detail[:400]}",
                category="invalid_request" if proc.returncode == 1 else "upstream_error",
                endpoint="local-document",
            )

        try:
            parsed_json = json.loads(stdout_text)
        except ValueError as exc:
            raise FirecrawlError(
                f"Firecrawl Local returned invalid JSON while parsing '{resolved.name}': {exc}",
                category="upstream_error",
                endpoint="local-document",
            ) from exc

        if isinstance(parsed_json, dict) and parsed_json.get("success") is False:
            err_msg = str(parsed_json.get("error") or "Document parse failed")
            raise FirecrawlError(
                f"Firecrawl Local failed to parse '{resolved.name}': {err_msg}",
                category="invalid_request",
                endpoint="local-document",
            )

        normalized = self._normalize_scrape_response(
            parsed_json if isinstance(parsed_json, dict) else {"data": {}},
            resolved.name,
        )
        normalized["url"] = resolved.name
        return normalized

    async def batch_scrape(
        self,
        urls: list[str],
        *,
        formats: list[str] | None = None,
        only_main_content: bool = True,
        max_concurrency: int = 8,
        poll_interval: float = 1.0,
        max_wait_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Batch scrape multiple URLs via Firecrawl Local POST /v2/batch/scrape."""
        cfg = self.config
        wait_budget = max_wait_seconds if max_wait_seconds is not None else cfg.crawl_max_wait_seconds
        clean_urls = [u.strip() for u in urls if u.strip()]
        if not clean_urls:
            raise ValueError("At least one URL is required for batch_scrape.")

        payload: dict[str, Any] = {
            "urls": clean_urls,
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
            "maxConcurrency": max(1, min(32, int(max_concurrency))),
        }
        start_resp = await self._request("POST", "/v2/batch/scrape", payload=payload)

        if isinstance(start_resp.get("data"), list) and not start_resp.get("id"):
            pages = self._normalize_crawl_pages(start_resp["data"])
            return {
                "job_id": None,
                "status": "completed",
                "total": len(pages),
                "completed": len(pages),
                "pages": pages,
            }

        job_id = str(start_resp.get("id") or "").strip()
        if not job_id:
            raise FirecrawlError(
                "Firecrawl /v2/batch/scrape did not return a job ID.",
                category="upstream_error",
                endpoint="/v2/batch/scrape",
            )

        status_path = f"/v2/batch/scrape/{job_id}"
        deadline = time.monotonic() + max(2.0, float(wait_budget))
        while True:
            status_resp = await self._request("GET", status_path, retries=1)
            state = str(status_resp.get("status") or "").strip().lower()
            raw_pages = status_resp.get("data") if isinstance(status_resp.get("data"), list) else []
            if state in ("completed", "done"):
                pages = self._normalize_crawl_pages(raw_pages)
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "total": int(status_resp.get("total") or len(pages)),
                    "completed": int(status_resp.get("completed") or len(pages)),
                    "pages": pages,
                    "errors": status_resp.get("errors") or [],
                }
            if state in ("failed", "cancelled", "error"):
                raise FirecrawlError(
                    f"Firecrawl batch scrape job {job_id} ended with status '{state}': "
                    f"{status_resp.get('error') or status_resp.get('errors') or 'unknown error'}",
                    category="upstream_error",
                    endpoint=status_path,
                )
            if time.monotonic() >= deadline:
                pages = self._normalize_crawl_pages(raw_pages)
                return {
                    "job_id": job_id,
                    "status": "partial_timeout",
                    "total": int(status_resp.get("total") or len(pages)),
                    "completed": int(status_resp.get("completed") or len(pages)),
                    "pages": pages,
                }
            await asyncio.sleep(max(0.2, min(5.0, float(poll_interval))))

    async def cancel_batch_scrape(self, job_id: str) -> dict[str, Any]:
        """Cancel a running batch scrape via DELETE /v2/batch/scrape/{id}."""
        return await self._request("DELETE", f"/v2/batch/scrape/{job_id}", retries=0)

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        scrape_content: bool = False,
        lang: str | None = None,
        country: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run web search via Firecrawl Local POST /v2/search."""
        clamped_limit = max(1, min(15, int(limit)))
        payload: dict[str, Any] = {
            "query": query.strip(),
            "limit": clamped_limit,
        }
        if scrape_content:
            payload["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
        if lang:
            payload["lang"] = lang
        if country:
            payload["country"] = country

        resp = await self._request("POST", "/v2/search", payload=payload)
        raw_items = resp.get("data")
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("web") or raw_items.get("results") or []
        if not isinstance(raw_items, list):
            raw_items = []

        results: list[dict[str, Any]] = []
        for item in raw_items[:clamped_limit]:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            item_url = str(item.get("url") or meta.get("sourceURL") or meta.get("url") or "").strip()
            title = str(item.get("title") or meta.get("title") or item_url).strip()
            description = str(item.get("description") or meta.get("description") or "").strip()
            markdown = str(item.get("markdown") or item.get("content") or "").strip()
            if not item_url and not title:
                continue
            results.append(
                {
                    "title": title,
                    "url": item_url,
                    "description": description,
                    "markdown": markdown,
                    "metadata": meta,
                }
            )
        return results

    async def map_urls(
        self,
        url: str,
        *,
        search: str | None = None,
        limit: int = 50,
        include_subdomains: bool = False,
    ) -> list[str]:
        """Discover URLs across a domain via Firecrawl Local POST /v2/map."""
        payload: dict[str, Any] = {
            "url": url,
            "limit": max(1, min(500, int(limit))),
            "includeSubdomains": bool(include_subdomains),
        }
        if search and search.strip():
            payload["search"] = search.strip()

        resp = await self._request("POST", "/v2/map", payload=payload)
        raw_links = resp.get("links")
        if raw_links is None and isinstance(resp.get("data"), list):
            raw_links = resp["data"]
        if not isinstance(raw_links, list):
            return []

        urls: list[str] = []
        for entry in raw_links:
            if isinstance(entry, str) and entry.strip():
                urls.append(entry.strip())
            elif isinstance(entry, dict):
                link_url = str(entry.get("url") or "").strip()
                if link_url:
                    urls.append(link_url)
        return urls

    async def crawl(
        self,
        url: str,
        *,
        limit: int = 5,
        max_depth: int = 2,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        poll_interval: float = 1.0,
        max_wait_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Crawl a site asynchronously via POST /v2/crawl and poll GET /v2/crawl/{id}."""
        cfg = self.config
        wait_budget = max_wait_seconds if max_wait_seconds is not None else cfg.crawl_max_wait_seconds
        clamped_limit = max(1, min(50, int(limit)))
        clamped_depth = max(0, min(5, int(max_depth)))

        payload: dict[str, Any] = {
            "url": url,
            "limit": clamped_limit,
            "maxDiscoveryDepth": clamped_depth,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        }
        if include_paths:
            payload["includePaths"] = [str(p) for p in include_paths if str(p).strip()]
        if exclude_paths:
            payload["excludePaths"] = [str(p) for p in exclude_paths if str(p).strip()]

        start_resp = await self._request("POST", "/v2/crawl", payload=payload)

        if isinstance(start_resp.get("data"), list) and not start_resp.get("id"):
            pages = self._normalize_crawl_pages(start_resp["data"])
            return {
                "job_id": None,
                "status": "completed",
                "total": len(pages),
                "completed": len(pages),
                "pages": pages,
            }

        job_id = str(start_resp.get("id") or "").strip()
        if not job_id:
            raise FirecrawlError(
                "Firecrawl /v2/crawl did not return a crawl job ID.",
                category="upstream_error",
                endpoint="/v2/crawl",
            )

        status_path = f"/v2/crawl/{job_id}"
        deadline = time.monotonic() + max(2.0, float(wait_budget))
        while True:
            status_resp = await self._request("GET", status_path, retries=1)
            state = str(status_resp.get("status") or "").strip().lower()
            raw_pages = status_resp.get("data") if isinstance(status_resp.get("data"), list) else []

            if state in ("completed", "done"):
                pages = self._normalize_crawl_pages(raw_pages)
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "total": int(status_resp.get("total") or len(pages)),
                    "completed": int(status_resp.get("completed") or len(pages)),
                    "pages": pages,
                }
            if state in ("failed", "cancelled", "error"):
                raise FirecrawlError(
                    f"Firecrawl crawl job {job_id} ended with status '{state}': "
                    f"{status_resp.get('error') or status_resp.get('errors') or 'unknown error'}",
                    category="upstream_error",
                    endpoint=status_path,
                )
            if time.monotonic() >= deadline:
                pages = self._normalize_crawl_pages(raw_pages)
                return {
                    "job_id": job_id,
                    "status": "partial_timeout",
                    "total": int(status_resp.get("total") or len(pages)),
                    "completed": int(status_resp.get("completed") or len(pages)),
                    "pages": pages,
                    "note": f"Crawl reached wait limit ({wait_budget:.0f}s); returning {len(pages)} page(s) collected so far.",
                }
            await asyncio.sleep(max(0.2, min(5.0, float(poll_interval))))

    async def cancel_crawl(self, job_id: str) -> dict[str, Any]:
        """Cancel a running crawl via DELETE /v2/crawl/{id}."""
        return await self._request("DELETE", f"/v2/crawl/{job_id}", retries=0)

    async def extract(
        self,
        urls: list[str],
        *,
        prompt: str | None = None,
        schema: dict[str, Any] | None = None,
        poll_interval: float = 1.5,
        max_wait_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Extract structured data from one or more URLs via POST /v2/extract."""
        cfg = self.config
        wait_budget = max_wait_seconds if max_wait_seconds is not None else cfg.crawl_max_wait_seconds
        payload: dict[str, Any] = {"urls": [u.strip() for u in urls if u.strip()]}
        if prompt and prompt.strip():
            payload["prompt"] = prompt.strip()
        if schema:
            payload["schema"] = schema

        start_resp = await self._request("POST", "/v2/extract", payload=payload)
        if "data" in start_resp and not start_resp.get("id"):
            return {
                "job_id": None,
                "status": "completed",
                "data": start_resp.get("data"),
                "sources": start_resp.get("sources") or payload["urls"],
            }

        job_id = str(start_resp.get("id") or "").strip()
        if not job_id:
            return {
                "job_id": None,
                "status": "completed",
                "data": start_resp.get("data"),
                "sources": payload["urls"],
            }

        status_path = f"/v2/extract/{job_id}"
        deadline = time.monotonic() + max(2.0, float(wait_budget))
        while True:
            status_resp = await self._request("GET", status_path, retries=1)
            state = str(status_resp.get("status") or "").strip().lower()
            if state in ("completed", "done") or ("data" in status_resp and state != "processing"):
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "data": status_resp.get("data"),
                    "sources": status_resp.get("sources") or payload["urls"],
                }
            if state in ("failed", "cancelled", "error"):
                raise FirecrawlError(
                    f"Firecrawl extract job {job_id} failed: {status_resp.get('error') or 'unknown error'}",
                    category="upstream_error",
                    endpoint=status_path,
                )
            if time.monotonic() >= deadline:
                raise FirecrawlError(
                    f"Firecrawl extract job {job_id} timed out after {wait_budget:.0f}s.",
                    category="timeout",
                    endpoint=status_path,
                )
            await asyncio.sleep(max(0.2, min(5.0, float(poll_interval))))

    @staticmethod
    def _normalize_crawl_pages(raw_pages: list[Any]) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        for item in raw_pages:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            page_url = str(meta.get("sourceURL") or meta.get("url") or item.get("url") or "").strip()
            title = str(meta.get("title") or item.get("title") or page_url).strip()
            markdown = str(item.get("markdown") or item.get("content") or "").strip()
            pages.append(
                {
                    "url": page_url,
                    "title": title,
                    "markdown": markdown,
                    "status_code": meta.get("statusCode"),
                    "metadata": meta,
                }
            )
        return pages


def parse_optional_json_schema(schema_json: str | None) -> dict[str, Any] | None:
    if not schema_json or not schema_json.strip():
        return None
    parsed = json.loads(schema_json)
    if not isinstance(parsed, dict):
        raise ValueError("schema_json must be a JSON object representing a JSON Schema.")
    return parsed
