"""Phase 4: Local Sandbox Tools (Local Hands).

Implements the six required local tools bound to a SandboxManager:
- bash_exec(command)
- read_file(path)
- write_file(path, content)
- edit_file(path, old_text, new_text)
- list_dir(path)
- web_fetch(url)

All file operations are strictly confined to /workspace; web_fetch converts
HTML into clean, compact Markdown instead of dumping raw HTML into context.
"""

from __future__ import annotations

import ipaddress
import re
from contextvars import ContextVar
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool
from markdownify import markdownify as html_to_markdown

from src.sandbox import (
    SandboxExecutionError,
    SandboxManager,
    SandboxSecurityError,
)

# Per-task ContextVar so concurrent LangGraph sessions automatically target their own sandbox session
current_session_id: ContextVar[str] = ContextVar("sandbox_session_id", default="default")


def html_to_clean_markdown(html: str, *, base_url: str = "") -> str:
    """Convert raw HTML into clean, readable Markdown stripped of scripts/styles/chrome."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "canvas"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split()).strip()

    # Prefer main/article content if present and substantive
    main_container = soup.find("main") or soup.find("article") or soup.body or soup
    raw_md = html_to_markdown(
        str(main_container),
        heading_style="ATX",
        strip=["img"],
    )

    # Collapse excessive blank lines and trailing spaces
    cleaned_lines = [line.rstrip() for line in raw_md.splitlines()]
    markdown = "\n".join(cleaned_lines)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()

    header_lines: list[str] = []
    if title:
        header_lines.append(f"# {title}")
    if base_url:
        header_lines.append(f"Source: {base_url}")
    if header_lines:
        return "\n".join(header_lines) + "\n\n" + markdown
    return markdown


def _validate_web_url(url: str, *, allow_private: bool = False) -> str:
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}; only http/https allowed.")
    if not parsed.hostname:
        raise ValueError(f"Invalid URL (missing hostname): {url!r}")

    if not allow_private:
        host = parsed.hostname.lower()
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):  # noqa: S104
            raise SandboxSecurityError(
                f"Access denied: web_fetch cannot target loopback address {host!r}."
            )
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise SandboxSecurityError(
                    f"Access denied: web_fetch cannot target private/internal IP {host!r}."
                )
        except ValueError:
            pass

    return cleaned


async def fetch_webpage_markdown(
    url: str,
    *,
    timeout_seconds: float = 20.0,
    max_chars: int = 20_000,
    allow_private: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Fetch a webpage and return clean Markdown or plain text."""
    validated_url = _validate_web_url(url, allow_private=allow_private)
    headers = {
        "User-Agent": "AutonomousLangGraphAgent/1.0 (+https://localhost; research-fetch)",
        "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5",
    }

    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=True,
        max_redirects=5,
        transport=transport,
    ) as client:
        response = await client.get(validated_url, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        body = response.text

    if "html" in content_type or "<html" in body[:500].lower():
        extracted = html_to_clean_markdown(body, base_url=str(response.url))
    else:
        extracted = f"Source: {response.url}\n\n{body.strip()}"

    if len(extracted) > max_chars:
        extracted = (
            extracted[:max_chars].rstrip()
            + f"\n\n[Truncated at {max_chars:,} characters from {len(extracted):,} total]"
        )
    return extracted


def build_local_tools(
    sandbox: SandboxManager,
    *,
    allow_private_web_fetch: bool = False,
) -> list[BaseTool]:
    """Create the 6 LangChain local execution tools bound to ``sandbox``."""

    @tool("bash_exec")
    async def bash_exec(command: str, timeout: int | None = None) -> str:
        """Execute a shell or Python command inside the isolated /workspace sandbox."""
        session_id = current_session_id.get()
        try:
            res = await sandbox.exec_command(command, session_id=session_id, timeout=timeout)
            return res.format_output()
        except (SandboxSecurityError, SandboxExecutionError) as exc:
            return f"ERROR: {exc}"

    @tool("read_file")
    def read_file(path: str) -> str:
        """Read a UTF-8 text file strictly inside /workspace."""
        session_id = current_session_id.get()
        try:
            return sandbox.read_file(path, session_id=session_id)
        except (SandboxSecurityError, FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool("write_file")
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file strictly inside /workspace."""
        session_id = current_session_id.get()
        try:
            return sandbox.write_file(path, content, session_id=session_id)
        except (SandboxSecurityError, OSError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool("edit_file")
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace a unique matching snippet inside an existing file in /workspace."""
        session_id = current_session_id.get()
        try:
            return sandbox.edit_file(path, old_text, new_text, session_id=session_id)
        except (SandboxSecurityError, FileNotFoundError, ValueError, OSError) as exc:
            return f"ERROR: {exc}"

    @tool("list_dir")
    def list_dir(path: str = "/workspace") -> str:
        """List files and subdirectories inside /workspace."""
        session_id = current_session_id.get()
        try:
            entries = sandbox.list_dir(path, session_id=session_id)
            if not entries:
                return f"(empty directory: {path})"
            return "\n".join(entries)
        except (SandboxSecurityError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool("web_fetch")
    async def web_fetch(url: str) -> str:
        """Fetch a webpage and extract clean Markdown text (stripped of scripts and raw HTML)."""
        try:
            return await fetch_webpage_markdown(
                url,
                allow_private=allow_private_web_fetch,
            )
        except (SandboxSecurityError, ValueError) as exc:
            return f"ERROR: {exc}"
        except httpx.TimeoutException:
            return f"ERROR: web_fetch timed out requesting {url}"
        except httpx.HTTPStatusError as exc:
            return f"ERROR: HTTP {exc.response.status_code} fetching {url}"
        except httpx.HTTPError as exc:
            return f"ERROR: Network error fetching {url}: {type(exc).__name__}"

    return [bash_exec, read_file, write_file, edit_file, list_dir, web_fetch]
