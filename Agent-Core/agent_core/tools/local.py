"""Local Sandbox & Official Firecrawl Local Tools: bash_exec, file tools, web_search, web_fetch, web_map, web_crawl, web_extract."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, ConfigDict, Field

from agent_core.firecrawl import (
    FirecrawlClient,
    FirecrawlError,
    is_firecrawl_document_path,
    parse_optional_json_schema,
)
from agent_core.sandbox import (
    SandboxExecutionError,
    SandboxManager,
    SandboxSecurityError,
    resolve_workspace_path,
    to_virtual_workspace_path,
)

current_session_id: ContextVar[str] = ContextVar("sandbox_session_id", default="default")


def _is_blocked_web_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    )


def validate_public_url(url: str, *, allow_private: bool = False) -> str:
    """Validate an http(s) URL and block loopback/private/metadata SSRF targets unless explicitly allowed."""
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}; only http/https allowed.")
    if not parsed.hostname:
        raise ValueError(f"Invalid URL (missing hostname): {url!r}")
    if parsed.username or parsed.password:
        raise SandboxSecurityError("Access denied: web URLs must not contain embedded credentials.")

    if not allow_private:
        host = parsed.hostname.lower().rstrip(".")
        if (
            host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "metadata.google.internal")  # noqa: S104
            or host.endswith((".localhost", ".local", ".internal", ".localdomain"))
        ):
            raise SandboxSecurityError(
                f"Access denied: web tools cannot target internal or loopback host {host!r}."
            )
        try:
            literal_ip = ipaddress.ip_address(host)
        except ValueError:
            literal_ip = None

        if literal_ip is not None:
            if _is_blocked_web_ip(literal_ip):
                raise SandboxSecurityError(
                    f"Access denied: web tools cannot target private/internal IP {host!r}."
                )
        else:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except socket.gaierror:
                addr_infos = []
            for _family, _socktype, _proto, _canon, sockaddr in addr_infos:
                try:
                    resolved_ip = ipaddress.ip_address(sockaddr[0])
                except ValueError:
                    continue
                if _is_blocked_web_ip(resolved_ip):
                    raise SandboxSecurityError(
                        f"Access denied: web host {host!r} resolves to private/internal IP {resolved_ip}."
                    )

    return cleaned


_validate_web_url = validate_public_url


def _format_firecrawl_search_markdown(
    query: str,
    results: list[dict[str, object]],
) -> str:
    if not results:
        return (
            f"Firecrawl web search for `{query}` returned no results. "
            "Try refining the search query or use `web_map` / `web_fetch` on a known URL."
        )
    lines: list[str] = [f"Web search results for: **{query}** (source: firecrawl)\n"]
    for idx, item in enumerate(results, start=1):
        title = str(item.get("title") or item.get("url") or f"Result {idx}").strip()
        url = str(item.get("url") or "").strip()
        description = str(item.get("description") or "").strip()
        markdown = str(item.get("markdown") or "").strip()
        lines.append(f"{idx}. **{title}**")
        if url:
            lines.append(f"   Source URL: {url}")
        if description:
            lines.append(f"   Summary: {description}")
        if markdown:
            excerpt = markdown[:700].rstrip()
            if len(markdown) > 700:
                excerpt += " ..."
            lines.append(f"   Excerpt:\n   {excerpt.replace(chr(10), chr(10) + '   ')}")
    lines.append(
        "\nCitation note: Cite the exact Source URL(s) above. Use `web_fetch(url=...)` to read a full page, "
        "`web_map(url=...)` to discover site URLs, or `web_crawl(url=...)` for multi-page crawling."
    )
    return "\n".join(lines)


def _format_firecrawl_scrape_markdown(scraped: dict[str, object]) -> str:
    url = str(scraped.get("url") or "").strip()
    title = str(scraped.get("title") or "").strip()
    description = str(scraped.get("description") or "").strip()
    markdown = str(scraped.get("markdown") or "").strip()
    status_code = scraped.get("status_code")
    engine = str(scraped.get("engine") or "").strip()
    num_pages = scraped.get("num_pages")
    content_type = str(scraped.get("content_type") or "").strip()
    warning = str(scraped.get("warning") or "").strip()

    header_lines: list[str] = [f"Source: {url} (via Firecrawl Local)"]
    if engine:
        header_lines.append(f"Engine: {engine}")
    if title:
        header_lines.append(f"Title: {title}")
    if description:
        header_lines.append(f"Description: {description}")
    if content_type:
        header_lines.append(f"Content-Type: {content_type}")
    if isinstance(num_pages, int):
        header_lines.append(f"Pages: {num_pages}")
    if status_code is not None:
        header_lines.append(f"HTTP Status: {status_code}")
    if warning:
        header_lines.append(f"Warning: {warning}")
    header = "\n".join(header_lines)
    body = markdown if markdown else "(No extractable textual content returned by Firecrawl.)"
    return f"{header}\n\n{body}"


def _format_paginated_document_markdown(
    vpath: str,
    parsed: dict[str, Any],
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    max_lines: int = 250,
    max_chars: int = 24_000,
) -> str:
    """Format Firecrawl-parsed PDF/Office Markdown with deterministic line-range pagination."""
    engine = str(parsed.get("engine") or "firecrawl-local").strip()
    num_pages = parsed.get("num_pages")
    title = str(parsed.get("title") or "").strip()
    content_type = str(parsed.get("content_type") or "").strip()
    warning = str(parsed.get("warning") or "").strip()
    markdown = str(parsed.get("markdown") or "").strip()

    meta_bits = [f"DOCUMENT: {vpath}", f"parser=firecrawl-local/{engine}"]
    if isinstance(num_pages, int):
        meta_bits.append(f"pages={num_pages}")
    if content_type:
        meta_bits.append(f"type={content_type}")
    if title:
        meta_bits.append(f"title={title!r}")

    all_lines = markdown.splitlines()
    total_lines = len(all_lines)
    if total_lines == 0:
        header = f"[{' | '.join(meta_bits)} | 0 extractable lines]"
        if warning:
            return f"{header}\nWarning: {warning}\n\n(No extractable text found in document.)"
        return f"{header}\n\n(No extractable text found in document.)"

    req_start = max(1, int(start_line)) if start_line is not None else 1
    effective_max_lines = max(1, min(1000, int(max_lines)))
    effective_max_chars = max(2_000, min(100_000, int(max_chars)))
    if end_line is not None:
        req_end = min(total_lines, max(req_start, int(end_line)))
    else:
        req_end = total_lines
    target_end = min(req_end, req_start + effective_max_lines - 1)

    if req_start > total_lines:
        return (
            f"[{' | '.join(meta_bits)} | requested start_line={req_start} exceeds "
            f"total_lines={total_lines}. Valid line range is 1-{total_lines}.]"
        )

    line_width = max(4, len(str(total_lines)))
    formatted_lines: list[str] = []
    char_count = 0
    returned_end = req_start - 1
    for lineno in range(req_start, target_end + 1):
        entry = f"{lineno:>{line_width}} | {all_lines[lineno - 1]}"
        if formatted_lines and char_count + len(entry) + 1 > effective_max_chars:
            break
        formatted_lines.append(entry)
        char_count += len(entry) + 1
        returned_end = lineno

    has_more = returned_end < total_lines
    remaining = max(0, total_lines - returned_end)
    status_str = f"has_more=true, next_start_line={returned_end + 1}, remaining_lines={remaining}" if has_more else "EOF reached"
    meta_bits.append(f"lines {req_start}-{returned_end} of {total_lines}")
    meta_bits.append(status_str)
    header = f"[{' | '.join(meta_bits)}]"
    warning_line = f"\nWarning: {warning}" if warning else ""
    footer = (
        f"\n[More document content remains: call read_file(path={vpath!r}, start_line={returned_end + 1}) to continue.]"
        if has_more
        else "\n[End of parsed document]"
    )
    return f"{header}{warning_line}\n" + "\n".join(formatted_lines) + footer


def _strip_json_code_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_llm_json_payload(raw_text: str) -> Any:
    cleaned = _strip_json_code_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_obj = cleaned.find("{")
        start_arr = cleaned.find("[")
        starts = [i for i in (start_obj, start_arr) if i != -1]
        if not starts:
            raise
        start = min(starts)
        end_char = "}" if cleaned[start] == "{" else "]"
        end = cleaned.rfind(end_char)
        if end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _extract_message_text(raw_content: Any) -> str:
    """Safely extract text from LangChain message content (string or structured content blocks)."""
    if isinstance(raw_content, list):
        parts: list[str] = []
        for part in raw_content:
            if isinstance(part, dict):
                text_val = part.get("text")
                if text_val is not None:
                    parts.append(str(text_val))
            elif part is not None:
                parts.append(str(part))
        return "".join(parts)
    return str(raw_content or "")


async def extract_structured_with_llm(
    fetched_pages: list[tuple[str, str]],
    *,
    prompt: str,
    schema: dict[str, Any] | None,
    model: Any,
) -> Any:
    """Extract strict structured JSON from Firecrawl Local scraped Markdown using the session's active LLM."""
    combined_sections = []
    for src_url, md in fetched_pages:
        combined_sections.append(f"=== SOURCE: {src_url} ===\n{md[:12_000]}")
    context_blob = "\n\n".join(combined_sections)[:32_000]

    schema_instruction = (
        f"\nYou MUST conform strictly to this JSON Schema:\n{json.dumps(schema, indent=2)}\n"
        if schema
        else ""
    )
    sys_msg = SystemMessage(
        content=(
            "You are a precise structured data extraction engine. "
            "Extract facts explicitly present in the provided source pages, including any Page Title "
            "and Page Description metadata headers. "
            "Do not fabricate values; use null only if a field is genuinely absent from both metadata and body. "
            "Return ONLY valid JSON with no surrounding commentary or markdown fences."
            + schema_instruction
        )
    )
    user_msg = HumanMessage(
        content=f"Extraction instructions:\n{prompt}\n\nSource pages:\n{context_blob}"
    )

    response = await model.ainvoke([sys_msg, user_msg])
    raw_text = _extract_message_text(getattr(response, "content", response))

    try:
        return _parse_llm_json_payload(raw_text)
    except Exception:  # noqa: BLE001
        repair_msg = HumanMessage(
            content=(
                "Your previous response was not valid JSON. "
                "Output ONLY the corrected valid JSON matching the requested extraction and schema.\n\n"
                f"Previous output:\n{raw_text[:4000]}"
            )
        )
        repaired = await model.ainvoke([sys_msg, repair_msg])
        rep_text = _extract_message_text(getattr(repaired, "content", repaired))
        return _parse_llm_json_payload(rep_text)


class _WebExtractArgs(BaseModel):
    """Input schema for ``web_extract`` using alias ``schema_json`` to avoid Pydantic v2 BaseModel shadowing."""

    model_config = ConfigDict(populate_by_name=True)

    urls: list[str] = Field(
        ...,
        description="List of public http(s) URLs (or wildcard URL patterns) to extract from.",
    )
    prompt: str = Field(
        ...,
        description="Clear natural-language description of the structured facts to extract.",
    )
    json_schema: str | None = Field(
        default=None,
        alias="schema_json",
        description="Optional JSON Schema string constraining the extracted output object.",
    )


def build_local_tools(
    sandbox: SandboxManager,
    *,
    allow_private_web_fetch: bool = False,
    firecrawl_client: FirecrawlClient | None = None,
    extract_model_factory: Callable[[], Any] | None = None,
    include_web_tools: bool = True,
) -> list[BaseTool]:
    """Create local sandbox tools and (optionally) official Firecrawl Local web research tools bound to ``sandbox``."""
    fc_client = firecrawl_client or FirecrawlClient()

    @tool("bash_exec")
    async def bash_exec(command: str, timeout: int | None = None) -> str:
        """Execute a shell or Python command starting inside the workspace directory (/workspace).

        The command starts with the workspace root as its current working directory (CWD) and
        sets the WORKSPACE environment variable to the workspace directory. When
        ALLOW_UNSAFE_LOCAL_BASH is enabled, commands have full unrestricted access to the host
        machine (including /root and host binaries) while still starting in /workspace.

        Args:
            command: Shell command to run with working directory set to the workspace root (e.g. `python3 script.py`).
            timeout: Optional timeout in seconds (defaults to sandbox limit).
        """
        session_id = current_session_id.get()
        try:
            res = await sandbox.exec_command(command, session_id=session_id, timeout=timeout)
            return res.format_output()
        except (SandboxSecurityError, SandboxExecutionError) as exc:
            return f"ERROR: {exc}"

    def _read_file_sync(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_lines: int = 250,
    ) -> str:
        session_id = current_session_id.get()
        try:
            ws = sandbox.get_session_workspace(session_id)
            resolved = resolve_workspace_path(
                path,
                ws,
                allow_host_access=sandbox.allow_unsafe_local_bash,
            )
            if resolved.is_file() and is_firecrawl_document_path(resolved):
                vpath = to_virtual_workspace_path(resolved, ws)
                parsed = asyncio.run(fc_client.parse_local_document(resolved))
                return _format_paginated_document_markdown(
                    vpath,
                    parsed,
                    start_line=start_line,
                    end_line=end_line,
                    max_lines=max_lines,
                )
            page = sandbox.read_file_paginated(
                path,
                session_id=session_id,
                start_line=start_line,
                end_line=end_line,
                max_lines=max_lines,
            )
            return str(page["formatted_output"])
        except (SandboxSecurityError, FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return f"ERROR: {exc}"
        except FirecrawlError as exc:
            return f"ERROR [firecrawl/{exc.category}]: {exc}"

    async def _read_file_async(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_lines: int = 250,
    ) -> str:
        session_id = current_session_id.get()
        try:
            ws = sandbox.get_session_workspace(session_id)
            resolved = resolve_workspace_path(
                path,
                ws,
                allow_host_access=sandbox.allow_unsafe_local_bash,
            )
            if resolved.is_file() and is_firecrawl_document_path(resolved):
                vpath = to_virtual_workspace_path(resolved, ws)
                parsed = await fc_client.parse_local_document(resolved)
                return _format_paginated_document_markdown(
                    vpath,
                    parsed,
                    start_line=start_line,
                    end_line=end_line,
                    max_lines=max_lines,
                )
            page = sandbox.read_file_paginated(
                path,
                session_id=session_id,
                start_line=start_line,
                end_line=end_line,
                max_lines=max_lines,
            )
            return str(page["formatted_output"])
        except (SandboxSecurityError, FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return f"ERROR: {exc}"
        except FirecrawlError as exc:
            return f"ERROR [firecrawl/{exc.category}]: {exc}"

    read_file = StructuredTool.from_function(
        func=_read_file_sync,
        coroutine=_read_file_async,
        name="read_file",
        description=(
            "Read a UTF-8 text file or parse a PDF/Office document (.pdf, .docx, .doc, .xlsx, .xls, "
            ".pptx, .ppt, .odt, .ods, .odp, .rtf, .epub) into numbered, paginated Markdown via official "
            "Firecrawl Local. Accepts relative paths, '/workspace/...', or absolute host paths (including "
            "'/root/...') when ALLOW_UNSAFE_LOCAL_BASH is enabled. Returns explicit pagination metadata "
            "(lines shown, total lines, has_more, next_start_line) without truncating mid-line."
        ),
    )

    @tool("write_file")
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file (defaults to /workspace; accepts host paths when ALLOW_UNSAFE_LOCAL_BASH is enabled).

        For very large files (>200 lines), prefer writing modular files or building in chunks
        with write_file + edit_file so tool-call JSON stays compact and never truncates.
        Inside scripts you write, use relative paths (e.g. 'data.txt') or os.environ.get('WORKSPACE', '.')
        rather than hardcoding '/workspace/...'.

        Args:
            path: Target file path (e.g. 'main.py', '/workspace/main.py', or host path when enabled).
            content: Full UTF-8 text content to write.
        """
        session_id = current_session_id.get()
        try:
            return sandbox.write_file(path, content, session_id=session_id)
        except (SandboxSecurityError, OSError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool("edit_file")
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace a unique matching snippet inside an existing file (defaults to /workspace; accepts host paths when ALLOW_UNSAFE_LOCAL_BASH is enabled).

        Args:
            path: Existing file path (relative to /workspace, '/workspace/...', or host path when enabled).
            old_text: Exact unique substring to replace.
            new_text: Replacement text.
        """
        session_id = current_session_id.get()
        try:
            return sandbox.edit_file(path, old_text, new_text, session_id=session_id)
        except (SandboxSecurityError, FileNotFoundError, ValueError, OSError) as exc:
            return f"ERROR: {exc}"

    @tool("list_dir")
    def list_dir(path: str = "/workspace") -> str:
        """List files and subdirectories (defaults to '/workspace'; accepts host paths such as '/root' when ALLOW_UNSAFE_LOCAL_BASH is enabled).

        Args:
            path: Directory path to list (defaults to '/workspace').
        """
        session_id = current_session_id.get()
        try:
            entries = sandbox.list_dir(path, session_id=session_id)
            if not entries:
                return f"(empty directory: {path})"
            return "\n".join(entries)
        except (SandboxSecurityError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool("web_search")
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the live web using official Firecrawl Local (/v2/search).

        Args:
            query: Search query string (keywords or natural-language question).
            max_results: Maximum number of results to return (1-15, defaults to 5).
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return "ERROR: Search query must not be empty."
        try:
            results = await fc_client.search(clean_query, limit=max_results, scrape_content=False)
            return _format_firecrawl_search_markdown(clean_query, results)
        except FirecrawlError as exc:
            return f"ERROR [firecrawl/{exc.category}]: {exc}"
        except ValueError as exc:
            return f"ERROR: {exc}"

    @tool("web_fetch")
    async def web_fetch(
        url: str,
        only_main_content: bool = True,
        wait_for_ms: int | None = None,
    ) -> str:
        """Scrape a webpage, remote PDF, or remote Office document via official Firecrawl Local (/v2/scrape).

        Natively parses HTML (with automatic Chromium JS rendering when needed), PDFs (.pdf via
        pdf-inspector), and Office/ebook documents (.docx, .doc, .xlsx, .xls, .pptx, .ppt, .odt,
        .ods, .odp, .rtf, .epub via anydoc) into clean GitHub-Flavored Markdown.

        Args:
            url: Public http:// or https:// URL to a webpage, PDF, or Office document.
            only_main_content: When True, strip site chrome and focus on primary article/doc content.
            wait_for_ms: Optional milliseconds to wait after page load in Chromium for JS-heavy SPAs.
        """
        try:
            validate_public_url(url, allow_private=allow_private_web_fetch)
            scraped = await fc_client.scrape(
                url,
                only_main_content=only_main_content,
                wait_for=wait_for_ms,
            )
            return _format_firecrawl_scrape_markdown(scraped)
        except (SandboxSecurityError, ValueError) as exc:
            return f"ERROR: {exc}"
        except FirecrawlError as exc:
            return f"ERROR [firecrawl/{exc.category}]: {exc}"

    @tool("web_map")
    async def web_map(url: str, search: str | None = None, limit: int = 50) -> str:
        """Discover and map URLs across a website using official Firecrawl Local (/v2/map).

        Args:
            url: Public root or section http(s) URL to map.
            search: Optional keyword filter to prioritize matching URLs.
            limit: Maximum number of URLs to return (1-500, defaults to 50).
        """
        try:
            validate_public_url(url, allow_private=allow_private_web_fetch)
            links = await fc_client.map_urls(url, search=search, limit=limit)
            if not links:
                return f"Firecrawl map for {url} discovered 0 URLs."
            lines = [f"Discovered {len(links)} URL(s) under {url} (source: firecrawl/map):"]
            for idx, link in enumerate(links, start=1):
                lines.append(f"{idx}. {link}")
            return "\n".join(lines)
        except (SandboxSecurityError, ValueError) as exc:
            return f"ERROR: {exc}"
        except FirecrawlError as exc:
            return f"ERROR [firecrawl/{exc.category}]: {exc}"

    @tool("web_crawl")
    async def web_crawl(
        url: str,
        limit: int = 5,
        max_depth: int = 2,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> str:
        """Crawl multiple linked pages under a website using official Firecrawl Local (/v2/crawl).

        Args:
            url: Public starting http(s) URL to crawl.
            limit: Maximum pages to crawl (1-50, defaults to 5).
            max_depth: Maximum discovery depth from the starting URL (0-5, defaults to 2).
            include_paths: Optional list of URL path regex patterns to include.
            exclude_paths: Optional list of URL path regex patterns to exclude.
        """
        try:
            validate_public_url(url, allow_private=allow_private_web_fetch)
            res = await fc_client.crawl(
                url,
                limit=limit,
                max_depth=max_depth,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
            )
            pages = res.get("pages") or []
            header = (
                f"Firecrawl crawl of {url} — status={res.get('status')} "
                f"({len(pages)} page(s) returned, completed={res.get('completed')}/{res.get('total')})"
            )
            if res.get("note"):
                header += f"\nNote: {res['note']}"
            if not pages:
                return f"{header}\n\n(No crawlable pages returned.)"
            sections: list[str] = [header]
            for idx, page in enumerate(pages, start=1):
                p_url = str(page.get("url") or url)
                p_title = str(page.get("title") or p_url)
                p_md = str(page.get("markdown") or "").strip()
                excerpt = p_md[:4000].rstrip()
                if len(p_md) > 4000:
                    excerpt += "\n... [page excerpt truncated; use web_fetch on this exact URL for full page]"
                sections.append(f"## {idx}. {p_title}\nSource URL: {p_url}\n\n{excerpt}")
            return "\n\n---\n\n".join(sections)
        except (SandboxSecurityError, ValueError) as exc:
            return f"ERROR: {exc}"
        except FirecrawlError as exc:
            return f"ERROR [firecrawl/{exc.category}]: {exc}"

    async def _web_extract(
        urls: list[str],
        prompt: str,
        json_schema: str | None = None,
    ) -> str:
        schema_json = json_schema
        if not urls:
            return "ERROR: At least one URL is required for web_extract."

        cleaned_urls: list[str] = []
        try:
            for u in urls:
                base_check = u.replace("/*", "")
                validate_public_url(base_check, allow_private=allow_private_web_fetch)
                cleaned_urls.append(base_check)
            schema = parse_optional_json_schema(schema_json)
        except (SandboxSecurityError, ValueError) as exc:
            return f"ERROR: {exc}"

        try:
            res = await fc_client.extract(urls=urls, prompt=prompt, schema=schema)
            sources = res.get("sources") or urls
            data_str = json.dumps(res.get("data"), indent=2, ensure_ascii=False, default=str)
            src_lines = "\n".join(f"- {s}" for s in sources)
            return (
                f"Firecrawl structured extraction (status={res.get('status')}):\n"
                f"Sources:\n{src_lines}\n\n"
                f"Extracted Data:\n```json\n{data_str}\n```"
            )
        except FirecrawlError as exc:
            # On keyless firecrawl-local, /v2/extract returns 401 or 501 CLOUD_ONLY.
            # Scrape via firecrawl-local /v2/scrape and extract structured JSON with the session LLM.
            if exc.category not in ("cloud_only", "auth_error", "payment_required"):
                return f"ERROR [firecrawl/{exc.category}]: {exc}"

            scraped_pages: list[tuple[str, str]] = []
            for target in cleaned_urls[:5]:
                try:
                    scraped = await fc_client.scrape(target, only_main_content=True)
                    md = str(scraped.get("markdown") or "").strip()
                    if not md:
                        # Retry full-page scrape when main-content extraction stripped everything
                        scraped_full = await fc_client.scrape(target, only_main_content=False)
                        if str(scraped_full.get("markdown") or "").strip():
                            scraped = scraped_full
                            md = str(scraped.get("markdown") or "").strip()
                    meta_obj = scraped.get("metadata") if isinstance(scraped.get("metadata"), dict) else {}
                    title = str(scraped.get("title") or meta_obj.get("title") or "").strip()
                    desc = str(scraped.get("description") or meta_obj.get("description") or "").strip()
                    meta_lines: list[str] = []
                    if title:
                        meta_lines.append(f"Page Title: {title}")
                    if desc:
                        meta_lines.append(f"Page Description: {desc}")
                    if meta_lines or md:
                        body_parts = ["\n".join(meta_lines)] if meta_lines else []
                        if md:
                            body_parts.append(md)
                        scraped_pages.append(
                            (str(scraped.get("url") or target), "\n\n".join(body_parts))
                        )
                except FirecrawlError as scrape_exc:
                    return f"ERROR [firecrawl/{scrape_exc.category}]: {scrape_exc}"

            if not scraped_pages:
                return f"ERROR [firecrawl/{exc.category}]: {exc}"

            active_model = None
            if extract_model_factory is not None:
                try:
                    active_model = extract_model_factory()
                except Exception:  # noqa: BLE001
                    active_model = None

            if active_model is not None:
                try:
                    extracted_obj = await extract_structured_with_llm(
                        scraped_pages,
                        prompt=prompt,
                        schema=schema,
                        model=active_model,
                    )
                    data_str = json.dumps(extracted_obj, indent=2, ensure_ascii=False, default=str)
                    src_lines = "\n".join(f"- {u}" for u, _ in scraped_pages)
                    return (
                        f"Structured extraction (source: firecrawl-local+llm, status=completed):\n"
                        f"Sources:\n{src_lines}\n\n"
                        f"Extracted Data:\n```json\n{data_str}\n```"
                    )
                except Exception as llm_exc:  # noqa: BLE001
                    return f"ERROR [firecrawl-local+llm]: Structured extraction failed: {llm_exc}"

            schema_section = f"\nRequested Schema:\n```json\n{schema_json}\n```\n" if schema_json else ""
            blocks = [f"### Source: {u}\n\n{md}" for u, md in scraped_pages]
            return (
                f"Firecrawl Local scraped source Markdown (source: firecrawl-local/scrape):\n\n"
                f"Extraction Prompt: {prompt}\n"
                f"{schema_section}\n"
                + "\n\n---\n\n".join(blocks)
            )

    web_extract = StructuredTool.from_function(
        coroutine=_web_extract,
        name="web_extract",
        description=(
            "Extract structured JSON data from one or more public URLs via Firecrawl Local. "
            "Uses /v2/extract when Firecrawl Cloud credentials are configured; when running keyless "
            "on Firecrawl Local, scrapes pages via /v2/scrape and structures JSON using the active session LLM."
        ),
        args_schema=_WebExtractArgs,
    )

    workspace_tools: list[BaseTool] = [
        bash_exec,
        read_file,
        write_file,
        edit_file,
        list_dir,
    ]
    if not include_web_tools:
        return workspace_tools
    return [
        *workspace_tools,
        web_search,
        web_fetch,
        web_map,
        web_crawl,
        web_extract,
    ]
