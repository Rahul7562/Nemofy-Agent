"""Local Sandbox Tools (Local Hands): bash_exec, read/write/edit_file, list_dir, web_search, web_fetch."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from contextvars import ContextVar
from html import unescape
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool
from markdownify import markdownify as html_to_markdown

from agent_core.sandbox import (
    SandboxExecutionError,
    SandboxManager,
    SandboxSecurityError,
)

current_session_id: ContextVar[str] = ContextVar("sandbox_session_id", default="default")


def html_to_clean_markdown(html: str, *, base_url: str = "") -> str:
    """Convert raw HTML into clean, readable Markdown stripped of scripts/styles/chrome."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "canvas"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split()).strip()

    main_container = soup.find("main") or soup.find("article") or soup.body or soup
    raw_md = html_to_markdown(
        str(main_container),
        heading_style="ATX",
        strip=["img"],
    )

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


def _validate_web_url(url: str, *, allow_private: bool = False) -> str:
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}; only http/https allowed.")
    if not parsed.hostname:
        raise ValueError(f"Invalid URL (missing hostname): {url!r}")
    if parsed.username or parsed.password:
        raise SandboxSecurityError("Access denied: web_fetch URLs must not contain embedded credentials.")

    if not allow_private:
        host = parsed.hostname.lower().rstrip(".")
        if (
            host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "metadata.google.internal")  # noqa: S104
            or host.endswith((".localhost", ".local", ".internal", ".localdomain"))
        ):
            raise SandboxSecurityError(
                f"Access denied: web_fetch cannot target internal or loopback host {host!r}."
            )
        try:
            literal_ip = ipaddress.ip_address(host)
        except ValueError:
            literal_ip = None

        if literal_ip is not None:
            if _is_blocked_web_ip(literal_ip):
                raise SandboxSecurityError(
                    f"Access denied: web_fetch cannot target private/internal IP {host!r}."
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
                        f"Access denied: web_fetch host {host!r} resolves to private/internal IP {resolved_ip}."
                    )

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
    current_url = _validate_web_url(url, allow_private=allow_private)
    headers = {
        "User-Agent": "NemofyAgentCore/2.0 (+https://nemofy.tech; research-fetch)",
        "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5",
    }

    max_redirects = 5
    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=False,
        transport=transport,
    ) as client:
        response: httpx.Response | None = None
        for _hop in range(max_redirects + 1):
            response = await client.get(current_url, headers=headers)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    break
                next_url = urljoin(current_url, location.strip())
                current_url = _validate_web_url(next_url, allow_private=allow_private)
                continue
            break
        else:
            raise httpx.TooManyRedirects(f"Exceeded {max_redirects} redirects while fetching {url!r}.")

        assert response is not None
        if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
            raise httpx.TooManyRedirects(f"Exceeded {max_redirects} redirects while fetching {url!r}.")
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


_DEFAULT_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _unwrap_search_url(raw_href: str) -> str:
    """Extract clean target http(s) URL from DuckDuckGo redirect links or raw hrefs."""
    href = unescape((raw_href or "").strip())
    if not href:
        return ""
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.hostname or "").lower() and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg")
        if uddg and uddg[0]:
            href = unquote(uddg[0])
            parsed = urlparse(href)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return ""
    return href


def _parse_ddg_html_results(html_text: str, max_results: int) -> list[dict[str, str]]:
    """Parse search results from DuckDuckGo's HTML endpoint."""
    soup = BeautifulSoup(html_text, "html.parser")
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for block in soup.select(".result, .web-result"):
        link = block.select_one("a.result__a") or block.select_one("h2 a")
        if not link:
            continue
        url = _unwrap_search_url(str(link.get("href") or ""))
        if not url or url in seen_urls:
            continue
        title = re.sub(r"\s+", " ", link.get_text(" ", strip=True)).strip() or url
        snippet_el = block.select_one(".result__snippet") or block.select_one("a.result__snippet")
        snippet = (
            re.sub(r"\s+", " ", snippet_el.get_text(" ", strip=True)).strip()
            if snippet_el
            else ""
        )
        seen_urls.add(url)
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _parse_ddg_lite_results(html_text: str, max_results: int) -> list[dict[str, str]]:
    """Parse search results from DuckDuckGo's Lite endpoint as a resilient fallback."""
    soup = BeautifulSoup(html_text, "html.parser")
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    links = soup.select("a.result-link")
    snippets = soup.select("td.result-snippet")
    for idx, link in enumerate(links):
        url = _unwrap_search_url(str(link.get("href") or ""))
        if not url or url in seen_urls:
            continue
        title = re.sub(r"\s+", " ", link.get_text(" ", strip=True)).strip() or url
        snippet = ""
        if idx < len(snippets):
            snippet = re.sub(r"\s+", " ", snippets[idx].get_text(" ", strip=True)).strip()
        seen_urls.add(url)
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


async def search_web_results(
    query: str,
    *,
    max_results: int = 5,
    timeout_seconds: float = 15.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[dict[str, str]], str]:
    """Run a live web search and return (results, provider_name).

    Uses TAVILY_API_KEY or BRAVE_SEARCH_API_KEY when configured in environment;
    otherwise uses zero-key DuckDuckGo HTML/Lite search with a Wikipedia OpenSearch fallback.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query must not be empty.")
    clamped = max(1, min(int(max_results), 10))
    headers = {
        "User-Agent": _DEFAULT_SEARCH_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds, connect=8.0),
        follow_redirects=True,
        max_redirects=5,
        headers=headers,
        transport=transport,
    ) as client:
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if tavily_key:
            try:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": clean_query,
                        "max_results": clamped,
                        "search_depth": "basic",
                    },
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    items = [
                        {
                            "title": str(it.get("title") or it.get("url") or "").strip(),
                            "url": str(it.get("url") or "").strip(),
                            "snippet": str(it.get("content") or "").strip(),
                        }
                        for it in (payload.get("results") or [])
                        if isinstance(it, dict)
                        and str(it.get("url") or "").startswith(("http://", "https://"))
                    ][:clamped]
                    if items:
                        return items, "tavily"
            except Exception:
                pass

        brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if brave_key:
            try:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": clean_query, "count": clamped},
                    headers={"X-Subscription-Token": brave_key, "Accept": "application/json"},
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    web_items = (payload.get("web") or {}).get("results") or []
                    items = [
                        {
                            "title": str(it.get("title") or it.get("url") or "").strip(),
                            "url": str(it.get("url") or "").strip(),
                            "snippet": str(it.get("description") or "").strip(),
                        }
                        for it in web_items
                        if isinstance(it, dict)
                        and str(it.get("url") or "").startswith(("http://", "https://"))
                    ][:clamped]
                    if items:
                        return items, "brave"
            except Exception:
                pass

        try:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": clean_query, "kl": "us-en"},
                headers={**headers, "Referer": "https://html.duckduckgo.com/"},
            )
            if resp.status_code == 200 and resp.text:
                parsed = _parse_ddg_html_results(resp.text, clamped)
                if parsed:
                    return parsed, "duckduckgo"
        except Exception:
            pass

        try:
            resp = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": clean_query},
                headers={**headers, "Referer": "https://lite.duckduckgo.com/"},
            )
            if resp.status_code == 200 and resp.text:
                parsed = _parse_ddg_lite_results(resp.text, clamped)
                if parsed:
                    return parsed, "duckduckgo_lite"
        except Exception:
            pass

        try:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": clean_query,
                    "limit": clamped,
                    "namespace": 0,
                    "format": "json",
                },
                headers={"User-Agent": "NemofyAgentCore/2.0", "Accept": "application/json"},
            )
            if resp.status_code == 200:
                arr = resp.json()
                if isinstance(arr, list) and len(arr) >= 4:
                    titles, descs, urls = arr[1], arr[2], arr[3]
                    items = [
                        {
                            "title": str(t or u).strip(),
                            "url": str(u).strip(),
                            "snippet": str(d or "").strip(),
                        }
                        for t, d, u in zip(titles, descs, urls)
                        if isinstance(u, str) and u.startswith(("http://", "https://"))
                    ][:clamped]
                    if items:
                        return items, "wikipedia"
        except Exception:
            pass

    return [], "duckduckgo"


def format_search_results_markdown(
    query: str,
    results: list[dict[str, str]],
    provider: str,
) -> str:
    """Format web search results into compact Markdown for the LLM."""
    if not results:
        return (
            f"Web search for `{query}` returned no results (source={provider}). "
            "Try rephrasing the query or fetch a known URL directly with `web_fetch`."
        )
    lines: list[str] = [f"Web search results for: **{query}** (source: {provider})\n"]
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or item.get("url") or f"Result {idx}"
        url = item.get("url") or ""
        snippet = item.get("snippet") or ""
        lines.append(f"{idx}. **{title}**")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
    lines.append(
        "\nTip: Call `web_fetch(url=...)` on any relevant URL above to read the full page content."
    )
    return "\n".join(lines)


def build_local_tools(
    sandbox: SandboxManager,
    *,
    allow_private_web_fetch: bool = False,
) -> list[BaseTool]:
    """Create the 7 LangChain local execution tools bound to ``sandbox``."""

    @tool("bash_exec")
    async def bash_exec(command: str, timeout: int | None = None) -> str:
        """Execute a shell or Python command inside the isolated workspace sandbox.

        The command starts with the workspace root as its current working directory (CWD) and
        sets the WORKSPACE environment variable to the workspace directory. Inside scripts/code
        you write, prefer relative paths (e.g. 'data.txt') or os.environ.get('WORKSPACE', '.')
        instead of hardcoding literal '/workspace/...' paths.

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

    @tool("read_file")
    def read_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read a UTF-8 text file inside /workspace, optionally sliced by 1-indexed line numbers.

        Args:
            path: File path inside /workspace (e.g. 'app.py' or '/workspace/app.py').
            start_line: Optional 1-indexed starting line number.
            end_line: Optional 1-indexed ending line number (inclusive).
        """
        session_id = current_session_id.get()
        try:
            return sandbox.read_file(
                path,
                session_id=session_id,
                start_line=start_line,
                end_line=end_line,
            )
        except (SandboxSecurityError, FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool("write_file")
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file strictly inside /workspace.

        For very large files (>200 lines), prefer writing modular files or building in chunks
        with write_file + edit_file so tool-call JSON stays compact and never truncates.
        Inside scripts you write, use relative paths (e.g. 'data.txt') or os.environ.get('WORKSPACE', '.')
        rather than hardcoding '/workspace/...'.

        Args:
            path: Target file path inside /workspace (e.g. 'main.py' or '/workspace/main.py').
            content: Full UTF-8 text content to write.
        """
        session_id = current_session_id.get()
        try:
            return sandbox.write_file(path, content, session_id=session_id)
        except (SandboxSecurityError, OSError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool("edit_file")
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace a unique matching snippet inside an existing file in /workspace.

        Args:
            path: Existing file path inside /workspace.
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
        """List files and subdirectories inside /workspace.

        Args:
            path: Directory path inside /workspace (defaults to '/workspace').
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
        """Search the live web for up-to-date information, documentation, news, and URLs.

        Args:
            query: Search query string (keywords or natural-language question).
            max_results: Maximum number of results to return (1-10, defaults to 5).
        """
        try:
            results, provider = await search_web_results(query, max_results=max_results)
            return format_search_results_markdown(query, results, provider)
        except ValueError as exc:
            return f"ERROR: {exc}"
        except httpx.TimeoutException:
            return f"ERROR: web_search timed out for query {query!r}"
        except httpx.HTTPError as exc:
            return f"ERROR: Network error searching for {query!r}: {type(exc).__name__}"

    @tool("web_fetch")
    async def web_fetch(url: str) -> str:
        """Fetch a webpage and extract clean Markdown text.

        Args:
            url: Public http:// or https:// URL to fetch.
        """
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

    return [bash_exec, read_file, write_file, edit_file, list_dir, web_search, web_fetch]
