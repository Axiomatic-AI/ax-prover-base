"""LeanSearch tool for searching Lean 4/Mathlib theorems and definitions."""

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import aiohttp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import get_logger
from .registry import register_tool, tool_name_from_type

logger = get_logger(__name__)

LEAN_SEARCH_TOOL_TYPE = "search_lean_search"
DEFAULT_LEAN_SEARCH_URL = "https://leansearch.net"


@dataclass
class SearchLeanSearchConfig:
    """Configuration for LeanSearch tool.

    Default URL is https://leansearch.net. Set server_url to override.
    """

    server_url: str = field(default=DEFAULT_LEAN_SEARCH_URL)
    max_results: int = 6
    timeout: int = 60
    max_retries: int = 3
    retry_delay: int = 2


_lean_search_session: aiohttp.ClientSession | None = None
_lean_search_session_lock: asyncio.Lock = asyncio.Lock()

_lean_search_warmup_result: bool | None = None
_lean_search_warmup_lock: asyncio.Lock = asyncio.Lock()


async def get_lean_search_session() -> aiohttp.ClientSession:
    """Get or create the global LeanSearch session with connection pooling.

    Safe to call from multiple concurrent tasks.
    """
    global _lean_search_session

    if _lean_search_session is not None and not _lean_search_session.closed:
        return _lean_search_session

    async with _lean_search_session_lock:
        if _lean_search_session is not None and not _lean_search_session.closed:
            return _lean_search_session

        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        _lean_search_session = aiohttp.ClientSession(connector=connector)
        logger.debug("Created global ClientSession for LeanSearch")

    return _lean_search_session


@contextlib.asynccontextmanager
async def lean_search_session_manager() -> AsyncIterator[None]:
    """Manage LeanSearch session lifecycle (creation/cleanup)."""
    try:
        yield
    finally:
        global _lean_search_session, _lean_search_warmup_result
        if _lean_search_session is not None and not _lean_search_session.closed:
            await _lean_search_session.close()
            logger.debug("Closed global LeanSearch session")
        _lean_search_session = None
        _lean_search_warmup_result = None


async def _retry_with_backoff(
    attempt: int, config: SearchLeanSearchConfig, error_detail: str
) -> None:
    """Wait with exponential backoff and jitter before retry."""
    wait_time = config.retry_delay * (2**attempt)
    wait_time += random.gauss(0, wait_time * 0.1)
    logger.warning(
        f"LeanSearch: Retry {attempt + 1}/{config.max_retries} after {wait_time:.1f}s - {error_detail}"
    )
    await asyncio.sleep(wait_time)


async def _make_lean_search_request_with_retry(
    query: str,
    config: SearchLeanSearchConfig,
) -> list[list[dict[str, Any]]]:
    """Make async HTTP request to LeanSearch API with retry logic."""
    url = f"{config.server_url}/search"
    session = await get_lean_search_session()

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ax-prover",
    }

    if "axleansearch" in config.server_url:
        from ..utils.google_auth import get_auth_token

        try:
            token = get_auth_token(config.server_url)
            headers["Authorization"] = f"Bearer {token}"
            logger.debug("Added authentication token for axleansearch")
        except Exception as e:
            logger.warning(f"Failed to get auth token, proceeding without auth: {e}")

    payload = {"query": [query], "num_results": config.max_results}
    timeout = aiohttp.ClientTimeout(total=config.timeout)

    for attempt in range(config.max_retries):
        try:
            logger.debug(f"Making request to {url} with payload: {payload}")
            async with session.post(
                url, json=payload, headers=headers, timeout=timeout
            ) as response:
                logger.debug(
                    f"Response status: {response.status}, size: {response.content_length} bytes"
                )
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            error_detail = f"HTTP {e.status}: {e.message}"
            should_retry = e.status == 429

            if should_retry and attempt < config.max_retries - 1:
                await _retry_with_backoff(attempt, config, error_detail)
                continue

            logger.error(f"LeanSearch failed after {attempt + 1} attempts: {error_detail}")
            raise

        except (TimeoutError, aiohttp.ClientError) as e:
            error_detail = f"Connection error: {str(e)}"

            if attempt < config.max_retries - 1:
                await _retry_with_backoff(attempt, config, error_detail)
                continue

            logger.error(f"LeanSearch failed after {attempt + 1} attempts: {error_detail}")
            raise

    raise RuntimeError(f"LeanSearch failed: No data received after {config.max_retries} attempts")


def _process_lean_search_response(
    query: str,
    response_data: list[list[dict[str, Any]]],
) -> str:
    """Process and format LeanSearch API response."""
    logger.debug(
        f"Processing response - type: {type(response_data)}, "
        f"length: {len(response_data) if response_data else 0}"
    )

    if not response_data or not response_data[0]:
        logger.info(f"LeanSearch: No results for '{query}'")
        return f"No results found for: {query}"

    matches = response_data[0]
    logger.info(f"LeanSearch: Found {len(matches)} matches for '{query}'")

    output = [f"=== {query} ({len(matches)} matches) ==="]

    for item in matches:
        result = item.get("result", {})

        name_raw = result.get("name", ["Unknown"])
        name = ".".join(name_raw) if isinstance(name_raw, list) else name_raw

        kind = result.get("kind", "")
        signature = result.get("signature", "")
        docstring = result.get("docstring", "") or ""

        output.append(f"\n• {name} [{kind}]")
        if signature:
            output.append(f"  {signature}")
        if docstring:
            output.append(f"  Doc: {docstring.strip()[:3000]}")

    return "\n".join(output)


async def lean_search(query: str, config: SearchLeanSearchConfig) -> str:
    """Search for Lean 4/Mathlib theorems using module paths or natural language."""
    logger.debug(
        f"lean_search() - server: {config.server_url}, "
        f"max_results: {config.max_results}, timeout: {config.timeout}s"
    )
    try:
        result_data = await _make_lean_search_request_with_retry(query=query, config=config)
        return _process_lean_search_response(query, result_data)
    except aiohttp.ClientError as e:
        logger.error(f"LeanSearch ClientError: {type(e).__name__} - {e}")
        if "127.0.0.1" in config.server_url or "localhost" in config.server_url:
            parsed = urlparse(config.server_url)
            port = parsed.port or 8765
            return (
                f"Cannot connect to LeanSearch server at {config.server_url}. "
                f"Make sure the server is running:\n"
                f"  uvicorn server:app --host 127.0.0.1 --port {port}"
            )
        return f"Cannot connect to LeanSearch server at {config.server_url}"
    except Exception as e:
        logger.error(f"LeanSearch error: {type(e).__name__} - {e}", exc_info=True)
        return str(e)


class SearchQueryInput(BaseModel):
    query: str = Field(..., description="Search query string")


@register_tool(LEAN_SEARCH_TOOL_TYPE, SearchLeanSearchConfig)
async def create_search_lean_search_tool(config: SearchLeanSearchConfig) -> StructuredTool | None:
    """Create a LeanSearch tool with warmup (once per process).

    Returns None if warmup fails.
    """
    global _lean_search_warmup_result

    async with _lean_search_warmup_lock:
        if _lean_search_warmup_result is False:
            return None

        if _lean_search_warmup_result is None:
            try:
                await warmup_lean_search(config)
                _lean_search_warmup_result = True
            except Exception as e:
                logger.warning(f"LeanSearch warm-up failed: {e}")
                _lean_search_warmup_result = False
                return None

    async def _search(query: str) -> str:
        logger.debug(f"LeanSearch tool invoked with query: '{query}'")
        return await lean_search(query, config)

    return StructuredTool(
        name=tool_name_from_type(LEAN_SEARCH_TOOL_TYPE),
        description="""Search for Lean theorems using module paths or natural language.

LeanSearch accepts both precise module paths and natural language descriptions.

Examples of module paths:
- "Mathlib.Analysis.InnerProductSpace.Adjoint"
- "Mathlib.Topology.Basic"
- "Mathlib.Data.Real.Basic"

Examples of natural language:
- "continuity of functions"
- "prime number theorems"
- "adjoint operators in Hilbert spaces"
""",
        coroutine=_search,
        args_schema=SearchQueryInput,
    )


async def warmup_lean_search(config: SearchLeanSearchConfig) -> None:
    """Warm up LeanSearch server with a test query."""
    from dataclasses import replace

    logger.info(f"Warming up LeanSearch server at {config.server_url}...")
    warmup_config = replace(config, timeout=120)
    await _make_lean_search_request_with_retry(query="Nat", config=warmup_config)
    logger.info("LeanSearch warm-up successful")
