"""LeanExplore tool for searching Lean 4/Mathlib theorems and definitions locally."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..runtime import Runtime
from ..utils import get_logger
from .registry import register_tool, tool_name_from_type

logger = get_logger(__name__)

LEAN_EXPLORE_TOOL_TYPE = "search_lean_explore"


@dataclass
class SearchLeanExploreConfig:
    """Configuration for LeanExplore tool."""

    max_results: int = 6
    rerank_top: int = 50
    packages: list[str] = field(default_factory=lambda: ["Mathlib"])
    max_concurrent_searches: int = 1


@dataclass
class LeanExploreResources:
    """Loaded LeanExplore service plus a semaphore serializing its searches.

    Service.search runs a synchronous FAISS index search and a torch
    cross-encoder inline; concurrent calls deadlock or produce inconsistent
    results, so all searches share a single semaphore.
    """

    service: Any
    search_semaphore: asyncio.Semaphore


@asynccontextmanager
async def _lean_explore_lifespan(
    config: SearchLeanExploreConfig,
) -> AsyncIterator[LeanExploreResources | None]:
    """Load the LeanExplore service once for the runtime.

    Yields None if the optional ``lean-explore`` package or its models are
    unavailable, so the tool is skipped rather than aborting the whole run.
    """
    try:
        from lean_explore.search import Service

        logger.info("Initializing LeanExplore service (loading models)...")
        service = await asyncio.to_thread(Service)
        logger.info("LeanExplore service ready")
    except Exception as e:
        logger.warning(f"LeanExplore initialization failed: {e}")
        yield None
        return

    yield LeanExploreResources(
        service=service,
        search_semaphore=asyncio.Semaphore(config.max_concurrent_searches),
    )


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _format_response(query: str, response) -> str:
    if not response.results:
        return f"No results found for: {query}"

    output = [f"=== {query} ({len(response.results)} matches) ==="]

    for r in response.results:
        output.append(f"\n• {r.name}  [{r.module}]")
        source = _truncate(r.source_text, 800)
        if source:
            output.append(f"  Source:\n    {source.replace(chr(10), chr(10) + '    ')}")
        if r.docstring:
            output.append(f"  Doc: {_truncate(r.docstring, 600)}")
        if r.informalization:
            output.append(f"  Informal: {_truncate(r.informalization, 600)}")

    return "\n".join(output)


async def lean_explore(
    query: str, config: SearchLeanExploreConfig, resources: LeanExploreResources
) -> str:
    """Search for Lean 4/Mathlib theorems via the local LeanExplore service."""
    logger.debug(
        f"lean_explore() - query: '{query}', max_results: {config.max_results}, "
        f"rerank_top: {config.rerank_top}, packages: {config.packages}"
    )
    try:
        async with resources.search_semaphore:
            response = await resources.service.search(
                query=query,
                limit=config.max_results,
                rerank_top=config.rerank_top,
                packages=config.packages,
            )
        return _format_response(query, response)
    except Exception as e:
        logger.error(f"LeanExplore error: {type(e).__name__} - {e}", exc_info=True)
        return f"LeanExplore error: {e}"


class SearchQueryInput(BaseModel):
    query: str = Field(..., description="Search query string")


@register_tool(LEAN_EXPLORE_TOOL_TYPE, SearchLeanExploreConfig, _lean_explore_lifespan)
def create_search_lean_explore_tool(
    config: SearchLeanExploreConfig, runtime: Runtime
) -> StructuredTool | None:
    """Create a LeanExplore tool, or None if the service failed to load."""
    resources = runtime.get_tool_resources(LEAN_EXPLORE_TOOL_TYPE)
    if resources is None:
        return None

    async def _search(query: str) -> str:
        logger.debug(f"LeanExplore tool invoked with query: '{query}'")
        return await lean_explore(query, config, resources)

    return StructuredTool(
        name=tool_name_from_type(LEAN_EXPLORE_TOOL_TYPE),
        description="""Search for Lean 4/Mathlib theorems and definitions using LeanExplore (local semantic search).

Accepts natural language descriptions or partial names. Returns matches with module path,
source code, docstring, and an informal mathematical description.

Examples:
- "Finset filter card sum"
- "continuity of composition"
- "adjoint operators in Hilbert spaces"
""",
        coroutine=_search,
        args_schema=SearchQueryInput,
    )
