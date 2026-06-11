"""LeanExplore tool for searching Lean 4/Mathlib theorems and definitions locally."""

import asyncio
from dataclasses import dataclass, field

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


_lean_explore_service = None
_lean_explore_service_lock: asyncio.Lock = asyncio.Lock()

# Service.search runs synchronous FAISS index search and a torch cross-encoder
# inline; concurrent calls deadlock or produce inconsistent results. Serialize.
_lean_explore_search_sem: asyncio.Semaphore | None = None


async def _get_service():
    """Get or create the global LeanExplore Service (loads models on first call)."""
    global _lean_explore_service

    if _lean_explore_service is not None:
        return _lean_explore_service

    async with _lean_explore_service_lock:
        if _lean_explore_service is not None:
            return _lean_explore_service

        from lean_explore.search import Service

        logger.info("Initializing LeanExplore service (loading models)...")
        _lean_explore_service = await asyncio.to_thread(Service)
        logger.info("LeanExplore service ready")

    return _lean_explore_service


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


async def lean_explore(query: str, config: SearchLeanExploreConfig) -> str:
    """Search for Lean 4/Mathlib theorems via local LeanExplore service."""
    logger.debug(
        f"lean_explore() - query: '{query}', max_results: {config.max_results}, "
        f"rerank_top: {config.rerank_top}, packages: {config.packages}"
    )
    global _lean_explore_search_sem
    if _lean_explore_search_sem is None:
        _lean_explore_search_sem = asyncio.Semaphore(config.max_concurrent_searches)

    try:
        service = await _get_service()
        async with _lean_explore_search_sem:
            response = await service.search(
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


def summarize_for_log(result: str, top_k: int = 3) -> str:
    """Compact rendering of LeanExplore output for the cross-iteration tool log.

    Keeps the first ``top_k`` `•` bullets (name + module) only; drops the
    Source / Doc / Informal blocks which are large. Falls back to first-line
    truncation for non-bullet outputs.
    """
    if not result or "•" not in result:
        return result.strip().splitlines()[0] if result else ""

    bullets: list[str] = []
    for line in result.splitlines():
        if line.startswith("• "):
            bullets.append(line)
            if len(bullets) >= top_k:
                break
    return "\n".join(bullets)


@register_tool(LEAN_EXPLORE_TOOL_TYPE, SearchLeanExploreConfig)
async def create_search_lean_explore_tool(
    config: SearchLeanExploreConfig,
    _: Runtime,
) -> StructuredTool | None:
    """Create a LeanExplore tool. Initializes the service eagerly so model load
    happens up-front rather than during the first agent step."""
    try:
        await _get_service()
    except Exception as e:
        logger.warning(f"LeanExplore initialization failed: {e}")
        return None

    async def _search(query: str) -> str:
        logger.debug(f"LeanExplore tool invoked with query: '{query}'")
        return await lean_explore(query, config)

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
        metadata={"summarize_for_log": summarize_for_log},
    )
