"""Loogle tool for searching Lean 4/Mathlib by signature/name patterns."""

from dataclasses import dataclass

import aiohttp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..runtime import Runtime
from ..utils import get_logger
from .registry import register_tool, tool_name_from_type

logger = get_logger(__name__)

LOOGLE_TOOL_TYPE = "search_loogle"
DEFAULT_LOOGLE_URL = "http://localhost:8088"


class LoogleNotReadyError(RuntimeError):
    """Raised when the Loogle backend is still starting up."""


@dataclass
class SearchLoogleConfig:
    """Configuration for Loogle tool."""

    server_url: str = DEFAULT_LOOGLE_URL
    max_results: int = 6
    timeout: int = 60


_LOADING_MARKER = "starting up"


async def _query_loogle(query: str, config: SearchLoogleConfig) -> dict:
    timeout = aiohttp.ClientTimeout(total=config.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{config.server_url}/json", params={"q": query}) as resp:
            resp.raise_for_status()
            return await resp.json()


def _format_response(query: str, data: dict, max_results: int) -> str:
    if not data:
        return f"No results for: {query}"

    error = data.get("error")
    if error:
        if _LOADING_MARKER in error.lower():
            raise LoogleNotReadyError(
                "Loogle backend is still starting up. Please wait a minute or two for "
                "Loogle to finish loading and try again."
            )
        suggestions = data.get("suggestions") or []
        msg = f"Loogle error: {error}"
        if suggestions:
            msg += f"\nSuggestions: {', '.join(suggestions)}"
        return msg

    hits = data.get("hits") or []
    if not hits:
        return f"No results for: {query}"

    count = data.get("count", len(hits))
    shown = min(len(hits), max_results)
    output = [f"=== {query} ({count} total, showing {shown}) ==="]
    header = (data.get("header") or "").strip()
    if header:
        output.append(header)

    for h in hits[:max_results]:
        name = h.get("name", "?")
        module = h.get("module", "")
        type_str = (h.get("type") or "").strip()
        doc = (h.get("doc") or "").strip()

        output.append(f"\n• {name}  [{module}]")
        if type_str:
            output.append(f"  : {type_str}")
        if doc:
            if len(doc) > 400:
                doc = doc[:400].rstrip() + "…"
            output.append(f"  Doc: {doc}")

    return "\n".join(output)


async def loogle_search(query: str, config: SearchLoogleConfig) -> str:
    logger.debug(f"loogle_search() - server: {config.server_url}, query: '{query}'")
    try:
        data = await _query_loogle(query, config)
    except aiohttp.ClientError as e:
        logger.error(f"Loogle connection error: {type(e).__name__} - {e}")
        return (
            f"Cannot connect to Loogle server at {config.server_url}. "
            "Make sure the local Loogle server is running."
        )
    return _format_response(query, data, config.max_results)


class SearchQueryInput(BaseModel):
    query: str = Field(..., description="Loogle query (see tool description for syntax)")


_DESCRIPTION = """Search Lean 4/Mathlib lemmas via Loogle. The query is NOT free text — it must follow Loogle's syntax.

Five filter kinds (combine with commas; results match ALL):

1. By constant — name an identifier the lemma must mention:
     Real.sin
     Real.sin, Real.cos
2. By name substring — a quoted string that must appear in the lemma name:
     "differ"
3. By subexpression — a pattern with `_` for anything and `?a` for named metavariables:
     _ * (_ ^ _)
     Real.sqrt ?a * Real.sqrt ?a
4. By main conclusion — prefix `|-`; matches the conclusion (or any hypothesis if it doesn't match the conclusion):
     |- tsum _ = _ * tsum _
5. By type — restrict to definitions vs theorems:
     |- (_ : Type _)    -- definitions
     |- (_ : Prop)      -- theorems

Examples of combined queries:
  Real.sin, Real.cos, "add"
  Real.sqrt ?a * Real.sqrt ?a
  List, |- _ = _ ++ _

Returns each hit's fully-qualified name, module, type signature, and (when present) docstring."""


@register_tool(LOOGLE_TOOL_TYPE, SearchLoogleConfig)
def create_search_loogle_tool(config: SearchLoogleConfig, _: Runtime) -> StructuredTool:
    async def _search(query: str) -> str:
        logger.debug(f"Loogle tool invoked with query: '{query}'")
        return await loogle_search(query, config)

    return StructuredTool(
        name=tool_name_from_type(LOOGLE_TOOL_TYPE),
        description=_DESCRIPTION,
        coroutine=_search,
        args_schema=SearchQueryInput,
    )
