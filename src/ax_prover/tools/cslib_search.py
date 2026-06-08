"""CSLib search tool.

Searches the vendored CSLib dependency (under `.lake/packages/cslib`), which the local
project search excludes and the Mathlib-only remote LeanSearch tool does not cover.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import get_logger
from .local_lean_search import (
    _search_root,
    _walk_down_for_roots,
    _walk_up_for_root,
)
from .registry import register_tool, tool_name_from_type

logger = get_logger(__name__)

CSLIB_SEARCH_TOOL_TYPE = "search_cslib"


@dataclass
class SearchCslibConfig:
    """Configuration for the CSLib search tool."""

    max_results: int = 6
    max_chars: int = 4000
    package_subpath: str = ".lake/packages/cslib"


class CslibSearcher:
    """Searches the CSLib package, resolved relative to the project's lake root."""

    def __init__(self, config: SearchCslibConfig, base_folder: str = "."):
        self.config = config
        self.base_folder = base_folder
        self._resolution: tuple[Path | None, str] | None = None

    def _resolve_cslib(self) -> tuple[Path | None, str]:
        if self._resolution is None:
            self._resolution = self._compute_cslib()
        return self._resolution

    def _compute_cslib(self) -> tuple[Path | None, str]:
        start = Path(self.base_folder).resolve()
        root = _walk_up_for_root(start)
        if root is None:
            down = _walk_down_for_roots(start)
            root = down[0] if len(down) == 1 else None
        if root is None:
            return None, f"CSLib search unavailable: no lakefile found at or under {start}."
        cslib = root / self.config.package_subpath
        if not cslib.is_dir():
            return None, (
                f"CSLib search unavailable: '{self.config.package_subpath}' not found under {root}."
            )
        return cslib, ""

    def search(self, query: str) -> str:
        query = query.strip()
        logger.debug(f"CslibSearch tool invoked with query: '{query}'")
        if not query:
            return "Please provide a non-empty keyword to search for."
        cslib, error = self._resolve_cslib()
        if cslib is None:
            logger.warning(f"CslibSearch: {error}")
            return error
        text, _decls = _search_root(cslib, query, self.config, label="CslibSearch")
        return text


class CslibSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Keyword(s) to match against CSLib declaration names (case-insensitive). "
            "Multiple words match names containing all of them, e.g. 'WellFounded ofTransGen'."
        ),
    )


@register_tool(CSLIB_SEARCH_TOOL_TYPE, SearchCslibConfig)
def create_search_cslib_tool(config: SearchCslibConfig, base_folder: str = ".") -> StructuredTool:
    """Create the CSLib search tool, scoped to the project's vendored cslib package."""
    searcher = CslibSearcher(config, base_folder=base_folder)
    return StructuredTool(
        name=tool_name_from_type(CSLIB_SEARCH_TOOL_TYPE),
        description="""Search the CSLib library for declarations by name.

Returns the full source of CSLib `def`/`theorem`/`lemma`/`structure`/etc. whose name
contains your keyword (case-insensitive). CSLib is a dependency the local project search
does not cover and the Mathlib LeanSearch tool does not index.

Pass a single keyword or multiple words (which must all appear in the qualified name, e.g.
"WellFounded ofTransGen"). If no name matches, falls back to matching identifiers used in
declaration bodies.""",
        func=searcher.search,
        args_schema=CslibSearchInput,
    )
