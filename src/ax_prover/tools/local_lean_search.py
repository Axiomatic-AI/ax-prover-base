"""Local Lean library search tool.

Searches the .lean files of the project being proven (not Mathlib / .lake
dependencies) and returns the full declaration blocks whose names match a
keyword. Complements the remote `lean_search` tool, which covers Mathlib.
"""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..models.declaration import DeclarationType
from ..utils import get_logger
from ..utils.lean_parsing import (
    DECL_NAME_END,
    LEAN_KEYWORDS,
    extract_function_from_content,
    list_all_declarations_in_lean_code,
)
from .registry import register_tool, tool_name_from_type

logger = get_logger(__name__)

LOCAL_LEAN_SEARCH_TOOL_TYPE = "search_lean_local"

# Files that mark a Lean project (lake) root.
LAKE_ROOT_MARKERS = ("lakefile.toml", "lakefile.lean", "lake-manifest.json")

# Build artifacts + vendored dependencies (Mathlib) live here; never searched.
EXCLUDED_DIR = ".lake"

# Declaration kinds worth returning. Excludes structural keywords (open, end,
# namespace, section, import) that DeclarationType also enumerates.
SEARCHABLE_TYPES = frozenset(
    {
        DeclarationType.Definition,
        DeclarationType.Theorem,
        DeclarationType.Lemma,
        DeclarationType.Instance,
        DeclarationType.Structure,
        DeclarationType.Class,
        DeclarationType.Inductive,
        DeclarationType.Axiom,
        DeclarationType.Abbrev,
        DeclarationType.NoncomputableDef,
        DeclarationType.NoncomputableAbbrev,
    }
)


@dataclass
class SearchLeanLocalConfig:
    """Configuration for the local Lean search tool."""

    max_results: int = 6
    max_chars: int = 4000


def _has_lake_marker(directory: Path) -> bool:
    return any((directory / marker).exists() for marker in LAKE_ROOT_MARKERS)


def _walk_up_for_root(start: Path) -> Path | None:
    """Nearest ancestor (including `start`) containing a lake marker, or None."""
    for directory in (start, *start.parents):
        if _has_lake_marker(directory):
            return directory
    return None


def _walk_down_for_roots(start: Path) -> list[Path]:
    """Lake-project directories at or below `start`, skipping `.lake/`.

    Does not descend into a project once found (nested lake projects are rare
    and would only add noise).
    """
    roots: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if d != EXCLUDED_DIR]
        if any(marker in filenames for marker in LAKE_ROOT_MARKERS):
            roots.append(Path(dirpath))
            dirnames[:] = []  # don't descend into a found project
    return roots


_KEYWORDS_PATTERN = "|".join(re.escape(keyword) for keyword in LEAN_KEYWORDS)


def _iter_lean_files(root: Path) -> Iterator[Path]:
    """Yield every .lean file under `root`, skipping the `.lake/` directory."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != EXCLUDED_DIR]
        for filename in filenames:
            if filename.endswith(".lean"):
                yield Path(dirpath) / filename


def _matching_declaration_names(content: str, query: str) -> list[str]:
    """Names of searchable declarations matching `query` (case-insensitive).

    A multi-word query matches names that contain every whitespace-separated
    token, so "Treap insert" matches `Treap.insert` (a single-word query keeps
    the original substring behaviour). Preserves source order, de-duplicated.
    """
    tokens = query.lower().split()
    if not tokens:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for declaration in list_all_declarations_in_lean_code(content):
        if declaration.declaration_type not in SEARCHABLE_TYPES:
            continue
        name_lower = declaration.name.lower()
        if all(token in name_lower for token in tokens) and declaration.name not in seen:
            seen.add(declaration.name)
            names.append(declaration.name)
    return names


def _declaration_line(content: str, name: str) -> int:
    """1-based line number where the declaration of `name` begins (keyword line).

    Falls back to 1 if the declaration keyword cannot be located.
    """
    pattern = rf"^\s*(?:{_KEYWORDS_PATTERN})\s+{re.escape(name)}{DECL_NAME_END}"
    match = re.search(pattern, content, re.MULTILINE)
    if match is None:
        return 1
    # With re.MULTILINE, `^` anchors at a line start and the following `\s*` can span
    # blank lines and indentation, so match.start() may precede the keyword line.
    # Advance past the matched leading whitespace to the keyword itself.
    keyword_offset = match.start() + len(match.group()) - len(match.group().lstrip())
    return content[:keyword_offset].count("\n") + 1


def _format_results(
    query: str,
    matches: list[tuple[str, Path, int, str]],
    config: SearchLeanLocalConfig,
) -> str:
    """Render matches, capping by max_results and max_chars; overflow listed by name."""
    header = f'Found {len(matches)} declaration(s) matching "{query}":'
    shown: list[str] = []
    overflow: list[str] = []
    total = len(header)
    for name, rel_path, line, block in matches:
        entry = f"-- {rel_path}:{line}\n{block}"
        within_budget = total + len(entry) + 2 <= config.max_chars
        if len(shown) < config.max_results and within_budget:
            shown.append(entry)
            total += len(entry) + 2
        else:
            overflow.append(f"{name} ({rel_path}:{line})")
    output = header + "\n\n" + "\n\n".join(shown)
    if overflow:
        output += "\n\nAdditional matches (not shown, refine your query): " + ", ".join(overflow)
    return output


class LocalLeanSearcher:
    """Searches a Lean project's own .lean files for declarations by name.

    The project root is resolved once (from the lakefile) and cached.
    """

    def __init__(self, config: SearchLeanLocalConfig, base_folder: str = "."):
        self.config = config
        self.base_folder = base_folder
        self._resolution: tuple[Path | None, str] | None = None

    def _resolve_root(self) -> tuple[Path | None, str]:
        if self._resolution is None:
            self._resolution = self._compute_root()
        return self._resolution

    def _compute_root(self) -> tuple[Path | None, str]:
        """Return (root, "") on success, or (None, error_message) on failure."""
        start = Path(self.base_folder).resolve()
        up = _walk_up_for_root(start)
        if up is not None:
            return up, ""
        down = _walk_down_for_roots(start)
        if len(down) == 1:
            return down[0], ""
        if not down:
            return None, (f"Local Lean search unavailable: no lakefile found at or under {start}.")
        listed = ", ".join(str(path) for path in sorted(down))
        return None, (
            f"Local Lean search found multiple Lean projects under {start}: {listed}. "
            f"Point --folder at one of them."
        )

    def search(self, query: str) -> str:
        query = query.strip()
        logger.debug(f"LocalLeanSearch tool invoked with query: '{query}'")
        if not query:
            return "Please provide a non-empty keyword to search for."

        root, error = self._resolve_root()
        if root is None:
            logger.warning(f"LocalLeanSearch: {error}")
            return error

        matches: list[tuple[str, Path, int, str]] = []
        for lean_file in _iter_lean_files(root):
            try:
                content = lean_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug(f"Skipping unreadable Lean file {lean_file}: {exc}")
                continue
            for name in _matching_declaration_names(content, query):
                block = extract_function_from_content(content, name)
                if block is None:
                    continue
                line = _declaration_line(content, name)
                matches.append((name, lean_file.relative_to(root), line, block))

        if not matches:
            logger.info(f"LocalLeanSearch: No results for '{query}'")
            return f'No declarations matching "{query}" found.'
        logger.info(f"LocalLeanSearch: Found {len(matches)} matches for '{query}' under {root}")
        return _format_results(query, matches, self.config)


class LocalLeanSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Keyword(s) to match against declaration names (case-insensitive). "
            "Multiple words match names containing all of them, e.g. 'Treap insert' "
            "finds `Treap.insert`."
        ),
    )


@register_tool(LOCAL_LEAN_SEARCH_TOOL_TYPE, SearchLeanLocalConfig)
def create_search_lean_local_tool(
    config: SearchLeanLocalConfig, base_folder: str = "."
) -> StructuredTool:
    """Create the local Lean search tool, scoped to `base_folder`'s Lean project."""
    searcher = LocalLeanSearcher(config, base_folder=base_folder)
    return StructuredTool(
        name=tool_name_from_type(LOCAL_LEAN_SEARCH_TOOL_TYPE),
        description="""Search the local Lean project for declarations by name.

Returns the full source of `def`/`theorem`/`lemma`/`structure`/etc. whose name
contains your keyword (case-insensitive), from the project's own .lean files.
Mathlib and other dependencies are excluded — use the lean_search tool for those.

Pass a single keyword (e.g. "Treap" or "extract_min"). Multiple words match names
containing all of them, so "Treap insert" finds `Treap.insert`.

Use this to retrieve project-local definitions you need to reference in a proof,
e.g. search "Treap" to get the definition of a local `Treap` structure.""",
        func=searcher.search,
        args_schema=LocalLeanSearchInput,
    )
