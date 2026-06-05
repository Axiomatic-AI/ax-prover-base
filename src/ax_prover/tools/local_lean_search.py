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
    DECL_PREFIX,
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


# Longest-first so compound keywords like `noncomputable def` win over the bare `def`.
_KEYWORDS_PATTERN = "|".join(
    re.escape(keyword) for keyword in sorted(LEAN_KEYWORDS, key=len, reverse=True)
)


# Characters that make up a Lean identifier; a body token matches only when flanked by
# non-identifier characters (so "query_aux" won't hit inside "query_auxN" and "insert"
# won't hit inside "Treap.insert").
_IDENT_CHAR = r"[0-9A-Za-z_'.]"


def _identifier_match(token: str, text: str) -> bool:
    """True if `token` appears in `text` as a whole identifier (case-insensitive)."""
    pattern = rf"(?<!{_IDENT_CHAR}){re.escape(token)}(?!{_IDENT_CHAR})"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _iter_lean_files(root: Path) -> Iterator[Path]:
    """Yield every .lean file under `root`, skipping the `.lake/` directory."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != EXCLUDED_DIR]
        for filename in filenames:
            if filename.endswith(".lean"):
                yield Path(dirpath) / filename


def _iter_searchable(content: str):
    """Yield (declaration, qualified_name, occurrence) for each searchable declaration.

    Tracks namespace/section scope to build the qualified name and counts per-simple-name
    occurrences (aligned with re.finditer order in extract_function_from_content /
    _declaration_line). Shared by name matching and the body-search fallback.
    """
    namespace_stack: list[str | None] = []
    name_counts: dict[str, int] = {}
    for declaration in list_all_declarations_in_lean_code(content):
        occurrence = name_counts.get(declaration.name, 0)
        name_counts[declaration.name] = occurrence + 1
        declaration_type = declaration.declaration_type
        if declaration_type == DeclarationType.Namespace:
            namespace_stack.append(declaration.name)
            continue
        if declaration_type == DeclarationType.Section:
            namespace_stack.append(None)  # sections do not contribute to the name
            continue
        if declaration_type == DeclarationType.End:
            if namespace_stack:
                namespace_stack.pop()
            continue
        if declaration_type not in SEARCHABLE_TYPES:
            continue
        prefix = ".".join(part for part in namespace_stack if part)
        if prefix and not declaration.name.startswith(f"{prefix}."):
            qualified = f"{prefix}.{declaration.name}"
        else:
            qualified = declaration.name
        yield declaration, qualified, occurrence


def _matching_declaration_names(content: str, query: str) -> list[tuple[str, str, int]]:
    """`(simple_name, qualified_name, occurrence)` for declarations matching `query`.

    Matching is case-insensitive against the namespace-QUALIFIED name, so a
    multi-word query like "BinaryTree insert" matches a `def insert` declared inside
    `namespace BinaryTree` as well as a top-level `def BinaryTree.insert`. The simple
    name (as written in source) is returned alongside, since block extraction and line
    lookup operate on the source text. `occurrence` is the 0-based index of this
    declaration among all declarations sharing its simple name in the file, so callers
    can disambiguate the correct block/line when the same simple name appears in several
    namespaces. Preserves source order, de-duplicated by qualified name.
    """
    tokens = query.lower().split()
    if not tokens:
        return []
    results: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for declaration, qualified, occurrence in _iter_searchable(content):
        if qualified in seen:
            continue
        if all(token in qualified.lower() for token in tokens):
            seen.add(qualified)
            results.append((declaration.name, qualified, occurrence))
    return results


def _body_matching_declarations(content: str, query: str) -> list[tuple[str, str, int]]:
    """`(simple_name, qualified_name, occurrence)` for declarations whose block text contains
    every query token as a whole identifier. Fallback for identifiers that are not declaration
    names themselves (e.g. `where`/`let rec` helpers, inductive constructors). De-duplicated by
    qualified name, source order preserved.
    """
    tokens = query.lower().split()
    if not tokens:
        return []
    results: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for declaration, qualified, occurrence in _iter_searchable(content):
        if qualified in seen:
            continue
        block = extract_function_from_content(content, declaration.name, occurrence)
        if block is None:
            continue
        if all(_identifier_match(token, block) for token in tokens):
            seen.add(qualified)
            results.append((declaration.name, qualified, occurrence))
    return results


def _declaration_line(content: str, name: str, occurrence: int = 0) -> int:
    """1-based line number where the declaration of `name` begins (keyword line).

    `occurrence` selects the N-th (0-based) declaration with this simple name, matching
    extract_function_from_content, so duplicates across namespaces resolve to the right
    line. Falls back to 1 if the declaration keyword cannot be located.
    """
    pattern = rf"^\s*{DECL_PREFIX}(?:{_KEYWORDS_PATTERN})\s+{re.escape(name)}{DECL_NAME_END}"
    matches = list(re.finditer(pattern, content, re.MULTILINE))
    if occurrence >= len(matches):
        return 1
    match = matches[occurrence]
    # With re.MULTILINE, `^` anchors at a line start and the following `\s*` can span
    # blank lines and indentation, so match.start() may precede the keyword line.
    # Advance past the matched leading whitespace to the keyword itself.
    keyword_offset = match.start() + len(match.group()) - len(match.group().lstrip())
    return content[:keyword_offset].count("\n") + 1


def _format_results(
    query: str,
    declarations: list[tuple[str, str, list[tuple[Path, int]]]],
    config: SearchLeanLocalConfig,
    body_match: bool = False,
) -> str:
    """Render unique declarations, capping by max_results and max_chars.

    Each declaration is (name, block, locations); locations is one or more
    (path, line) where identical content was found. The first location heads the
    entry and the rest are listed in an "(also: …)" suffix. Overflow (beyond the
    caps) is listed by name.
    """
    header = f'Found {len(declarations)} declaration(s) matching "{query}":'
    note = " (matched in body)" if body_match else ""
    shown: list[str] = []
    overflow: list[str] = []
    total = len(header)
    for name, block, locations in declarations:
        (primary_path, primary_line), *extra = locations
        location_header = f"-- {primary_path}:{primary_line}{note}"
        if extra:
            also = ", ".join(f"{path}:{line}" for path, line in extra)
            location_header += f" (also: {also})"
        entry = f"{location_header}\n{block}"
        within_budget = total + len(entry) + 2 <= config.max_chars
        if len(shown) < config.max_results and within_budget:
            shown.append(entry)
            total += len(entry) + 2
        else:
            overflow.append(f"{name} ({primary_path}:{primary_line})")
    output = header + "\n\n" + "\n\n".join(shown)
    if overflow:
        output += "\n\nAdditional matches (not shown, refine your query): " + ", ".join(overflow)
    return output


def _collect_matches(
    root: Path, query: str, *, body: bool
) -> list[tuple[str, str, list[tuple[Path, int]]]]:
    """Scan every .lean file under `root` and group matches by (qualified_name, block).

    `body=False` matches declaration names; `body=True` matches query tokens as whole
    identifiers inside declaration blocks (the fallback). Identical blocks copied across
    files collapse to one entry recording every location.
    """
    grouped: dict[tuple[str, str], list[tuple[Path, int]]] = {}
    for lean_file in _iter_lean_files(root):
        try:
            content = lean_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug(f"Skipping unreadable Lean file {lean_file}: {exc}")
            continue
        matches = (
            _body_matching_declarations(content, query)
            if body
            else _matching_declaration_names(content, query)
        )
        for simple_name, qualified_name, occurrence in matches:
            block = extract_function_from_content(content, simple_name, occurrence)
            if block is None:
                continue
            line = _declaration_line(content, simple_name, occurrence)
            grouped.setdefault((qualified_name, block), []).append(
                (lean_file.relative_to(root), line)
            )
    return [(name, block, locations) for (name, block), locations in grouped.items()]


def _search_root(
    root: Path, query: str, config: SearchLeanLocalConfig, *, label: str = "LocalLeanSearch"
) -> str:
    """Search `root` by declaration name; fall back to body search only if name finds nothing."""
    decls = _collect_matches(root, query, body=False)
    if decls:
        logger.info(f"{label}: Found {len(decls)} declarations for '{query}' under {root}")
        return _format_results(query, decls, config)
    body_decls = _collect_matches(root, query, body=True)
    if body_decls:
        logger.info(f"{label}: Found {len(body_decls)} body matches for '{query}' under {root}")
        return _format_results(query, body_decls, config, body_match=True)
    logger.info(f"{label}: No results for '{query}'")
    return f'No declarations matching "{query}" found.'


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

        return _search_root(root, query, self.config)


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
