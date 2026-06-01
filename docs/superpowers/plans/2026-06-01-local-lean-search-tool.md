# Local Lean Search Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `search_lean_local` tool that returns the full declaration blocks (by name keyword) from the project's own `.lean` files, so the prover can use project-local definitions (e.g. `Def_Treap.lean`) while proving.

**Architecture:** A new self-contained tool module (`local_lean_search.py`) following the existing `web_search.py` tool pattern. A `LocalLeanSearcher` class resolves the Lean project root from the lakefile (walk up, then down), scans `.lean` files excluding `.lake/`, name-matches declarations, and returns full blocks via the existing `lean_parsing` utilities. The agent's `base_folder` is threaded into the tool factory through a small, backward-compatible change to `create_tool`.

**Tech Stack:** Python 3.12+, LangChain `StructuredTool`, Pydantic, dataclasses, pytest (`pytest-asyncio`), ruff. Reuses `ax_prover.utils.lean_parsing` (`list_all_declarations_in_lean_code`, `extract_function_from_content`, `LEAN_KEYWORDS`) and `ax_prover.models.declaration.DeclarationType`.

**Spec:** `docs/superpowers/specs/2026-06-01-local-lean-search-tool-design.md`

---

## File Structure

- **Create** `src/ax_prover/tools/local_lean_search.py` — the tool: constants, `SearchLeanLocalConfig`, root-resolution helpers, file-scan helpers, `LocalLeanSearcher` class, args schema, and the `@register_tool` factory.
- **Create** `tests/unit/tools/test_local_lean_search.py` — unit tests for root resolution, name matching, full-block return, `.lake/` exclusion, caps, error messages, and the wiring injection.
- **Modify** `src/ax_prover/tools/registry.py` — `create_tool()` accepts `base_folder` and passes it to factories that declare it.
- **Modify** `src/ax_prover/tools/__init__.py` — import/export `create_search_lean_local_tool` (also triggers registration).
- **Modify** `src/ax_prover/prover/agent.py` — `_create_tools()` passes `self.base_folder` to `create_tool()`.
- **Modify** `configs/tools.yaml` — add the `search_lean_local` tool config.

Run tests with: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -v`

---

## Task 1: Tool module scaffolding — constants, config, args schema

**Files:**
- Create: `src/ax_prover/tools/local_lean_search.py`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for the local Lean library search tool."""

from pathlib import Path

import pytest

from ax_prover.models.declaration import DeclarationType
from ax_prover.tools.local_lean_search import (
    LOCAL_LEAN_SEARCH_TOOL_TYPE,
    SEARCHABLE_TYPES,
    SearchLeanLocalConfig,
)


class TestModuleConstants:
    def test_tool_type_value(self):
        assert LOCAL_LEAN_SEARCH_TOOL_TYPE == "search_lean_local"

    def test_config_defaults(self):
        config = SearchLeanLocalConfig()
        assert config.max_results == 6
        assert config.max_chars == 4000

    def test_searchable_types_include_defs_exclude_structural(self):
        assert DeclarationType.Definition in SEARCHABLE_TYPES
        assert DeclarationType.Theorem in SEARCHABLE_TYPES
        assert DeclarationType.Structure in SEARCHABLE_TYPES
        # Structural keywords must NOT be treated as searchable declarations.
        assert DeclarationType.Namespace not in SEARCHABLE_TYPES
        assert DeclarationType.Import not in SEARCHABLE_TYPES
        assert DeclarationType.Open not in SEARCHABLE_TYPES
        assert DeclarationType.End not in SEARCHABLE_TYPES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestModuleConstants -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ax_prover.tools.local_lean_search'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ax_prover/tools/local_lean_search.py`:

```python
"""Local Lean library search tool.

Searches the .lean files of the project being proven (not Mathlib / .lake
dependencies) and returns the full declaration blocks whose names match a
keyword. Complements the remote `lean_search` tool, which covers Mathlib.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..models.declaration import DeclarationType
from ..utils import get_logger
from ..utils.lean_parsing import (
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestModuleConstants -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): scaffold local Lean search module (constants + config)"
```

---

## Task 2: Project-root resolution (walk up, then down)

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import _walk_down_for_roots, _walk_up_for_root


def _make_lake_project(directory: Path) -> Path:
    """Create a minimal lake project (lakefile.toml) at `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lakefile.toml").write_text("name = \"demo\"\n")
    return directory


class TestRootResolution:
    def test_walk_up_when_base_is_root(self, tmp_path):
        root = _make_lake_project(tmp_path / "proj")
        assert _walk_up_for_root(root) == root

    def test_walk_up_from_subdirectory(self, tmp_path):
        root = _make_lake_project(tmp_path / "proj")
        sub = root / "Lib" / "Nested"
        sub.mkdir(parents=True)
        assert _walk_up_for_root(sub) == root

    def test_walk_up_returns_none_when_no_marker(self, tmp_path):
        plain = tmp_path / "nolake"
        plain.mkdir()
        assert _walk_up_for_root(plain) is None

    def test_walk_down_finds_single_project(self, tmp_path):
        root = _make_lake_project(tmp_path / "outer" / "challenges")
        assert _walk_down_for_roots(tmp_path / "outer") == [root]

    def test_walk_down_finds_multiple_projects(self, tmp_path):
        a = _make_lake_project(tmp_path / "outer" / "a")
        b = _make_lake_project(tmp_path / "outer" / "b")
        assert sorted(_walk_down_for_roots(tmp_path / "outer")) == sorted([a, b])

    def test_walk_down_skips_dot_lake(self, tmp_path):
        outer = tmp_path / "outer"
        # A lakefile buried inside .lake must be ignored.
        buried = outer / ".lake" / "packages" / "mathlib"
        _make_lake_project(buried)
        outer.mkdir(parents=True, exist_ok=True)
        assert _walk_down_for_roots(outer) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestRootResolution -v`
Expected: FAIL with `ImportError: cannot import name '_walk_down_for_roots'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/ax_prover/tools/local_lean_search.py` (after the config dataclass):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestRootResolution -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): add Lean project-root resolution (up then down)"
```

---

## Task 3: File scanning, name matching, and line numbers

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import (
    _declaration_line,
    _iter_lean_files,
    _matching_declaration_names,
)

SAMPLE_LEAN = """import Mathlib

namespace Treaps

/-- A treap node. -/
structure Treap where
  key : Nat
  priority : Nat

def Treap.insert (t : Treap) (k : Nat) : Treap :=
  t

theorem treap_insert_size (t : Treap) : True := by
  trivial

end Treaps
"""


class TestScanHelpers:
    def test_iter_lean_files_excludes_dot_lake(self, tmp_path):
        (tmp_path / "A.lean").write_text("def a := 1\n")
        sub = tmp_path / "Lib"
        sub.mkdir()
        (sub / "B.lean").write_text("def b := 1\n")
        buried = tmp_path / ".lake" / "packages" / "mathlib"
        buried.mkdir(parents=True)
        (buried / "M.lean").write_text("def m := 1\n")

        found = {p.name for p in _iter_lean_files(tmp_path)}
        assert found == {"A.lean", "B.lean"}

    def test_matching_names_case_insensitive_substring(self):
        names = _matching_declaration_names(SAMPLE_LEAN, "treap")
        assert names == ["Treap", "Treap.insert", "treap_insert_size"]

    def test_matching_names_excludes_structural_keywords(self):
        # "Treaps" (the namespace) must not be returned even though it matches.
        names = _matching_declaration_names(SAMPLE_LEAN, "Treap")
        assert "Treaps" not in names

    def test_matching_names_no_match_returns_empty(self):
        assert _matching_declaration_names(SAMPLE_LEAN, "nonexistent") == []

    def test_declaration_line_points_at_keyword(self):
        # `structure Treap` is on line 6 (1-based) in SAMPLE_LEAN.
        assert _declaration_line(SAMPLE_LEAN, "Treap") == 6

    def test_declaration_line_distinguishes_dotted_name(self):
        # `def Treap.insert` is on line 10.
        assert _declaration_line(SAMPLE_LEAN, "Treap.insert") == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestScanHelpers -v`
Expected: FAIL with `ImportError: cannot import name '_iter_lean_files'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/ax_prover/tools/local_lean_search.py`:

```python
_KEYWORDS_PATTERN = "|".join(re.escape(keyword) for keyword in LEAN_KEYWORDS)


def _iter_lean_files(root: Path):
    """Yield every .lean file under `root`, skipping the `.lake/` directory."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != EXCLUDED_DIR]
        for filename in filenames:
            if filename.endswith(".lean"):
                yield Path(dirpath) / filename


def _matching_declaration_names(content: str, query: str) -> list[str]:
    """Names of searchable declarations whose name contains `query` (case-insensitive).

    Preserves source order and de-duplicates repeated names.
    """
    needle = query.lower()
    names: list[str] = []
    seen: set[str] = set()
    for declaration in list_all_declarations_in_lean_code(content):
        if declaration.declaration_type not in SEARCHABLE_TYPES:
            continue
        if needle in declaration.name.lower() and declaration.name not in seen:
            seen.add(declaration.name)
            names.append(declaration.name)
    return names


def _declaration_line(content: str, name: str) -> int:
    """1-based line number where the declaration of `name` begins (keyword line).

    Falls back to 1 if the declaration keyword cannot be located.
    """
    pattern = rf"^\s*(?:{_KEYWORDS_PATTERN})\s+{re.escape(name)}\b"
    match = re.search(pattern, content, re.MULTILINE)
    if match is None:
        return 1
    return content[: match.start()].count("\n") + 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestScanHelpers -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): add Lean file scanning, name matching, line lookup"
```

---

## Task 4: `LocalLeanSearcher` — resolution caching, search, formatting, caps, errors

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import LocalLeanSearcher


@pytest.fixture
def treap_project(tmp_path):
    """A lake project with a Treap definition file and a buried Mathlib file."""
    root = tmp_path / "challenges"
    (root / "Challenges" / "Treap").mkdir(parents=True)
    (root / "lakefile.toml").write_text("name = \"challenges\"\n")
    (root / "Challenges" / "Treap" / "Def_Treap.lean").write_text(SAMPLE_LEAN)
    buried = root / ".lake" / "packages" / "mathlib" / "Mathlib"
    buried.mkdir(parents=True)
    (buried / "Shadow.lean").write_text("def OnlyInMathlib := 1\n")
    return root


class TestLocalLeanSearcher:
    def test_search_returns_full_declaration_block(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        result = searcher.search("Treap")
        assert "Found 3 declaration(s)" in result
        assert "structure Treap where" in result
        assert "key : Nat" in result
        assert "Challenges/Treap/Def_Treap.lean:" in result

    def test_search_is_case_insensitive(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert "structure Treap where" in searcher.search("treap")

    def test_search_excludes_dot_lake(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert searcher.search("OnlyInMathlib") == 'No declarations matching "OnlyInMathlib" found.'

    def test_search_no_match_message(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert searcher.search("Nonexistent") == 'No declarations matching "Nonexistent" found.'

    def test_search_empty_query_message(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert searcher.search("   ") == "Please provide a non-empty keyword to search for."

    def test_search_resolves_root_via_walk_down(self, tmp_path, treap_project):
        # base_folder is the OUTER dir; root is the `challenges` subdir.
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(tmp_path))
        assert "structure Treap where" in searcher.search("Treap")

    def test_search_no_lakefile_message(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(plain))
        result = searcher.search("Treap")
        assert "no lakefile found" in result.lower()

    def test_search_multiple_projects_message(self, tmp_path):
        for name in ("a", "b"):
            proj = tmp_path / "outer" / name
            proj.mkdir(parents=True)
            (proj / "lakefile.toml").write_text("name = \"x\"\n")
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(tmp_path / "outer"))
        result = searcher.search("Treap")
        assert "multiple Lean projects" in result

    def test_caps_overflow_lists_names_only(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(max_results=1), base_folder=str(treap_project))
        result = searcher.search("Treap")
        assert "Found 3 declaration(s)" in result
        shown_section, _, overflow_section = result.partition("Additional matches")
        assert overflow_section  # overflow present
        # The single shown block is the structure (first match).
        assert "structure Treap where" in shown_section
        # The two un-shown matches are named in the overflow, not rendered as blocks.
        assert "Treap.insert" in overflow_section
        assert "treap_insert_size" in overflow_section
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestLocalLeanSearcher -v`
Expected: FAIL with `ImportError: cannot import name 'LocalLeanSearcher'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/ax_prover/tools/local_lean_search.py`:

```python
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
        start = Path(self.base_folder).resolve()
        up = _walk_up_for_root(start)
        if up is not None:
            return up, ""
        down = _walk_down_for_roots(start)
        if len(down) == 1:
            return down[0], ""
        if not down:
            return None, (
                f"Local Lean search unavailable: no lakefile found at or under {start}."
            )
        listed = ", ".join(str(path) for path in sorted(down))
        return None, (
            f"Local Lean search found multiple Lean projects under {start}: {listed}. "
            f"Point --folder at one of them."
        )

    def search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Please provide a non-empty keyword to search for."

        root, error = self._resolve_root()
        if root is None:
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
            return f'No declarations matching "{query}" found.'
        return _format_results(query, matches, self.config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestLocalLeanSearcher -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): add LocalLeanSearcher (search, formatting, caps, errors)"
```

---

## Task 5: Register the tool and export it

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`
- Modify: `src/ax_prover/tools/__init__.py:1-25`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools/test_local_lean_search.py`:

```python
from langchain_core.tools import StructuredTool

from ax_prover.tools import create_search_lean_local_tool  # noqa: E402
from ax_prover.tools.registry import TOOL_REGISTRY


class TestRegistration:
    def test_tool_is_registered(self):
        # Importing ax_prover.tools triggers @register_tool.
        assert LOCAL_LEAN_SEARCH_TOOL_TYPE in TOOL_REGISTRY
        assert TOOL_REGISTRY[LOCAL_LEAN_SEARCH_TOOL_TYPE].config_class is SearchLeanLocalConfig

    def test_factory_builds_structured_tool(self):
        tool = create_search_lean_local_tool(SearchLeanLocalConfig(), base_folder=".")
        assert isinstance(tool, StructuredTool)
        assert tool.name == "search_lean_local_tool"

    def test_factory_tool_is_callable_with_query(self, tmp_path):
        proj = tmp_path / "challenges"
        proj.mkdir()
        (proj / "lakefile.toml").write_text("name = \"x\"\n")
        (proj / "Defs.lean").write_text("def myThing := 1\n")
        tool = create_search_lean_local_tool(SearchLeanLocalConfig(), base_folder=str(proj))
        result = tool.func("myThing")
        assert "def myThing" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestRegistration -v`
Expected: FAIL with `ImportError: cannot import name 'create_search_lean_local_tool' from 'ax_prover.tools'`

- [ ] **Step 3: Write minimal implementation**

Add the args schema and factory to the end of `src/ax_prover/tools/local_lean_search.py`:

```python
class LocalLeanSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Keyword to match against declaration names (case-insensitive substring).",
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

Use this to retrieve project-local definitions you need to reference in a proof,
e.g. search "Treap" to get the definition of a local `Treap` structure.""",
        func=searcher.search,
        args_schema=LocalLeanSearchInput,
    )
```

Then modify `src/ax_prover/tools/__init__.py`. Add the import after the existing `lean_search` import:

```python
from .lean_search import (
    create_search_lean_search_tool,
    lean_search_session_manager,
    warmup_lean_search,
)
from .local_lean_search import create_search_lean_local_tool
from .registry import TOOL_REGISTRY, create_tool, tool_name_from_type
from .web_search import create_search_web_tool
```

And add `"create_search_lean_local_tool"` to the `__all__` list in the same file:

```python
__all__ = [
    "TOOL_REGISTRY",
    "create_tool",
    "tool_name_from_type",
    "create_search_lean_search_tool",
    "create_search_lean_local_tool",
    "create_search_web_tool",
    "lean_search_session_manager",
    "warmup_lean_search",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestRegistration -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py src/ax_prover/tools/__init__.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): register and export search_lean_local tool"
```

---

## Task 6: Thread `base_folder` through `create_tool`

**Files:**
- Modify: `src/ax_prover/tools/registry.py:50-90`
- Test: `tests/unit/tools/test_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools/test_registry.py` (inside the file, after the existing `TestCreateTool` class):

```python
class TestCreateToolBaseFolder:
    """Tests for base_folder injection into factories that declare it."""

    @pytest.mark.asyncio
    async def test_passes_base_folder_when_factory_declares_it(self):
        received = {}

        def factory(config, base_folder="."):
            received["base_folder"] = base_folder
            return MagicMock()

        with patch.dict(
            TOOL_REGISTRY,
            {"needs_folder": ToolRegistration(factory=factory, config_class=SearchWebConfig)},
        ):
            await create_tool({"tool_type": "needs_folder"}, base_folder="/some/path")

        assert received["base_folder"] == "/some/path"

    @pytest.mark.asyncio
    async def test_omits_base_folder_when_factory_does_not_declare_it(self):
        def factory(config):
            return MagicMock()

        with patch.dict(
            TOOL_REGISTRY,
            {"no_folder": ToolRegistration(factory=factory, config_class=SearchWebConfig)},
        ):
            # Must not raise TypeError despite base_folder being supplied.
            result = await create_tool({"tool_type": "no_folder"}, base_folder="/x")

        assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_registry.py::TestCreateToolBaseFolder -v`
Expected: FAIL with `TypeError: create_tool() got an unexpected keyword argument 'base_folder'`

- [ ] **Step 3: Write minimal implementation**

In `src/ax_prover/tools/registry.py`, change the `create_tool` signature and the factory-call section. Replace the existing function signature line:

```python
async def create_tool(
    tool_config: dict[str, Any],
) -> BaseTool | None:
```

with:

```python
async def create_tool(
    tool_config: dict[str, Any],
    base_folder: str = ".",
) -> BaseTool | None:
```

Then replace this existing block:

```python
    config = registration.config_class(**tool_config)

    # Call factory (handle both sync and async)
    tool = registration.factory(config)
    if inspect.iscoroutine(tool):
        tool = await tool
```

with:

```python
    config = registration.config_class(**tool_config)

    # Call factory (handle both sync and async). Factories that declare a
    # `base_folder` parameter receive the runtime project folder; others don't.
    factory = registration.factory
    if "base_folder" in inspect.signature(factory).parameters:
        tool = factory(config, base_folder=base_folder)
    else:
        tool = factory(config)
    if inspect.iscoroutine(tool):
        tool = await tool
```

Also update the docstring `Args:` section of `create_tool` to mention `base_folder` (insert after the `tool_config:` arg line):

```python
        base_folder: Project base folder passed to factories that declare it.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/tools/test_registry.py -v`
Expected: PASS (all existing tests + 2 new ones; existing single-arg `create_tool(...)` calls still work via the default)

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/registry.py tests/unit/tools/test_registry.py
git commit -m "feat(tools): pass base_folder to factories that declare it"
```

---

## Task 7: Pass `base_folder` from the agent

**Files:**
- Modify: `src/ax_prover/prover/agent.py:140-149`
- Test: manual (agent construction requires LLM config; covered indirectly by Task 6 + Task 8)

- [ ] **Step 1: Make the change**

In `src/ax_prover/prover/agent.py`, find `_create_tools` (around line 140):

```python
    async def _create_tools(self) -> list:
        """Create tools asynchronously, filtering out any that failed to initialize."""
        tools = []
        for tool_config in self.config.proposer_tools.values():
            if tool_config is None:
                continue
            tool = await create_tool(tool_config)
            if tool is not None:
                tools.append(tool)
        return tools
```

Change the `create_tool` call to pass `self.base_folder`:

```python
    async def _create_tools(self) -> list:
        """Create tools asynchronously, filtering out any that failed to initialize."""
        tools = []
        for tool_config in self.config.proposer_tools.values():
            if tool_config is None:
                continue
            tool = await create_tool(tool_config, base_folder=self.base_folder)
            if tool is not None:
                tools.append(tool)
        return tools
```

- [ ] **Step 2: Verify nothing broke**

Run: `.venv/bin/pytest tests/unit -q`
Expected: PASS (full unit suite green; no regressions in prover/agent tests)

- [ ] **Step 3: Commit**

```bash
git add src/ax_prover/prover/agent.py
git commit -m "feat(prover): pass base_folder to tool creation"
```

---

## Task 8: Add tool config and verify end-to-end on a real project

**Files:**
- Modify: `configs/tools.yaml`

- [ ] **Step 1: Add the tool config**

In `configs/tools.yaml`, add a new entry under `tool_configs:` (after `search_lean_search_ax`):

```yaml
  search_lean_local:
    tool_type: search_lean_local
    max_results: 6
    max_chars: 4000
```

- [ ] **Step 2: Verify config parses and the tool builds from it**

Run:

```bash
.venv/bin/python -c "
import asyncio
from ax_prover.tools import create_tool
tool = asyncio.run(create_tool({'tool_type': 'search_lean_local', 'max_results': 6, 'max_chars': 4000}, base_folder='.'))
print(tool.name)
"
```

Expected output: `search_lean_local_tool`

- [ ] **Step 3: Real-project smoke test (no LLM, no lake build needed)**

Run against the AI4Math Treap case study to confirm it finds the local definitions:

```bash
.venv/bin/python -c "
from ax_prover.tools.local_lean_search import LocalLeanSearcher, SearchLeanLocalConfig
s = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder='/Users/krystian/Documents/Axiomatic/Baku/AI4Math/challenges')
print(s.search('Treap'))
"
```

Expected: a `Found N declaration(s) matching "Treap":` header followed by full declaration blocks drawn from `Challenges/Treap/Def_Treap.lean` (and **no** results from `.lake/packages/mathlib`).

- [ ] **Step 4: Lint and run the full unit suite**

Run:

```bash
ruff format . && ruff check --fix .
.venv/bin/pytest tests/unit -q
```

Expected: ruff clean; all unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/tools.yaml
git commit -m "feat(config): add search_lean_local tool config"
```

---

## How to use the new tool

Add the tool to a prover run's `proposer_tools` via OmegaConf interpolation, exactly like the existing tools (see `configs/tools.yaml` header for the pattern):

```yaml
prover:
  proposer_tools:
    search_lean_local: ${tool_configs.search_lean_local}
```

Then run, pointing `--folder` at (or above) the lakefile directory:

```bash
ax-prover prove Challenges.Treap.Challenge_Treap_11:<theorem> \
  --folder /Users/krystian/Documents/Axiomatic/Baku/AI4Math/challenges
```

The agent can then call `search_lean_local_tool` with a keyword like `Treap` to retrieve the definitions in `Def_Treap.lean` while proving.
