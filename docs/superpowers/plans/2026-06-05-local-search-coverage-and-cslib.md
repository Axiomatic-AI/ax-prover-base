# Local Search Coverage + CSLib Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the prover's local Lean search find identifiers used in declaration bodies (e.g. `where`/`let rec` helpers like `query_aux`) and add a `search_cslib` tool that searches the vendored CSLib dependency; separately bump the proposer prompt to Lean 4.28.

**Architecture:** Extract the per-directory search routine of `LocalLeanSearcher` into a shared module-level `_search_root`. Add a **body-search fallback** that runs only when name matching finds nothing, matching query tokens as whole identifiers inside declaration blocks. Add a sibling `CslibSearcher`/`search_cslib` tool that resolves `<lake_root>/.lake/packages/cslib` relative to the project and reuses `_search_root` (inheriting the fallback). The shared declaration parser is untouched.

**Tech Stack:** Python 3.12, pytest, ruff, OmegaConf YAML configs, LangChain `StructuredTool`.

**Branch:** `local-search-body-and-cslib` (already created off `local-lean-search-tool`). PR A = this plan. PR B (prompt 4.28) is a separate tiny branch off `main` — see the final section.

---

## File Structure

- `src/ax_prover/tools/local_lean_search.py` — MODIFY: extract `_iter_searchable` + `_search_root`; add `_identifier_match`, `_body_matching_declarations`; `_format_results` gains `body_match`; `LocalLeanSearcher.search` delegates to `_search_root`.
- `src/ax_prover/tools/cslib_search.py` — CREATE: `CSLIB_SEARCH_TOOL_TYPE`, `SearchCslibConfig`, `CslibSearcher`, `CslibSearchInput`, `create_search_cslib_tool`.
- `src/ax_prover/tools/__init__.py` — MODIFY: export `create_search_cslib_tool`.
- `src/ax_prover/configs/tools.yaml` — MODIFY: add `search_cslib` tool_config.
- `src/ax_prover/configs/default.yaml` — MODIFY: add `search_cslib` to `proposer_tools`.
- `tests/unit/tools/test_local_lean_search.py` — MODIFY: body-search tests.
- `tests/unit/tools/test_cslib_search.py` — CREATE: CSLib tricky-retrieval tests.
- `src/ax_prover/prover/prompts.py` — MODIFY (PR B, separate branch): Lean 4.24 → 4.28.

Run tests with `.venv/bin/pytest`; format/lint with `.venv/bin/ruff`.

---

## Task 1: Refactor — extract `_search_root` (no behavior change)

Pure refactor; the existing test suite is the safety net.

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`

- [ ] **Step 1: Add `_iter_searchable` helper above `_matching_declaration_names`**

Insert this function immediately before `def _matching_declaration_names` (currently line 107):

```python
def _iter_searchable(content: str):
    """Yield (declaration, qualified_name, occurrence) for each searchable declaration.

    Tracks namespace/section scope to build the qualified name and counts per-simple-name
    occurrences (aligned with re.finditer order in extract_function_from_content /
    _declaration_line). Shared by name matching and body-search fallback.
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
```

- [ ] **Step 2: Rewrite `_matching_declaration_names` to use it**

Replace the body of `_matching_declaration_names` (keep the signature and docstring) with:

```python
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
```

- [ ] **Step 3: Add module-level `_collect_matches` and `_search_root`**

Insert immediately before `class LocalLeanSearcher` (currently line 211):

```python
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
        logger.info(
            f"{label}: Found {len(body_decls)} body matches for '{query}' under {root}"
        )
        return _format_results(query, body_decls, config, body_match=True)
    logger.info(f"{label}: No results for '{query}'")
    return f'No declarations matching "{query}" found.'
```

Note: `_body_matching_declarations` and the `body_match` parameter of `_format_results` are added in Tasks 2–4; this task will not run its new code path until then, but the module must still import — so also do Step 4 now.

- [ ] **Step 4: Point `LocalLeanSearcher.search` at `_search_root` and keep the existing log message format**

Replace the body of `LocalLeanSearcher.search` (currently lines 244–283) from the `# Group by ...` comment onward with:

```python
        return _search_root(root, query, self.config)
```

So the method reads in full:

```python
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
```

- [ ] **Step 5: Add a temporary stub so the module imports before Tasks 2–4**

At the end of the module-level functions (just before `class LocalLeanSearcher`), add a stub that Task 3 will replace:

```python
def _body_matching_declarations(content: str, query: str) -> list[tuple[str, str, int]]:
    """Placeholder replaced in Task 3."""
    return []
```

And add `body_match: bool = False` to `_format_results`'s signature now (unused until Task 4) so `_search_root` calls type-check:

```python
def _format_results(
    query: str,
    declarations: list[tuple[str, str, list[tuple[Path, int]]]],
    config: SearchLeanLocalConfig,
    body_match: bool = False,
) -> str:
```

- [ ] **Step 6: Run the existing suite — must stay green (characterization)**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -q`
Expected: PASS (same count as before the refactor; existing log-assertion tests still pass because the name-match path logs `Found N declarations for '...' under ...` and `No results for '...'`).

- [ ] **Step 7: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py
git commit -m "refactor(tools): extract _search_root + _iter_searchable from LocalLeanSearcher"
```

---

## Task 2: `_identifier_match` (whole-identifier body matching)

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Add import + failing test**

In `tests/unit/tools/test_local_lean_search.py`, add `_body_matching_declarations` and `_identifier_match` to the existing `from ax_prover.tools.local_lean_search import (...)` block, then append this test class:

```python
class TestIdentifierMatch:
    def test_matches_standalone_identifier(self):
        assert _identifier_match("query_aux", "  x := query_aux n 0")

    def test_not_matched_inside_longer_identifier(self):
        # trailing 'N' continues the identifier, so it is not a whole-identifier match
        assert not _identifier_match("query_aux", "def query_auxN := 1")

    def test_dot_is_an_identifier_boundary_char(self):
        # '.' is part of a Lean identifier, so "insert" does NOT match inside "Treap.insert"
        assert not _identifier_match("insert", "y := Treap.insert t")

    def test_case_insensitive(self):
        assert _identifier_match("foo", "exact FOO")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestIdentifierMatch -q`
Expected: FAIL (`cannot import name '_identifier_match'`).

- [ ] **Step 3: Implement `_identifier_match`**

In `src/ax_prover/tools/local_lean_search.py`, add near the other module helpers (e.g. after `_KEYWORDS_PATTERN`):

```python
# Characters that make up a Lean identifier; a body token matches only when flanked by
# non-identifier characters (so "query_aux" won't hit inside "query_auxN" and "insert"
# won't hit inside "Treap.insert").
_IDENT_CHAR = r"[0-9A-Za-z_'.]"


def _identifier_match(token: str, text: str) -> bool:
    """True if `token` appears in `text` as a whole identifier (case-insensitive)."""
    pattern = rf"(?<!{_IDENT_CHAR}){re.escape(token)}(?!{_IDENT_CHAR})"
    return re.search(pattern, text, re.IGNORECASE) is not None
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestIdentifierMatch -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): add whole-identifier body match helper"
```

---

## Task 3: `_body_matching_declarations`

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Append failing tests**

In `tests/unit/tools/test_local_lean_search.py`, add this fixture constant near the other inline samples and the test class:

```python
WHERE_BLOCK_LEAN = """import Mathlib

def query (n : Nat) : Nat :=
  query_aux n 0   where query_aux (j acc : Nat) : Nat :=
    if j = 0 then acc else query_aux (j - 1) (acc + j)
"""


class TestBodyMatchingDeclarations:
    def test_name_search_misses_where_helper(self):
        # `query_aux` is not a top-level declaration, so name search returns nothing.
        assert _matching_declaration_names(WHERE_BLOCK_LEAN, "query_aux") == []

    def test_body_search_returns_enclosing_declaration(self):
        assert _body_matching_declarations(WHERE_BLOCK_LEAN, "query_aux") == [
            ("query", "query", 0)
        ]

    def test_body_search_requires_all_tokens(self):
        assert _body_matching_declarations(WHERE_BLOCK_LEAN, "query_aux missing") == []

    def test_body_search_respects_identifier_boundary(self):
        # 'quer' is a prefix, not a whole identifier in the body → no match.
        assert _body_matching_declarations(WHERE_BLOCK_LEAN, "quer") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestBodyMatchingDeclarations -q`
Expected: FAIL (`_body_matching_declarations` returns `[]` from the Task-1 stub, so `test_body_search_returns_enclosing_declaration` fails).

- [ ] **Step 3: Replace the stub with the real implementation**

In `src/ax_prover/tools/local_lean_search.py`, replace the `_body_matching_declarations` stub from Task 1 with:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestBodyMatchingDeclarations -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): body-search fallback finds identifiers used in declaration bodies"
```

---

## Task 4: Wire the fallback end-to-end + `body_match` annotation

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py`
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Append failing integration tests**

In `tests/unit/tools/test_local_lean_search.py`, append:

```python
@pytest.fixture
def where_project(tmp_path):
    root = tmp_path / "proj"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "p"\n')
    (root / "Challenges" / "Q.lean").write_text(WHERE_BLOCK_LEAN)
    return root


class TestBodySearchEndToEnd:
    def test_search_falls_back_to_body_for_where_helper(self, where_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(where_project))
        out = searcher.search("query_aux")
        assert "def query" in out
        assert "matched in body" in out
        assert "Challenges/Q.lean:" in out

    def test_name_match_does_not_use_body_fallback(self, where_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(where_project))
        out = searcher.search("query")
        assert "def query" in out
        assert "matched in body" not in out

    def test_no_match_message_unchanged(self, where_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(where_project))
        assert searcher.search("zzz_nope") == 'No declarations matching "zzz_nope" found.'
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::TestBodySearchEndToEnd -q`
Expected: FAIL on `test_search_falls_back_to_body_for_where_helper` — `"matched in body"` is absent because `_format_results` ignores `body_match`.

- [ ] **Step 3: Implement the `body_match` annotation in `_format_results`**

In `_format_results`, just after computing `header`, add the note; and append it to each primary location header. The function body becomes:

```python
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
```

- [ ] **Step 4: Run to verify pass + full file suite**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -q`
Expected: PASS (all, including the three new end-to-end tests and every pre-existing test).

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat(tools): annotate body-match results and wire fallback through search"
```

---

## Task 5: `search_cslib` tool (scaffold + registration + wiring)

**Files:**
- Create: `src/ax_prover/tools/cslib_search.py`
- Modify: `src/ax_prover/tools/__init__.py`, `src/ax_prover/configs/tools.yaml`, `src/ax_prover/configs/default.yaml`
- Test: `tests/unit/tools/test_cslib_search.py`

- [ ] **Step 1: Write the first failing test**

Create `tests/unit/tools/test_cslib_search.py`:

```python
"""Unit tests for the CSLib search tool."""

import pytest

from ax_prover.tools import create_search_cslib_tool
from ax_prover.tools.cslib_search import (
    CSLIB_SEARCH_TOOL_TYPE,
    CslibSearcher,
    SearchCslibConfig,
)
from ax_prover.tools.registry import TOOL_REGISTRY, create_tool


def _make_project_with_cslib(tmp_path):
    """A lake project whose .lake/packages/cslib holds one namespaced theorem."""
    root = tmp_path / "proj"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "p"\n')
    (root / "Challenges" / "Def_Local.lean").write_text("def ProjectOnlyThing : Nat := 0\n")
    cslib = root / ".lake" / "packages" / "cslib" / "Cslib"
    cslib.mkdir(parents=True)
    (cslib / "Relation.lean").write_text(
        "namespace WellFounded\n\ntheorem ofTransGen (h : True) : True := h\n\nend WellFounded\n"
    )
    return root


class TestCslibBasic:
    def test_finds_namespaced_theorem_qualified(self, tmp_path):
        root = _make_project_with_cslib(tmp_path)
        searcher = CslibSearcher(SearchCslibConfig(), base_folder=str(root))
        out = searcher.search("ofTransGen")
        assert "WellFounded.ofTransGen" in out
        assert "theorem ofTransGen" in out

    def test_tool_type_constant(self):
        assert CSLIB_SEARCH_TOOL_TYPE == "search_cslib"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/unit/tools/test_cslib_search.py -q`
Expected: FAIL (`ModuleNotFoundError: ax_prover.tools.cslib_search`).

- [ ] **Step 3: Create the tool module**

Create `src/ax_prover/tools/cslib_search.py`:

```python
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
        return _search_root(cslib, query, self.config, label="CslibSearch")


class CslibSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Keyword(s) to match against CSLib declaration names (case-insensitive). "
            "Multiple words match names containing all of them, e.g. 'WellFounded ofTransGen'."
        ),
    )


@register_tool(CSLIB_SEARCH_TOOL_TYPE, SearchCslibConfig)
def create_search_cslib_tool(
    config: SearchCslibConfig, base_folder: str = "."
) -> StructuredTool:
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
```

- [ ] **Step 4: Export the factory**

In `src/ax_prover/tools/__init__.py`, add the import and `__all__` entry:

```python
from .cslib_search import create_search_cslib_tool
```

and add `"create_search_cslib_tool",` to the `__all__` list (next to `"create_search_lean_local_tool",`).

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/unit/tools/test_cslib_search.py::TestCslibBasic -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Add a registration test**

Append to `tests/unit/tools/test_cslib_search.py`:

```python
class TestCslibRegistration:
    def test_registered_in_registry(self):
        assert CSLIB_SEARCH_TOOL_TYPE in TOOL_REGISTRY

    @pytest.mark.asyncio
    async def test_create_tool_builds_named_tool(self, tmp_path):
        root = _make_project_with_cslib(tmp_path)
        tool = await create_tool({"tool_type": "search_cslib"}, base_folder=str(root))
        assert tool is not None
        assert tool.name == "search_cslib_tool"
```

Run: `.venv/bin/pytest tests/unit/tools/test_cslib_search.py::TestCslibRegistration -q`
Expected: PASS. (If `test_create_tool_builds_named_tool` errors with "async def not natively supported", confirm the repo's pytest-asyncio config; other async tests in `tests/unit/tools/test_lean_search.py` use the same style, so it is configured.)

- [ ] **Step 7: Wire into bundled configs**

In `src/ax_prover/configs/tools.yaml`, add under `tool_configs:` (after the `search_lean_local` block):

```yaml
  search_cslib:
    tool_type: search_cslib
    max_results: 6
    max_chars: 4000
```

In `src/ax_prover/configs/default.yaml`, add to `proposer_tools:`:

```yaml
    search_cslib: ${tool_configs.search_cslib}
```

- [ ] **Step 8: Commit**

```bash
git add src/ax_prover/tools/cslib_search.py src/ax_prover/tools/__init__.py \
        src/ax_prover/configs/tools.yaml src/ax_prover/configs/default.yaml \
        tests/unit/tools/test_cslib_search.py
git commit -m "feat(tools): add search_cslib tool and wire into bundled configs"
```

---

## Task 6: CSLib tricky-retrieval test matrix

The tool exists; these tests pin down the required behaviors. If any fails, fix the cause (likely in `cslib_search.py` or the shared `_search_root`) before moving on.

**Files:**
- Test: `tests/unit/tools/test_cslib_search.py`

- [ ] **Step 1: Add tricky fixtures + tests**

Append to `tests/unit/tools/test_cslib_search.py`:

```python
from ax_prover.tools.local_lean_search import LocalLeanSearcher, SearchLeanLocalConfig

# CSLib-shaped content: module header, public import, namespace, doc + own-line @[simp],
# noncomputable, private. `measure` here collides (different namespace/body) with the SKI file.
CSLIB_REL = """module

public import Cslib.Init

namespace WellFounded

/-- Transitive-closure well-foundedness. -/
@[simp]
theorem ofTransGen (h : True) : True := h

noncomputable def measure : Nat := 0

private theorem secret : True := trivial

end WellFounded
"""

# Inductive with distinctive constructor names (Iterm is only a constructor, not a decl).
CSLIB_SKI = """module
namespace CombinatoryLogic

inductive SKI where
  | Sterm | Kterm | Iterm
  | app : SKI -> SKI -> SKI

def measure : SKI -> Nat
  | _ => 1

end CombinatoryLogic
"""


def _make_rich_cslib(tmp_path, *, with_mathlib=False):
    root = tmp_path / "proj"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "p"\n')
    (root / "Challenges" / "Def_Local.lean").write_text("def ProjectOnlyThing : Nat := 0\n")
    cslib = root / ".lake" / "packages" / "cslib" / "Cslib"
    cslib.mkdir(parents=True)
    (cslib / "Relation.lean").write_text(CSLIB_REL)
    (cslib / "SKI.lean").write_text(CSLIB_SKI)
    if with_mathlib:
        mathlib = root / ".lake" / "packages" / "mathlib" / "Mathlib"
        mathlib.mkdir(parents=True)
        (mathlib / "X.lean").write_text("def MathlibOnly : Nat := 1\n")
    return root


class TestCslibIsolation:
    def test_cslib_tool_excludes_project_files(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("ProjectOnlyThing")
        assert out == 'No declarations matching "ProjectOnlyThing" found.'

    def test_cslib_tool_excludes_sibling_mathlib(self, tmp_path):
        root = _make_rich_cslib(tmp_path, with_mathlib=True)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("MathlibOnly")
        assert out == 'No declarations matching "MathlibOnly" found.'

    def test_local_tool_excludes_cslib(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root)).search("ofTransGen")
        assert out == 'No declarations matching "ofTransGen" found.'

    def test_local_tool_finds_project_file(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root)).search(
            "ProjectOnlyThing"
        )
        assert "def ProjectOnlyThing" in out


class TestCslibModuleShapes:
    def test_own_line_attribute_decl_found_and_block_includes_attr(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("ofTransGen")
        assert "WellFounded.ofTransGen" in out
        assert "theorem ofTransGen" in out
        # Block extraction backscans to the doc comment, pulling the own-line @[simp] in too.
        assert "@[simp]" in out

    def test_module_and_public_import_do_not_create_bogus_results(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("Cslib")
        # `public import Cslib.Init` is an import, not a searchable declaration.
        assert out == 'No declarations matching "Cslib" found.'


class TestCslibNamespaceAndDuplicates:
    def test_multiword_qualified_query(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search(
            "WellFounded ofTransGen"
        )
        assert "WellFounded.ofTransGen" in out

    def test_duplicate_simple_name_across_files_resolves_distinctly(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("measure")
        assert "WellFounded.measure" in out
        assert "CombinatoryLogic.measure" in out
        assert ":= 0" in out  # WellFounded.measure body
        assert "SKI -> Nat" in out  # CombinatoryLogic.measure body


class TestCslibModifiers:
    def test_private_theorem_found(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("secret")
        assert "WellFounded.secret" in out

    def test_noncomputable_def_found(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("measure")
        assert "noncomputable def measure" in out


class TestCslibBodyAndConstructors:
    def test_inductive_constructor_found_via_body(self, tmp_path):
        # `Iterm` is a constructor, not a top-level decl: name search misses, body search hits SKI.
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("Iterm")
        assert "inductive SKI" in out
        assert "matched in body" in out


class TestCslibEdges:
    def test_missing_cslib_dir_message(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "lakefile.toml").write_text('name = "p"\n')
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("anything")
        assert "CSLib search unavailable" in out
        assert ".lake/packages/cslib" in out

    def test_resolves_from_subdirectory_walk_up(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        sub = root / "Challenges"
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(sub)).search("ofTransGen")
        assert "WellFounded.ofTransGen" in out

    def test_resolves_from_parent_walk_down(self, tmp_path):
        _make_rich_cslib(tmp_path)  # creates tmp_path/proj
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(tmp_path)).search("ofTransGen")
        assert "WellFounded.ofTransGen" in out

    def test_caps_overflow_lists_extras(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(
            SearchCslibConfig(max_results=1), base_folder=str(root)
        ).search("measure")
        assert "Additional matches" in out
```

- [ ] **Step 2: Run the full CSLib suite**

Run: `.venv/bin/pytest tests/unit/tools/test_cslib_search.py -q`
Expected: PASS. If `test_module_and_public_import_do_not_create_bogus_results` fails because `public import Cslib.Init` is being surfaced, that is acceptable behavior to adjust — the import is parsed as an `import` declaration (not in `SEARCHABLE_TYPES`), so it should already be excluded; if not, investigate `_iter_searchable`. If `test_inductive_constructor_found_via_body` fails, verify the body-search fallback runs for the cslib root (it shares `_search_root`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/tools/test_cslib_search.py
git commit -m "test(tools): tricky CSLib retrieval matrix (isolation, module shapes, dups, body, edges)"
```

---

## Task 7: Full verification + push + PR A

**Files:** none (verification + git)

- [ ] **Step 1: Full unit suite**

Run: `.venv/bin/pytest tests/unit -q`
Expected: PASS (all green).

- [ ] **Step 2: Format + lint**

Run:
```bash
.venv/bin/ruff format src/ax_prover tests/unit
.venv/bin/ruff check src/ax_prover/tools/local_lean_search.py src/ax_prover/tools/cslib_search.py tests/unit/tools/test_cslib_search.py tests/unit/tools/test_local_lean_search.py
```
Expected: "All checks passed!" (commit any reformat with `git commit -am "style: ruff format"`).

- [ ] **Step 3: End-to-end smoke against the real dataset (read-only)**

Run:
```bash
.venv/bin/python -c "
from ax_prover.tools.local_lean_search import LocalLeanSearcher, SearchLeanLocalConfig
from ax_prover.tools.cslib_search import CslibSearcher, SearchCslibConfig
base='/Users/krystian/Documents/Axiomatic/Baku/AI4Math/challenges'
print('--- query_aux (body fallback) ---')
print(LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=base).search('query_aux')[:400])
print('--- CSLib WellFounded ---')
print(CslibSearcher(SearchCslibConfig(), base_folder=base).search('WellFounded')[:400])
"
```
Expected: `query_aux` returns the `query` declaration flagged `(matched in body)`; the CSLib query returns a real CSLib declaration block.

- [ ] **Step 4: Push and open PR A**

```bash
git push -u origin local-search-body-and-cslib
gh pr create --base local-lean-search-tool --head local-search-body-and-cslib \
  --title "Local search: body-search fallback + CSLib search tool" \
  --body "Adds a body-search fallback (finds where/let-rec helpers and other body identifiers when no declaration name matches) and a new search_cslib tool that searches the vendored CSLib dependency. Includes a tricky CSLib retrieval test matrix. Design: docs/superpowers/specs/2026-06-05-local-search-coverage-and-cslib-design.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Follow-up: PR B — proposer prompt Lean 4.28 (separate branch off `main`)

Not part of PR A. On a fresh branch off `main`:

- [ ] In `src/ax_prover/prover/prompts.py` line 6, change `(Lean version 4.24)` → `(Lean version 4.28)`.
- [ ] In `src/ax_prover/prover/prompts.py` line 127, change `(Lean version 4.24)` → `(Lean version 4.28)`.
- [ ] `.venv/bin/pytest tests/unit -q` (green), commit `fix(prompts): target Lean 4.28 to match the AI4Math dataset`, push, open PR B → `main`.

## Integration (after PR A + PR B merge)

- [ ] Merge both into `run_AI4Math_Challange`.
- [ ] On `run_AI4Math_Challange`, add `search_cslib: ${tool_configs.search_cslib}` to `proposer_tools` in the exp-only `configs/opus48_local.yaml`, and add the `search_cslib` tool_config to the repo-level `configs/tools.yaml` (the exp branch's copies), so the next experiment run uses the CSLib tool. Verify with `merge_configs(['opus48_local.yaml'], folder='.')`.
