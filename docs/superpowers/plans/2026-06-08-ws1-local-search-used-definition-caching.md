# WS1 — Local-search used-definition caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the prover re-querying the same local definitions every iteration by caching, verbatim and append-only, the local-search results the proof actually used, and injecting them into each subsequent proposer prompt.

**Architecture:** The persistent `LocalLeanSearcher` records every declaration it returns this run. After each attempt, the memory node deterministically intersects that pool with the identifiers used in the proposed code (excluding the target), appends the verbatim definitions to a run-scoped `used_definitions` cache on the graph state, and the proposer renders them in a `<local-definitions>` block. The cache bypasses the experience-summarizer LLM so Lean source stays byte-exact.

**Tech Stack:** Python 3.12, LangChain/LangGraph, Pydantic, pytest. Branch off `local-lean-search-tool`; PR back into `local-lean-search-tool`.

**Spec:** `docs/superpowers/specs/2026-06-08-local-search-caching-and-fuzzy-design.md`

---

## File structure

- `src/ax_prover/tools/local_lean_search.py` — `_search_root` returns `(text, decls)`; `LocalLeanSearcher.returned_declarations` + `_record_returned`; pure helpers `identifier_in_code`, `format_cached_definition_entry`, `accumulate_used_definitions`; caps `MAX_CACHED_DEFINITIONS`, `MAX_CACHED_DEFINITION_CHARS`.
- `src/ax_prover/tools/cslib_search.py` — unpack `(text, _)` from `_search_root` (CSLib results are never cached).
- `src/ax_prover/models/proving.py` — new `used_definitions: dict[str, str]` state field.
- `src/ax_prover/prover/prompts.py` — `LOCAL_DEFINITIONS_USER_PROMPT` template + one nudge line in the iterative proposer system prompt.
- `src/ax_prover/prover/agent.py` — capture `self._local_searcher`; `_find_local_searcher`; `_accumulate_used_definitions`; merge in `_memory_processor_node`; inject the block in `_proposer_node`.
- Tests: `tests/unit/tools/test_local_lean_search.py` (extend), `tests/unit/prover/test_used_definition_caching.py` (new).

Before starting: create the branch.

```bash
git checkout local-lean-search-tool && git pull --ff-only
git checkout -b ws1-local-search-used-definition-caching
```

---

## Task 1: `_search_root` returns structured declarations alongside text

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py` (function `_search_root`, ~lines 309-327)
- Modify: `src/ax_prover/tools/cslib_search.py` (method `CslibSearcher.search`, ~line 72)
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/tools/test_local_lean_search.py`:

```python
from pathlib import Path

from ax_prover.tools.local_lean_search import SearchLeanLocalConfig, _search_root


def _write(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_search_root_returns_text_and_decls_on_name_match(tmp_path):
    _write(tmp_path, "Def.lean", "def extract_min : Nat := 0\n")
    text, decls = _search_root(tmp_path, "extract_min", SearchLeanLocalConfig())
    assert "extract_min" in text
    assert len(decls) == 1
    qualified_name, block, locations = decls[0]
    assert qualified_name == "extract_min"
    assert "def extract_min" in block
    assert locations and locations[0][1] == 1  # (path, line)


def test_search_root_returns_empty_decls_on_no_match(tmp_path):
    _write(tmp_path, "Def.lean", "def foo : Nat := 0\n")
    text, decls = _search_root(tmp_path, "nonexistent_name", SearchLeanLocalConfig())
    assert "No declarations matching" in text
    assert decls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py::test_search_root_returns_text_and_decls_on_name_match -v`
Expected: FAIL — `ValueError: too many values to unpack` (currently returns a bare `str`).

- [ ] **Step 3: Change `_search_root` to return a tuple**

Replace the body of `_search_root` (keep the docstring) so every return path yields `(text, decls)`:

```python
def _search_root(
    root: Path, query: str, config: SearchLeanLocalConfig, *, label: str = "LocalLeanSearch"
) -> tuple[str, list[tuple[str, str, list[tuple[Path, int]]]]]:
    """Search `root` by declaration name; fall back to body search only if name finds nothing.

    Returns the formatted text plus the structured declarations it formatted (empty on no match).
    Walks the tree and reads each file once; the body fallback reuses the cached contents.
    """
    files = _read_lean_files(root)
    decls = _collect_matches(files, query, body=False)
    if decls:
        logger.info(f"{label}: Found {len(decls)} declarations for '{query}' under {root}")
        return _format_results(query, decls, config), decls
    body_decls = _collect_matches(files, query, body=True)
    if body_decls:
        logger.info(f"{label}: Found {len(body_decls)} body matches for '{query}' under {root}")
        return _format_results(query, body_decls, config, body_match=True), body_decls
    logger.info(f"{label}: No results for '{query}'")
    return f'No declarations matching "{query}" found.', []
```

- [ ] **Step 4: Update both callers to unpack the tuple**

In `src/ax_prover/tools/local_lean_search.py`, `LocalLeanSearcher.search` currently ends with `return _search_root(root, query, self.config)`. Change that line to:

```python
        text, _decls = _search_root(root, query, self.config)
        return text
```

(Task 3 replaces `_decls` with real recording; this keeps the build green meanwhile.)

In `src/ax_prover/tools/cslib_search.py`, `CslibSearcher.search` currently ends with `return _search_root(cslib, query, self.config, label="CslibSearch")`. Change that line to:

```python
        text, _decls = _search_root(cslib, query, self.config, label="CslibSearch")
        return text
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py tests/unit/tools/test_cslib_search.py -q`
Expected: PASS (new tests pass; all existing local/cslib search tests still pass).

- [ ] **Step 6: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py src/ax_prover/tools/cslib_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "refactor: _search_root returns (text, decls); callers unpack"
```

---

## Task 2: Pure caching helpers (`identifier_in_code`, `format_cached_definition_entry`, `accumulate_used_definitions`)

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py` (add caps + 3 functions, near the other module-level helpers, e.g. after `_identifier_match`)
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import (
    MAX_CACHED_DEFINITIONS,
    accumulate_used_definitions,
    format_cached_definition_entry,
    identifier_in_code,
)


def test_identifier_in_code_matches_qualified_and_simple():
    assert identifier_in_code("BinaryHeap.extract_min", "  exact extract_min h")
    assert identifier_in_code("BinaryHeap.extract_min", "  exact BinaryHeap.extract_min h")
    assert not identifier_in_code("extract_min", "  exact extract_minimum h")  # not whole-identifier
    assert not identifier_in_code("heapify", "  simp")


def test_format_cached_definition_entry_has_location_header_and_block():
    entry = format_cached_definition_entry("A.foo", "def foo := 1", Path("Def.lean"), 3)
    assert entry == "-- A.foo — Def.lean:3\ndef foo := 1"


def _pool():
    return {
        "BinaryHeap.extract_min": ("def extract_min : Nat := 0", Path("Def.lean"), 5),
        "BinaryHeap.heapify": ("def heapify : Nat := 1", Path("Def.lean"), 9),
    }


def test_accumulate_adds_only_used_definitions():
    code = "theorem t : True := by have := extract_min; trivial"
    result = accumulate_used_definitions({}, _pool(), code, target_name="t")
    assert "BinaryHeap.extract_min" in result
    assert "BinaryHeap.heapify" not in result  # returned but not used in code


def test_accumulate_excludes_target_by_simple_name():
    pool = {"Foo.extract_min": ("def extract_min := 0", Path("Def.lean"), 1)}
    code = "theorem extract_min : True := by exact extract_min"
    result = accumulate_used_definitions({}, pool, code, target_name="extract_min")
    assert result == {}  # the only candidate shares the target's simple name


def test_accumulate_is_monotonic_and_dedups():
    prior = {"BinaryHeap.heapify": "-- cached earlier"}
    code = "theorem t : True := by exact extract_min"  # heapify NOT used now
    result = accumulate_used_definitions(prior, _pool(), code, target_name="t")
    assert result["BinaryHeap.heapify"] == "-- cached earlier"  # preserved, not wiped
    assert "BinaryHeap.extract_min" in result  # newly used, added
    # Re-running with the same pool does not duplicate or change existing entries.
    again = accumulate_used_definitions(result, _pool(), code, target_name="t")
    assert again == result


def test_accumulate_respects_count_cap():
    pool = {
        f"N.def_{i}": (f"def def_{i} := {i}", Path("Def.lean"), i + 1)
        for i in range(MAX_CACHED_DEFINITIONS + 5)
    }
    code = " ".join(f"def_{i}" for i in range(MAX_CACHED_DEFINITIONS + 5))
    result = accumulate_used_definitions({}, pool, code, target_name="t")
    assert len(result) == MAX_CACHED_DEFINITIONS


def test_accumulate_respects_char_cap():
    big_block = "x" * 9000  # two of these exceed the 12000-char cap
    pool = {
        "N.a": (big_block, Path("Def.lean"), 1),
        "N.b": (big_block, Path("Def.lean"), 2),
    }
    code = "a b"  # both 'a' and 'b' referenced as whole identifiers
    result = accumulate_used_definitions({}, pool, code, target_name="t")
    assert len(result) == 1  # second entry would blow the char budget
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "identifier_in_code or format_cached or accumulate" -v`
Expected: FAIL — `ImportError: cannot import name 'accumulate_used_definitions'`.

- [ ] **Step 3: Implement the caps and helpers**

In `src/ax_prover/tools/local_lean_search.py`, add the caps near the other module constants (after `EXCLUDED_DIR`):

```python
# Caps for the run-scoped cache of used local definitions (consumed by the memory node).
MAX_CACHED_DEFINITIONS = 24
MAX_CACHED_DEFINITION_CHARS = 12000
```

Add these functions after `_identifier_match` (which they reuse):

```python
def identifier_in_code(name: str, code: str) -> bool:
    """True if `name` or its final dotted segment appears in `code` as a whole identifier."""
    candidates = {name, name.rsplit(".", 1)[-1]}
    return any(_identifier_match(candidate, code) for candidate in candidates)


def format_cached_definition_entry(qualified_name: str, block: str, path: Path, line: int) -> str:
    """Render one cached definition as a located, verbatim source entry."""
    return f"-- {qualified_name} — {path}:{line}\n{block}"


def accumulate_used_definitions(
    cached: dict[str, str],
    returned_declarations: dict[str, tuple[str, "Path", int]],
    code: str,
    target_name: str | None,
) -> dict[str, str]:
    """Append local-search results that `code` actually used to the run-scoped `cached` map.

    Append-only and deduped by qualified name. An entry is added only when the local-search tool
    returned it (it is in `returned_declarations`) AND it is referenced in `code` as a whole
    identifier. The target theorem's own simple name is excluded. Respects MAX_CACHED_DEFINITIONS
    and MAX_CACHED_DEFINITION_CHARS; existing entries are never removed.
    """
    merged = dict(cached)
    total_chars = sum(len(entry) for entry in merged.values())
    target_simple = target_name.rsplit(".", 1)[-1] if target_name else None
    for qualified_name, (block, path, line) in returned_declarations.items():
        if qualified_name in merged:
            continue
        if len(merged) >= MAX_CACHED_DEFINITIONS:
            break
        if target_simple and qualified_name.rsplit(".", 1)[-1] == target_simple:
            continue
        if not identifier_in_code(qualified_name, code):
            continue
        entry = format_cached_definition_entry(qualified_name, block, path, line)
        if total_chars + len(entry) > MAX_CACHED_DEFINITION_CHARS:
            break
        merged[qualified_name] = entry
        total_chars += len(entry)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "identifier_in_code or format_cached or accumulate" -v`
Expected: PASS (all 7 new tests).

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat: pure helpers for run-scoped used-definition cache"
```

---

## Task 3: `LocalLeanSearcher` records returned declarations

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py` (`LocalLeanSearcher.__init__` ~line 336, `.search` ~line 363; add `_record_returned`)
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import LocalLeanSearcher


def _make_lake_project(tmp_path) -> str:
    (tmp_path / "lakefile.toml").write_text("name = \"demo\"\n", encoding="utf-8")
    (tmp_path / "Def.lean").write_text(
        "def extract_min : Nat := 0\ndef heapify : Nat := 1\n", encoding="utf-8"
    )
    return str(tmp_path)


def test_searcher_accumulates_returned_declarations_across_calls(tmp_path):
    base = _make_lake_project(tmp_path)
    searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=base)

    searcher.search("extract_min")
    assert "extract_min" in searcher.returned_declarations
    block, path, line = searcher.returned_declarations["extract_min"]
    assert "def extract_min" in block

    searcher.search("heapify")
    # First result is retained; second is added (accumulation, not replacement).
    assert set(searcher.returned_declarations) == {"extract_min", "heapify"}


def test_searcher_records_nothing_on_miss(tmp_path):
    base = _make_lake_project(tmp_path)
    searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=base)
    searcher.search("totally_absent_name")
    assert searcher.returned_declarations == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "searcher_accumulates or searcher_records_nothing" -v`
Expected: FAIL — `AttributeError: 'LocalLeanSearcher' object has no attribute 'returned_declarations'`.

- [ ] **Step 3: Implement recording**

In `LocalLeanSearcher.__init__`, add the field after `self._resolution = None`:

```python
        self.returned_declarations: dict[str, tuple[str, Path, int]] = {}
```

Replace the final two lines of `LocalLeanSearcher.search` (`text, _decls = _search_root(...)` / `return text` from Task 1) with:

```python
        text, decls = _search_root(root, query, self.config)
        self._record_returned(decls)
        return text
```

Add the method to `LocalLeanSearcher`:

```python
    def _record_returned(
        self, decls: list[tuple[str, str, list[tuple[Path, int]]]]
    ) -> None:
        """Accumulate returned declarations for the run, keyed by qualified name (first-seen wins)."""
        for qualified_name, block, locations in decls:
            if qualified_name in self.returned_declarations or not locations:
                continue
            path, line = locations[0]
            self.returned_declarations[qualified_name] = (block, path, line)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -q`
Expected: PASS (full file green).

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat: LocalLeanSearcher records returned declarations for the run"
```

---

## Task 4: `used_definitions` state field

**Files:**
- Modify: `src/ax_prover/models/proving.py` (after the `experience` field, ~line 73)
- Test: `tests/unit/prover/test_used_definition_caching.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/prover/test_used_definition_caching.py`:

```python
"""Tests for the WS1 used-definition caching feature."""

from ax_prover.models.proving import ProverAgentState
from ax_prover.models.proving import TargetItem


def _state() -> ProverAgentState:
    return ProverAgentState(item=TargetItem(title="t"))


def test_used_definitions_defaults_to_empty_dict():
    state = _state()
    assert state.used_definitions == {}


def test_used_definitions_accepts_mapping():
    state = ProverAgentState(
        item=TargetItem(title="t"),
        used_definitions={"A.foo": "-- A.foo — Def.lean:1\ndef foo := 1"},
    )
    assert "A.foo" in state.used_definitions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/prover/test_used_definition_caching.py -v`
Expected: FAIL — `ValidationError` / `AttributeError` (`used_definitions` not a field).

- [ ] **Step 3: Add the field**

In `src/ax_prover/models/proving.py`, insert after the `experience` field (before `summary`):

```python
    used_definitions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Run-scoped cache of local-search definitions the proof actually used, keyed by "
            "qualified name → rendered verbatim entry. Append-only across iterations."
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/prover/test_used_definition_caching.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/models/proving.py tests/unit/prover/test_used_definition_caching.py
git commit -m "feat: add used_definitions field to ProverAgentState"
```

---

## Task 5: Proposer prompt template + nudge

**Files:**
- Modify: `src/ax_prover/prover/prompts.py` (add `LOCAL_DEFINITIONS_USER_PROMPT`; add one nudge line in `PROPOSER_SYSTEM_PROMPT`)
- Test: `tests/unit/prover/test_used_definition_caching.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/prover/test_used_definition_caching.py`:

```python
def test_local_definitions_prompt_wraps_definitions():
    from ax_prover.prover.prompts import LOCAL_DEFINITIONS_USER_PROMPT

    rendered = LOCAL_DEFINITIONS_USER_PROMPT.format(definitions="-- A.foo\ndef foo := 1")
    assert "<local-definitions>" in rendered
    assert "</local-definitions>" in rendered
    assert "def foo := 1" in rendered


def test_iterative_system_prompt_mentions_local_definitions():
    from ax_prover.prover.prompts import PROPOSER_SYSTEM_PROMPT

    assert "<local-definitions>" in PROPOSER_SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/prover/test_used_definition_caching.py -k "local_definitions_prompt or system_prompt_mentions" -v`
Expected: FAIL — `ImportError` for `LOCAL_DEFINITIONS_USER_PROMPT`.

- [ ] **Step 3: Add the template and the nudge**

In `src/ax_prover/prover/prompts.py`, add after `PROPOSER_USER_PROMPT` (the block ending at the closing `"""` near line 220):

```python
LOCAL_DEFINITIONS_USER_PROMPT = """
These project definitions were referenced in your previous attempts and are provided verbatim, so you do NOT need to search for them again:

<local-definitions>
{definitions}
</local-definitions>
"""
```

In `PROPOSER_SYSTEM_PROMPT` only (the iterative prompt), inside the `<requirements>` "Quality Standards" list, add a bullet immediately after the existing "When searching for lemmas, match the search tool to the file's imports…" line:

```
    - Project definitions you have already used in previous attempts are provided verbatim in the <local-definitions> section of the message — reference them directly and do NOT search for them again.
```

(Do not add this to `PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT`: single-shot runs never populate `used_definitions`, so the section never appears there.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/prover/test_used_definition_caching.py -k "local_definitions_prompt or system_prompt_mentions" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/prover/prompts.py tests/unit/prover/test_used_definition_caching.py
git commit -m "feat: local-definitions prompt template and proposer nudge"
```

---

## Task 6: Wire the searcher, accumulation, and injection into the agent

**Files:**
- Modify: `src/ax_prover/prover/agent.py` (imports; `__init__` ~line 94; `create` ~line 135; new `_find_local_searcher` and `_accumulate_used_definitions`; `_memory_processor_node` ~line 238; `_proposer_node` experience block ~line 280)
- Test: `tests/unit/prover/test_used_definition_caching.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/prover/test_used_definition_caching.py`:

```python
import inspect
from pathlib import Path
from types import SimpleNamespace


def test_find_local_searcher_locates_the_searcher(tmp_path):
    from ax_prover.prover.agent import ProverAgent
    from ax_prover.tools.local_lean_search import (
        LocalLeanSearcher,
        SearchLeanLocalConfig,
        create_search_lean_local_tool,
    )

    (tmp_path / "lakefile.toml").write_text("name = \"demo\"\n", encoding="utf-8")
    tool = create_search_lean_local_tool(SearchLeanLocalConfig(), base_folder=str(tmp_path))
    fake = SimpleNamespace(proposer_tools=[tool])

    searcher = ProverAgent._find_local_searcher(fake)
    assert isinstance(searcher, LocalLeanSearcher)


def test_find_local_searcher_returns_none_when_absent():
    from ax_prover.prover.agent import ProverAgent

    fake = SimpleNamespace(proposer_tools=[])
    assert ProverAgent._find_local_searcher(fake) is None


def test_memory_node_wires_in_accumulation():
    from ax_prover.prover.agent import ProverAgent

    source = inspect.getsource(ProverAgent._memory_processor_node)
    assert "_accumulate_used_definitions(" in source
    assert "used_definitions" in source


def test_proposer_node_injects_local_definitions():
    from ax_prover.prover.agent import ProverAgent

    source = inspect.getsource(ProverAgent._proposer_node)
    assert "state.used_definitions" in source
    assert "LOCAL_DEFINITIONS_USER_PROMPT" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/prover/test_used_definition_caching.py -k "find_local_searcher or memory_node_wires or injects_local" -v`
Expected: FAIL — `AttributeError: type object 'ProverAgent' has no attribute '_find_local_searcher'` (and source assertions fail).

- [ ] **Step 3: Add imports**

In `src/ax_prover/prover/agent.py`, near the existing `from ..tools import create_tool` import, add:

```python
from ..tools.local_lean_search import (
    LOCAL_LEAN_SEARCH_TOOL_TYPE,
    LocalLeanSearcher,
    accumulate_used_definitions,
)
from ..tools.registry import tool_name_from_type
```

In the prompts import block (where `PROPOSER_SYSTEM_PROMPT`, `PROPOSER_USER_PROMPT`, etc. are imported from `.prompts`), add `LOCAL_DEFINITIONS_USER_PROMPT` to that import list.

- [ ] **Step 4: Initialise `_local_searcher` and capture it after tool creation**

In `__init__`, immediately after `self.proposer_tools = []` (~line 94) add:

```python
        self._local_searcher: LocalLeanSearcher | None = None
```

In the `create` classmethod, after `instance.proposer_tools = await instance._create_tools()` and before `instance.app = instance._build_graph()`:

```python
        instance._local_searcher = instance._find_local_searcher()
```

- [ ] **Step 5: Add `_find_local_searcher` and `_accumulate_used_definitions` methods**

Add these two methods to `ProverAgent` (e.g. just above `_memory_processor_node`):

```python
    def _find_local_searcher(self) -> LocalLeanSearcher | None:
        """Locate the live LocalLeanSearcher behind the local-search tool, if configured."""
        local_tool_name = tool_name_from_type(LOCAL_LEAN_SEARCH_TOOL_TYPE)
        for tool in self.proposer_tools:
            if tool.name == local_tool_name:
                bound = getattr(getattr(tool, "func", None), "__self__", None)
                if isinstance(bound, LocalLeanSearcher):
                    return bound
        return None

    def _accumulate_used_definitions(self, state: ProverAgentState) -> dict[str, str]:
        """Append local-search results referenced by the latest proposal to the run cache."""
        if self._local_searcher is None or not state.last_proposal:
            return dict(state.used_definitions)
        target_name = state.item.location.name if state.item.location else None
        return accumulate_used_definitions(
            state.used_definitions,
            self._local_searcher.returned_declarations,
            state.last_proposal.code,
            target_name,
        )
```

- [ ] **Step 6: Merge accumulation into the memory node**

Replace `_memory_processor_node`:

```python
    async def _memory_processor_node(self, state: ProverAgentState) -> dict:
        """Process memory using the configured strategy, plus accumulate used local definitions."""
        result = await self.memory.process(state)
        result["used_definitions"] = self._accumulate_used_definitions(state)
        return result
```

- [ ] **Step 7: Inject the block in the proposer node**

In `_proposer_node`, right after the existing experience block:

```python
        if state.experience:
            query = "\n\n".join([query, state.experience])
```

add:

```python
        if state.used_definitions:
            definitions_block = LOCAL_DEFINITIONS_USER_PROMPT.format(
                definitions="\n\n".join(state.used_definitions.values())
            )
            query = "\n\n".join([query, definitions_block])
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/prover/test_used_definition_caching.py -v`
Expected: PASS (all WS1 prover tests).

- [ ] **Step 9: Commit**

```bash
git add src/ax_prover/prover/agent.py tests/unit/prover/test_used_definition_caching.py
git commit -m "feat: wire used-definition cache through memory and proposer nodes"
```

---

## Task 7: Full verification

- [ ] **Step 1: Run the whole unit suite**

Run: `.venv/bin/pytest tests/unit -q`
Expected: PASS (no regressions; previously ~395 tests + the new ones).

- [ ] **Step 2: Lint and format**

Run: `.venv/bin/ruff format . && .venv/bin/ruff check --fix .`
Expected: "All checks passed!" and files formatted. Commit any formatting changes:

```bash
git add -A && git commit -m "style: ruff format/lint" || echo "nothing to format"
```

- [ ] **Step 3: Push and open the PR into `local-lean-search-tool`**

```bash
git push -u origin ws1-local-search-used-definition-caching
gh pr create --base local-lean-search-tool --title "feat: cache used local-search definitions across iterations (WS1)" --body "$(cat <<'EOF'
Caches, verbatim and append-only, the local-search results the proof actually used, and injects them into each subsequent proposer prompt so the agent stops re-querying the same definitions every iteration.

- LocalLeanSearcher records returned declarations for the run
- Memory node intersects that pool with identifiers used in the proposed code (excludes the target), accumulating into a new `used_definitions` state field
- Proposer renders a `<local-definitions>` block; verbatim source bypasses the experience-summarizer LLM
- CSLib/Mathlib results are never cached

Spec: docs/superpowers/specs/2026-06-08-local-search-caching-and-fuzzy-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes (author)

- **Spec coverage:** WS1 components 1-6 map to Tasks 3, 6, 4, 6, 5, 2 respectively; caps in Task 2; accumulation guarantee tested in `test_accumulate_is_monotonic_and_dedups`. CSLib exclusion covered by Task 1 Step 4 (CslibSearcher discards decls) and verified by existing cslib tests.
- **Type consistency:** `returned_declarations: dict[str, tuple[str, Path, int]]` (qualified → block, path, line) is produced by `_record_returned` (Task 3) and consumed by `accumulate_used_definitions` (Task 2) with the identical shape. `_search_root` decls shape `(qualified_name, block, locations)` matches `_collect_matches`'s existing output.
- **No build break between tasks:** Task 1 Step 4 unpacks into `_decls` so callers compile before Task 3 introduces real recording.
