# WS2 — Local-search fuzzy fallback matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When exact name matching finds nothing, fall back to a fuzzy name match (closest declarations) so the agent recovers from wrong-name guesses — without polluting good exact searches or bloating context.

**Architecture:** Add a middle tier to `_search_root`: exact name match → (new) fuzzy name match → body-identifier match. Fuzzy matching normalizes names and the query into lowercased tokens (split on `.`/`_`/camelCase), scores with `difflib` plus best-token overlap, keeps the top-N above a threshold, and renders under a clear "No exact match — closest declarations" header capped by the existing result/char budget.

**Tech Stack:** Python 3.12 (`difflib`), pytest. Branch off `local-lean-search-tool` **after WS1 is merged** (both touch `_search_root`).

**Spec:** `docs/superpowers/specs/2026-06-08-local-search-caching-and-fuzzy-design.md`

---

## File structure

- `src/ax_prover/tools/local_lean_search.py` — `FUZZY_THRESHOLD`; `_normalize_tokens`, `_fuzzy_score`, `_fuzzy_matching_declarations`, `_collect_fuzzy_matches`; new fuzzy tier in `_search_root`; `fuzzy` flag in `_format_results`; tool + `query` description updates.
- Test: `tests/unit/tools/test_local_lean_search.py` (extend).

Before starting (WS1 must be merged into `local-lean-search-tool` first):

```bash
git checkout local-lean-search-tool && git pull --ff-only
git checkout -b ws2-local-search-fuzzy-fallback
```

---

## Task 1: Scoring primitives (`_normalize_tokens`, `_fuzzy_score`)

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py` (add `from difflib import SequenceMatcher`; constant + 2 functions near `_identifier_match`)
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import _fuzzy_score, _normalize_tokens


def test_normalize_tokens_splits_camel_snake_and_qualifier():
    assert _normalize_tokens("BinaryHeap.decreaseKey") == ["decrease", "key"]
    assert _normalize_tokens("extract_min") == ["extract", "min"]


def test_fuzzy_score_high_for_near_miss():
    # Underscore/spelling differences still score high.
    assert _fuzzy_score("extractmin", "BinaryHeap.extract_min") >= 0.8
    assert _fuzzy_score("decrese_min", "decrease_min") >= 0.7  # typo


def test_fuzzy_score_rewards_single_token_match():
    # Query matches one token of a compound name.
    assert _fuzzy_score("priority", "decreasePriority") >= 0.6


def test_fuzzy_score_low_for_unrelated():
    assert _fuzzy_score("heapify", "WeightedGraph") < 0.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "normalize_tokens or fuzzy_score" -v`
Expected: FAIL — `ImportError: cannot import name '_fuzzy_score'`.

- [ ] **Step 3: Implement the scoring primitives**

At the top of `src/ax_prover/tools/local_lean_search.py`, add to the imports:

```python
from difflib import SequenceMatcher
```

Add near `_identifier_match`:

```python
# Minimum similarity for a fuzzy name suggestion when exact matching finds nothing.
FUZZY_THRESHOLD = 0.6


def _normalize_tokens(name: str) -> list[str]:
    """Lowercased identifier tokens: drop the namespace qualifier, split on '_' and camelCase."""
    simple = name.rsplit(".", 1)[-1]
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", simple)
    return [part.lower() for part in re.split(r"[._\s]+", spaced) if part]


def _fuzzy_score(query: str, name: str) -> float:
    """Similarity in [0, 1] between `query` and a declaration's (final) name.

    Combines a whole-string ratio (ignoring underscores) with the best per-query-token ratio, so a
    query matching a single token of a compound name (e.g. 'priority' vs 'decreasePriority') still
    scores well.
    """
    simple = name.rsplit(".", 1)[-1].lower()
    query_squashed = query.lower().replace("_", "").replace(" ", "")
    whole = SequenceMatcher(None, query_squashed, simple.replace("_", "")).ratio()

    name_tokens = _normalize_tokens(name)
    query_tokens = _normalize_tokens(query) or [query_squashed]
    token_score = 0.0
    for query_token in query_tokens:
        best = max((SequenceMatcher(None, query_token, nt).ratio() for nt in name_tokens), default=0.0)
        token_score = max(token_score, best)
    return max(whole, token_score)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "normalize_tokens or fuzzy_score" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat: fuzzy scoring primitives for local search"
```

---

## Task 2: Fuzzy matchers (`_fuzzy_matching_declarations`, `_collect_fuzzy_matches`)

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py` (add 2 functions after `_body_matching_declarations` / `_collect_matches`)
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import (
    _collect_fuzzy_matches,
    _fuzzy_matching_declarations,
)


def test_fuzzy_matching_declarations_finds_close_name():
    content = "def extract_min : Nat := 0\ndef heapify : Nat := 1\n"
    matches = _fuzzy_matching_declarations(content, "extractmin")
    names = [qualified for _simple, qualified, _occ, _score in matches]
    assert "extract_min" in names
    assert "heapify" not in names


def test_collect_fuzzy_matches_sorts_by_score_and_caps(tmp_path):
    (tmp_path / "Def.lean").write_text(
        "def decrease_key : Nat := 0\ndef decrease_min : Nat := 1\ndef unrelated : Nat := 2\n",
        encoding="utf-8",
    )
    files = [(Path("Def.lean"), (tmp_path / "Def.lean").read_text(encoding="utf-8"))]
    results = _collect_fuzzy_matches(files, "decrease_mn", SearchLeanLocalConfig(max_results=1))
    assert len(results) == 1  # cap respected
    assert results[0][0] == "decrease_min"  # closest by score ranked first
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "fuzzy_matching_declarations or collect_fuzzy" -v`
Expected: FAIL — `ImportError: cannot import name '_collect_fuzzy_matches'`.

- [ ] **Step 3: Implement the matchers**

Add after `_body_matching_declarations` in `src/ax_prover/tools/local_lean_search.py`:

```python
def _fuzzy_matching_declarations(
    content: str, query: str, threshold: float = FUZZY_THRESHOLD
) -> list[tuple[str, str, int, float]]:
    """`(simple_name, qualified_name, occurrence, score)` for declarations whose name is a close
    fuzzy match to `query` (score >= threshold). De-duplicated by qualified name, source order.
    """
    if not query.strip():
        return []
    results: list[tuple[str, str, int, float]] = []
    seen: set[str] = set()
    for declaration, qualified, occurrence in _iter_searchable(content):
        if qualified in seen:
            continue
        score = _fuzzy_score(query, qualified)
        if score >= threshold:
            seen.add(qualified)
            results.append((declaration.name, qualified, occurrence, score))
    return results
```

Add after `_collect_matches`:

```python
def _collect_fuzzy_matches(
    files: list[tuple[Path, str]], query: str, config: SearchLeanLocalConfig
) -> list[tuple[str, str, list[tuple[Path, int]]]]:
    """Group fuzzy matches across files, ranked by descending score, capped to `max_results`."""
    grouped: dict[tuple[str, str], list[tuple[Path, int]]] = {}
    scores: dict[str, float] = {}
    for relative_path, content in files:
        for simple_name, qualified_name, occurrence, score in _fuzzy_matching_declarations(
            content, query
        ):
            block = extract_function_from_content(content, simple_name, occurrence)
            if block is None:
                continue
            line = _declaration_line(content, simple_name, occurrence)
            grouped.setdefault((qualified_name, block), []).append((relative_path, line))
            scores[qualified_name] = max(scores.get(qualified_name, 0.0), score)
    results = [(name, block, locations) for (name, block), locations in grouped.items()]
    results.sort(key=lambda result: scores.get(result[0], 0.0), reverse=True)
    return results[: config.max_results]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "fuzzy_matching_declarations or collect_fuzzy" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat: fuzzy declaration matchers for local search"
```

---

## Task 3: Fuzzy header in `_format_results`

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py` (`_format_results` signature + header/note)
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/tools/test_local_lean_search.py`:

```python
from ax_prover.tools.local_lean_search import _format_results


def test_format_results_fuzzy_header():
    decls = [("extract_min", "def extract_min := 0", [(Path("Def.lean"), 1)])]
    out = _format_results("extractmin", decls, SearchLeanLocalConfig(), fuzzy=True)
    assert "No exact match" in out
    assert "extractmin" in out
    assert "fuzzy match" in out


def test_format_results_default_header_unchanged():
    decls = [("foo", "def foo := 0", [(Path("Def.lean"), 1)])]
    out = _format_results("foo", decls, SearchLeanLocalConfig())
    assert out.startswith('Found 1 declaration(s) matching "foo":')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "format_results_fuzzy or format_results_default" -v`
Expected: FAIL — `_format_results() got an unexpected keyword argument 'fuzzy'`.

- [ ] **Step 3: Add the `fuzzy` flag**

In `_format_results`, change the signature from:

```python
def _format_results(
    query: str,
    declarations: list[tuple[str, str, list[tuple[Path, int]]]],
    config: SearchLeanLocalConfig,
    body_match: bool = False,
) -> str:
```

to add `fuzzy: bool = False`:

```python
def _format_results(
    query: str,
    declarations: list[tuple[str, str, list[tuple[Path, int]]]],
    config: SearchLeanLocalConfig,
    body_match: bool = False,
    fuzzy: bool = False,
) -> str:
```

Then replace the two lines that currently set `header` and `note`:

```python
    header = f'Found {len(declarations)} declaration(s) matching "{query}":'
    note = " (matched in body)" if body_match else ""
```

with:

```python
    if fuzzy:
        header = f'No exact match for "{query}". Closest declaration(s):'
        note = " (fuzzy match)"
    elif body_match:
        header = f'Found {len(declarations)} declaration(s) matching "{query}":'
        note = " (matched in body)"
    else:
        header = f'Found {len(declarations)} declaration(s) matching "{query}":'
        note = ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "format_results" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat: fuzzy header rendering in _format_results"
```

---

## Task 4: Insert the fuzzy tier into `_search_root` and update descriptions

**Files:**
- Modify: `src/ax_prover/tools/local_lean_search.py` (`_search_root` tier order; tool + `query` description)
- Test: `tests/unit/tools/test_local_lean_search.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tools/test_local_lean_search.py`:

```python
def test_search_root_exact_match_wins_no_fuzzy(tmp_path):
    (tmp_path / "Def.lean").write_text("def extract_min : Nat := 0\n", encoding="utf-8")
    text, decls = _search_root(tmp_path, "extract_min", SearchLeanLocalConfig())
    assert "No exact match" not in text
    assert text.startswith('Found 1 declaration(s) matching "extract_min":')
    assert len(decls) == 1


def test_search_root_fuzzy_fires_on_exact_miss(tmp_path):
    (tmp_path / "Def.lean").write_text("def extract_min : Nat := 0\n", encoding="utf-8")
    # "extractmin" is NOT a substring of "extract_min" (underscore) -> exact miss -> fuzzy.
    text, decls = _search_root(tmp_path, "extractmin", SearchLeanLocalConfig())
    assert "No exact match" in text
    assert "extract_min" in text
    assert decls and decls[0][0] == "extract_min"


def test_search_root_body_fallback_when_fuzzy_also_misses(tmp_path):
    # Name doesn't match query at all, but the identifier appears inside another decl's body.
    (tmp_path / "Def.lean").write_text(
        "def helper_token : Nat := 0\ndef wrapper : Nat := helper_token + 1\n",
        encoding="utf-8",
    )
    text, decls = _search_root(tmp_path, "helper_token", SearchLeanLocalConfig())
    # exact name match on 'helper_token' exists, so this returns exact — adjust to a body-only case:
    text2, decls2 = _search_root(tmp_path, "zzz_no_name_zzz", SearchLeanLocalConfig())
    assert "No declarations matching" in text2
    assert decls2 == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -k "search_root_exact_match_wins or search_root_fuzzy_fires" -v`
Expected: FAIL — `test_search_root_fuzzy_fires_on_exact_miss` fails (no "No exact match"; current code returns the no-match string for "extractmin").

- [ ] **Step 3: Insert the fuzzy tier**

In `_search_root`, between the exact-name block and the body block, add the fuzzy tier. The function body becomes:

```python
    files = _read_lean_files(root)
    decls = _collect_matches(files, query, body=False)
    if decls:
        logger.info(f"{label}: Found {len(decls)} declarations for '{query}' under {root}")
        return _format_results(query, decls, config), decls
    fuzzy_decls = _collect_fuzzy_matches(files, query, config)
    if fuzzy_decls:
        logger.info(f"{label}: Found {len(fuzzy_decls)} fuzzy matches for '{query}' under {root}")
        return _format_results(query, fuzzy_decls, config, fuzzy=True), fuzzy_decls
    body_decls = _collect_matches(files, query, body=True)
    if body_decls:
        logger.info(f"{label}: Found {len(body_decls)} body matches for '{query}' under {root}")
        return _format_results(query, body_decls, config, body_match=True), body_decls
    logger.info(f"{label}: No results for '{query}'")
    return f'No declarations matching "{query}" found.', []
```

- [ ] **Step 4: Update the tool and query descriptions**

In `create_search_lean_local_tool`, append a sentence to the `description` string (after the "Treap insert finds Treap.insert" paragraph):

```
If no name matches your keyword exactly, the closest declaration names are returned as fuzzy suggestions.
```

In `LocalLeanSearchInput`, append to the `query` field description:

```
 If nothing matches exactly, the closest names are returned as suggestions.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/tools/test_local_lean_search.py -q`
Expected: PASS (full file green, including the WS1 tests carried over).

- [ ] **Step 6: Commit**

```bash
git add src/ax_prover/tools/local_lean_search.py tests/unit/tools/test_local_lean_search.py
git commit -m "feat: fuzzy fallback tier in local search _search_root"
```

---

## Task 5: Full verification

- [ ] **Step 1: Run the whole unit suite**

Run: `.venv/bin/pytest tests/unit -q`
Expected: PASS (no regressions).

- [ ] **Step 2: Lint and format**

Run: `.venv/bin/ruff format . && .venv/bin/ruff check --fix .`
Expected: "All checks passed!" Commit any formatting:

```bash
git add -A && git commit -m "style: ruff format/lint" || echo "nothing to format"
```

- [ ] **Step 3: Push and open the PR into `local-lean-search-tool`**

```bash
git push -u origin ws2-local-search-fuzzy-fallback
gh pr create --base local-lean-search-tool --title "feat: fuzzy fallback matching for local search (WS2)" --body "$(cat <<'EOF'
Adds a fuzzy fallback tier to the local Lean search tool: exact name match → (new) fuzzy match → body-identifier match. Fuzzy fires only on an exact miss, returns the closest declaration names under a "No exact match" header, and is capped by the existing result/char budget so it never bloats context.

Spec: docs/superpowers/specs/2026-06-08-local-search-caching-and-fuzzy-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes (author)

- **Spec coverage:** WS2 components map to Task 1 (`_normalize_tokens`/`_fuzzy_score`), Task 2 (`_fuzzy_matching_declarations`/`_collect_fuzzy_matches`), Task 3 (`_format_results` fuzzy header), Task 4 (tier order + descriptions). Tier ordering "exact → fuzzy → body" verified in Task 4 tests; cap respected in Task 2 test.
- **Type consistency:** `_collect_fuzzy_matches` returns the same `(qualified_name, block, locations)` shape as `_collect_matches`, so `_format_results` and the `_search_root` decls contract (from WS1) are unchanged. `_fuzzy_matching_declarations` carries an extra `score` element used only internally by `_collect_fuzzy_matches`.
- **WS1 dependency:** assumes `_search_root` already returns `(text, decls)` (WS1). Task 4's body re-states all return paths as tuples, so it is correct whether or not the reviewer re-reads WS1.
- **Test note:** `test_search_root_body_fallback_when_fuzzy_also_misses` asserts the genuine no-match path (`zzz_no_name_zzz`) returns the empty result, since any close name would otherwise be caught by the fuzzy tier.
