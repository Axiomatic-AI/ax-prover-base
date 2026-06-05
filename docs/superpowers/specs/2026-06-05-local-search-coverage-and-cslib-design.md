# Local search coverage + CSLib search + prompt Lean 4.28 — Design

Date: 2026-06-05
Status: Approved (brainstorming)

## Context

The prover's local Lean search (`search_lean_local`) is the agent's main way to understand the
custom project environment. Checking it against the official AI4Math / Codabench dataset structure
(Lean **4.28.0** + Mathlib + **CSLib**; `Def_*` files supply definitions, `Challenge_*` files each
hold one `sorry`) revealed three gaps:

1. **Identifiers defined inside `where` / `let rec` blocks are unfindable.** The parser only sees
   column-anchored top-level keywords, so a helper like `query_aux` (defined in a `where` block in
   `Challenges/Segment_Tree/Def_Query.lean`) returns no result. More generally, the tool can only
   find a declaration by its *name*, never by something referenced in its *body*.
2. **CSLib is unsearchable.** It is vendored at `<project>/.lake/packages/cslib/` (≈109 files, 189
   namespaces). Local search excludes `.lake/` by design, and the remote LeanSearch tool is
   Mathlib-only, so CSLib lemmas have no search path — a blind spot for challenges that use it
   (notably Phase 2 Splay Tree / Spanner).
3. **The proposer prompt states the wrong Lean version** ("Lean version 4.24") while the dataset is
   4.28.0.

Grounding confirmed CSLib declarations use no modifiers the parser misses (`noncomputable`,
`private`/`partial`, inline `@[...]`, and namespaces are all already handled), so CSLib needs only
to be *pointed at* — not parsed differently.

## Design

### 1. Body-search fallback (replaces a fragile where/let-rec parser)

Rather than special-case `where`/`let rec` syntax (fragile, and only `query_aux` exists in the
current dataset), generalize: search declaration **bodies** when a name search finds nothing.

- Extract the per-root search body of `LocalLeanSearcher.search` into a shared module-level
  `_search_root(root, query, config)` in `local_lean_search.py` (the grouping + `_format_results`
  logic). `LocalLeanSearcher.search` keeps resolving the project root and delegates to it.
- **Pass 1 — name match (unchanged):** namespace-qualified declaration-name matching, exactly as
  today (modifiers/occurrence already handled).
- **Pass 2 — body match (new; runs ONLY when Pass 1 returns zero):** for each declaration, if every
  query token appears as a **whole identifier** (identifier-boundary, not raw substring) anywhere in
  that declaration's block text, return the enclosing declaration's block. The result header is
  annotated so the agent knows it was a body hit, e.g. `-- <path>:<line> (matched in body)`. Same
  `max_results` / `max_chars` caps as name matches.
- Net effect: `search("query_aux")` returns the `query` declaration block (query_aux is referenced
  in its body). Generalizes to any identifier with no syntax special-casing.

Identifier-boundary match: a token matches when bounded by non-identifier characters (Lean
identifiers include letters, digits, `_`, `'`, and `.`), so `query_aux` does not match inside
`query_auxiliary`.

### 2. CSLib search — separate `search_cslib` tool

A new tool, semantically distinct from project search, reusing the shared core.

- New `src/ax_prover/tools/cslib_search.py`:
  - `CSLIB_SEARCH_TOOL_TYPE = "search_cslib"`, `SearchCslibConfig` (`max_results`, `max_chars`,
    and `package_subpath: str = ".lake/packages/cslib"`).
  - `CslibSearcher`: reuse `LocalLeanSearcher`'s lake-root resolution
    (`_walk_up_for_root` / `_walk_down_for_roots`), then search `root / package_subpath`, resolved
    **relative to the project** (no hardcoded absolute path). If that directory is absent, return a
    clear "CSLib not found under .lake/packages" message. Reuses `_search_root` + `_iter_lean_files`
    (pointing the root *inside* cslib makes `.lake` exclusion irrelevant).
  - `@register_tool(CSLIB_SEARCH_TOOL_TYPE, SearchCslibConfig)` factory `create_search_cslib_tool`
    with a `base_folder` parameter so `create_tool` injects the project folder.
- Inherits the body-search fallback automatically via the shared core.
- Wiring (this PR, bundled configs): export in `src/ax_prover/tools/__init__.py`; add a
  `search_cslib` `tool_config` to `src/ax_prover/configs/tools.yaml`; add `search_cslib` to
  `proposer_tools` in `src/ax_prover/configs/default.yaml`.

### 3. Prompt → Lean 4.28

In `src/ax_prover/prover/prompts.py`, change the two "Lean version 4.24" occurrences to
"Lean version 4.28". Leave generic "Lean 4" mentions unchanged.

## Components / boundaries

- `_search_root(root, query, config)` — pure "search one directory tree" unit; both tools depend on
  it; independently testable.
- `LocalLeanSearcher` — resolves the project root, delegates to `_search_root`.
- `CslibSearcher` — resolves the project root, derives the cslib subdir, delegates to `_search_root`.
- Body-search lives entirely inside `_search_root` (and a small identifier-match helper); the shared
  parser `lean_parsing.py` is untouched, preserving builder/reviewer/stripped-lemma behavior.

## Files

- `src/ax_prover/tools/local_lean_search.py` — extract `_search_root`; add body-search fallback + identifier-match helper
- `src/ax_prover/tools/cslib_search.py` — new tool
- `src/ax_prover/tools/__init__.py`, `src/ax_prover/configs/tools.yaml`, `src/ax_prover/configs/default.yaml` — CSLib wiring
- `src/ax_prover/prover/prompts.py` — Lean 4.28
- `tests/unit/tools/test_local_lean_search.py` — body-search tests
- `tests/unit/tools/test_cslib_search.py` — new CSLib tests

## Tests (TDD: failing first)

- Body-search (`test_local_lean_search.py`): a `where`-block fixture (the `query_aux` shape) —
  `search("query_aux")` returns the enclosing `query` block, header flagged as a body match; a body
  hit on an ordinary referenced identifier; identifier-boundary guard (token does not match inside a
  longer identifier); and confirmation that body-search does NOT run when a name match exists.
- CSLib (`test_cslib_search.py`): tmp lake project with `.lake/packages/cslib/Cslib/X.lean`
  containing a namespaced `theorem` — found and namespace-qualified; missing-cslib-dir yields the
  clear message; the cslib tool does NOT return the project's own (non-cslib) declarations.

## Verification

1. `.venv/bin/pytest tests/unit -q` green incl. new tests; `ruff format` + `ruff check` clean.
2. End-to-end against the real dataset (read-only):
   ```
   .venv/bin/python -c "
   from ax_prover.tools.local_lean_search import LocalLeanSearcher, SearchLeanLocalConfig
   from ax_prover.tools.cslib_search import CslibSearcher, SearchCslibConfig
   base='/Users/krystian/Documents/Axiomatic/Baku/AI4Math/challenges'
   print(LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=base).search('query_aux')[:400])
   print(CslibSearcher(SearchCslibConfig(), base_folder=base).search('WellFounded')[:400])
   "
   ```
   Expect: `query_aux` returns the `query` block (flagged body match); CSLib query returns a CSLib
   declaration.

## Packaging / branching

- **PR A** (this branch, `local-search-body-and-cslib`, based on `local-lean-search-tool`):
  workstreams 1 + 2 — body-search fallback and the `search_cslib` tool, wired into the **bundled**
  package configs.
- **PR B** (separate, off `main`): workstream 3 — the Lean 4.28 prompt change.
- Both merge into `run_AI4Math_Challange`. At exp-integration, also add `search_cslib` to the
  exp-only `configs/opus48_local.yaml` `proposer_tools` so the next experiment run uses it (that
  file does not exist on this feature branch).

## Out of scope / deferred

- No `where`/`let rec` AST parsing — the body-search fallback subsumes the need.
- No caching of CSLib scans (≈5 MB per query). Acceptable for now; revisit if call latency matters.
- CSLib path assumes the standard `.lake/packages/cslib` layout; configurable via `package_subpath`.
