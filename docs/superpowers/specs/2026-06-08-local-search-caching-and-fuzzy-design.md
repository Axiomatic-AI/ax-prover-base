# Local-search efficiency & recall — Design

**Date:** 2026-06-08
**Branch:** features off `local-lean-search-tool` (a.k.a. "local search"); each merged back into
`local-lean-search-tool`, which then merges into the experiment branch `run_AI4Math_Challange`.

## Motivation

An experiment run (3 challenges: Dijkstra_3, Kruskal, BinaryHeap_2) surfaced two problems with the
**local Lean project search** tool (`search_lean_local`):

1. **Extreme redundancy / context bloat.** The agent re-queried the same local definitions
   (`extract_min`, `heapify`, `IsMST`, …) in almost every iteration. Root cause (confirmed in
   `agent.py` / `utils/llm.py`): each proposer iteration runs a *fresh* `agentic_loop` whose
   intermediate `ToolMessage`s are discarded — only the final proposal survives into graph state.
   So previously-fetched definitions are genuinely absent from the next iteration's context, and the
   agent is *structurally forced* to re-search them. This repeated flooding of large definitions was
   identified as the primary trigger for `missing_target_theorem` failures (the LLM loses track of
   formatting/structure constraints under the bloat).

2. **No fuzzy matching.** Matching requires every query token to be a literal substring of the
   declaration's qualified name. When the agent guessed a wrong name (e.g. `decrease_priority` for a
   function actually named differently), it got zero results and had to keep guessing.

This is **not** a Mathlib problem. The remote Mathlib `lean_search` tool does not exhibit this
redundancy, and Mathlib definitions are never gathered by anything below — only **local project**
declarations are in scope.

## Scope

Two independent workstreams, two separate PRs:

- **WS1 — Used-definition caching.** Eliminate redundant re-querying by persisting, verbatim, the
  local-search results the proof actually used.
- **WS2 — Fuzzy fallback matching.** Add recall when the agent guesses a wrong name, without
  polluting good exact searches or inflating context.

Both branch off `local-lean-search-tool`. They both touch `_search_root`, so they are sequenced
**WS1 → WS2** (WS2 rebases on WS1).

---

## WS1 — Used-definition caching

### Principle

Cache the **intersection of two sets**, accumulated across the whole run:

1. **Local-search results returned this run** — i.e. declarations the `search_lean_local` tool
   actually surfaced. This is the repetitive payload and the real source of bloat. (CSLib and remote
   Mathlib results are explicitly *not* recorded.)
2. **Declarations actually used in the proposed code** — the relevance filter.

The cache is therefore *"local-search results that the proof actually referenced."* This naturally
excludes:
- Mathlib / remote search results (never recorded).
- Challenge-file-local helpers the agent never searched for (already visible in the prompt, which
  always carries the complete challenge file).
- Search results the agent looked at but never used.

### Accumulation guarantee (critical)

The cache is **append-only across iterations**. Each memory step reads the prior
`state.used_definitions`, adds any newly used-and-returned definitions, and writes back the **union**
(deduped by qualified name). A definition captured in iteration 2 remains available through every
later iteration even if a subsequent attempt stops referencing it. The list is **never wiped**.

### Components

1. **Searcher records its results.**
   `LocalLeanSearcher` gains `returned_declarations: dict[str, tuple[str, Path, int]]` keyed by
   qualified name → `(block, path, line)`, populated on every `search()` call. To obtain the
   structured results, `_search_root(...)` is changed to return `(text, decls)` instead of just
   `text`, where `decls` is the same list of `(qualified_name, block, locations)` tuples it formats
   (whichever tier — exact, fuzzy, or body — produced the returned results). The searcher derives
   each pool entry from a decl's qualified name + block + its first location.
   - `CslibSearcher.search` unpacks and **ignores** `decls` (so CSLib results never enter any cache).
   - `LocalLeanSearcher.search` merges `decls` into `self.returned_declarations`.
   - The instance persists across iterations (tools are created once at agent setup), so the pool
     accumulates naturally over the run.

2. **Agent captures the live searcher.**
   After `_create_tools()`, the agent locates the local-search tool in `self.proposer_tools` by name
   (`tool_name_from_type(LOCAL_LEAN_SEARCH_TOOL_TYPE)`) and stores the bound searcher instance as
   `self._local_searcher` (via the tool's bound `func.__self__`). If local search is not configured,
   `self._local_searcher is None` and the entire feature is a clean no-op.

3. **State field.**
   `ProverAgentState.used_definitions: dict[str, str]` (default empty). Maps qualified name → the
   rendered verbatim entry (location header + source block). Overwrite semantics at the graph level:
   the memory node returns the full merged dict each iteration.

4. **Memory-node augmentation (deterministic, no LLM).**
   `_memory_processor_node` calls `self.memory.process(state)` as today, then merges in
   `self._accumulate_used_definitions(state)`:
   - If `self._local_searcher is None` or there is no `last_proposal`/`location`, return the existing
     `used_definitions` unchanged.
   - Read `pool = self._local_searcher.returned_declarations` and `code = state.last_proposal.code`.
   - Start from `dict(state.used_definitions)` (the accumulation base).
   - For each `(qualified, (block, path, line))` in the pool not already cached: skip if it is the
     target theorem's own name; otherwise include it iff the declaration is referenced in `code` as a
     whole identifier — checking both the qualified name and its final dotted segment (the source
     simple name) via `_identifier_match`. Respect caps (below).
   - Return the merged dict under key `"used_definitions"`.
   - This runs **independent of the configured memory strategy** (works even with
     `MemorylessProcessor`) and **bypasses the experience-summarizer LLM**, so Lean source stays
     byte-exact (a paraphrased definition would cause "unknown identifier" miscites).

5. **Proposer injection.**
   In `_proposer_node`, when `state.used_definitions` is non-empty, append a rendered
   `<local-definitions>` section to the query (after the experience block). Add one line to the
   proposer system prompt(s): definitions already used are provided in `<local-definitions>` and must
   not be re-searched.

   Rendered shape:
   ```
   <local-definitions>
   These project definitions were referenced in your previous attempts and are provided verbatim,
   so you do NOT need to search for them again:

   -- <qualified_name> — <path>:<line>
   <verbatim block>

   -- <qualified_name> — <path>:<line>
   <verbatim block>
   </local-definitions>
   ```

6. **Caps (module constants, promotable to config later).**
   `MAX_CACHED_DEFINITIONS` (≈24) and `MAX_CACHED_DEFINITION_CHARS` (≈12000) bound growth. When a cap
   is reached, stop adding new definitions (already-cached ones are retained — accumulation is never
   reduced).

### Files (WS1)

- `src/ax_prover/tools/local_lean_search.py` — `returned_declarations` on `LocalLeanSearcher`;
  `_search_root` returns `(text, decls)`; caps constants; a small render helper for cached entries.
- `src/ax_prover/tools/cslib_search.py` — unpack `(text, _)` from `_search_root`.
- `src/ax_prover/models/proving.py` — `used_definitions: dict[str, str]` field.
- `src/ax_prover/prover/agent.py` — capture `self._local_searcher`; `_accumulate_used_definitions`;
  `_memory_processor_node` merge; proposer injection of `<local-definitions>`.
- `src/ax_prover/prover/prompts.py` — `<local-definitions>` template + system-prompt nudge.

### Tests (WS1, TDD)

- `_search_root` returns both text and structured decls for exact / fuzzy / body tiers; CSLib path
  unpacks without error and records nothing.
- `LocalLeanSearcher.returned_declarations` accumulates across multiple `search()` calls and dedups.
- `_accumulate_used_definitions`:
  - caches a pooled def referenced in the code (whole-identifier), with location header;
  - ignores a pooled def **not** referenced in the code;
  - excludes the target theorem's own name;
  - is monotonic — a def cached in an earlier step survives a later step whose code no longer uses it;
  - respects `MAX_CACHED_DEFINITIONS` / `MAX_CACHED_DEFINITION_CHARS`;
  - is a no-op when `self._local_searcher is None`.
- Proposer injects `<local-definitions>` into the query when `used_definitions` is non-empty
  (source-level guard that the wiring exists, mirroring existing proposer-wiring tests).

---

## WS2 — Fuzzy fallback matching

### Behavior

A new matching tier inside `_search_root`, **firing only when exact name matching returns zero**:

```
exact name match  →  (NEW) fuzzy name match  →  body-identifier match
```

This keeps exact searches clean and fast and only spends recall effort on a genuine miss — important
because WS1's whole purpose is reducing context bloat, so fuzzy must not blend approximate matches
into otherwise-good exact results.

### Components

- `_fuzzy_name_matches(names, query, threshold)`: normalize both candidate names and the query
  (case-fold; split on `.`, `_`, and camelCase boundaries into token sequences); score with
  `difflib.SequenceMatcher` ratio combined with token-set overlap; keep the top-N candidates above a
  threshold (≈0.6). Returns the same `(simple_name, qualified_name, occurrence)` tuple shape the
  existing tiers use, so `_collect_matches`-style grouping and `_format_results` are reused.
- `_format_results` gains a fuzzy flag that renders a clear header:
  `No exact match — closest names:` (analogous to the existing `body_match` note). Capped by the same
  `max_results` / `max_chars` budget so it cannot bloat context.
- Tool description and the `query` field description mention the fuzzy fallback.

### Files (WS2)

- `src/ax_prover/tools/local_lean_search.py` — `_fuzzy_name_matches`, normalization helper, fuzzy
  tier in `_search_root`, `_format_results` fuzzy label, tool/`query` description updates.

### Tests (WS2, TDD)

- Exact match still wins — no fuzzy results when an exact substring match exists.
- Fuzzy fires only on an exact miss; a near-miss query (e.g. `extractmin` → `extract_min`, or a typo
  `decrese_min` → `decrease_min`) surfaces the closest names under the `No exact match` header.
- Body-identifier fallback still works when both exact and fuzzy miss.
- Fuzzy results respect the result/char caps.

---

## Out of scope / non-goals

- No change to the Mathlib remote `lean_search` tool or the CSLib `search_cslib` tool behavior
  (CSLib only loses the now-unused second tuple element from `_search_root`).
- No change to import/opens handling, the `restrict_to_proof_body` lock, or the reviewer.
- Caps are not surfaced as config in this iteration (constants only).

## Branching & sequencing

1. Branch `wsX` off `local-lean-search-tool` for WS1 → PR into `local-lean-search-tool`.
2. Branch off the updated `local-lean-search-tool` for WS2 → PR into `local-lean-search-tool`.
3. Merge `local-lean-search-tool` → `run_AI4Math_Challange` (experiment branch).

`main` is not touched by this work beyond the existing open PR #13 for `local-lean-search-tool`.
