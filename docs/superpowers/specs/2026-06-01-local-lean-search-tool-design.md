# Local Lean Library Search Tool — Design

**Date:** 2026-06-01
**Status:** Approved (pending implementation plan)

## Problem

While proving, the agent can search Mathlib via the remote `leansearch` tool and
the web via `web_search`, but it **cannot search the local Lean library** of the
project it is proving in. This blocks proofs that depend on project-local
definitions.

**Case study:** proving `challenges/Challenges/Treap/Challenge_Treap_11.lean`
requires the definitions in `challenges/Challenges/Treap/Def_Treap.lean`. The
agent has no way to retrieve those definitions today.

## Goal

A simple, local tool that, given a keyword, returns the **full declarations**
matching that keyword from the project's own `.lean` files. Assumes the agent
runs locally with the project checked out. Must work for **any** Lean repo with
no hardcoded paths.

## Scope decisions (resolved during brainstorming)

| Decision | Choice |
|---|---|
| What it returns | **Full declaration blocks** (signature + body), not raw grep lines |
| What the keyword matches | **Declaration names only**, case-insensitive substring |
| Search root | The **lakefile directory**, inferred (not hardcoded) |
| Dependencies | **Excluded** — skip `.lake/` (build artifacts + vendored Mathlib) |

Note on "repo" vs "project root": the outer repo (e.g. `AI4Math/`) may not contain
the lakefile. In the case study the Lean project root is `AI4Math/challenges/`
(where `lakefile.toml`, `lake-manifest.json`, `lean-toolchain` live). The tool
treats the **lakefile directory** as the search root, which may be a subdirectory
of the repo.

## Architecture

New tool module `src/ax_prover/tools/local_lean_search.py`, following the existing
tool pattern in `web_search.py`:

- a `dataclass` config,
- a `pydantic` `args_schema` (single `query: str`),
- a `@register_tool(...)`-decorated factory returning a `StructuredTool`.

`tool_type = "search_lean_local"`, name follows the convention
(`search_lean_local_tool`).

### Reused infrastructure

- `utils/lean_parsing.list_all_declarations_in_lean_code(raw_code) -> list[Declaration]`
  — enumerate declarations (type + name + content) per file.
- `utils/lean_parsing.extract_function_from_content(content, name) -> str | None`
  — pull the full declaration block by name.
- `models/declaration.Declaration` — declaration model.

No new Lean parsing is written; the tool assembles these.

## Search-root resolution (no hardcoding)

Resolved once per tool instance and cached:

1. **Walk up** from the agent's `base_folder` to the nearest directory containing
   any of `lakefile.toml`, `lakefile.lean`, `lake-manifest.json`.
2. If not found above (e.g. `base_folder` is `AI4Math` with the lakefile in a
   child), do a **bounded downward search** for lakefile directories:
   - exactly one → use it;
   - several → return a clear message listing them so the run config can point
     more precisely;
   - none → search unavailable (see Errors).
3. Always **exclude `.lake/`** from both the downward search and the file scan.

## Wiring: injecting `base_folder` into the tool

Today `ProverAgent._create_tools()` passes only the static YAML config dict to
`tools.registry.create_tool()`. The agent already holds `self.base_folder`.

**Change (minimal, backward-compatible):** `create_tool()` inspects each factory's
signature; if the factory declares a `base_folder` parameter, `create_tool()`
passes `base_folder` through. `_create_tools()` forwards `self.base_folder`.

- `web_search` / `lean_search` factories do **not** declare `base_folder` → untouched.
- `search_lean_local`'s factory opts in by declaring it.
- No per-tool special-casing in the agent; no hardcoded paths.

## Search algorithm

Given `query`:

1. Resolve root (above). On failure, return the error message.
2. Enumerate `*.lean` under root, excluding `.lake/`.
3. For each file, `list_all_declarations_in_lean_code()`; keep declarations whose
   **name** contains `query` (case-insensitive substring).
4. For each match, extract the full block via `extract_function_from_content()`
   and record `file:line` (path relative to root).
5. Apply caps (below) and format output.

## Output format

Tight, to control token use:

```
Found 2 declaration(s) matching "Treap":

-- challenges/Challenges/Treap/Def_Treap.lean:14
structure Treap where
  ...

-- challenges/Challenges/Treap/Def_Treap.lean:31
def Treap.insert (t : Treap) (x : Nat) : Treap := ...
```

### Config (`configs/tools.yaml`, no `root_path` — inferred)

```yaml
  search_lean_local:
    tool_type: search_lean_local
    max_results: 6
    max_chars: 4000
```

- `max_results` — max declarations returned with full bodies.
- `max_chars` — total output cap. If exceeded, remaining matches are listed by
  **name only** (so the agent knows they exist) instead of full bodies.

## Error / edge handling

- No lakefile found → message stating local search is unavailable for this project.
- Multiple lakefile dirs found in downward search → list them; do not guess.
- No matches → `No declarations matching "<query>" found`.
- Empty/whitespace query → message asking for a non-empty keyword.
- Unreadable / malformed `.lean` files → skipped, logged at debug.
- Root resolved once and cached on the tool instance.

## Testing

Pytest unit tests using a throwaway fixture repo (temp dir with a `lakefile.toml`,
a couple of `.lean` files, and a dummy `.lake/` dir to prove exclusion):

- **Root inference:** walk-up hit; walk-down fallback (single dir); walk-down with
  multiple dirs (returns listing message); no-lakefile case.
- **Name match:** case-insensitive substring; returns full declaration block;
  multiple matches across files.
- **`.lake/` exclusion:** declarations under `.lake/` are never returned.
- **Caps:** results beyond `max_results` / `max_chars` are listed by name only.
- **No-match** and **empty-query** handling.
- **Wiring:** `create_tool()` passes `base_folder` to a factory that declares it,
  and omits it for one that does not.

## Out of scope (YAGNI)

- Body/comment full-text search (names only).
- Regex / fuzzy matching.
- Searching `.lake/` dependencies or Mathlib (the remote `leansearch` tool covers that).
- Indexing / caching across runs.
