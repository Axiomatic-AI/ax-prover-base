# Restrict edits to the proof body (challenge-file immutability)

**Date:** 2026-06-05
**Status:** Approved design — ready for implementation plan

## Problem

The AI4Math / Codabench TCS proving competition imposes hard rules on what a
submission may change in a challenge file:

> 1. Only fill in the `sorry` placeholders in the challenge files. Do not modify
>    any other part of those files.
> 2. Do not change the import statements in the challenge files.

ax-prover today violates both. The proposer LLM returns `imports` and `opens`
lists, and `TemporaryProposal` writes them into the file before compilation:

- `agent.py:315-316` copies `result.imports` / `result.opens` into the `ProposalMessage`.
- `build.py:417-418` calls `edit_imports(...)` whenever `proposal.imports` is non-empty.
- `build.py:423-424` calls `edit_opens(...)` whenever `proposal.opens` is non-empty.
- `files.py:148 edit_imports` merges new imports into the file's import block.

The prompt actively *encourages* the violation: both system-prompt variants
(`PROPOSER_SYSTEM_PROMPT`, `PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT`) say
`"Add imports as needed for any lemmas or tactics you use"`, and the `imports`
output field is described as `"Required imports NOT already in the file"`.

A file-level `open Foo` is just as much a Rule 1 violation as a new `import`,
because it modifies a part of the file other than the `sorry` placeholder.

## Goal

When enabled, the agent edits only the proof body. It never writes a new
`import` or a new file-level `open` statement into the challenge file. This
satisfies Rules 1 and 2. The capability is opt-in via config, so ax-prover
remains a general-purpose Lean prover (where adding imports/opens is legitimate)
by default.

## Non-goals

- **Auxiliary files (Rule 4):** out of scope. The competition allows creating new
  auxiliary files, but since the challenge file cannot `import` them (Rule 2),
  they cannot be referenced from the proof. Inlining helpers with
  `have`/`let`/`let rec`/`where` remains the only viable strategy, so the existing
  stripped-lemma guidance still applies unchanged.
- **Axiom restriction (Rule 5 — only `propext`, `Quot.sound`, `Classical.choice`):**
  out of scope here. Candidate for a separate reviewer enhancement later.

## Decisions (locked)

- **Block scope:** new imports AND new file-level `open` statements. The model
  must use `open … in` inside the proof body or fully-qualify names instead.
- **Control:** config flag `restrict_to_proof_body: bool`, default `False`
  (preserves current general behavior). The challenge profile enables it.
- **On violation:** drop the proposed imports/opens (never write them) AND warn
  the proposer in the build feedback, mirroring the existing stripped-lemma
  warning, so the model adapts instead of looping.
- **Output schema:** keep the `imports`/`opens` fields (shared, used in general
  mode); instruct empty lists via the prompt and drop anything non-empty
  mechanically. Do not dynamically remove fields from the structured schema.

## Design

### 1. Config (`src/ax_prover/config.py`)

Add `restrict_to_proof_body: bool = False` to `ProverConfig`. Default `False`
keeps general behavior; the challenge profile sets it `true`.

### 2. Mechanical enforcement (`src/ax_prover/utils/build.py` → `TemporaryProposal`)

- New constructor parameter `restrict_to_proof_body: bool = False`.
- In `__enter__`, when `True`: skip the `edit_imports` call (currently 417-418)
  AND the `edit_opens` call (currently 423-424). Proposed imports/opens are never
  written. Code application is unchanged — it is already scoped to the target
  function via `extract_function_from_content`, so `open … in`, `have`, `let`,
  `let rec`, and `where` inside the proof body survive normally.
- Both `agent.py` construction sites — the builder node (~352) and the reviewer's
  `apply_permanently` path (~488) — pass `self.config.restrict_to_proof_body`.
  The standalone `build.py:343` helper keeps the `False` default.

### 3. Proposer feedback (`src/ax_prover/prover/agent.py` builder node)

**Baseline note:** On `main`, `BuildFailedFeedback` has only `error_output` — it
has no `warning` field, and there is no stripped-declaration logic or
`find_stripped_declaration_names`. That infrastructure was introduced by PR #15
("telling_agent_not_to_add_lemmas"), which lives on the experiment branch, not on
`main`. To keep this feature independent of PR #15, Branch B does **not** modify
`messages.py`.

When the flag is on and `state.last_proposal.imports or state.last_proposal.opens`
is non-empty (a trivial truthiness check — no parsing needed), the builder node
prepends a clearly-delimited warning block to the `error_output` string passed to
`BuildFailedFeedback`. Example text:

> WARNING — IMPORTS/OPENS IGNORED: you proposed import(s) `X` and open(s) `Y`,
> but this file is locked — only the proof body may change. They were not
> applied. All necessary imports are already present; use `open … in` inside the
> proof or fully-qualify names.

The warning attaches to `BuildFailedFeedback` only. When a build succeeds despite
dropped imports/opens, they were unnecessary, so no warning is needed.

**Exp-merge composition:** PR #15 routes its stripped-lemma warning through a
dedicated `warning` field on `BuildFailedFeedback`. When Branch B merges into the
experiment branch, the single expected conflict is the builder-node feedback
construction (any feature touching that node hits this); resolve it by routing
both warnings through the same channel. Because Branch B leaves `messages.py`
untouched, there is no `messages.py` conflict.

### 4. Prompt (`src/ax_prover/prover/prompts.py` + `agent.py`)

- Add a new constant `PROOF_BODY_RESTRICTION_PROMPT` fragment in `prompts.py`.
- In `_proposer_node`, append the fragment to the selected system prompt **only
  when `self.config.restrict_to_proof_body` is `True`**. This covers both the
  iterative `PROPOSER_SYSTEM_PROMPT` and `PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT`,
  which are static constants selected by `max_iterations` (agent.py:261-263) and
  are not `.format()`ed — so appending avoids brace-escaping issues with Lean
  code in the prompt.
- The fragment explicitly overrides the base prompts' general "Add imports as
  needed" guidance for this task. Suggested wording:

  > This file is locked: you may only edit the proof body. Do NOT output any
  > `imports` or `opens` — leave both lists empty. The file's imports are fixed
  > and already complete. If you need a namespace, use `open … in` within the
  > proof body or fully-qualify names. Anything placed in the `imports`/`opens`
  > fields is ignored.

- Base prompts' general guidance stays intact, so general (flag-off) mode is
  unaffected.

### 5. Activation config (experiment branch only)

Set `restrict_to_proof_body: true` in `configs/opus48_local.yaml`. This file is
exp-only (not on `main`), so the activation is applied on the experiment branch
after merge — the same pattern used to wire the cslib search tool. The bundled
`src/ax_prover/configs/default.yaml` stays at the `False` default.

## Components and responsibilities

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `ProverConfig.restrict_to_proof_body` | Single source of truth for the flag | — |
| `TemporaryProposal(restrict_to_proof_body=...)` | Hard enforcement: never write imports/opens when set | config flag (passed in) |
| builder node warning | Tell the proposer its imports/opens were dropped | flag + `last_proposal` |
| `PROOF_BODY_RESTRICTION_PROMPT` + `_proposer_node` | Tell the proposer up front not to emit imports/opens | flag |
| `opus48_local.yaml` activation | Turn the feature on for the competition | feature merged to exp |

## Data flow

1. Config resolves `restrict_to_proof_body` (true for the challenge profile).
2. `_proposer_node` appends the restriction fragment to the system prompt when on.
3. The proposer ideally returns empty `imports`/`opens`; if not, the lists still
   flow into the `ProposalMessage`.
4. The builder node constructs `TemporaryProposal(..., restrict_to_proof_body=True)`,
   which skips `edit_imports`/`edit_opens` — the file's preamble is untouched.
5. If the proposal had non-empty imports/opens and the build fails, the builder
   node appends the IMPORTS/OPENS IGNORED warning to the build feedback.

## Error handling

- Dropping imports/opens cannot fail (it is the absence of an action).
- If the model relied on a dropped import, the build fails with
  "unknown identifier"; the warning explains why and points to `open … in` /
  fully-qualified names.
- Flag-off behavior is byte-for-byte the current behavior (no new code path runs).

## Testing (TDD)

- **TemporaryProposal:**
  - flag on → a proposal with `imports`/`opens` leaves the temp file's preamble
    unchanged (neither is written).
  - flag off (default) → imports/opens are written (guards existing behavior).
- **Builder node:**
  - flag on + proposal has imports/opens + build fails → the `BuildFailedFeedback`
    content (via `error_output`) names the dropped imports/opens.
  - flag off → no such warning even if imports/opens are present.
- **Prompt selection:** `_proposer_node`'s system prompt includes
  `PROOF_BODY_RESTRICTION_PROMPT` iff the flag is on (both iterative and
  single-shot).
- Full unit suite stays green; `ruff format` / `ruff check` clean.

## Risks

- **`open … in` reliability:** steering the model to local `open … in` instead of
  file-level opens is the main behavioral risk. Observable in build logs; the
  prompt fragment can be tightened if the model fumbles syntax.

## Branching

- New branch `restrict-edits-to-proof-body`, based on `main` (one PR = one
  feature, reviewed independently). The PR ships the general feature with the
  flag defaulting `False`; it targets `main`.
- After review, merge into `run_AI4Math_Challange` and apply the
  `restrict_to_proof_body: true` activation in `opus48_local.yaml` on the
  experiment branch. Do not merge into `main` as part of the experiment prep.
