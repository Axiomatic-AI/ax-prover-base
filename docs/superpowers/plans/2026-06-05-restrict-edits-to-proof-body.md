# Restrict Edits to the Proof Body — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in config flag that stops the prover from writing new `import` or file-level `open` statements into a challenge file, so the agent only ever edits the proof body (AI4Math competition Rules 1 & 2).

**Architecture:** A single `restrict_to_proof_body` flag on `ProverConfig` (default `False`) flows to three places: (1) `TemporaryProposal` skips `edit_imports`/`edit_opens` when set; (2) the builder node prepends a "your imports/opens were ignored" warning into the build-failure `error_output`; (3) the proposer's system prompt gains a "this file is locked" fragment. Default-off preserves ax-prover's general behavior. No `messages.py` changes, keeping this independent of the exp-only PR #15.

**Tech Stack:** Python 3, dataclasses, Pydantic models, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-06-05-restrict-edits-to-proof-body-design.md`

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `src/ax_prover/config.py` | Modify | Add `restrict_to_proof_body: bool = False` to `ProverConfig` |
| `src/ax_prover/utils/build.py` | Modify | `TemporaryProposal` gains the flag; skips writing imports/opens when set |
| `src/ax_prover/prover/prompts.py` | Modify | New `PROOF_BODY_RESTRICTION_PROMPT` fragment + `build_proposer_system_prompt()` helper |
| `src/ax_prover/prover/agent.py` | Modify | Pass flag to both `TemporaryProposal` sites; prepend dropped-preamble warning; use the prompt helper |
| `tests/unit/utils/test_config.py` | Modify | Assert the new flag's default |
| `tests/unit/utils/test_build.py` | Modify | `TemporaryProposal` honors the flag (writes vs skips) |
| `tests/unit/prover/test_proof_body_restriction.py` | Create | Prompt-assembly helper + dropped-preamble warning helper |

**Out of scope (per spec):** auxiliary files (Rule 4), axiom checking (Rule 5), and the `opus48_local.yaml` activation (applied on the experiment branch after merge — this PR ships with the flag defaulting `False`).

---

## Task 1: Config flag

**Files:**
- Modify: `src/ax_prover/config.py:85-95` (`ProverConfig` dataclass)
- Test: `tests/unit/utils/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/utils/test_config.py` (it already imports from `ax_prover.config`; add `ProverConfig` to that import or import inline as shown):

```python
def test_prover_config_restrict_to_proof_body_defaults_false():
    from ax_prover.config import ProverConfig

    assert ProverConfig().restrict_to_proof_body is False
    assert ProverConfig(restrict_to_proof_body=True).restrict_to_proof_body is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_config.py::test_prover_config_restrict_to_proof_body_defaults_false -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'restrict_to_proof_body'` (and the attribute access would also fail).

- [ ] **Step 3: Add the field**

In `src/ax_prover/config.py`, add the field to `ProverConfig` right after `user_comments`:

```python
    user_comments: str | None = None
    restrict_to_proof_body: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_config.py::test_prover_config_restrict_to_proof_body_defaults_false -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/config.py tests/unit/utils/test_config.py
git commit -m "feat: add restrict_to_proof_body flag to ProverConfig"
```

---

## Task 2: TemporaryProposal skips imports/opens when locked

**Files:**
- Modify: `src/ax_prover/utils/build.py:358-377` (`__init__`), `:417-427` (`__enter__` import/open application)
- Test: `tests/unit/utils/test_build.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/utils/test_build.py` (top-level imports needed — add if absent):

```python
from pathlib import Path

from ax_prover.models.files import Location
from ax_prover.models.messages import ProposalMessage
from ax_prover.utils.build import TemporaryProposal


def _make_demo_project(tmp_path) -> Location:
    """Write a minimal Lean file and return its Location."""
    (tmp_path / "Demo.lean").write_text(
        "import Existing.Module\n\ntheorem thm : True := trivial\n",
        encoding="utf-8",
    )
    return Location(name="thm", module_path="Demo", is_external=False)


def test_temporary_proposal_applies_imports_and_opens_by_default(tmp_path):
    location = _make_demo_project(tmp_path)
    proposal = ProposalMessage(
        reasoning="r",
        code="",
        location=location,
        imports=["Mathlib.Tactic"],
        opens=["Nat"],
    )
    with TemporaryProposal(str(tmp_path), location, proposal) as applier:
        assert applier.success
        content = (Path(tmp_path) / applier.location.path).read_text(encoding="utf-8")
    assert "import Mathlib.Tactic" in content
    assert "open Nat" in content


def test_temporary_proposal_skips_imports_and_opens_when_locked(tmp_path):
    location = _make_demo_project(tmp_path)
    proposal = ProposalMessage(
        reasoning="r",
        code="",
        location=location,
        imports=["Mathlib.Tactic"],
        opens=["Nat"],
    )
    with TemporaryProposal(
        str(tmp_path), location, proposal, restrict_to_proof_body=True
    ) as applier:
        assert applier.success
        content = (Path(tmp_path) / applier.location.path).read_text(encoding="utf-8")
    assert "import Mathlib.Tactic" not in content
    assert "open Nat" not in content
    # The original (already-present) import is untouched.
    assert "import Existing.Module" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/utils/test_build.py -k temporary_proposal -v`
Expected: `test_..._when_locked` FAILS with `TypeError: __init__() got an unexpected keyword argument 'restrict_to_proof_body'`. (The default-behavior test should already PASS — it guards existing behavior.)

- [ ] **Step 3: Add the constructor parameter**

In `src/ax_prover/utils/build.py`, update `TemporaryProposal.__init__` (currently lines 358-377):

```python
    def __init__(
        self,
        base_folder: str,
        original_location: Location | None,
        proposal: "ProposalMessage",
        restrict_to_proof_body: bool = False,
    ):
        """Initialize the temporary proposal applier.

        Args:
            base_folder: Base folder path
            original_location: Location object for the original file (None means no location set)
            proposal: ProposalMessage with imports, opens, and code to apply
            restrict_to_proof_body: When True, do not write new imports or file-level
                opens to the file — only the proof body is edited.
        """
        self.base_folder = base_folder
        self.original_location = original_location
        self.proposal = proposal
        self.restrict_to_proof_body = restrict_to_proof_body
        self.location: Location | None = None  # Temp location, set in __enter__
        self.error: str = ""
        self.success: bool = False
        self._temp_file = None
```

- [ ] **Step 4: Gate the import/open application**

In `src/ax_prover/utils/build.py`, wrap the existing import/open blocks (currently lines 417-427) in the flag check. Code application (the `if self.proposal.code:` block) is left unchanged:

```python
            if not self.restrict_to_proof_body:
                if self.proposal.imports:
                    success = edit_imports(
                        self.base_folder, self.location.path, self.proposal.imports
                    )
                    if not success:
                        self.error = "Failed to apply imports to temp file"
                        return self

                if self.proposal.opens:
                    success = edit_opens(
                        self.base_folder, self.location.path, self.proposal.opens
                    )
                    if not success:
                        self.error = "Failed to apply opens to temp file"
                        return self
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/utils/test_build.py -k temporary_proposal -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ax_prover/utils/build.py tests/unit/utils/test_build.py
git commit -m "feat: TemporaryProposal skips imports/opens when restrict_to_proof_body"
```

---

## Task 3: Builder/reviewer wiring + dropped-preamble warning

**Files:**
- Modify: `src/ax_prover/prover/agent.py` — add `_dropped_preamble_warning()` helper; pass the flag at both `TemporaryProposal` sites (`:335`, `:457`); prepend the warning at the build-failed path (`:412`)
- Test: `tests/unit/prover/test_proof_body_restriction.py`

- [ ] **Step 1: Write the failing test for the warning helper**

Create `tests/unit/prover/test_proof_body_restriction.py`:

```python
"""Tests for the restrict-to-proof-body feature: prompt assembly and warning helper."""

from ax_prover.prover.agent import _dropped_preamble_warning


class TestDroppedPreambleWarning:
    def test_empty_when_flag_off(self):
        assert _dropped_preamble_warning(False, ["Mathlib.Tactic"], ["Nat"]) == ""

    def test_empty_when_nothing_proposed(self):
        assert _dropped_preamble_warning(True, [], []) == ""

    def test_names_imports_and_opens_when_locked(self):
        warning = _dropped_preamble_warning(True, ["Mathlib.Tactic"], ["Nat"])
        assert "IMPORTS/OPENS IGNORED" in warning
        assert "Mathlib.Tactic" in warning
        assert "Nat" in warning

    def test_imports_only(self):
        warning = _dropped_preamble_warning(True, ["Mathlib.Tactic"], [])
        assert "Mathlib.Tactic" in warning
        assert "open(s)" not in warning

    def test_opens_only(self):
        warning = _dropped_preamble_warning(True, [], ["Nat"])
        assert "Nat" in warning
        assert "import(s)" not in warning
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/prover/test_proof_body_restriction.py -v`
Expected: FAIL with `ImportError: cannot import name '_dropped_preamble_warning'`.

- [ ] **Step 3: Add the helper**

In `src/ax_prover/prover/agent.py`, add this module-level function (place it just above the `ProverAgent` class definition, after the imports):

```python
def _dropped_preamble_warning(
    restrict_to_proof_body: bool,
    imports: list[str],
    opens: list[str],
) -> str:
    """Warn the proposer that proposed imports/opens were dropped under a locked file.

    Returns an empty string when the restriction is off or nothing was proposed.
    """
    if not restrict_to_proof_body or not (imports or opens):
        return ""
    parts = []
    if imports:
        parts.append("import(s) " + ", ".join(f"`{name}`" for name in imports))
    if opens:
        parts.append("open(s) " + ", ".join(f"`{name}`" for name in opens))
    proposed = " and ".join(parts)
    return (
        f"WARNING — IMPORTS/OPENS IGNORED: you proposed {proposed}, but this file is "
        "locked — only the proof body may change, so they were NOT applied. All necessary "
        "imports are already present; use `open ... in` inside the proof body or "
        "fully-qualify names instead of adding file-level imports/opens."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/prover/test_proof_body_restriction.py -v`
Expected: all 5 `TestDroppedPreambleWarning` tests PASS.

- [ ] **Step 5: Pass the flag to both `TemporaryProposal` sites**

In `src/ax_prover/prover/agent.py`, the builder-node site (currently lines 335-337):

```python
        with TemporaryProposal(
            self.base_folder,
            state.item.location,
            state.last_proposal,
            restrict_to_proof_body=self.config.restrict_to_proof_body,
        ) as applier:
```

And the reviewer `apply_permanently` site (currently lines 457-459) — this is critical so the *final committed* file also excludes imports/opens:

```python
            with TemporaryProposal(
                self.base_folder,
                state.item.location,
                state.last_proposal,
                restrict_to_proof_body=self.config.restrict_to_proof_body,
            ) as applier:
                applier.apply_permanently()
```

- [ ] **Step 6: Prepend the warning at the build-failed path**

In `src/ax_prover/prover/agent.py`, replace the build-failed feedback construction (currently line 412):

```python
        feedback = BuildFailedFeedback(error_output=self._build_error_processing(message))
```

with:

```python
        error_output = self._build_error_processing(message)
        preamble_warning = _dropped_preamble_warning(
            self.config.restrict_to_proof_body,
            state.last_proposal.imports,
            state.last_proposal.opens,
        )
        if preamble_warning:
            error_output = f"{preamble_warning}\n\n{error_output}"
        feedback = BuildFailedFeedback(error_output=error_output)
```

- [ ] **Step 7: Run the prover test module + a quick import sanity check**

Run: `.venv/bin/pytest tests/unit/prover/ -v`
Expected: PASS (including existing `test_agent_routing.py`, `test_memory.py`).

Run: `.venv/bin/python -c "import ax_prover.prover.agent"`
Expected: no error (confirms the edits parse).

- [ ] **Step 8: Commit**

```bash
git add src/ax_prover/prover/agent.py tests/unit/prover/test_proof_body_restriction.py
git commit -m "feat: warn proposer and pass lock flag through builder/reviewer"
```

---

## Task 4: Prompt fragment + system-prompt assembly helper

**Files:**
- Modify: `src/ax_prover/prover/prompts.py` — add `PROOF_BODY_RESTRICTION_PROMPT` and `build_proposer_system_prompt()`
- Modify: `src/ax_prover/prover/agent.py:256-263` — use the helper
- Test: `tests/unit/prover/test_proof_body_restriction.py`

- [ ] **Step 1: Write the failing tests for the prompt helper**

Append to `tests/unit/prover/test_proof_body_restriction.py`:

```python
from ax_prover.prover.prompts import (
    PROOF_BODY_RESTRICTION_PROMPT,
    PROPOSER_SYSTEM_PROMPT,
    PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT,
    build_proposer_system_prompt,
)


class TestBuildProposerSystemPrompt:
    def test_iterative_selected_for_multi_iteration(self):
        prompt = build_proposer_system_prompt(max_iterations=50)
        assert prompt.startswith(PROPOSER_SYSTEM_PROMPT)

    def test_single_shot_selected_for_one_iteration(self):
        prompt = build_proposer_system_prompt(max_iterations=1)
        assert prompt.startswith(PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT)

    def test_restriction_fragment_absent_by_default(self):
        prompt = build_proposer_system_prompt(max_iterations=50)
        assert PROOF_BODY_RESTRICTION_PROMPT not in prompt

    def test_restriction_fragment_present_when_locked(self):
        prompt = build_proposer_system_prompt(
            max_iterations=50, restrict_to_proof_body=True
        )
        assert PROOF_BODY_RESTRICTION_PROMPT in prompt

    def test_restriction_fragment_present_in_single_shot_when_locked(self):
        prompt = build_proposer_system_prompt(
            max_iterations=1, restrict_to_proof_body=True
        )
        assert PROOF_BODY_RESTRICTION_PROMPT in prompt

    def test_user_comments_appended(self):
        prompt = build_proposer_system_prompt(
            max_iterations=50, user_comments="be terse"
        )
        assert "<user-comments>\nbe terse\n</user-comments>" in prompt


def test_restriction_fragment_mentions_key_rules():
    assert "import" in PROOF_BODY_RESTRICTION_PROMPT
    assert "open" in PROOF_BODY_RESTRICTION_PROMPT
    assert "locked" in PROOF_BODY_RESTRICTION_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/prover/test_proof_body_restriction.py -k "BuildProposerSystemPrompt or restriction_fragment_mentions" -v`
Expected: FAIL with `ImportError: cannot import name 'PROOF_BODY_RESTRICTION_PROMPT'` / `build_proposer_system_prompt`.

- [ ] **Step 3: Add the fragment and helper to prompts.py**

In `src/ax_prover/prover/prompts.py`, after the `PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT` definition (and before `PROPOSER_USER_PROMPT`), add:

```python
PROOF_BODY_RESTRICTION_PROMPT = """

<locked-file>
IMPORTANT — THIS FILE IS LOCKED: you may ONLY edit the proof body of the target theorem.
- Do NOT add new imports. Leave the `imports` field empty (`[]`). The file's imports are fixed and already complete for this problem.
- Do NOT add file-level `open` statements. Leave the `opens` field empty (`[]`).
- If you need a namespace, use `open ... in` immediately before the term or tactic block inside the proof body, or fully-qualify names.
Anything placed in the `imports` or `opens` fields is IGNORED and never written to the file, so relying on it only causes "unknown identifier" errors.
</locked-file>"""


def build_proposer_system_prompt(
    *,
    max_iterations: int,
    restrict_to_proof_body: bool = False,
    user_comments: str | None = None,
) -> str:
    """Assemble the proposer system prompt from config-driven options.

    Selects the single-shot vs iterative base prompt, appends the locked-file
    restriction when enabled, then any user comments.
    """
    prompt = (
        PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT
        if max_iterations == 1
        else PROPOSER_SYSTEM_PROMPT
    )
    if restrict_to_proof_body:
        prompt += PROOF_BODY_RESTRICTION_PROMPT
    if user_comments:
        prompt += f"\n\n<user-comments>\n{user_comments}\n</user-comments>"
    return prompt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/prover/test_proof_body_restriction.py -k "BuildProposerSystemPrompt or restriction_fragment_mentions" -v`
Expected: all tests PASS.

- [ ] **Step 5: Use the helper in the proposer node**

In `src/ax_prover/prover/agent.py`, replace the inline prompt selection + user-comments append (currently lines 256-263):

```python
        system_prompt = (
            PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT
            if self.config.max_iterations == 1
            else PROPOSER_SYSTEM_PROMPT
        )

        if self.config.user_comments:
            system_prompt += f"\n\n<user-comments>\n{self.config.user_comments}\n</user-comments>"
```

with:

```python
        system_prompt = build_proposer_system_prompt(
            max_iterations=self.config.max_iterations,
            restrict_to_proof_body=self.config.restrict_to_proof_body,
            user_comments=self.config.user_comments,
        )
```

- [ ] **Step 6: Update the prompts import in agent.py**

In `src/ax_prover/prover/agent.py`, the import block currently brings in `PROPOSER_SYSTEM_PROMPT` and `PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT` (around lines 61-62). Add `build_proposer_system_prompt` to that same `from ..prover.prompts import (...)` group:

```python
    PROPOSER_SYSTEM_PROMPT,
    PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT,
    build_proposer_system_prompt,
```

Note: `PROPOSER_SYSTEM_PROMPT` and `PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT` are no longer referenced directly in `agent.py` after Step 5, but keep them in the import only if still used elsewhere; otherwise remove them to satisfy ruff (Step 7 will catch an unused import). Verify with: `grep -n "PROPOSER_SYSTEM_PROMPT" src/ax_prover/prover/agent.py` — if the only remaining hits are the import lines, delete those two names from the import.

- [ ] **Step 7: Run the prover tests + lint**

Run: `.venv/bin/pytest tests/unit/prover/ -v`
Expected: PASS.

Run: `.venv/bin/ruff check src/ax_prover/prover/agent.py src/ax_prover/prover/prompts.py`
Expected: no errors (no unused imports).

- [ ] **Step 8: Commit**

```bash
git add src/ax_prover/prover/prompts.py src/ax_prover/prover/agent.py tests/unit/prover/test_proof_body_restriction.py
git commit -m "feat: locked-file prompt fragment via build_proposer_system_prompt"
```

---

## Task 5: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/pytest tests/unit -q`
Expected: all tests pass (no regressions).

- [ ] **Step 2: Format and lint the whole change**

Run: `.venv/bin/ruff format . && .venv/bin/ruff check .`
Expected: clean. If `ruff format` reflows any edited file, re-run the suite (Step 1) and amend the last commit:
```bash
git add -A && git commit --amend --no-edit
```

- [ ] **Step 3: Confirm default-off behavior is untouched**

Run: `.venv/bin/python -c "from ax_prover.config import ProverConfig; assert ProverConfig().restrict_to_proof_body is False; print('default off: OK')"`
Expected: `default off: OK`

- [ ] **Step 4: Push the branch and open the PR (targeting main)**

```bash
git push -u origin restrict-edits-to-proof-body
gh pr create --base main --head restrict-edits-to-proof-body \
  --title "feat: restrict edits to the proof body (lock challenge-file imports/opens)" \
  --body "Adds opt-in \`restrict_to_proof_body\` (default false). When enabled, the agent never writes new imports or file-level opens into the file — only the proof body changes — satisfying AI4Math Rules 1 & 2. Drops violating imports/opens in TemporaryProposal, warns the proposer in build feedback, and adds a locked-file prompt fragment. No messages.py changes (independent of PR #15). See docs/superpowers/specs/2026-06-05-restrict-edits-to-proof-body-design.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

**Do NOT merge into main.** After review, this branch is merged into `run_AI4Math_Challange`, and `restrict_to_proof_body: true` is then set in `configs/opus48_local.yaml` on the experiment branch (that file is exp-only).

---

## Self-Review

**1. Spec coverage:**
- Config flag (spec §1) → Task 1. ✓
- Mechanical enforcement, both call sites incl. reviewer apply_permanently (spec §2) → Task 2 + Task 3 Step 5. ✓
- Dropped-preamble warning via `error_output`, no `messages.py` change (spec §3) → Task 3. ✓
- Prompt fragment + conditional append, both prompt variants (spec §4) → Task 4. ✓
- Output fields kept, instructed empty (spec §5) → fragment text in Task 4 Step 3; no schema change. ✓
- Activation config (spec §6) → explicitly deferred to exp branch (Task 5 Step 4 note). ✓
- Testing (spec §7) → Tasks 1-4 tests + Task 5 full suite. ✓
- Branching (spec §8) → Task 5 Step 4 (base main, PR to main, no merge to main). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**3. Type consistency:** `restrict_to_proof_body: bool` used identically across `ProverConfig`, `TemporaryProposal.__init__`, `_dropped_preamble_warning`, and `build_proposer_system_prompt`. Helper names `_dropped_preamble_warning` and `build_proposer_system_prompt` are referenced consistently in tests and call sites. ✓
