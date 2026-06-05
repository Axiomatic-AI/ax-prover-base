"""Tests for the restrict-to-proof-body feature: prompt assembly and warning helper."""

from ax_prover.prover.agent import _dropped_preamble_warning
from ax_prover.prover.prompts import (
    PROOF_BODY_RESTRICTION_PROMPT,
    PROPOSER_SYSTEM_PROMPT,
    PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT,
    build_proposer_system_prompt,
)


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
