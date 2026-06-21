"""Tests for the Declaration model."""

import pytest
from lean_interact.interface import Tactic

from ax_prover.models.declaration import Declaration


def _make_tactic(tactic: str) -> Tactic:
    """Build a minimal Tactic carrying only the applied tactic string."""
    return Tactic(
        pos={"line": 1, "column": 0},
        endPos={"line": 1, "column": len(tactic)},
        goals="",
        tactic=tactic,
    )


def _make_declaration(tactics: list[str]) -> Declaration:
    """Build a Declaration with the given tactic strings, skipping DeclarationInfo."""
    return Declaration.model_construct(tactics=[_make_tactic(t) for t in tactics])


# Tactics that ARE Lean search tactics (ground truth).
SEARCH_TACTICS = [
    # exact? and its surface variants
    "exact?",
    "exact? using h₁ h₂",
    "exact? says exact foo",
    "exact? [-lemma₁, -lemma₂]",
    "exact? (config := { x := 1 })",
    "exact? +grind",
    "exact? +try?",
    "exact? -star",
    "exact? +all",
    "exact?%",
    # apply? and its surface variants
    "apply?",
    "apply? using h₁ h₂",
    "apply? says apply foo",
    "apply? [-lemma₁]",
    "apply? (config := { x := 1 })",
    "apply? +all",
    "apply?%",
    # rw?
    "rw?",
    "rw? at h",
    "rw? [-lemma₁, -lemma₂]",
    # simp-family discovery tactics
    "simp?",
    "simp?!",
    "simp_all?",
    "decide?",
    "gcongr?",
    # other tactic-level discovery
    "observe?",
    "observe",
    "hint",
    # More convoluted syntaxes that are still search tactics
    "intro x; exact?",
    "· exact?",
    "simp <;> exact?",
    "constructor <;> simp?",
    "case foo => exact?",
]

# Tactics that are NOT Lean search tactics (ground truth).
NON_SEARCH_TACTICS = [
    # ordinary tactics with no trailing '?'
    "simp",
    "simp only [foo]",
    "exact foo",
    "apply foo",
    "rw [foo]",
    "intro x",
    "rfl",
    "decide",
    # discovery tools that have no '?' suffix
    "polyrith",
    "select_premises",
    "suggest_tactics",
    # elaborator commands (start with '#', run outside `by` blocks)
    "#find _ + _",
    '#leansearch "query"',
    '#moogle "query"',
    "#loogle Nat.succ",
    # metavariables / goal tags: contain '?' but no preceding identifier
    "refine ⟨?_, ?_⟩",
    "exact ?foo",
    "refine f ?a ?b",
    "cases h with | inl h => ?_",
]


class TestSearchTactics:
    """Tests for Declaration.search_tactics."""

    @pytest.mark.parametrize("tactic", SEARCH_TACTICS)
    def test_recognizes_search_tactic(self, tactic):
        """Each known search-tactic surface form is extracted, even when embedded."""
        declaration = _make_declaration([tactic])
        assert [t.tactic for t in declaration.search_tactics] == [tactic]

    @pytest.mark.parametrize("tactic", NON_SEARCH_TACTICS)
    def test_ignores_non_search_tactic(self, tactic):
        """Non-search tactics, commands, and metavariables are excluded."""
        declaration = _make_declaration([tactic])
        assert declaration.search_tactics == []

    def test_extracts_only_search_tactics_from_mixed_list(self):
        """A mixed proof yields exactly the search tactics, preserving order."""
        tactics = SEARCH_TACTICS + NON_SEARCH_TACTICS
        declaration = _make_declaration(tactics)
        assert [t.tactic for t in declaration.search_tactics] == SEARCH_TACTICS

    def test_empty_when_no_tactics(self):
        """A declaration with no tactics has no search tactics."""
        declaration = _make_declaration([])
        assert declaration.search_tactics == []
