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
