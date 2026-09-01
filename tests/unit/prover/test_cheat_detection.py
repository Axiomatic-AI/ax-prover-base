"""Tests for the reviewer's cheat detection (agent._detect_cheats_in_code)."""

from types import SimpleNamespace

from lean_interact.interface import Tactic

from ax_prover.models.declaration import Declaration
from ax_prover.models.messages import SearchTacticsDetectedFeedback
from ax_prover.prover.agent import _detect_cheats_in_code


def _make_tactic(line: int, tactic: str) -> Tactic:
    return Tactic(
        pos={"line": line, "column": 0},
        endPos={"line": line, "column": len(tactic)},
        goals="",
        tactic=tactic,
    )


def _make_declaration(tactics: list[Tactic], *, kind: str = "theorem") -> Declaration:
    """Declaration carrying only what cheat detection reads (info.kind, tactics)."""
    return Declaration.model_construct(info=SimpleNamespace(kind=kind), tactics=tactics)


class TestSearchTacticsFeedback:
    """The search-tactics rejection must pinpoint the offending tactics."""

    async def test_reports_each_offending_tactic_by_line(self):
        proposed = _make_declaration(
            [
                _make_tactic(line=42, tactic="simp?"),
                _make_tactic(line=50, tactic="exact? using h"),
            ]
        )

        feedback = await _detect_cheats_in_code(proposed, [], [])

        assert isinstance(feedback, SearchTacticsDetectedFeedback)
        assert feedback.count == 2
        assert "line 42: simp?" in feedback.locations
        assert "line 50: exact? using h" in feedback.locations

    async def test_does_not_flag_innocent_tactics(self):
        """field_simp and bare apply are ordinary tactics, not search tactics."""
        proposed = _make_declaration(
            [_make_tactic(line=1, tactic="field_simp"), _make_tactic(line=2, tactic="apply foo")]
        )

        assert await _detect_cheats_in_code(proposed, [], []) is None
