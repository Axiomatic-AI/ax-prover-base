"""Tests for Lean code parsing utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from lean_interact.interface import (
    DeclarationInfo,
    DeclModifiers,
    DeclSignature,
    Pos,
    Range,
    ScopeInfo,
    Sorry,
    Tactic,
)

from ax_prover.models.declaration import Declaration
from ax_prover.utils.lean_parsing import (
    count_pattern,
    find_declaration_at_line,
    find_declaration_by_name,
    format_goal_state_at_sorries,
    list_declarations_from_code,
    read_declaration_source_code,
    strip_comments,
)

NESTED_COMMENT_CODE = """\
/- outer /- inner -/ still outer -/
def foo := 1
"""


class TestStripComments:
    """Tests for strip_comments function."""

    def test_no_comments(self):
        """Code without comments is unchanged."""
        src = "def foo := 1\ndef bar := 2"
        assert strip_comments(src) == src

    def test_line_comment_removed(self):
        """Line comments (--) are replaced with spaces."""
        src = "def foo := 1 -- this is a comment"
        result = strip_comments(src)
        assert "comment" not in result
        assert result.startswith("def foo := 1")

    def test_block_comment_removed(self):
        """Block comments (/- ... -/) are removed."""
        src = "/- hello -/ def foo := 1"
        result = strip_comments(src)
        assert "hello" not in result
        assert "def foo := 1" in result

    def test_nested_block_comments(self):
        """Nested block comments are handled correctly."""
        result = strip_comments(NESTED_COMMENT_CODE)
        assert "outer" not in result
        assert "inner" not in result
        assert "def foo := 1" in result

    def test_string_literal_preserved(self):
        """String literals are not treated as comments."""
        src = 'def s := "not -- a comment"'
        result = strip_comments(src)
        assert '"not -- a comment"' in result

    def test_preserves_line_count(self):
        """Output has same number of lines as input."""
        src = "/- multi\nline\ncomment -/\ndef foo := 1"
        result = strip_comments(src)
        assert result.count("\n") == src.count("\n")

    def test_preserves_byte_count_per_line(self):
        """Each line in output has same length as corresponding input line."""
        src = "def foo := 1 -- comment here"
        result = strip_comments(src)
        for orig_line, stripped_line in zip(src.splitlines(), result.splitlines(), strict=True):
            assert len(stripped_line) == len(orig_line)

    def test_empty_input(self):
        """Empty string returns empty string."""
        assert strip_comments("") == ""

    def test_doc_comment_stripped(self):
        """Lean4 doc comments (/-- ... -/) are also stripped."""
        src = "/-- My doc comment. -/\ndef foo := 1"
        result = strip_comments(src)
        assert "My doc comment" not in result
        assert "def foo := 1" in result


class TestCountPattern:
    """Tests for count_pattern function."""

    SORRY_PATTERN = r"\b(sorry|admit)\b"

    def test_no_sorries(self):
        """Clean code returns count 0."""
        code = "def foo := 42\ndef bar := 1 + 2"
        count, locations = count_pattern(code, pattern=self.SORRY_PATTERN)
        assert count == 0
        assert locations == []

    def test_single_sorry(self):
        """One sorry is detected with correct line number."""
        code = "def foo := by\n  sorry"
        count, locations = count_pattern(code, pattern=self.SORRY_PATTERN)
        assert count == 1
        assert locations[0][0] == 2  # line number

    def test_multiple_sorries(self):
        """Multiple sorries on different lines are all found."""
        code = "def foo := by\n  sorry\ndef bar := by\n  sorry"
        count, _ = count_pattern(code, pattern=self.SORRY_PATTERN)
        assert count == 2

    def test_sorry_and_admit(self):
        """Both 'sorry' and 'admit' are detected."""
        code = "def foo := by\n  sorry\ndef bar := by\n  admit"
        count, locations = count_pattern(code, pattern=self.SORRY_PATTERN)
        assert count == 2

    def test_context_lines(self):
        """Context lines around sorry are included."""
        code = "-- before\ndef foo := by\n  sorry\n-- after"
        _, locations = count_pattern(code, pattern=self.SORRY_PATTERN, context_lines=1)
        context_text = locations[0][1]
        assert "def foo" in context_text
        assert "sorry" in context_text

    def test_sorry_in_word_not_counted(self):
        """Words containing 'sorry' (e.g., 'sorry_lemma') are not counted."""
        code = "def sorry_lemma := 42"
        count, _ = count_pattern(code, pattern=self.SORRY_PATTERN)
        assert count == 0

    def test_custom_pattern_detects_axiom(self):
        """Custom pattern detects axiom declarations."""
        code = "axiom myAxiom : Nat → Nat\ntheorem foo : True := trivial"
        count, locations = count_pattern(code, pattern=r"\baxiom\b")
        assert count == 1
        assert locations[0][0] == 1

    def test_custom_pattern_detects_search_tactics(self):
        """Custom pattern detects apply? and exact?."""
        code = "theorem foo : P := by\n  apply?\n  exact?"
        count, _ = count_pattern(code, pattern=r"\b(apply|exact)\?")
        assert count == 2

    def test_custom_pattern_does_not_flag_apply_without_question_mark(self):
        """apply (without ?) is not flagged by the search tactics pattern."""
        code = "theorem foo : P := by\n  apply some_lemma"
        count, _ = count_pattern(code, pattern=r"\b(apply|exact)\?")
        assert count == 0

    def test_sorry_pattern_does_not_match_axiom(self):
        """Sorry/admit pattern does not match axiom."""
        code = "axiom myAxiom : Nat → Nat"
        count, _ = count_pattern(code, pattern=self.SORRY_PATTERN)
        assert count == 0


def _make_decl_info(
    name: str,
    start: tuple[int, int],
    finish: tuple[int, int],
    *,
    kind: str = "def",
    pp: str | None = None,
) -> DeclarationInfo:
    """Build a minimal DeclarationInfo for unit tests.

    Only the fields read by the code under test (`name`, `range`, `pp`) need realistic
    values; the rest are filled with defaults accepted by `lean_interact`'s models.
    """
    start_pos = Pos(line=start[0], column=start[1])
    finish_pos = Pos(line=finish[0], column=finish[1])
    decl_range = Range(synthetic=False, start=start_pos, finish=finish_pos)
    return DeclarationInfo(
        pp=pp or f"{kind} {name} := sorry",
        range=decl_range,
        scope=ScopeInfo(currNamespace=""),
        name=name,
        full_name=name,
        kind=kind,
        modifiers=DeclModifiers(),
        signature=DeclSignature(pp="", constants=[], range=decl_range),
    )


def _make_sorry(line: int, column: int, goal: str = "⊢ False") -> Sorry:
    return Sorry(
        pos=Pos(line=line, column=column),
        endPos=Pos(line=line, column=column + 5),
        goal=goal,
    )


def _make_tactic(line: int, column: int, tactic: str, goals: str = "⊢ True") -> Tactic:
    return Tactic(
        pos=Pos(line=line, column=column),
        endPos=Pos(line=line, column=column + len(tactic)),
        goals=goals,
        tactic=tactic,
    )


class TestListDeclarationsFromCode:
    """Tests for list_declarations_from_code.

    Given a Lean interact server response, we build Declaration objects whose statement (`pp`/`kind`)
    and `sorries` match what the response contains.
    """

    @pytest.fixture
    def scenarios(self) -> list[tuple[DeclarationInfo, list[Sorry]]]:
        """Single source of truth: each DeclarationInfo paired with the sorries
        that fall inside its range. The mock server response and the expected
        output are both derived from this list, so editing a scenario does not
        require rewriting any assertion.
        """
        return [
            (
                _make_decl_info(
                    "add",
                    start=(1, 0),
                    finish=(1, 48),
                    kind="def",
                    pp="noncomputable def add (a b : Nat) : Nat := a + b",
                ),
                [],
                [],
            ),
            (
                _make_decl_info(
                    "add_zero_proven",
                    start=(3, 0),
                    finish=(3, 54),
                    kind="theorem",
                    pp="theorem add_zero_proven (a : Nat) : add a 0 = a := rfl",
                ),
                [],
                [],
            ),
            (
                _make_decl_info(
                    "with_sorry",
                    start=(5, 0),
                    finish=(5, 54),
                    kind="theorem",
                    pp="theorem with_sorry (a : Nat) : add a 0 = a := by sorry",
                ),
                [_make_sorry(line=5, column=49, goal="⊢ add a 0 = a")],
                [
                    _make_tactic(
                        line=5,
                        column=49,
                        tactic="sorry",
                        goals="⊢ add a 0 = a",
                    )
                ],
            ),
            (
                _make_decl_info(
                    "Κατ.Μοδ.αβ_γ'δε₀₁₂_ℕtoℤ_φψ''ωΩ_über_café_Δ?!",
                    start=(25, 2),
                    finish=(25, 56),
                    kind="def",
                    pp="def Κατ.Μοδ.αβ_γ'δε₀₁₂_ℕtoℤ_φψ''ωΩ_über_café_Δ?! := 42",
                ),
                [],
                [],
            ),
            (
                _make_decl_info(
                    "double_sorry",
                    start=(25, 0),
                    finish=(28, 7),
                    kind="theorem",
                    pp="theorem double_sorry{n : Nat} : n + 0 = n := by\n have h : n + 0 = n := by\n    sorry\n apply?",
                ),
                [
                    _make_sorry(line=27, column=6, goal="n : Nat\n⊢ n + 0 = n"),
                ],
                [
                    _make_tactic(
                        line=26,
                        column=2,
                        tactic="have h : n + 0 = n := by sorry",
                        goals="n : Nat\n⊢ n + 0 = n",
                    ),
                    _make_tactic(line=27, column=4, tactic="sorry", goals="n : Nat\n⊢ n + 0 = n"),
                    _make_tactic(
                        line=28,
                        column=2,
                        tactic="apply?",
                        goals=" n : Nat\nh : n + 0 = n\n⊢ n + 0 = n",
                    ),
                ],
            ),
        ]

    @pytest.fixture
    def fake_server(self, scenarios):
        declarations = [info for info, _, _ in scenarios]
        sorries = [s for _, decl_sorries, _ in scenarios for s in decl_sorries]
        tactics = [t for _, _, decl_tactics in scenarios for t in decl_tactics]
        response = MagicMock(declarations=declarations, sorries=sorries, tactics=tactics)
        return MagicMock(run=AsyncMock(return_value=response))

    async def test_builds_one_declaration_per_response_entry_with_its_sorries(
        self, fake_server, scenarios
    ):
        """Output mirrors the response: one Declaration per entry (statement preserved),
        with each sorry attached to the declaration whose range contains it."""
        result = await list_declarations_from_code(fake_server, "...")

        expected = [
            Declaration(info=info, sorries=sorries, tactics=tactics)
            for info, sorries, tactics in scenarios
        ]
        assert result == expected


class TestFormatGoalStateAtSorries:
    """Tests for format_goal_state_at_sorries function."""

    def test_empty_returns_no_sorries_message(self):
        """An empty list yields the sentinel message."""
        assert format_goal_state_at_sorries([]) == "No sorries found in code."

    def test_formats_with_index_position_and_goal(self):
        """Each sorry produces an entry with 1-based index, line/column, and goal text."""
        result = format_goal_state_at_sorries(
            [
                _make_sorry(line=5, column=10, goal="⊢ x + 0 = x"),
                _make_sorry(line=7, column=2, goal="⊢ True"),
            ]
        )
        assert result == (
            "Sorry #1 at line 5, column 10:\n⊢ x + 0 = x\n\nSorry #2 at line 7, column 2:\n⊢ True\n"
        )


class TestFindDeclarationByName:
    """Tests for find_declaration_by_name function."""

    @pytest.fixture
    def declarations(self):
        return [
            Declaration(info=_make_decl_info("foo", start=(1, 0), finish=(2, 0))),
            Declaration(info=_make_decl_info("bar", start=(3, 0), finish=(4, 0), kind="theorem")),
        ]

    def test_finds_existing(self, declarations):
        """Returns the declaration when found."""
        result = find_declaration_by_name(declarations, "foo")
        assert result is not None
        assert result.info.name == "foo"

    def test_returns_none_for_missing(self, declarations):
        """Returns None when name not found."""
        assert find_declaration_by_name(declarations, "baz") is None

    def test_empty_list(self):
        """Returns None for empty declarations list."""
        assert find_declaration_by_name([], "foo") is None


class TestFindDeclarationAtLine:
    """Tests for find_declaration_at_line function.

    Each case provides its own declarations and the exact declaration expected to be
    returned (compared by identity), so there are no magic names to keep in sync and
    new scenarios can be added by appending a row.
    """

    FIRST = Declaration(info=_make_decl_info("first", start=(1, 0), finish=(3, 5)))
    SECOND = Declaration(info=_make_decl_info("second", start=(5, 0), finish=(8, 5)))
    OUTER = Declaration(info=_make_decl_info("outer", start=(1, 0), finish=(10, 15)))
    INNER = Declaration(info=_make_decl_info("inner", start=(4, 0), finish=(6, 18)))

    @pytest.mark.parametrize(
        "declarations, line, expected",
        [
            ([FIRST, SECOND], 1, FIRST),  # start boundary is inclusive
            ([FIRST, SECOND], 2, FIRST),  # interior line
            ([FIRST, SECOND], 3, FIRST),  # finish boundary is inclusive
            ([FIRST, SECOND], 5, SECOND),  # start boundary of a later declaration
            ([FIRST, SECOND], 8, SECOND),  # finish boundary of a later declaration
            ([FIRST, SECOND], 4, None),  # gap between declarations
            ([FIRST, SECOND], 9, None),  # beyond all declarations
            ([OUTER, INNER], 5, INNER),  # nested: smallest containing range wins
            ([OUTER, INNER], 2, OUTER),  # nested: only the outer range contains the line
            ([], 1, None),  # empty list
        ],
    )
    def test_find_declaration_at_line(self, declarations, line, expected):
        """Returns the smallest-range declaration containing the line, or None."""
        assert find_declaration_at_line(declarations, line) is expected


class TestReadDeclarationSourceCode:
    """Tests for read_declaration_source_code.

    The contract: the returned text must be exactly the slice of the file occupied by the
    declaration's line range (start.line..finish.line, both 1-indexed and inclusive), with no
    characters added, dropped, or duplicated. The expected source for every case is written out
    by hand (never computed from the content) so a shared off-by-one in both the helper and the
    function under test cannot make a broken implementation look correct.
    """

    @staticmethod
    def _decl(start_line: int, finish_line: int) -> Declaration:
        # Only the line numbers of the range matter here; columns are irrelevant because the
        # function works line-by-line. Use distinctive columns to make that explicit.
        return Declaration(
            info=_make_decl_info("d", start=(start_line, 0), finish=(finish_line, 5))
        )

    # (content, start_line, finish_line, expected_source) — content always uses '\n' line endings,
    # and expected_source is spelled out literally rather than sliced from content.
    CASES = {
        "single_line_at_top": (
            "theorem foo : True := trivial\ndef bar := 1\n",
            1,
            1,
            "theorem foo : True := trivial\n",
        ),
        "single_line_in_middle": (
            "def a := 1\ntheorem t : True := trivial\ndef b := 2\n",
            2,
            2,
            "theorem t : True := trivial\n",
        ),
        "single_line_at_eof_no_trailing_newline": (
            "def a := 1\ndef b := 2",
            2,
            2,
            "def b := 2",
        ),
        "multi_line_in_middle": (
            "import Mathlib\n\ntheorem foo : True := by\n  trivial\n\ndef after := 0\n",
            3,
            4,
            "theorem foo : True := by\n  trivial\n",
        ),
        "multi_line_with_internal_blank_line": (
            "def before := 0\ntheorem foo : True := by\n\n  trivial\ndef after := 1\n",
            2,
            4,
            "theorem foo : True := by\n\n  trivial\n",
        ),
        "indentation_and_tabs_preserved": (
            "namespace N\ntheorem foo : True := by\n\t\thave h := trivial\n        exact h\nend N\n",
            2,
            4,
            "theorem foo : True := by\n\t\thave h := trivial\n        exact h\n",
        ),
        "spans_whole_file": (
            "theorem foo : True := by\n  trivial\n",
            1,
            2,
            "theorem foo : True := by\n  trivial\n",
        ),
        "multi_line_at_eof_no_trailing_newline": (
            "def a := 1\ntheorem foo : True := by\n  trivial",
            2,
            3,
            "theorem foo : True := by\n  trivial",
        ),
        "blank_lines_inside_range_are_kept": (
            "a\n\n\nb\n",
            1,
            4,
            "a\n\n\nb\n",
        ),
    }

    @pytest.mark.parametrize(
        "content, start_line, finish_line, expected_source", list(CASES.values()), ids=list(CASES)
    )
    def test_returns_exact_original_source(
        self, tmp_path, content, start_line, finish_line, expected_source
    ):
        """The returned text equals the exact original source of the declaration's lines."""
        file_path = tmp_path / "Thing.lean"
        file_path.write_text(content, encoding="utf-8")

        result = read_declaration_source_code(self._decl(start_line, finish_line), file_path)

        assert result in content  # The result is a substring of the original content
        assert result == expected_source

    def test_does_not_insert_blank_lines_between_lines(self, tmp_path):
        """Regression: adjacent declaration lines stay adjacent (no doubled separators)."""
        content = "theorem foo : True := by\n  have h := trivial\n  exact h\n"
        file_path = tmp_path / "Thing.lean"
        file_path.write_text(content, encoding="utf-8")

        result = read_declaration_source_code(self._decl(1, 3), file_path)

        assert "\n\n" not in result
        assert result.splitlines() == content.splitlines()
