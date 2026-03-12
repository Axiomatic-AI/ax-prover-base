"""Tests for Lean code parsing utilities."""

import pytest

from ax_prover.models.declaration import Declaration, DeclarationType
from ax_prover.utils.lean_parsing import (
    count_sorries,
    extract_function_from_content,
    extract_theorem_name,
    find_declaration_by_name,
    list_all_declarations_in_lean_code,
    normalize_location,
    strip_comments,
)

SAMPLE_LEAN_CODE = """\
import Mathlib.Topology.Basic

/-- Addition of naturals. -/
def add (a b : Nat) : Nat :=
  a + b

/-- Commutativity of addition. -/
theorem add_comm (a b : Nat) : add a b = add b a := by
  simp [add]
  omega

lemma helper_lemma(n : Nat) : n + 0 = n := by
  sorry
"""

# Trailing whitespace in content comes from strip_comments replacing doc comments
# and block comments with spaces (to preserve byte offsets), then the parser
# appending those whitespace-only lines as part of the preceding declaration's content.
EXPECTED_DECLARATIONS: list[Declaration] = [
    Declaration(
        declaration_type=DeclarationType.Import,
        name="Mathlib",
        content=". Topology . Basic\n\n                            ",
    ),
    Declaration(
        declaration_type=DeclarationType.Definition,
        name="add",
        content="( a b : Nat ) : Nat : =\n  a + b\n\n                                 ",
    ),
    Declaration(
        declaration_type=DeclarationType.Theorem,
        name="add_comm",
        content="( a b : Nat ) : add a b = add b a : = by\n  simp [add]\n  omega\n",
    ),
    Declaration(
        declaration_type=DeclarationType.Lemma,
        name="helper_lemma",
        content="( n : Nat ) : n + 0 = n : = by\n  sorry\n",
    ),
]

EXPECTED_FUNCTION_EXTRACTIONS: list[tuple[str, str | None]] = [
    ("add", "/-- Addition of naturals. -/\ndef add (a b : Nat) : Nat :=\n  a + b"),
    (
        "add_comm",
        "/-- Commutativity of addition. -/\n"
        "theorem add_comm (a b : Nat) : add a b = add b a := by\n"
        "  simp [add]\n"
        "  omega",
    ),
    ("helper_lemma", "lemma helper_lemma(n : Nat) : n + 0 = n := by\n  sorry"),
    ("nonexistent", None),
]

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


class TestCountSorries:
    """Tests for count_sorries function."""

    def test_no_sorries(self):
        """Clean code returns count 0."""
        code = "def foo := 42\ndef bar := 1 + 2"
        count, locations = count_sorries(code)
        assert count == 0
        assert locations == []

    def test_single_sorry(self):
        """One sorry is detected with correct line number."""
        code = "def foo := by\n  sorry"
        count, locations = count_sorries(code)
        assert count == 1
        assert locations[0][0] == 2  # line number

    def test_multiple_sorries(self):
        """Multiple sorries on different lines are all found."""
        code = "def foo := by\n  sorry\ndef bar := by\n  sorry"
        count, _ = count_sorries(code)
        assert count == 2

    def test_sorry_and_admit(self):
        """Both 'sorry' and 'admit' are detected."""
        code = "def foo := by\n  sorry\ndef bar := by\n  admit"
        count, locations = count_sorries(code)
        assert count == 2

    def test_context_lines(self):
        """Context lines around sorry are included."""
        code = "-- before\ndef foo := by\n  sorry\n-- after"
        _, locations = count_sorries(code, context_lines=1)
        context_text = locations[0][1]
        assert "def foo" in context_text
        assert "sorry" in context_text

    def test_sorry_in_word_not_counted(self):
        """Words containing 'sorry' (e.g., 'sorry_lemma') are not counted."""
        code = "def sorry_lemma := 42"
        count, _ = count_sorries(code)
        assert count == 0


class TestExtractFunctionFromContent:
    """Tests for extract_function_from_content function."""

    @pytest.mark.parametrize(
        "name, expected",
        EXPECTED_FUNCTION_EXTRACTIONS,
        ids=[name for name, _ in EXPECTED_FUNCTION_EXTRACTIONS],
    )
    def test_extract_function(self, name, expected):
        """Extracts the exact expected text for each declaration."""
        assert extract_function_from_content(SAMPLE_LEAN_CODE, name) == expected

    def test_namespaced_function(self):
        """Functions with dots in names can be extracted."""
        code = "theorem Poly.not_principal : P := by sorry"
        assert extract_function_from_content(code, "Poly.not_principal") == code


class TestExtractTheoremName:
    """Tests for extract_theorem_name function."""

    @pytest.mark.parametrize(
        "stmt, expected",
        [
            ("theorem foo : P := sorry", "foo"),
            ("lemma bar(n : Nat) : n > 0 := by sorry", "bar"),
            ("def baz := 42", "baz"),
            (
                "theorem Polynomial.not_isPrincipalIdealRing : P := sorry",
                "Polynomial.not_isPrincipalIdealRing",
            ),
            ("-- just a comment", None),
            ("", None),
            ("instance myInstance : Foo := {}", "myInstance"),
        ],
    )
    def test_extract_theorem_name(self, stmt, expected):
        """Extracts theorem name from various declaration types."""
        assert extract_theorem_name(stmt) == expected


class TestListAllDeclarationsInLeanCode:
    """Tests for list_all_declarations_in_lean_code function."""

    def test_finds_all_declarations(self):
        """Finds exactly the expected declarations in order."""
        declarations = list_all_declarations_in_lean_code(SAMPLE_LEAN_CODE)
        names = [d.name for d in declarations]
        expected_names = [d.name for d in EXPECTED_DECLARATIONS]
        assert names == expected_names

    @pytest.mark.parametrize(
        "expected",
        EXPECTED_DECLARATIONS,
        ids=[d.name for d in EXPECTED_DECLARATIONS],
    )
    def test_declaration_type_correct(self, expected):
        """Each declaration has the correct type."""
        declarations = list_all_declarations_in_lean_code(SAMPLE_LEAN_CODE)
        by_name = {d.name: d for d in declarations}
        assert expected.name in by_name
        assert by_name[expected.name].declaration_type == expected.declaration_type

    @pytest.mark.parametrize(
        "expected",
        EXPECTED_DECLARATIONS,
        ids=[d.name for d in EXPECTED_DECLARATIONS],
    )
    def test_declaration_content_correct(self, expected):
        """Each declaration has the expected content (ignoring trailing whitespace)."""
        declarations = list_all_declarations_in_lean_code(SAMPLE_LEAN_CODE)
        by_name = {d.name: d for d in declarations}
        assert expected.name in by_name
        # Compare the content ignoring the trailing whitespaces
        assert by_name[expected.name].content.strip() == expected.content.strip()

    def test_import_detected(self):
        """Import statements are listed as declarations."""
        declarations = list_all_declarations_in_lean_code(SAMPLE_LEAN_CODE)
        expected_imports = [
            d for d in EXPECTED_DECLARATIONS if d.declaration_type == DeclarationType.Import
        ]
        actual_imports = [d for d in declarations if d.declaration_type == DeclarationType.Import]
        assert len(actual_imports) == len(expected_imports)
        for actual, expected in zip(actual_imports, expected_imports, strict=True):
            assert actual.name == expected.name

    def test_empty_code(self):
        """Empty code returns empty list."""
        assert list_all_declarations_in_lean_code("") == []

    def test_comments_ignored(self):
        """Declarations inside comments are not detected."""
        code = "/- def hidden := 42 -/\ndef visible := 1"
        declarations = list_all_declarations_in_lean_code(code)
        names = [d.name for d in declarations]
        assert "visible" in names
        assert "hidden" not in names


class TestNormalizeLocation:
    """Tests for normalize_location function."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("Module.Path:func", "Module.Path:func"),
            ("path/to/file.lean:func", "path.to.file:func"),
            ("no_colon_at_all", "no_colon_at_all"),
            ("A/B.lean:foo", "A.B:foo"),
        ],
    )
    def test_normalize_location(self, input_str, expected):
        """Normalizes file paths to module paths."""
        assert normalize_location(input_str) == expected


class TestFindDeclarationByName:
    """Tests for find_declaration_by_name function."""

    @pytest.fixture
    def declarations(self):
        """Sample declarations list."""
        return [
            Declaration(declaration_type=DeclarationType.Definition, name="foo", content="42"),
            Declaration(declaration_type=DeclarationType.Theorem, name="bar", content=": P"),
        ]

    def test_finds_existing(self, declarations):
        """Returns the declaration when found."""
        result = find_declaration_by_name(declarations, "foo")
        assert result is not None
        assert result.name == "foo"

    def test_returns_none_for_missing(self, declarations):
        """Returns None when name not found."""
        result = find_declaration_by_name(declarations, "baz")
        assert result is None

    def test_empty_list(self):
        """Returns None for empty declarations list."""
        assert find_declaration_by_name([], "foo") is None
