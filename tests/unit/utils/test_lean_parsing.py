"""Tests for Lean code parsing utilities."""

import pytest

from ax_prover.models.declaration import Declaration, DeclarationType
from ax_prover.utils.lean_parsing import (
    count_pattern,
    extract_function_from_content,
    extract_theorem_name,
    find_declaration_at_line,
    find_declaration_by_name,
    list_all_declarations_in_lean_code,
    normalize_location,
    strip_comments,
)

SAMPLE_LEAN_CODE = r"""
import Mathlib.Topology.Basic

/-- Addition of naturals. -/
def add (a b : Nat) : Nat :=
  a + b

/-- Commutativity of addition. -/
theorem add_comm (a b : Nat) : add a b = add b a := by
  simp [add]
  omega

lemma helper_lemma{n : Nat} : n + 0 = n := by
  sorry

def Κατ.Μοδ.αβ_γ'δε₀₁₂³_ℕtoℤ_φψ''ωΩ_über_café_∂Δ_Привет?! := 42

theorem Some.Very.«Nested.Theorem»?: P :=
    sorry
"""

# Trailing whitespace in content comes from strip_comments replacing doc comments
# and block comments with spaces (to preserve byte offsets), then the parser
# appending those whitespace-only lines as part of the preceding declaration's content.
EXPECTED_DECLARATIONS: list[Declaration] = [
    Declaration(
        declaration_type=DeclarationType.Import,
        name="Mathlib.Topology.Basic",
        content="",
    ),
    Declaration(
        declaration_type=DeclarationType.Definition,
        name="add",
        content="(a b : Nat) : Nat :=\n  a + b",
    ),
    Declaration(
        declaration_type=DeclarationType.Theorem,
        name="add_comm",
        content="(a b : Nat) : add a b = add b a := by\n  simp [add]\n  omega",
    ),
    Declaration(
        declaration_type=DeclarationType.Lemma,
        name="helper_lemma",
        content=r"{n : Nat} : n + 0 = n := by" + "\n  sorry",
    ),
    Declaration(
        declaration_type=DeclarationType.Definition,
        name="Κατ.Μοδ.αβ_γ'δε₀₁₂³_ℕtoℤ_φψ''ωΩ_über_café_∂Δ_Привет?!",
        content=":= 42",
    ),
    Declaration(
        declaration_type=DeclarationType.Theorem,
        name="Some.Very.«Nested.Theorem»?",
        content=": P :=\n    sorry",
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
    ("helper_lemma", "lemma helper_lemma{n : Nat} : n + 0 = n := by" + "\n  sorry"),
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

    def test_prefix_name_not_confused_with_longer_namespaced_name(self):
        """Extracting 'Treap' must not match an earlier 'Treap.insert' declaration."""
        code = (
            "def Treap.insert (t : Treap) : Treap :=\n  t\n\nstructure Treap where\n  key : Nat\n"
        )
        treap = extract_function_from_content(code, "Treap")
        assert treap is not None
        assert treap.startswith("structure Treap where")
        assert "def Treap.insert" not in treap
        # The longer name is still extractable on its own.
        insert = extract_function_from_content(code, "Treap.insert")
        assert insert is not None
        assert insert.startswith("def Treap.insert")

    def test_universe_polymorphic_name(self):
        """A name followed by a universe binder `.{u}` is extractable."""
        code = "theorem foo.{u} (x : Type u) : x = x := by rfl"
        block = extract_function_from_content(code, "foo")
        assert block is not None
        assert block.startswith("theorem foo.{u}")

    def test_universe_binder_does_not_match_qualified_name(self):
        """Searching `foo` must not match a different decl `foo.bar`."""
        code = "theorem foo.bar : True := trivial"
        assert extract_function_from_content(code, "foo") is None

    def test_includes_leading_open_in_prefix(self):
        """A contiguous `open … in` prefix that binds to the decl is included."""
        code = "open Nat in\ntheorem target (n : Nat) : n = n := by rfl"
        block = extract_function_from_content(code, "target")
        assert block is not None
        assert "open Nat in" in block
        assert "theorem target" in block

    def test_includes_multiple_in_prefixes(self):
        """Several stacked `… in` prefix commands are all included."""
        code = (
            "set_option maxHeartbeats 400000 in\nopen Nat in\ntheorem target : True := by trivial"
        )
        block = extract_function_from_content(code, "target")
        assert block is not None
        assert block.startswith("set_option maxHeartbeats 400000 in")
        assert "open Nat in" in block
        assert "theorem target" in block

    def test_does_not_pull_in_preceding_unrelated_decl(self):
        """An ordinary preceding declaration is not pulled into the block."""
        code = "theorem other : True := trivial\ntheorem target : True := trivial"
        block = extract_function_from_content(code, "target")
        assert block is not None
        assert "other" not in block
        assert block.startswith("theorem target")

    def test_stops_at_blank_line_before_in_prefix(self):
        """A blank line separates an `… in` prefix that does not bind to the decl."""
        code = "open Nat in\n\ntheorem target : True := by trivial"
        block = extract_function_from_content(code, "target")
        assert block is not None
        assert "open Nat in" not in block
        assert block.startswith("theorem target")

    def test_commented_decl_skipped_block_comment(self):
        """A `def foo` inside a block comment must not be selected as occurrence 0."""
        code = "/-\ndef foo := 1\n-/\ndef foo := 2\n"
        block = extract_function_from_content(code, "foo", 0)
        assert block is not None
        assert block.startswith("def foo := 2")
        assert ":= 1" not in block

    def test_commented_decl_skipped_line_comment(self):
        """A `def foo` after a `--` line comment must not be selected as occurrence 0."""
        code = "-- def foo := 1\ndef foo := 2\n"
        block = extract_function_from_content(code, "foo", 0)
        assert block is not None
        assert block.startswith("def foo := 2")
        assert ":= 1" not in block

    def test_no_commented_decls_occurrence_unchanged(self):
        """With no commented decls, occurrence indexing is unchanged (regression guard)."""
        code = "def foo := 1\n\ndef foo := 2\n"
        first = extract_function_from_content(code, "foo", 0)
        second = extract_function_from_content(code, "foo", 1)
        assert first is not None and first.startswith("def foo := 1")
        assert second is not None and second.startswith("def foo := 2")


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
        assert str(by_name[expected.name]) == str(expected)

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


class TestModifiersAndAttributes:
    """Declarations carrying modifiers (`noncomputable`/`partial`/`private`/...) or inline
    attributes (`@[simp]`) must be parsed, extracted, and located like any other decl.

    These were previously dropped because the parser keyed off the first whitespace-delimited
    word, which for `noncomputable def foo` was `noncomputable` (not a keyword).
    """

    @pytest.mark.parametrize(
        ("code", "expected_type", "expected_name"),
        [
            (
                "noncomputable def dijkstra_rec (g : G) : T := foo",
                DeclarationType.NoncomputableDef,
                "dijkstra_rec",
            ),
            (
                "noncomputable abbrev weightSum : Nat := 0",
                DeclarationType.NoncomputableAbbrev,
                "weightSum",
            ),
            ("partial def dRec (x : Nat) : Nat := dRec x", DeclarationType.Definition, "dRec"),
            ("private def secret : Nat := 1", DeclarationType.Definition, "secret"),
            ("protected def prot : Nat := 2", DeclarationType.Definition, "prot"),
            ("unsafe def danger : Nat := 0", DeclarationType.Definition, "danger"),
            ("@[simp] def tagged : Nat := 0", DeclarationType.Definition, "tagged"),
            ("@[simp, reducible] private def both : Nat := 0", DeclarationType.Definition, "both"),
        ],
    )
    def test_modified_declaration_detected(self, code, expected_type, expected_name):
        decls = list_all_declarations_in_lean_code(code)
        assert len(decls) == 1
        assert decls[0].declaration_type == expected_type
        assert decls[0].name == expected_name

    def test_names_are_correct_for_modified_decls(self):
        code = (
            "noncomputable def relax_neighbors (g : G) : T := foo\n"
            "private def secret : Nat := 1\n"
            "@[simp] def tagged : Nat := 0\n"
        )
        names = [d.name for d in list_all_declarations_in_lean_code(code)]
        assert names == ["relax_neighbors", "secret", "tagged"]

    def test_modifier_word_in_body_is_not_a_false_positive(self):
        # A body line beginning with a modifier-like identifier must not be parsed as a decl.
        code = "theorem foo : T := by\n  private_helper := bar\n  exact x"
        names = [d.name for d in list_all_declarations_in_lean_code(code)]
        assert names == ["foo"]

    def test_extract_noncomputable_def_block(self):
        code = (
            "/-- Relaxes neighbours. -/\n"
            "noncomputable def relax_neighbors (g : G) : T :=\n"
            "  foo\n\n"
            "def other : Nat := 0\n"
        )
        block = extract_function_from_content(code, "relax_neighbors")
        assert block is not None
        assert block.startswith("/-- Relaxes neighbours. -/")
        assert "noncomputable def relax_neighbors" in block
        assert "def other" not in block  # stops before the next declaration

    def test_extract_stops_before_next_attributed_decl(self):
        code = "def first : Nat := 0\n@[simp] def second : Nat := 1\n"
        block = extract_function_from_content(code, "first")
        assert block is not None
        assert "second" not in block

    def test_extract_attributed_def_block(self):
        code = "@[simp] def tagged : Nat := 0\n"
        block = extract_function_from_content(code, "tagged")
        assert block is not None
        assert block.startswith("@[simp] def tagged")

    def test_find_declaration_by_name_finds_noncomputable_target(self):
        # Coupling guard: the builder verifies the target via find_declaration_by_name; a
        # `noncomputable def` target used to be invisible -> false MissingTargetTheorem.
        decls = list_all_declarations_in_lean_code("noncomputable def foo := 1")
        assert find_declaration_by_name(decls, "foo") is not None

    def test_sorry_inside_noncomputable_def_attributed_to_it(self):
        code = "noncomputable def foo : Nat := by\n  sorry"
        assert find_declaration_at_line(code, 2) == "foo"
