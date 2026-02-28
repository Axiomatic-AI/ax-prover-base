"""Tests for file utilities: find_declaration_at_line and edit_function."""

from pathlib import Path

import pytest

from ax_prover.models.files import Location
from ax_prover.utils.files import edit_function
from ax_prover.utils.lean_parsing import find_declaration_at_line


@pytest.fixture
def test_base_folder():
    """Return the path to the test fixtures directory."""
    return str(Path(__file__).parent.parent / "fixtures")


class TestFindDeclarationAtLine:
    """Tests for find_declaration_at_line function."""

    def test_line_in_definition_body(self, test_base_folder):
        """Test finding declaration when line is in function body."""
        content = Path(test_base_folder, "sample.lean").read_text()
        result = find_declaration_at_line(content, 6)
        assert result == "add"

    def test_line_at_declaration_keyword(self, test_base_folder):
        """Test finding declaration when line is at the keyword."""
        content = Path(test_base_folder, "sample.lean").read_text()
        result = find_declaration_at_line(content, 5)
        assert result == "add"

    def test_line_in_theorem_body(self, test_base_folder):
        """Test finding theorem when line is in proof body."""
        content = Path(test_base_folder, "sample.lean").read_text()
        result = find_declaration_at_line(content, 19)
        assert result == "add_comm"

    def test_line_in_lemma(self, test_base_folder):
        """Test finding lemma declaration."""
        content = Path(test_base_folder, "sample.lean").read_text()
        result = find_declaration_at_line(content, 23)
        assert result == "mul_zero"

    def test_line_out_of_bounds(self):
        """Test line number beyond file length."""
        content = "def foo := 1"
        result = find_declaration_at_line(content, 100)
        assert result is None

    def test_line_zero(self):
        """Test line number 0 (invalid)."""
        content = "def foo := 1"
        result = find_declaration_at_line(content, 0)
        assert result is None

    def test_line_negative(self):
        """Test negative line number."""
        content = "def foo := 1"
        result = find_declaration_at_line(content, -1)
        assert result is None

    def test_single_line_def(self):
        """Test finding a single-line definition."""
        content = "def foo := 1"
        result = find_declaration_at_line(content, 1)
        assert result == "foo"

    def test_multiple_declarations(self):
        """Test finding declarations in multi-declaration file."""
        content = """def first := 1

def second := 2

theorem third : True := trivial"""
        assert find_declaration_at_line(content, 1) == "first"
        assert find_declaration_at_line(content, 3) == "second"
        assert find_declaration_at_line(content, 5) == "third"

    def test_namespace_declaration(self, test_base_folder):
        """Test that namespace is found at its line."""
        content = Path(test_base_folder, "sample.lean").read_text()
        result = find_declaration_at_line(content, 1)
        assert result == "TestNamespace"

    def test_last_line_before_end(self):
        """Test that the last line before 'end' is correctly attributed."""
        content = """theorem foo : True := by
  sorry
end Namespace"""
        # Line 2 (sorry) should be inside foo, not excluded
        assert find_declaration_at_line(content, 1) == "foo"
        assert find_declaration_at_line(content, 2) == "foo"
        assert find_declaration_at_line(content, 3) == "Namespace"


class TestEditFunctionPreservesComments:
    """Tests that edit_function preserves comments above function definitions."""

    def test_line_comments_above_theorem_preserved(self, tmp_path):
        """Line comments (--) above a theorem are not removed when replacing it."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text(
            "-- This comment explains the theorem\n"
            "-- It should not disappear\n"
            "theorem my_theorem : True := by\n"
            "  sorry\n"
        )
        location = Location(module_path="Test", name="my_theorem", is_external=False)
        new_code = "theorem my_theorem : True := by\n  trivial"

        result = edit_function(str(tmp_path), location, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "-- This comment explains the theorem" in content
        assert "-- It should not disappear" in content
        assert "trivial" in content
        assert "sorry" not in content

    def test_line_comments_between_functions_preserved(self, tmp_path):
        """Comments between two functions survive when the second is replaced."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text(
            "def helper := 42\n"
            "\n"
            "-- Important context for the next theorem\n"
            "-- Do not remove this\n"
            "theorem target : True := by\n"
            "  sorry\n"
        )
        location = Location(module_path="Test", name="target", is_external=False)
        new_code = "theorem target : True := by\n  trivial"

        result = edit_function(str(tmp_path), location, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "def helper := 42" in content
        assert "-- Important context for the next theorem" in content
        assert "-- Do not remove this" in content
        assert "trivial" in content
        assert "sorry" not in content

    def test_block_comments_above_theorem_preserved(self, tmp_path):
        """Block comments (/- ... -/) above a theorem are preserved."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text(
            "/- This is a block comment\n"
            "   explaining the theorem below -/\n"
            "theorem my_theorem : True := by\n"
            "  sorry\n"
        )
        location = Location(module_path="Test", name="my_theorem", is_external=False)
        new_code = "theorem my_theorem : True := by\n  trivial"

        result = edit_function(str(tmp_path), location, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "This is a block comment" in content
        assert "explaining the theorem below" in content
        assert "trivial" in content
        assert "sorry" not in content

    def test_doc_comment_preserved_when_new_code_omits_it(self, tmp_path):
        """Doc comment is preserved when the replacement code has no doc comment."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text(
            "/-- Important documentation about the theorem. -/\n"
            "theorem my_theorem : True := by\n"
            "  sorry\n"
        )
        location = Location(module_path="Test", name="my_theorem", is_external=False)
        # LLM proposal typically omits the doc comment
        new_code = "theorem my_theorem : True := by\n  trivial"

        result = edit_function(str(tmp_path), location, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "Important documentation about the theorem" in content
        assert "trivial" in content
        assert "sorry" not in content

    def test_doc_comment_replaced_when_new_code_has_one(self, tmp_path):
        """Doc comment is replaced if the new code provides a new doc comment."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text("/-- Old doc comment. -/\ntheorem my_theorem : True := by\n  sorry\n")
        location = Location(module_path="Test", name="my_theorem", is_external=False)
        new_code = "/-- New doc comment. -/\ntheorem my_theorem : True := by\n  trivial"

        result = edit_function(str(tmp_path), location, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "New doc comment" in content
        assert "Old doc comment" not in content
        assert "trivial" in content

    def test_line_comments_above_with_doc_comment_preserved(self, tmp_path):
        """Line comments above a doc comment + theorem are preserved."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text(
            "-- Section: basic theorems\n"
            "-- These are foundational results\n"
            "/-- A trivial theorem. -/\n"
            "theorem my_theorem : True := by\n"
            "  sorry\n"
        )
        location = Location(module_path="Test", name="my_theorem", is_external=False)
        # New code omits the doc comment — both line comments and doc comment should survive
        new_code = "theorem my_theorem : True := by\n  trivial"

        result = edit_function(str(tmp_path), location, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "-- Section: basic theorems" in content
        assert "-- These are foundational results" in content
        assert "/-- A trivial theorem. -/" in content
        assert "trivial" in content
        assert "sorry" not in content
