"""Tests for find_declaration_at_line in ax_agent/utils/lean_parsing.py"""

from pathlib import Path

import pytest

from ax_agent.utils.lean_parsing import find_declaration_at_line


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
