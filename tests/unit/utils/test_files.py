"""Tests for file utilities: replace_in_file."""

from ax_prover.utils.files import replace_in_file


class TestReplaceInFile:
    """Tests for replace_in_file."""

    def test_replaces_declaration(self, tmp_path):
        """The matched declaration is replaced by the new text."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text("theorem my_theorem : True := by\n  sorry\n")
        original = "theorem my_theorem : True := by\n  sorry"
        new_code = "theorem my_theorem : True := by\n  trivial"

        result = replace_in_file(lean_file, original, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "trivial" in content
        assert "sorry" not in content

    def test_returns_false_when_file_missing(self, tmp_path):
        """A non-existent file is reported as a failure without raising."""
        missing = tmp_path / "Missing.lean"
        result = replace_in_file(missing, "theorem foo : True := trivial", "anything")
        assert result is False

    def test_returns_false_when_original_not_found(self, tmp_path):
        """When the original text is absent, the file is left untouched."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text("theorem foo : True := by\n  sorry\n")
        result = replace_in_file(
            lean_file, "theorem bar : False := by\n  sorry", "theorem bar := by trivial"
        )
        assert result is False
        assert lean_file.read_text() == "theorem foo : True := by\n  sorry\n"

    def test_matches_despite_leading_and_trailing_whitespace(self, tmp_path):
        """DeclarationInfo.pp can carry a leading space / extra newlines; matching is stripped."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text("theorem my_theorem : True := by\n  sorry\n")
        # Simulate the leading-space artifact seen on synthetic-range pp output.
        original = " theorem my_theorem : True := by\n  sorry\n"
        new_code = "theorem my_theorem : True := by\n  trivial\n"

        result = replace_in_file(lean_file, original, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "trivial" in content
        assert "sorry" not in content

    def test_only_first_occurrence_replaced(self, tmp_path):
        """Only the first match is replaced when the text appears more than once."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text("axiom foo : Nat\naxiom foo : Nat\n")

        result = replace_in_file(lean_file, "axiom foo : Nat", "axiom bar : Nat")

        assert result is True
        content = lean_file.read_text()
        assert content == "axiom bar : Nat\naxiom foo : Nat\n"

    def test_line_comments_above_theorem_preserved(self, tmp_path):
        """Line comments (--) above a theorem are not part of the original, so they survive."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text(
            "-- This comment explains the theorem\n"
            "-- It should not disappear\n"
            "theorem my_theorem : True := by\n"
            "  sorry\n"
        )
        original = "theorem my_theorem : True := by\n  sorry"
        new_code = "theorem my_theorem : True := by\n  trivial"

        result = replace_in_file(lean_file, original, new_code)

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
        original = "theorem target : True := by\n  sorry"
        new_code = "theorem target : True := by\n  trivial"

        result = replace_in_file(lean_file, original, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "def helper := 42" in content
        assert "-- Important context for the next theorem" in content
        assert "-- Do not remove this" in content
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
        # pp includes the doc comment as part of the declaration source.
        original = (
            "/-- Important documentation about the theorem. -/\n"
            "theorem my_theorem : True := by\n"
            "  sorry"
        )
        # LLM proposal typically omits the doc comment.
        new_code = "theorem my_theorem : True := by\n  trivial"

        result = replace_in_file(lean_file, original, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "Important documentation about the theorem" in content
        assert "trivial" in content
        assert "sorry" not in content

    def test_doc_comment_replaced_when_new_code_has_one(self, tmp_path):
        """Doc comment is replaced if the new code provides a new doc comment."""
        lean_file = tmp_path / "Test.lean"
        lean_file.write_text("/-- Old doc comment. -/\ntheorem my_theorem : True := by\n  sorry\n")
        original = "/-- Old doc comment. -/\ntheorem my_theorem : True := by\n  sorry"
        new_code = "/-- New doc comment. -/\ntheorem my_theorem : True := by\n  trivial"

        result = replace_in_file(lean_file, original, new_code)

        assert result is True
        content = lean_file.read_text()
        assert "New doc comment" in content
        assert "Old doc comment" not in content
        assert "trivial" in content
