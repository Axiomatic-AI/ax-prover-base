"""Unit tests for the local Lean library search tool."""

from pathlib import Path

import pytest
from langchain_core.tools import StructuredTool

from ax_prover.models.declaration import DeclarationType
from ax_prover.tools import create_search_lean_local_tool
from ax_prover.tools.local_lean_search import (
    LOCAL_LEAN_SEARCH_TOOL_TYPE,
    SEARCHABLE_TYPES,
    LocalLeanSearcher,
    SearchLeanLocalConfig,
    _declaration_line,
    _iter_lean_files,
    _matching_declaration_names,
    _walk_down_for_roots,
    _walk_up_for_root,
)
from ax_prover.tools.registry import TOOL_REGISTRY


class TestModuleConstants:
    def test_tool_type_value(self):
        assert LOCAL_LEAN_SEARCH_TOOL_TYPE == "search_lean_local"

    def test_config_defaults(self):
        config = SearchLeanLocalConfig()
        assert config.max_results == 6
        assert config.max_chars == 4000

    def test_searchable_types_include_defs_exclude_structural(self):
        assert DeclarationType.Definition in SEARCHABLE_TYPES
        assert DeclarationType.Theorem in SEARCHABLE_TYPES
        assert DeclarationType.Structure in SEARCHABLE_TYPES
        # Structural keywords must NOT be treated as searchable declarations.
        assert DeclarationType.Namespace not in SEARCHABLE_TYPES
        assert DeclarationType.Import not in SEARCHABLE_TYPES
        assert DeclarationType.Open not in SEARCHABLE_TYPES
        assert DeclarationType.End not in SEARCHABLE_TYPES


def _make_lake_project(directory: Path) -> Path:
    """Create a minimal lake project (lakefile.toml) at `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lakefile.toml").write_text('name = "demo"\n')
    return directory


class TestRootResolution:
    def test_walk_up_when_base_is_root(self, tmp_path):
        root = _make_lake_project(tmp_path / "proj")
        assert _walk_up_for_root(root) == root

    def test_walk_up_from_subdirectory(self, tmp_path):
        root = _make_lake_project(tmp_path / "proj")
        sub = root / "Lib" / "Nested"
        sub.mkdir(parents=True)
        assert _walk_up_for_root(sub) == root

    def test_walk_up_returns_none_when_no_marker(self, tmp_path):
        plain = tmp_path / "nolake"
        plain.mkdir()
        assert _walk_up_for_root(plain) is None

    def test_walk_down_finds_single_project(self, tmp_path):
        root = _make_lake_project(tmp_path / "outer" / "challenges")
        assert _walk_down_for_roots(tmp_path / "outer") == [root]

    def test_walk_down_finds_multiple_projects(self, tmp_path):
        a = _make_lake_project(tmp_path / "outer" / "a")
        b = _make_lake_project(tmp_path / "outer" / "b")
        assert sorted(_walk_down_for_roots(tmp_path / "outer")) == sorted([a, b])

    def test_walk_down_skips_dot_lake(self, tmp_path):
        outer = tmp_path / "outer"
        # A lakefile buried inside .lake must be ignored.
        buried = outer / ".lake" / "packages" / "mathlib"
        _make_lake_project(buried)
        outer.mkdir(parents=True, exist_ok=True)
        assert _walk_down_for_roots(outer) == []

    @pytest.mark.parametrize("marker", ["lakefile.lean", "lake-manifest.json"])
    def test_alternate_root_markers_recognized(self, tmp_path, marker):
        root = tmp_path / "proj"
        root.mkdir()
        (root / marker).write_text("")
        assert _walk_up_for_root(root) == root
        assert _walk_down_for_roots(tmp_path) == [root]


SAMPLE_LEAN = """import Mathlib

namespace Treaps

/-- A treap node. -/
structure Treap where
  key : Nat
  priority : Nat

def Treap.insert (t : Treap) (k : Nat) : Treap :=
  t

theorem treap_insert_size (t : Treap) : True := by
  trivial

end Treaps
"""


class TestScanHelpers:
    def test_iter_lean_files_excludes_dot_lake(self, tmp_path):
        (tmp_path / "A.lean").write_text("def a := 1\n")
        sub = tmp_path / "Lib"
        sub.mkdir()
        (sub / "B.lean").write_text("def b := 1\n")
        buried = tmp_path / ".lake" / "packages" / "mathlib"
        buried.mkdir(parents=True)
        (buried / "M.lean").write_text("def m := 1\n")

        found = {p.name for p in _iter_lean_files(tmp_path)}
        assert found == {"A.lean", "B.lean"}

    def test_matching_names_case_insensitive_substring(self):
        names = _matching_declaration_names(SAMPLE_LEAN, "treap")
        assert names == ["Treap", "Treap.insert", "treap_insert_size"]

    def test_matching_names_excludes_structural_keywords(self):
        # "Treaps" (the namespace) must not be returned even though it matches.
        names = _matching_declaration_names(SAMPLE_LEAN, "Treap")
        assert "Treaps" not in names

    def test_matching_names_no_match_returns_empty(self):
        assert _matching_declaration_names(SAMPLE_LEAN, "nonexistent") == []

    def test_matching_names_multiword_query_requires_all_tokens(self):
        # "Treap insert" should match names containing BOTH "treap" and "insert",
        # not require the literal two-word substring (which never matches a name).
        names = _matching_declaration_names(SAMPLE_LEAN, "Treap insert")
        assert names == ["Treap.insert", "treap_insert_size"]

    def test_matching_names_multiword_ignores_extra_whitespace(self):
        assert _matching_declaration_names(SAMPLE_LEAN, "  Treap   insert ") == [
            "Treap.insert",
            "treap_insert_size",
        ]

    def test_matching_names_multiword_excludes_partial_token_match(self):
        # "Treap" matches all three, but "Treap missing" matches none of them
        # because no name contains "missing".
        assert _matching_declaration_names(SAMPLE_LEAN, "Treap missing") == []

    def test_declaration_line_points_at_keyword(self):
        # `structure Treap` is on line 6 (1-based) in SAMPLE_LEAN.
        assert _declaration_line(SAMPLE_LEAN, "Treap") == 6

    def test_declaration_line_distinguishes_dotted_name(self):
        # `def Treap.insert` is on line 10.
        assert _declaration_line(SAMPLE_LEAN, "Treap.insert") == 10

    def test_declaration_line_prefix_name_not_confused(self):
        # `def Treap.insert` (line 1) must not be matched when locating `Treap` (line 4).
        code = (
            "def Treap.insert (t : Treap) : Treap :=\n  t\n\nstructure Treap where\n  key : Nat\n"
        )
        assert _declaration_line(code, "Treap") == 4
        assert _declaration_line(code, "Treap.insert") == 1


@pytest.fixture
def treap_project(tmp_path):
    """A lake project with a Treap definition file and a buried Mathlib file."""
    root = tmp_path / "challenges"
    (root / "Challenges" / "Treap").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "challenges"\n')
    (root / "Challenges" / "Treap" / "Def_Treap.lean").write_text(SAMPLE_LEAN)
    buried = root / ".lake" / "packages" / "mathlib" / "Mathlib"
    buried.mkdir(parents=True)
    (buried / "Shadow.lean").write_text("def OnlyInMathlib := 1\n")
    return root


class TestLocalLeanSearcher:
    def test_search_returns_full_declaration_block(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        result = searcher.search("Treap")
        assert "Found 3 declaration(s)" in result
        assert "structure Treap where" in result
        assert "key : Nat" in result
        assert "Challenges/Treap/Def_Treap.lean:" in result

    def test_search_is_case_insensitive(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert "structure Treap where" in searcher.search("treap")

    def test_search_excludes_dot_lake(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert searcher.search("OnlyInMathlib") == 'No declarations matching "OnlyInMathlib" found.'

    def test_search_no_match_message(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert searcher.search("Nonexistent") == 'No declarations matching "Nonexistent" found.'

    def test_search_empty_query_message(self, treap_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        assert searcher.search("   ") == "Please provide a non-empty keyword to search for."

    def test_search_resolves_root_via_walk_down(self, tmp_path, treap_project):
        # `treap_project` created `tmp_path/"challenges"`; base_folder is the parent `tmp_path`.
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(tmp_path))
        assert "structure Treap where" in searcher.search("Treap")

    def test_search_no_lakefile_message(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(plain))
        result = searcher.search("Treap")
        assert "no lakefile found" in result.lower()

    def test_search_multiple_projects_message(self, tmp_path):
        for name in ("a", "b"):
            proj = tmp_path / "outer" / name
            proj.mkdir(parents=True)
            (proj / "lakefile.toml").write_text('name = "x"\n')
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(tmp_path / "outer"))
        result = searcher.search("Treap")
        assert "multiple Lean projects" in result

    def test_caps_overflow_lists_names_only(self, treap_project):
        searcher = LocalLeanSearcher(
            SearchLeanLocalConfig(max_results=1), base_folder=str(treap_project)
        )
        result = searcher.search("Treap")
        assert "Found 3 declaration(s)" in result
        shown_section, _, overflow_section = result.partition("Additional matches")
        assert overflow_section  # overflow present
        # The single shown block is the structure (first match).
        assert "structure Treap where" in shown_section
        # The two un-shown matches are named in the overflow, not rendered as blocks.
        assert "Treap.insert" in overflow_section
        assert "treap_insert_size" in overflow_section

    def test_caps_overflow_via_max_chars(self, treap_project):
        # A tiny char budget forces overflow even though max_results is high.
        searcher = LocalLeanSearcher(
            SearchLeanLocalConfig(max_results=10, max_chars=60), base_folder=str(treap_project)
        )
        result = searcher.search("Treap")
        assert "Found 3 declaration(s)" in result
        assert "Additional matches" in result

    def test_search_logs_match_count_at_info(self, treap_project, caplog):
        import logging

        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        with caplog.at_level(logging.INFO):
            searcher.search("Treap")
        assert any(
            "LocalLeanSearch: Found 3 declarations for 'Treap'" in r.message for r in caplog.records
        )

    def test_search_logs_no_results_at_info(self, treap_project, caplog):
        import logging

        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(treap_project))
        with caplog.at_level(logging.INFO):
            searcher.search("Nonexistent")
        assert any(
            "LocalLeanSearch: No results for 'Nonexistent'" in r.message for r in caplog.records
        )


class TestDeduplication:
    """Identical declarations copied into multiple files collapse to one entry."""

    DECL = "def shared (n : Nat) : Nat :=\n  n + 1\n"

    def _project_with_two_copies(self, tmp_path):
        root = tmp_path / "challenges"
        (root / "Challenges" / "Mod").mkdir(parents=True)
        (root / "Challenges_Remainder" / "Mod").mkdir(parents=True)
        (root / "lakefile.toml").write_text('name = "challenges"\n')
        (root / "Challenges" / "Mod" / "A.lean").write_text(self.DECL)
        (root / "Challenges_Remainder" / "Mod" / "A.lean").write_text(self.DECL)
        return root

    def test_identical_declaration_deduplicated(self, tmp_path):
        root = self._project_with_two_copies(tmp_path)
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root))
        result = searcher.search("shared")
        # One unique declaration, both paths recorded.
        assert "Found 1 declaration(s)" in result
        assert result.count("def shared") == 1
        assert "(also:" in result
        assert "Challenges/Mod/A.lean:" in result
        assert "Challenges_Remainder/Mod/A.lean:" in result

    def test_same_name_different_body_not_merged(self, tmp_path):
        root = tmp_path / "challenges"
        (root / "Challenges").mkdir(parents=True)
        (root / "lakefile.toml").write_text('name = "challenges"\n')
        (root / "Challenges" / "A.lean").write_text("def shared (n : Nat) : Nat :=\n  n + 1\n")
        (root / "Challenges" / "B.lean").write_text("def shared (n : Nat) : Nat :=\n  n + 2\n")
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root))
        result = searcher.search("shared")
        # Different bodies -> genuinely different declarations, both shown.
        assert "Found 2 declaration(s)" in result
        assert "(also:" not in result


class TestRegistration:
    def test_tool_is_registered(self):
        # Importing ax_prover.tools triggers @register_tool.
        assert LOCAL_LEAN_SEARCH_TOOL_TYPE in TOOL_REGISTRY
        assert TOOL_REGISTRY[LOCAL_LEAN_SEARCH_TOOL_TYPE].config_class is SearchLeanLocalConfig

    def test_factory_builds_structured_tool(self):
        tool = create_search_lean_local_tool(SearchLeanLocalConfig(), base_folder=".")
        assert isinstance(tool, StructuredTool)
        assert tool.name == "search_lean_local_tool"

    def test_factory_tool_is_callable_with_query(self, tmp_path):
        proj = tmp_path / "challenges"
        proj.mkdir()
        (proj / "lakefile.toml").write_text('name = "x"\n')
        (proj / "Defs.lean").write_text("def myThing := 1\n")
        tool = create_search_lean_local_tool(SearchLeanLocalConfig(), base_folder=str(proj))
        result = tool.func("myThing")
        assert "def myThing" in result
