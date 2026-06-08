"""Unit tests for the local Lean library search tool."""

from pathlib import Path

import pytest
from langchain_core.tools import StructuredTool

from ax_prover.models.declaration import DeclarationType
from ax_prover.tools import create_search_lean_local_tool
from ax_prover.tools.local_lean_search import (
    LOCAL_LEAN_SEARCH_TOOL_TYPE,
    MAX_CACHED_DEFINITIONS,
    SEARCHABLE_TYPES,
    LocalLeanSearcher,
    SearchLeanLocalConfig,
    _body_matching_declarations,
    _collect_fuzzy_matches,
    _declaration_line,
    _fuzzy_matching_declarations,
    _fuzzy_score,
    _identifier_match,
    _iter_lean_files,
    _matching_declaration_names,
    _normalize_tokens,
    _search_root,
    _walk_down_for_roots,
    _walk_up_for_root,
    accumulate_used_definitions,
    format_cached_definition_entry,
    identifier_in_code,
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
        # Declarations live inside `namespace Treaps`, so results are qualified.
        names = _matching_declaration_names(SAMPLE_LEAN, "treap")
        assert names == [
            ("Treap", "Treaps.Treap", 0),
            ("Treap.insert", "Treaps.Treap.insert", 0),
            ("treap_insert_size", "Treaps.treap_insert_size", 0),
        ]

    def test_matching_names_excludes_structural_keywords(self):
        # "Treaps" (the namespace) must not be returned as a declaration.
        names = _matching_declaration_names(SAMPLE_LEAN, "Treap")
        assert "Treaps" not in [simple for simple, *_ in names]

    def test_matching_names_no_match_returns_empty(self):
        assert _matching_declaration_names(SAMPLE_LEAN, "nonexistent") == []

    def test_matching_names_multiword_query_requires_all_tokens(self):
        # "Treap insert" should match names containing BOTH "treap" and "insert",
        # not require the literal two-word substring (which never matches a name).
        names = _matching_declaration_names(SAMPLE_LEAN, "Treap insert")
        assert names == [
            ("Treap.insert", "Treaps.Treap.insert", 0),
            ("treap_insert_size", "Treaps.treap_insert_size", 0),
        ]

    def test_matching_names_multiword_ignores_extra_whitespace(self):
        assert _matching_declaration_names(SAMPLE_LEAN, "  Treap   insert ") == [
            ("Treap.insert", "Treaps.Treap.insert", 0),
            ("treap_insert_size", "Treaps.treap_insert_size", 0),
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

    def test_single_oversized_match_is_truncated_not_dropped(self, tmp_path):
        # A single match whose block exceeds max_chars must still surface some of the
        # body plus a truncation marker, NOT an empty body with only a "not shown" note.
        root = tmp_path / "challenges"
        (root / "Challenges").mkdir(parents=True)
        (root / "lakefile.toml").write_text('name = "challenges"\n')
        body_lines = "\n".join(f"  step_{i} := {i}" for i in range(50))
        (root / "Challenges" / "Big.lean").write_text(f"def bigDecl : Nat :=\n{body_lines}\n  0\n")
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(max_chars=200), base_folder=str(root))
        result = searcher.search("bigDecl")
        assert "Found 1 declaration(s)" in result
        # The whole oversized block must NOT be shown (it was truncated).
        assert "step_49" not in result
        # Part of the actual declaration body must appear.
        assert "def bigDecl" in result
        # A truncation marker must be present.
        assert "truncated" in result
        # It must NOT claim the only match is "not shown".
        assert "Additional matches (not shown" not in result

    def test_normal_small_result_not_truncated(self, tmp_path):
        # A normal small result that fits the budget is rendered fully, no marker.
        root = tmp_path / "challenges"
        (root / "Challenges").mkdir(parents=True)
        (root / "lakefile.toml").write_text('name = "challenges"\n')
        (root / "Challenges" / "Small.lean").write_text("def smallDecl : Nat :=\n  1\n")
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root))
        result = searcher.search("smallDecl")
        assert "Found 1 declaration(s)" in result
        assert "def smallDecl : Nat :=" in result
        assert "truncated" not in result
        assert "Additional matches" not in result

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


# A project exercising namespace qualification and modifier/attribute handling.
NAMESPACED_LEAN = """import Mathlib

namespace BinaryTree

def insert (t : BinaryTree) (k : Nat) : BinaryTree :=
  t

noncomputable def dijkstra_rec (g : G) (s : V) : T :=
  foo

end BinaryTree

@[simp] private def topLevelTagged : Nat := 0

section Helpers
def in_section : Nat := 1
end Helpers
"""


@pytest.fixture
def namespaced_project(tmp_path):
    root = tmp_path / "challenges"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "challenges"\n')
    (root / "Challenges" / "Defs.lean").write_text(NAMESPACED_LEAN)
    return root


class TestNamespaceQualification:
    def test_qualifies_namespaced_declaration(self):
        # `def insert` inside `namespace BinaryTree` is reachable via "BinaryTree insert".
        names = _matching_declaration_names(NAMESPACED_LEAN, "BinaryTree insert")
        assert ("insert", "BinaryTree.insert", 0) in names

    def test_simple_name_preserved_for_extraction(self):
        # The simple (source) name must remain available for block extraction.
        names = _matching_declaration_names(NAMESPACED_LEAN, "BinaryTree insert")
        simple, qualified, _ = next(t for t in names if t[1] == "BinaryTree.insert")
        assert simple == "insert"

    def test_section_does_not_contribute_to_name(self):
        # `section Helpers` must NOT prefix `in_section`.
        names = _matching_declaration_names(NAMESPACED_LEAN, "in_section")
        assert names == [("in_section", "in_section", 0)]

    def test_no_double_qualification(self):
        code = "namespace A.B\n\ndef A.B.f : Nat := 0\n\nend A.B\n"
        names = _matching_declaration_names(code, "f")
        assert names == [("A.B.f", "A.B.f", 0)]

    def test_nested_dotted_namespace(self):
        code = "namespace A.B\n\ndef f : Nat := 0\n\nend A.B\n"
        names = _matching_declaration_names(code, "f")
        assert names == [("f", "A.B.f", 0)]

    def test_search_displays_qualified_name_for_namespaced_def(self, namespaced_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(namespaced_project))
        result = searcher.search("BinaryTree insert")
        assert 'matching "BinaryTree insert"' in result
        assert "def insert (t : BinaryTree)" in result  # block is the source text


class TestModifiedDeclarationSearch:
    def test_search_finds_noncomputable_def(self, namespaced_project):
        # Regression: `noncomputable def` declarations were invisible to search.
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(namespaced_project))
        result = searcher.search("dijkstra_rec")
        assert "noncomputable def dijkstra_rec" in result

    def test_search_finds_attributed_private_def(self, namespaced_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(namespaced_project))
        result = searcher.search("topLevelTagged")
        assert "def topLevelTagged" in result


# Two declarations sharing a simple name in different namespaces, same file.
DUP_SIMPLE_NAME_LEAN = """namespace A

def insert (t : T) : T := AAA

end A

namespace B

def insert (t : T) : T := BBB

end B
"""


@pytest.fixture
def dup_name_project(tmp_path):
    root = tmp_path / "challenges"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "challenges"\n')
    (root / "Challenges" / "Dup.lean").write_text(DUP_SIMPLE_NAME_LEAN)
    return root


class TestAmbiguousSimpleName:
    def test_matching_names_reports_distinct_occurrences(self):
        names = _matching_declaration_names(DUP_SIMPLE_NAME_LEAN, "insert")
        # Each match carries (simple, qualified, occurrence) so the right block can be found.
        assert names == [("insert", "A.insert", 0), ("insert", "B.insert", 1)]

    def test_search_shows_correct_block_and_line_per_namespace(self, dup_name_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(dup_name_project))
        result = searcher.search("insert")
        # Both bodies must appear, each at its own line — not the first block twice.
        assert ":= AAA" in result
        assert ":= BBB" in result
        assert "Challenges/Dup.lean:3" in result  # A.insert
        assert "Challenges/Dup.lean:9" in result  # B.insert


# A real `def foo` preceded by a commented-out `def foo` (line-start, inside a block
# comment). The raw-content matcher would otherwise pick the commented one first.
COMMENTED_DECL_LEAN = """import Mathlib

/-
def foo := 1
-/
def foo := 2
"""


@pytest.fixture
def commented_decl_project(tmp_path):
    root = tmp_path / "challenges"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "challenges"\n')
    (root / "Challenges" / "Commented.lean").write_text(COMMENTED_DECL_LEAN)
    return root


class TestCommentedDeclaration:
    def test_declaration_line_skips_commented_decl(self):
        # The real `def foo := 2` is on line 6; the commented one (line 4) must be ignored.
        assert _declaration_line(COMMENTED_DECL_LEAN, "foo", 0) == 6

    def test_search_returns_real_block_not_commented(self, commented_decl_project):
        searcher = LocalLeanSearcher(
            SearchLeanLocalConfig(), base_folder=str(commented_decl_project)
        )
        result = searcher.search("foo")
        assert "def foo := 2" in result
        assert ":= 1" not in result
        assert "Challenges/Commented.lean:6" in result


class TestIdentifierMatch:
    def test_matches_standalone_identifier(self):
        assert _identifier_match("query_aux", "  x := query_aux n 0")

    def test_not_matched_inside_longer_identifier(self):
        # trailing 'N' continues the identifier, so it is not a whole-identifier match
        assert not _identifier_match("query_aux", "def query_auxN := 1")

    def test_dot_is_an_identifier_boundary_char(self):
        # '.' is part of a Lean identifier, so "insert" does NOT match inside "Treap.insert"
        assert not _identifier_match("insert", "y := Treap.insert t")

    def test_case_insensitive(self):
        assert _identifier_match("foo", "exact FOO")


WHERE_BLOCK_LEAN = """import Mathlib

def query (n : Nat) : Nat :=
  query_aux n 0   where query_aux (j acc : Nat) : Nat :=
    if j = 0 then acc else query_aux (j - 1) (acc + j)
"""


class TestBodyMatchingDeclarations:
    def test_name_search_misses_where_helper(self):
        # `query_aux` is not a top-level declaration, so name search returns nothing.
        assert _matching_declaration_names(WHERE_BLOCK_LEAN, "query_aux") == []

    def test_body_search_returns_enclosing_declaration(self):
        assert _body_matching_declarations(WHERE_BLOCK_LEAN, "query_aux") == [("query", "query", 0)]

    def test_body_search_requires_all_tokens(self):
        assert _body_matching_declarations(WHERE_BLOCK_LEAN, "query_aux missing") == []

    def test_body_search_respects_identifier_boundary(self):
        # 'quer' is a prefix, not a whole identifier in the body -> no match.
        assert _body_matching_declarations(WHERE_BLOCK_LEAN, "quer") == []


def _write(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_search_root_returns_text_and_decls_on_name_match(tmp_path):
    _write(tmp_path, "Def.lean", "def extract_min : Nat := 0\n")
    text, decls = _search_root(tmp_path, "extract_min", SearchLeanLocalConfig())
    assert "extract_min" in text
    assert len(decls) == 1
    qualified_name, block, locations = decls[0]
    assert qualified_name == "extract_min"
    assert "def extract_min" in block
    assert locations and locations[0][1] == 1  # (path, line)


def test_search_root_returns_empty_decls_on_no_match(tmp_path):
    _write(tmp_path, "Def.lean", "def foo : Nat := 0\n")
    text, decls = _search_root(tmp_path, "nonexistent_name", SearchLeanLocalConfig())
    assert "No declarations matching" in text
    assert decls == []


@pytest.fixture
def where_project(tmp_path):
    root = tmp_path / "proj"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "p"\n')
    (root / "Challenges" / "Q.lean").write_text(WHERE_BLOCK_LEAN)
    return root


class TestBodySearchEndToEnd:
    def test_search_falls_back_to_body_for_where_helper(self, where_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(where_project))
        out = searcher.search("query_aux")
        assert "def query" in out
        assert "matched in body" in out
        assert "Challenges/Q.lean:" in out

    def test_name_match_does_not_use_body_fallback(self, where_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(where_project))
        out = searcher.search("query")
        assert "def query" in out
        assert "matched in body" not in out

    def test_no_match_message_unchanged(self, where_project):
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(where_project))
        assert searcher.search("zzz_nope") == 'No declarations matching "zzz_nope" found.'

    def test_body_fallback_reads_each_file_once(self, where_project, monkeypatch):
        # A name-miss that succeeds only via the body fallback must NOT re-walk and
        # re-read the tree: each .lean file is read exactly once per search.
        import pathlib

        read_counts: dict[str, int] = {}
        original = pathlib.Path.read_text

        def counting_read_text(self, *args, **kwargs):
            if str(self).endswith(".lean"):
                read_counts[str(self)] = read_counts.get(str(self), 0) + 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", counting_read_text)
        searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(where_project))
        out = searcher.search("query_aux")  # name miss -> body fallback hit
        assert "matched in body" in out
        assert read_counts  # at least one .lean file was read
        assert all(count == 1 for count in read_counts.values()), read_counts


def test_identifier_in_code_matches_qualified_and_simple():
    assert identifier_in_code("BinaryHeap.extract_min", "  exact extract_min h")
    assert identifier_in_code("BinaryHeap.extract_min", "  exact BinaryHeap.extract_min h")
    assert not identifier_in_code(
        "extract_min", "  exact extract_minimum h"
    )  # not whole-identifier
    assert not identifier_in_code("heapify", "  simp")


def test_format_cached_definition_entry_has_location_header_and_block():
    entry = format_cached_definition_entry("A.foo", "def foo := 1", Path("Def.lean"), 3)
    assert entry == "-- A.foo — Def.lean:3\ndef foo := 1"


def _pool():
    return {
        "BinaryHeap.extract_min": ("def extract_min : Nat := 0", Path("Def.lean"), 5),
        "BinaryHeap.heapify": ("def heapify : Nat := 1", Path("Def.lean"), 9),
    }


def test_accumulate_adds_only_used_definitions():
    code = "theorem t : True := by have := extract_min; trivial"
    result = accumulate_used_definitions({}, _pool(), code, target_name="t")
    assert "BinaryHeap.extract_min" in result
    assert "BinaryHeap.heapify" not in result  # returned but not used in code


def test_accumulate_excludes_target_by_simple_name():
    pool = {"Foo.extract_min": ("def extract_min := 0", Path("Def.lean"), 1)}
    code = "theorem extract_min : True := by exact extract_min"
    result = accumulate_used_definitions({}, pool, code, target_name="extract_min")
    assert result == {}  # the only candidate shares the target's simple name


def test_accumulate_is_monotonic_and_dedups():
    prior = {"BinaryHeap.heapify": "-- cached earlier"}
    code = "theorem t : True := by exact extract_min"  # heapify NOT used now
    result = accumulate_used_definitions(prior, _pool(), code, target_name="t")
    assert result["BinaryHeap.heapify"] == "-- cached earlier"  # preserved, not wiped
    assert "BinaryHeap.extract_min" in result  # newly used, added
    again = accumulate_used_definitions(result, _pool(), code, target_name="t")
    assert again == result


def test_accumulate_respects_count_cap():
    pool = {
        f"N.def_{i}": (f"def def_{i} := {i}", Path("Def.lean"), i + 1)
        for i in range(MAX_CACHED_DEFINITIONS + 5)
    }
    code = " ".join(f"def_{i}" for i in range(MAX_CACHED_DEFINITIONS + 5))
    result = accumulate_used_definitions({}, pool, code, target_name="t")
    assert len(result) == MAX_CACHED_DEFINITIONS


def test_accumulate_respects_char_cap():
    big_block = "x" * 9000  # two of these exceed the 12000-char cap
    pool = {
        "N.a": (big_block, Path("Def.lean"), 1),
        "N.b": (big_block, Path("Def.lean"), 2),
    }
    code = "a b"  # both 'a' and 'b' referenced as whole identifiers
    result = accumulate_used_definitions({}, pool, code, target_name="t")
    assert len(result) == 1  # second entry would blow the char budget


def test_accumulate_none_target_and_empty_pool_are_safe():
    # None target_name must not crash; empty returned_declarations returns the cache unchanged.
    assert accumulate_used_definitions({}, {}, "some code", None) == {}
    prior = {"A.foo": "-- A.foo — Def.lean:1\ndef foo := 1"}
    assert accumulate_used_definitions(prior, {}, "foo", None) == prior


def test_accumulate_none_target_still_adds_used_definition():
    returned = {"BinaryHeap.extract_min": ("def extract_min : Nat := 0", Path("Def.lean"), 5)}
    code = "theorem t : True := by exact extract_min"
    result = accumulate_used_definitions(
        cached={}, returned_declarations=returned, code=code, target_name=None
    )
    assert "BinaryHeap.extract_min" in result


def test_searcher_accumulates_returned_declarations_across_calls(tmp_path):
    root = _make_lake_project(tmp_path)
    (root / "Def.lean").write_text(
        "def extract_min : Nat := 0\ndef heapify : Nat := 1\n", encoding="utf-8"
    )
    searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root))

    searcher.search("extract_min")
    assert "extract_min" in searcher.returned_declarations
    block, path, line = searcher.returned_declarations["extract_min"]
    assert "def extract_min" in block

    searcher.search("heapify")
    # First result is retained; second is added (accumulation, not replacement).
    assert set(searcher.returned_declarations) == {"extract_min", "heapify"}


def test_searcher_first_seen_wins_on_repeat_search(tmp_path):
    root = _make_lake_project(tmp_path)
    (root / "Def.lean").write_text(
        "def extract_min : Nat := 0\ndef heapify : Nat := 1\n", encoding="utf-8"
    )
    searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root))
    searcher.search("extract_min")
    first_value = searcher.returned_declarations["extract_min"]
    # Searching the same name again must not replace the recorded entry.
    searcher.search("extract_min")
    assert searcher.returned_declarations["extract_min"] == first_value


def test_searcher_records_nothing_on_miss(tmp_path):
    root = _make_lake_project(tmp_path)
    (root / "Def.lean").write_text(
        "def extract_min : Nat := 0\ndef heapify : Nat := 1\n", encoding="utf-8"
    )
    searcher = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root))
    searcher.search("totally_absent_name")
    assert searcher.returned_declarations == {}


def test_normalize_tokens_splits_camel_snake_and_qualifier():
    assert _normalize_tokens("BinaryHeap.decreaseKey") == ["decrease", "key"]
    assert _normalize_tokens("extract_min") == ["extract", "min"]


def test_fuzzy_score_high_for_near_miss():
    # Underscore/spelling differences still score high.
    assert _fuzzy_score("extractmin", "BinaryHeap.extract_min") >= 0.8
    assert _fuzzy_score("decrese_min", "decrease_min") >= 0.7  # typo


def test_fuzzy_score_rewards_single_token_match():
    # Query matches one token of a compound name.
    assert _fuzzy_score("priority", "decreasePriority") >= 0.6


def test_fuzzy_score_low_for_unrelated():
    # An unrelated name must score below the suggestion gate, so it is never suggested.
    from ax_prover.tools.local_lean_search import FUZZY_THRESHOLD

    assert _fuzzy_score("heapify", "WeightedGraph") < FUZZY_THRESHOLD


def test_fuzzy_matching_declarations_finds_close_name():
    content = "def extract_min : Nat := 0\ndef heapify : Nat := 1\n"
    matches = _fuzzy_matching_declarations(content, "extractmin")
    names = [qualified for _simple, qualified, _occ, _score in matches]
    assert "extract_min" in names
    assert "heapify" not in names


def test_collect_fuzzy_matches_sorts_by_score_and_caps(tmp_path):
    (tmp_path / "Def.lean").write_text(
        "def decrease_key : Nat := 0\ndef decrease_min : Nat := 1\ndef unrelated : Nat := 2\n",
        encoding="utf-8",
    )
    files = [(Path("Def.lean"), (tmp_path / "Def.lean").read_text(encoding="utf-8"))]
    results = _collect_fuzzy_matches(files, "decrease_mn", SearchLeanLocalConfig(max_results=1))
    assert len(results) == 1  # cap respected
    assert results[0][0] == "decrease_min"  # closest by score ranked first


def test_fuzzy_score_preserves_recall_for_single_token_of_long_name():
    from ax_prover.tools.local_lean_search import FUZZY_THRESHOLD

    # A short query that exactly matches one token of a long compound name must stay suggestible.
    assert _fuzzy_score("fold", "Algebra.leftFoldOverMonoidWithIdentity") >= FUZZY_THRESHOLD


def test_collect_fuzzy_matches_ranks_closer_suffix_first(tmp_path):
    (tmp_path / "Def.lean").write_text(
        "def decrease_key : Nat := 0\ndef decrease_min : Nat := 1\n", encoding="utf-8"
    )
    files = [(Path("Def.lean"), (tmp_path / "Def.lean").read_text(encoding="utf-8"))]
    results = _collect_fuzzy_matches(files, "decrease_mn", SearchLeanLocalConfig(max_results=2))
    names = [name for name, _block, _locs in results]
    assert names == ["decrease_min", "decrease_key"]  # closer suffix ranked first
