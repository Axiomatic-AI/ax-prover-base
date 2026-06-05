"""Unit tests for the CSLib search tool."""

import pytest

from ax_prover.tools import create_search_cslib_tool
from ax_prover.tools.cslib_search import (
    CSLIB_SEARCH_TOOL_TYPE,
    CslibSearcher,
    SearchCslibConfig,
)
from ax_prover.tools.registry import TOOL_REGISTRY, create_tool


def _make_project_with_cslib(tmp_path):
    """A lake project whose .lake/packages/cslib holds one namespaced theorem."""
    root = tmp_path / "proj"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "p"\n')
    (root / "Challenges" / "Def_Local.lean").write_text("def ProjectOnlyThing : Nat := 0\n")
    cslib = root / ".lake" / "packages" / "cslib" / "Cslib"
    cslib.mkdir(parents=True)
    (cslib / "Relation.lean").write_text(
        "namespace WellFounded\n\ntheorem ofTransGen (h : True) : True := h\n\nend WellFounded\n"
    )
    return root


class TestCslibBasic:
    def test_finds_namespaced_theorem_qualified(self, tmp_path):
        root = _make_project_with_cslib(tmp_path)
        searcher = CslibSearcher(SearchCslibConfig(), base_folder=str(root))
        out = searcher.search("ofTransGen")
        assert "WellFounded.ofTransGen" in out
        assert "theorem ofTransGen" in out

    def test_tool_type_constant(self):
        assert CSLIB_SEARCH_TOOL_TYPE == "search_cslib"


class TestCslibRegistration:
    def test_registered_in_registry(self):
        assert CSLIB_SEARCH_TOOL_TYPE in TOOL_REGISTRY

    @pytest.mark.asyncio
    async def test_create_tool_builds_named_tool(self, tmp_path):
        root = _make_project_with_cslib(tmp_path)
        tool = await create_tool({"tool_type": "search_cslib"}, base_folder=str(root))
        assert tool is not None
        assert tool.name == "search_cslib_tool"


from ax_prover.tools.local_lean_search import LocalLeanSearcher, SearchLeanLocalConfig

# CSLib-shaped content: module header, public import, namespace, doc + own-line @[simp],
# noncomputable, private. `measure` here collides (different namespace/body) with the SKI file.
CSLIB_REL = """module

public import Cslib.Init

namespace WellFounded

/-- Transitive-closure well-foundedness. -/
@[simp]
theorem ofTransGen (h : True) : True := h

noncomputable def measure : Nat := 0

private theorem secret : True := trivial

end WellFounded
"""

# Inductive with distinctive constructor names (Iterm is only a constructor, not a decl).
CSLIB_SKI = """module
namespace CombinatoryLogic

inductive SKI where
  | Sterm | Kterm | Iterm
  | app : SKI -> SKI -> SKI

def measure : SKI -> Nat
  | _ => 1

end CombinatoryLogic
"""


def _make_rich_cslib(tmp_path, *, with_mathlib=False):
    root = tmp_path / "proj"
    (root / "Challenges").mkdir(parents=True)
    (root / "lakefile.toml").write_text('name = "p"\n')
    (root / "Challenges" / "Def_Local.lean").write_text("def ProjectOnlyThing : Nat := 0\n")
    cslib = root / ".lake" / "packages" / "cslib" / "Cslib"
    cslib.mkdir(parents=True)
    (cslib / "Relation.lean").write_text(CSLIB_REL)
    (cslib / "SKI.lean").write_text(CSLIB_SKI)
    if with_mathlib:
        mathlib = root / ".lake" / "packages" / "mathlib" / "Mathlib"
        mathlib.mkdir(parents=True)
        (mathlib / "X.lean").write_text("def MathlibOnly : Nat := 1\n")
    return root


class TestCslibIsolation:
    def test_cslib_tool_excludes_project_files(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("ProjectOnlyThing")
        assert out == 'No declarations matching "ProjectOnlyThing" found.'

    def test_cslib_tool_excludes_sibling_mathlib(self, tmp_path):
        root = _make_rich_cslib(tmp_path, with_mathlib=True)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("MathlibOnly")
        assert out == 'No declarations matching "MathlibOnly" found.'

    def test_local_tool_excludes_cslib(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root)).search("ofTransGen")
        assert out == 'No declarations matching "ofTransGen" found.'

    def test_local_tool_finds_project_file(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = LocalLeanSearcher(SearchLeanLocalConfig(), base_folder=str(root)).search(
            "ProjectOnlyThing"
        )
        assert "def ProjectOnlyThing" in out


class TestCslibModuleShapes:
    def test_own_line_attribute_decl_found_and_block_includes_attr(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("ofTransGen")
        assert "WellFounded.ofTransGen" in out
        assert "theorem ofTransGen" in out
        assert "@[simp]" in out

    def test_module_and_public_import_do_not_create_bogus_results(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("Cslib")
        assert out == 'No declarations matching "Cslib" found.'


class TestCslibNamespaceAndDuplicates:
    def test_multiword_qualified_query(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search(
            "WellFounded ofTransGen"
        )
        assert "WellFounded.ofTransGen" in out

    def test_duplicate_simple_name_across_files_resolves_distinctly(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("measure")
        assert "WellFounded.measure" in out
        assert "CombinatoryLogic.measure" in out
        assert ":= 0" in out
        assert "SKI -> Nat" in out


class TestCslibModifiers:
    def test_private_theorem_found(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("secret")
        assert "WellFounded.secret" in out

    def test_noncomputable_def_found(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("measure")
        assert "noncomputable def measure" in out


class TestCslibBodyAndConstructors:
    def test_inductive_constructor_found_via_body(self, tmp_path):
        # `Iterm` is a constructor, not a top-level decl: name search misses, body search hits SKI.
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("Iterm")
        assert "inductive SKI" in out
        assert "matched in body" in out


class TestCslibEdges:
    def test_missing_cslib_dir_message(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "lakefile.toml").write_text('name = "p"\n')
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(root)).search("anything")
        assert "CSLib search unavailable" in out
        assert ".lake/packages/cslib" in out

    def test_resolves_from_subdirectory_walk_up(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        sub = root / "Challenges"
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(sub)).search("ofTransGen")
        assert "WellFounded.ofTransGen" in out

    def test_resolves_from_parent_walk_down(self, tmp_path):
        _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(), base_folder=str(tmp_path)).search("ofTransGen")
        assert "WellFounded.ofTransGen" in out

    def test_caps_overflow_lists_extras(self, tmp_path):
        root = _make_rich_cslib(tmp_path)
        out = CslibSearcher(SearchCslibConfig(max_results=1), base_folder=str(root)).search(
            "measure"
        )
        assert "Additional matches" in out
