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
