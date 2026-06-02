"""Unit tests for the local Lean library search tool."""

from pathlib import Path  # noqa: F401

import pytest  # noqa: F401

from ax_prover.models.declaration import DeclarationType
from ax_prover.tools.local_lean_search import (
    LOCAL_LEAN_SEARCH_TOOL_TYPE,
    SEARCHABLE_TYPES,
    SearchLeanLocalConfig,
)


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


from ax_prover.tools.local_lean_search import _walk_down_for_roots, _walk_up_for_root


def _make_lake_project(directory: Path) -> Path:
    """Create a minimal lake project (lakefile.toml) at `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lakefile.toml").write_text("name = \"demo\"\n")
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
