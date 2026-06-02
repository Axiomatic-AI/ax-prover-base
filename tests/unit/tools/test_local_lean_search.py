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
