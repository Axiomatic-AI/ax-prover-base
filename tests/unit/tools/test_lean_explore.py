"""Unit tests for LeanExplore tool.

These target the pure formatting/summarization helpers plus the `lean_explore`
entry point with a mocked Service, so they run without the optional
`lean-explore` package or its FAISS/torch models installed.
"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ax_prover.tools import lean_explore


def _make_result(name, module, source_text=None, docstring=None, informalization=None):
    return SimpleNamespace(
        name=name,
        module=module,
        source_text=source_text,
        docstring=docstring,
        informalization=informalization,
    )


@pytest.fixture
def mock_config():
    return lean_explore.SearchLeanExploreConfig(max_results=6, rerank_top=50)


class TestTruncate:
    def test_short_text_unchanged(self):
        assert lean_explore._truncate("hello", 100) == "hello"

    def test_long_text_truncated_with_ellipsis(self):
        result = lean_explore._truncate("A" * 100, 10)
        assert result.endswith("…")
        assert len(result) <= 11

    def test_none_returns_empty(self):
        assert lean_explore._truncate(None, 10) == ""


class TestFormatResponse:
    def test_with_results(self):
        response = SimpleNamespace(
            results=[
                _make_result(
                    "Real.add_comm",
                    "Mathlib.Data.Real.Basic",
                    source_text="theorem add_comm ...",
                    docstring="Addition is commutative.",
                    informalization="a plus b equals b plus a",
                ),
                _make_result("Nat.succ", "Init.Prelude"),
            ]
        )
        result = lean_explore._format_response("add_comm", response)

        assert "add_comm (2 matches)" in result
        assert "Real.add_comm  [Mathlib.Data.Real.Basic]" in result
        assert "Source:" in result
        assert "Doc: Addition is commutative." in result
        assert "Informal: a plus b equals b plus a" in result
        assert "Nat.succ  [Init.Prelude]" in result

    def test_no_results(self):
        response = SimpleNamespace(results=[])
        assert lean_explore._format_response("q", response) == "No results found for: q"

    def test_optional_fields_omitted(self):
        response = SimpleNamespace(results=[_make_result("T", "M")])
        result = lean_explore._format_response("q", response)

        assert "T  [M]" in result
        assert "Source:" not in result
        assert "Doc:" not in result
        assert "Informal:" not in result


def _make_resources(service):
    return lean_explore.LeanExploreResources(
        service=service, search_semaphore=asyncio.Semaphore(1)
    )


class TestLeanExplore:
    """Tests for the `lean_explore` entry point with a mocked Service."""

    async def test_success(self, mock_config):
        response = SimpleNamespace(results=[_make_result("Real.add_comm", "M", docstring="doc")])
        service = SimpleNamespace(search=AsyncMock(return_value=response))

        result = await lean_explore.lean_explore("add_comm", mock_config, _make_resources(service))

        assert "Real.add_comm" in result
        service.search.assert_awaited_once()

    async def test_error_returns_message(self, mock_config):
        service = SimpleNamespace(search=AsyncMock(side_effect=RuntimeError("search failed")))

        result = await lean_explore.lean_explore("add_comm", mock_config, _make_resources(service))

        assert "LeanExplore error" in result
        assert "search failed" in result


class TestLifespan:
    """Tests for the `_lean_explore_lifespan` context manager."""

    @staticmethod
    def _install_fake_service(monkeypatch, service_factory):
        """Register a fake `lean_explore.search` module so the lazy import resolves."""
        parent = types.ModuleType("lean_explore")
        search_mod = types.ModuleType("lean_explore.search")
        search_mod.Service = service_factory
        parent.search = search_mod
        monkeypatch.setitem(sys.modules, "lean_explore", parent)
        monkeypatch.setitem(sys.modules, "lean_explore.search", search_mod)

    async def test_yields_resources_on_success(self, mock_config, monkeypatch):
        self._install_fake_service(monkeypatch, lambda: SimpleNamespace(search=AsyncMock()))

        async with lean_explore._lean_explore_lifespan(mock_config) as resources:
            assert isinstance(resources, lean_explore.LeanExploreResources)
            assert resources.search_semaphore._value == mock_config.max_concurrent_searches

    async def test_yields_none_when_service_fails(self, mock_config, monkeypatch):
        def boom():
            raise RuntimeError("models not downloaded")

        self._install_fake_service(monkeypatch, boom)

        async with lean_explore._lean_explore_lifespan(mock_config) as resources:
            assert resources is None
