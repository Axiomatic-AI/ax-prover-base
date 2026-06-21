"""Unit tests for Loogle tool."""

from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from ax_prover.tools import loogle


@pytest.fixture
def mock_config():
    """A real SearchLoogleConfig instance (small, plain dataclass)."""
    return loogle.SearchLoogleConfig(
        server_url="http://test-server:8088", max_results=15, timeout=30
    )


class TestDefaults:
    def test_default_url(self):
        assert loogle.DEFAULT_LOOGLE_URL == "http://localhost:8088"

    def test_default_config_values(self):
        config = loogle.SearchLoogleConfig()
        assert config.server_url == loogle.DEFAULT_LOOGLE_URL
        assert config.max_results == 6
        assert config.timeout == 60


class TestFormatResponse:
    """Tests for `_format_response` (pure formatting)."""

    def test_with_hits(self):
        data = {
            "count": 2,
            "header": "Found stuff",
            "hits": [
                {
                    "name": "Real.add_comm",
                    "module": "Mathlib.Data.Real.Basic",
                    "type": "a + b = b + a",
                    "doc": "Addition is commutative.",
                },
                {"name": "Nat.succ", "module": "Init.Prelude"},
            ],
        }
        result = loogle._format_response("add_comm", data, max_results=15)

        assert "add_comm (2 total, showing 2)" in result
        assert "Found stuff" in result
        assert "Real.add_comm  [Mathlib.Data.Real.Basic]" in result
        assert ": a + b = b + a" in result
        assert "Doc: Addition is commutative." in result
        assert "Nat.succ  [Init.Prelude]" in result

    def test_respects_max_results(self):
        data = {"hits": [{"name": f"thm{i}"} for i in range(10)]}
        result = loogle._format_response("q", data, max_results=3)

        assert "showing 3" in result
        assert "thm0" in result
        assert "thm2" in result
        assert "thm3" not in result

    def test_truncates_long_docstring(self):
        data = {"hits": [{"name": "T", "doc": "A" * 1000}]}
        result = loogle._format_response("q", data, max_results=15)

        assert "…" in result
        assert "A" * 1000 not in result

    def test_no_hits(self):
        assert loogle._format_response("q", {"hits": []}, 15) == "No results for: q"

    def test_empty_data(self):
        assert loogle._format_response("q", {}, 15) == "No results for: q"

    def test_error_with_suggestions(self):
        data = {"error": "unknown identifier", "suggestions": ["Real.sin", "Real.cos"]}
        result = loogle._format_response("q", data, 15)

        assert "Loogle error: unknown identifier" in result
        assert "Suggestions: Real.sin, Real.cos" in result

    def test_starting_up_raises_not_ready(self):
        data = {"error": "Loogle is still starting up, try again later"}
        with pytest.raises(loogle.LoogleNotReadyError):
            loogle._format_response("q", data, 15)


class TestLoogleSearch:
    """Tests for the `loogle_search` entry point."""

    async def test_success(self, mock_config):
        data = {"hits": [{"name": "Real.add_comm", "module": "M", "type": "a + b = b + a"}]}
        with patch(
            "ax_prover.tools.loogle._query_loogle", new_callable=AsyncMock, return_value=data
        ):
            result = await loogle.loogle_search("add_comm", mock_config)

        assert "Real.add_comm" in result
        assert "a + b = b + a" in result

    async def test_connection_error_returns_friendly_message(self, mock_config):
        with patch(
            "ax_prover.tools.loogle._query_loogle",
            new_callable=AsyncMock,
            side_effect=aiohttp.ClientError("refused"),
        ):
            result = await loogle.loogle_search("add_comm", mock_config)

        assert "Cannot connect to Loogle server" in result
        assert mock_config.server_url in result
