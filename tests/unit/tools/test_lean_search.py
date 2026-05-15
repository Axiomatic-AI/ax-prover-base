"""Unit tests for LeanSearch tool."""

import random
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from ax_prover.tools import lean_search


@pytest.fixture
def mock_config():
    """Create a mock SearchLeanSearchConfig."""
    config = Mock()
    config.server_url = "http://test-server.com"
    config.max_results = 10
    config.timeout = 30
    config.max_retries = 3
    config.retry_delay = 1.0
    return config


@pytest.fixture
def mock_session():
    """Create a mock aiohttp.ClientSession suitable for `async with session.post(...)`."""
    return AsyncMock(spec=aiohttp.ClientSession)


def _build_post_cm(response_data=None, status=200, raise_exc=None):
    """Build a mock for `session.post(...)` used as `async with session.post(...) as resp`."""
    resp = MagicMock()
    resp.status = status
    resp.content_length = 100
    resp.json = AsyncMock(return_value=response_data)
    if raise_exc is None:
        resp.raise_for_status = Mock()
    else:
        resp.raise_for_status = Mock(side_effect=raise_exc)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class TestURLSelection:
    """Tests for default URL selection."""

    def test_get_default_url(self):
        """Default URL is leansearch.net."""
        assert lean_search.DEFAULT_LEAN_SEARCH_URL == "https://leansearch.net"


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    async def test_retry_with_backoff(self, mock_config):
        """Retry with backoff calculates correct wait time."""
        with patch("ax_prover.tools.lean_search.asyncio.sleep") as mock_sleep:
            random.seed(42)
            await lean_search._retry_with_backoff(0, mock_config, "Test error")

            call_args = mock_sleep.call_args[0][0]
            assert 0.8 <= call_args <= 1.2  # Allow for jitter

    async def test_retry_with_backoff_exponential(self, mock_config):
        """Retry with backoff increases exponentially."""
        with patch("ax_prover.tools.lean_search.asyncio.sleep") as mock_sleep:
            random.seed(42)
            await lean_search._retry_with_backoff(2, mock_config, "Test error")

            call_args = mock_sleep.call_args[0][0]
            assert 3.2 <= call_args <= 4.8  # Allow for larger jitter range


class TestHTTPRequests:
    """Tests for HTTP request handling against an injected session."""

    async def test_make_request_success(self, mock_config, mock_session):
        """Successful API request returns parsed JSON."""
        response_data = [[{"result": {"name": "Test"}}]]
        mock_session.post = Mock(return_value=_build_post_cm(response_data=response_data))

        result = await lean_search._make_lean_search_request_with_retry(
            query="test query", config=mock_config, session=mock_session
        )

        assert result == response_data

    async def test_make_request_with_axleansearch_url_auth(self, mock_config, mock_session):
        """Request to an axleansearch URL adds an Authorization header."""
        mock_config.server_url = "https://axleansearch-sgyxphaitq-uc.a.run.app"
        response_data = [[{"result": {"name": "Test"}}]]
        mock_session.post = Mock(return_value=_build_post_cm(response_data=response_data))

        with patch("ax_prover.utils.google_auth.get_auth_token", return_value="test-token"):
            result = await lean_search._make_lean_search_request_with_retry(
                query="test query", config=mock_config, session=mock_session
            )

        assert result == response_data
        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"

    async def test_make_request_retry_on_429(self, mock_config, mock_session):
        """Request retries on 429 status code."""
        error = aiohttp.ClientResponseError(
            request_info=Mock(), history=(), status=429, message="Rate limited"
        )
        cm_error = _build_post_cm(status=429, raise_exc=error)
        cm_success = _build_post_cm(response_data=[[{"result": {"name": "Test"}}]])
        mock_session.post = Mock(side_effect=[cm_error, cm_success])

        with patch("ax_prover.tools.lean_search.asyncio.sleep", new_callable=AsyncMock):
            result = await lean_search._make_lean_search_request_with_retry(
                query="test query", config=mock_config, session=mock_session
            )

        assert result == [[{"result": {"name": "Test"}}]]
        assert mock_session.post.call_count == 2

    async def test_make_request_timeout_retry(self, mock_config, mock_session):
        """Request retries on timeout."""
        cm_success = _build_post_cm(response_data=[[{"result": {"name": "Test"}}]])
        mock_session.post = Mock(side_effect=[TimeoutError("Request timeout"), cm_success])

        with patch("ax_prover.tools.lean_search.asyncio.sleep", new_callable=AsyncMock):
            result = await lean_search._make_lean_search_request_with_retry(
                query="test query", config=mock_config, session=mock_session
            )

        assert result == [[{"result": {"name": "Test"}}]]
        assert mock_session.post.call_count == 2

    async def test_make_request_max_retries_exceeded(self, mock_config, mock_session):
        """Request fails after max retries."""
        mock_session.post = Mock(side_effect=TimeoutError("Request timeout"))

        with (
            patch("ax_prover.tools.lean_search.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(TimeoutError),
        ):
            await lean_search._make_lean_search_request_with_retry(
                query="test query", config=mock_config, session=mock_session
            )

        assert mock_session.post.call_count == mock_config.max_retries


class TestResponseProcessing:
    """Tests for response processing and formatting."""

    def test_process_response_with_results(self):
        response_data = [
            [
                {
                    "result": {
                        "name": ["Mathlib", "Analysis", "Basic"],
                        "kind": "theorem",
                        "signature": "theorem_signature",
                        "docstring": "Test docstring",
                    }
                },
                {
                    "result": {
                        "name": "SimpleTheorem",
                        "kind": "lemma",
                        "signature": "lemma_signature",
                        "docstring": None,
                    }
                },
            ]
        ]

        result = lean_search._process_lean_search_response("test query", response_data)

        assert "test query (2 matches)" in result
        assert "Mathlib.Analysis.Basic [theorem]" in result
        assert "theorem_signature" in result
        assert "Test docstring" in result
        assert "SimpleTheorem [lemma]" in result
        assert "lemma_signature" in result

    def test_process_response_no_results(self):
        result = lean_search._process_lean_search_response("test query", [[]])
        assert result == "No results found for: test query"

    def test_process_response_empty_data(self):
        result = lean_search._process_lean_search_response("test query", [])
        assert result == "No results found for: test query"

    def test_process_response_truncates_long_docstring(self):
        long_docstring = "A" * 5000
        response_data = [
            [
                {
                    "result": {
                        "name": "TestTheorem",
                        "kind": "theorem",
                        "signature": "sig",
                        "docstring": long_docstring,
                    }
                }
            ]
        ]

        result = lean_search._process_lean_search_response("test query", response_data)

        assert "A" * 3000 in result
        assert len(result) < len(long_docstring)

    def test_process_response_missing_fields(self):
        response_data = [[{"result": {"name": "TestTheorem"}}]]
        result = lean_search._process_lean_search_response("test query", response_data)
        assert "TestTheorem []" in result


class TestSearchFunction:
    """Tests for the main `lean_search` entry point."""

    async def test_lean_search_success(self, mock_config, mock_session):
        response_data = [
            [
                {
                    "result": {
                        "name": "TestTheorem",
                        "kind": "theorem",
                        "signature": "test_sig",
                        "docstring": "Test doc",
                    }
                }
            ]
        ]

        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            return_value=response_data,
        ):
            result = await lean_search.lean_search("test query", mock_config, mock_session)

        assert "TestTheorem [theorem]" in result
        assert "test_sig" in result

    async def test_lean_search_connection_error_localhost(self, mock_config, mock_session):
        mock_config.server_url = "http://127.0.0.1:8765"
        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            side_effect=aiohttp.ClientError("Connection failed"),
        ):
            result = await lean_search.lean_search("test query", mock_config, mock_session)

        assert "Cannot connect to LeanSearch server" in result
        assert "uvicorn server:app" in result
        assert "8765" in result

    async def test_lean_search_connection_error_remote(self, mock_config, mock_session):
        mock_config.server_url = "https://remote-server.com"
        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            side_effect=aiohttp.ClientError("Connection failed"),
        ):
            result = await lean_search.lean_search("test query", mock_config, mock_session)

        assert "Cannot connect to LeanSearch server" in result
        assert "remote-server.com" in result
        assert "uvicorn" not in result

    async def test_lean_search_generic_error(self, mock_config, mock_session):
        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            side_effect=ValueError("Some error"),
        ):
            result = await lean_search.lean_search("test query", mock_config, mock_session)

        assert "Some error" in result


class TestLifespan:
    """Tests for the `_lean_search_lifespan` context manager."""

    @pytest.fixture
    def real_config(self):
        """A real dataclass instance is required because warmup uses dataclasses.replace()."""
        return lean_search.SearchLeanSearchConfig(server_url="http://test-server.com")

    async def test_lifespan_yields_open_session_and_closes_it(self, real_config):
        """The lifespan yields an open ClientSession and closes it on exit."""
        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            return_value=[[]],
        ):
            async with lean_search._lean_search_lifespan(real_config) as session:
                assert isinstance(session, aiohttp.ClientSession)
                assert not session.closed

        assert session.closed

    async def test_lifespan_raises_when_warmup_fails(self, real_config):
        """If warmup fails, the lifespan propagates the exception (fails fast)."""
        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            side_effect=aiohttp.ClientError("boom"),
        ):
            with pytest.raises(aiohttp.ClientError):
                async with lean_search._lean_search_lifespan(real_config):
                    pass
