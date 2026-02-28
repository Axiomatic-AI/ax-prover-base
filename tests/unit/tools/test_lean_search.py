"""Unit tests for LeanSearch tool."""

import random
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from ax_prover.tools import lean_search


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    lean_search._lean_search_session = None
    lean_search._lean_search_warmup_result = None
    yield
    lean_search._lean_search_session = None
    lean_search._lean_search_warmup_result = None


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


class TestURLSelection:
    """Tests for default URL selection."""

    def test_get_default_url(self):
        """Test default URL is leansearch.net."""
        assert lean_search.DEFAULT_LEAN_SEARCH_URL == "https://leansearch.net"


class TestSessionManagement:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_get_lean_search_session_creates_new(self):
        """Test that get_lean_search_session creates a new session."""
        session = await lean_search.get_lean_search_session()

        assert session is not None
        assert isinstance(session, aiohttp.ClientSession)
        assert not session.closed

        await session.close()

    @pytest.mark.asyncio
    async def test_get_lean_search_session_reuses_existing(self):
        """Test that get_lean_search_session reuses existing session."""
        session1 = await lean_search.get_lean_search_session()
        session2 = await lean_search.get_lean_search_session()

        assert session1 is session2

        await session1.close()

    @pytest.mark.asyncio
    async def test_get_lean_search_session_creates_new_if_closed(self):
        """Test that a new session is created if the existing one is closed."""
        session1 = await lean_search.get_lean_search_session()
        await session1.close()

        session2 = await lean_search.get_lean_search_session()

        assert session1 is not session2
        assert not session2.closed

        await session2.close()

    @pytest.mark.asyncio
    async def test_lean_search_session_manager_cleanup(self):
        """Test that session manager properly cleans up."""
        async with lean_search.lean_search_session_manager():
            session = await lean_search.get_lean_search_session()
            assert not session.closed

        # After exiting context, session should be closed
        assert session.closed
        assert lean_search._lean_search_session is None
        assert lean_search._lean_search_warmup_result is None


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, mock_config):
        """Test retry with backoff calculates correct wait time."""
        with patch("ax_prover.tools.lean_search.asyncio.sleep") as mock_sleep:
            random.seed(42)
            await lean_search._retry_with_backoff(0, mock_config, "Test error")

            call_args = mock_sleep.call_args[0][0]
            assert 0.8 <= call_args <= 1.2  # Allow for jitter

    @pytest.mark.asyncio
    async def test_retry_with_backoff_exponential(self, mock_config):
        """Test retry with backoff increases exponentially."""
        with patch("ax_prover.tools.lean_search.asyncio.sleep") as mock_sleep:
            random.seed(42)
            await lean_search._retry_with_backoff(2, mock_config, "Test error")

            call_args = mock_sleep.call_args[0][0]
            assert 3.2 <= call_args <= 4.8  # Allow for larger jitter range


class TestHTTPRequests:
    """Tests for HTTP request handling."""

    @pytest.mark.asyncio
    async def test_make_request_success(self, mock_config):
        """Test successful API request."""
        mock_response_data = [[{"result": {"name": "Test"}}]]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = Mock()
        mock_response.content_length = 100
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with patch(
            "ax_prover.tools.lean_search.get_lean_search_session", return_value=mock_session
        ):
            result = await lean_search._make_lean_search_request_with_retry(
                "test query", mock_config
            )

        assert result == mock_response_data

    @pytest.mark.asyncio
    async def test_make_request_with_axleansearch_url_auth(self, mock_config):
        """Test API request with auth when URL contains 'axleansearch'."""
        # Set URL containing "axleansearch" to trigger auth
        mock_config.server_url = "https://axleansearch-sgyxphaitq-uc.a.run.app"
        mock_response_data = [[{"result": {"name": "Test"}}]]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = Mock()
        mock_response.content_length = 100
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_cm)

        with (
            patch("ax_prover.tools.lean_search.get_lean_search_session", return_value=mock_session),
            patch("ax_prover.utils.google_auth.get_auth_token", return_value="test-token"),
        ):
            result = await lean_search._make_lean_search_request_with_retry(
                "test query", mock_config
            )

        assert result == mock_response_data
        # Verify Authorization header was added
        call_kwargs = mock_session.post.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_make_request_retry_on_429(self, mock_config):
        """Test request retries on 429 status code."""
        mock_response_error = MagicMock()
        mock_response_error.status = 429
        mock_response_error.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=Mock(), history=(), status=429, message="Rate limited"
        )
        mock_response_error.__aenter__ = AsyncMock(return_value=mock_response_error)
        mock_response_error.__aexit__ = AsyncMock(return_value=None)

        mock_response_success = MagicMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value=[[{"result": {"name": "Test"}}]])
        mock_response_success.raise_for_status = Mock()
        mock_response_success.content_length = 100
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_post_cm_error = MagicMock()
        mock_post_cm_error.__aenter__ = AsyncMock(return_value=mock_response_error)
        mock_post_cm_error.__aexit__ = AsyncMock(return_value=None)

        mock_post_cm_success = MagicMock()
        mock_post_cm_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_post_cm_success.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(side_effect=[mock_post_cm_error, mock_post_cm_success])

        with (
            patch("ax_prover.tools.lean_search.get_lean_search_session", return_value=mock_session),
            patch("ax_prover.tools.lean_search.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await lean_search._make_lean_search_request_with_retry(
                "test query", mock_config
            )

        assert result == [[{"result": {"name": "Test"}}]]
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_make_request_timeout_retry(self, mock_config):
        """Test request retries on timeout."""
        mock_response_success = MagicMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value=[[{"result": {"name": "Test"}}]])
        mock_response_success.raise_for_status = Mock()
        mock_response_success.content_length = 100
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_post_cm_success = MagicMock()
        mock_post_cm_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_post_cm_success.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(
            side_effect=[TimeoutError("Request timeout"), mock_post_cm_success]
        )

        with (
            patch("ax_prover.tools.lean_search.get_lean_search_session", return_value=mock_session),
            patch("ax_prover.tools.lean_search.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await lean_search._make_lean_search_request_with_retry(
                "test query", mock_config
            )

        assert result == [[{"result": {"name": "Test"}}]]
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_make_request_max_retries_exceeded(self, mock_config):
        """Test request fails after max retries."""
        mock_session = AsyncMock()
        mock_session.post = Mock(side_effect=TimeoutError("Request timeout"))

        with (
            patch("ax_prover.tools.lean_search.get_lean_search_session", return_value=mock_session),
            patch("ax_prover.tools.lean_search.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(TimeoutError),
        ):
            await lean_search._make_lean_search_request_with_retry("test query", mock_config)

        assert mock_session.post.call_count == mock_config.max_retries


class TestResponseProcessing:
    """Tests for response processing and formatting."""

    def test_process_response_with_results(self):
        """Test processing response with valid results."""
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
        """Test processing response with no results."""
        response_data = [[]]

        result = lean_search._process_lean_search_response("test query", response_data)

        assert result == "No results found for: test query"

    def test_process_response_empty_data(self):
        """Test processing empty response data."""
        response_data = []

        result = lean_search._process_lean_search_response("test query", response_data)

        assert result == "No results found for: test query"

    def test_process_response_truncates_long_docstring(self):
        """Test that long docstrings are truncated."""
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
        """Test processing response with missing fields."""
        response_data = [
            [
                {
                    "result": {
                        "name": "TestTheorem",
                        # kind, signature, and docstring missing
                    }
                }
            ]
        ]

        result = lean_search._process_lean_search_response("test query", response_data)

        assert "TestTheorem []" in result


class TestSearchFunction:
    """Tests for the main search function."""

    @pytest.mark.asyncio
    async def test_lean_search_success(self, mock_config):
        """Test successful search."""
        mock_response_data = [
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
            return_value=mock_response_data,
        ):
            result = await lean_search.lean_search("test query", mock_config)

        assert "TestTheorem [theorem]" in result
        assert "test_sig" in result

    @pytest.mark.asyncio
    async def test_lean_search_connection_error_localhost(self, mock_config):
        """Test search with connection error to localhost."""
        mock_config.server_url = "http://127.0.0.1:8765"

        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            side_effect=aiohttp.ClientError("Connection failed"),
        ):
            result = await lean_search.lean_search("test query", mock_config)

        assert "Cannot connect to LeanSearch server" in result
        assert "uvicorn server:app" in result
        assert "8765" in result

    @pytest.mark.asyncio
    async def test_lean_search_connection_error_remote(self, mock_config):
        """Test search with connection error to remote server."""
        mock_config.server_url = "https://remote-server.com"

        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            side_effect=aiohttp.ClientError("Connection failed"),
        ):
            result = await lean_search.lean_search("test query", mock_config)

        assert "Cannot connect to LeanSearch server" in result
        assert "remote-server.com" in result
        assert "uvicorn" not in result  # No local server instructions for remote

    @pytest.mark.asyncio
    async def test_lean_search_generic_error(self, mock_config):
        """Test search with generic error."""
        with patch(
            "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
            new_callable=AsyncMock,
            side_effect=ValueError("Some error"),
        ):
            result = await lean_search.lean_search("test query", mock_config)

        assert "Some error" in result


class TestWarmup:
    """Tests for warmup functionality."""

    @pytest.mark.asyncio
    async def test_warmup_lean_search(self, mock_config):
        """Test successful warmup."""
        warmup_config_result = Mock()
        warmup_config_result.timeout = 120

        with (
            patch(
                "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
                new_callable=AsyncMock,
                return_value=[[{"result": {"name": "Nat"}}]],
            ) as mock_request,
            patch("dataclasses.replace", return_value=warmup_config_result),
        ):
            await lean_search.warmup_lean_search(mock_config)

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["query"] == "Nat"
            assert call_args[1]["config"].timeout == 120

    @pytest.mark.asyncio
    async def test_warmup_lean_search_failure(self, mock_config):
        """Test warmup failure propagates exception."""
        warmup_config_result = Mock()
        warmup_config_result.timeout = 120

        with (
            patch(
                "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
                new_callable=AsyncMock,
                side_effect=TimeoutError("Warmup timeout"),
            ),
            patch("dataclasses.replace", return_value=warmup_config_result),
            pytest.raises(TimeoutError),
        ):
            await lean_search.warmup_lean_search(mock_config)


class TestToolCreation:
    """Tests for tool creation with warmup."""

    @pytest.mark.asyncio
    async def test_create_tool_success(self, mock_config):
        """Test successful tool creation after warmup."""
        with patch("ax_prover.tools.lean_search.warmup_lean_search", new_callable=AsyncMock):
            tool = await lean_search.create_search_lean_search_tool(mock_config)

        assert tool is not None
        assert tool.name == "search_lean_search_tool"
        assert "LeanSearch" in tool.description

    @pytest.mark.asyncio
    async def test_create_tool_warmup_failure(self, mock_config):
        """Test tool creation returns None when warmup fails."""
        with patch(
            "ax_prover.tools.lean_search.warmup_lean_search",
            new_callable=AsyncMock,
            side_effect=TimeoutError("Warmup failed"),
        ):
            tool = await lean_search.create_search_lean_search_tool(mock_config)

        assert tool is None

    @pytest.mark.asyncio
    async def test_create_tool_caches_warmup_result(self, mock_config):
        """Test that warmup result is cached across tool creations."""
        with patch(
            "ax_prover.tools.lean_search.warmup_lean_search", new_callable=AsyncMock
        ) as mock_warmup:
            tool1 = await lean_search.create_search_lean_search_tool(mock_config)
            tool2 = await lean_search.create_search_lean_search_tool(mock_config)

        assert tool1 is not None
        assert tool2 is not None
        assert mock_warmup.call_count == 1

    @pytest.mark.asyncio
    async def test_create_tool_respects_cached_failure(self, mock_config):
        """Test that cached warmup failure prevents tool creation."""
        lean_search._lean_search_warmup_result = False

        tool = await lean_search.create_search_lean_search_tool(mock_config)

        assert tool is None

    @pytest.mark.asyncio
    async def test_tool_invocation(self, mock_config):
        """Test that created tool can be invoked."""
        mock_response_data = [[{"result": {"name": "TestTheorem", "kind": "theorem"}}]]

        with (
            patch("ax_prover.tools.lean_search.warmup_lean_search", new_callable=AsyncMock),
            patch(
                "ax_prover.tools.lean_search._make_lean_search_request_with_retry",
                new_callable=AsyncMock,
                return_value=mock_response_data,
            ),
        ):
            tool = await lean_search.create_search_lean_search_tool(mock_config)
            result = await tool.ainvoke({"query": "test query"})

        assert "TestTheorem [theorem]" in result
