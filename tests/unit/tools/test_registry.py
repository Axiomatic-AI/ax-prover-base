"""Unit tests for tool registry and create_tool dispatcher."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ax_prover.tools.lean_search import SearchLeanSearchConfig
from ax_prover.tools.registry import (
    TOOL_REGISTRY,
    ToolRegistration,
    create_tool,
    create_tool_lifespans,
    register_tool,
)
from ax_prover.tools.web_search import SearchWebConfig


class TestRegisterToolDecorator:
    """Tests for the @register_tool decorator."""

    def test_register_tool_adds_to_registry(self):
        """@register_tool populates TOOL_REGISTRY (triggered by tools/__init__.py imports)."""
        assert "search_web" in TOOL_REGISTRY
        assert "search_lean_search" in TOOL_REGISTRY

    def test_registered_entry_has_correct_config_class(self):
        """Test that the registered entry has the correct config class."""
        assert TOOL_REGISTRY["search_web"].config_class is SearchWebConfig
        assert TOOL_REGISTRY["search_lean_search"].config_class is SearchLeanSearchConfig

    def test_registered_entry_has_callable_factory(self):
        """Test that the registered entry has a callable factory."""
        assert callable(TOOL_REGISTRY["search_web"].factory)
        assert callable(TOOL_REGISTRY["search_lean_search"].factory)

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        with pytest.raises(ValueError, match="Duplicate tool registration"):

            @register_tool("search_web", SearchWebConfig)
            def duplicate_factory(config, runtime):
                pass

    def test_register_tool_default_lifespan_is_none(self):
        """Tools registered without a lifespan have lifespan=None."""
        assert TOOL_REGISTRY["search_web"].lifespan is None

    def test_register_tool_stores_lifespan(self):
        """A lifespan passed to @register_tool is stored on the registration."""

        @asynccontextmanager
        async def fake_lifespan(cfg):
            yield "resource"

        try:

            @register_tool("registry_test_tool_with_lifespan", SearchWebConfig, fake_lifespan)
            def factory(cfg, runtime):
                return None

            assert TOOL_REGISTRY["registry_test_tool_with_lifespan"].lifespan is fake_lifespan
        finally:
            TOOL_REGISTRY.pop("registry_test_tool_with_lifespan", None)


class TestCreateTool:
    """Tests for the create_tool dispatcher."""

    async def test_create_tool_missing_tool_type(self):
        """Test missing tool type raises ValueError."""
        with pytest.raises(ValueError, match="missing 'tool_type'"):
            await create_tool({"max_results": 3}, MagicMock())

    async def test_create_tool_unknown_tool_type(self):
        """Test that unknown tool_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool_type: 'nonexistent'"):
            await create_tool({"tool_type": "nonexistent"}, MagicMock())

    async def test_create_tool_builds_typed_config(self):
        """Test that create_tool constructs the typed config from dict."""
        mock_tool = MagicMock()
        mock_factory = MagicMock(return_value=mock_tool)
        runtime = MagicMock()

        with patch.dict(
            TOOL_REGISTRY,
            {"test_tool": ToolRegistration(factory=mock_factory, config_class=SearchWebConfig)},
        ):
            result = await create_tool(
                {"tool_type": "test_tool", "max_results": 5, "timeout": 20},
                runtime,
            )

        assert result is mock_tool
        config, passed_runtime = mock_factory.call_args[0]
        assert isinstance(config, SearchWebConfig)
        assert config.max_results == 5
        assert config.timeout == 20
        assert passed_runtime is runtime

    async def test_create_tool_handles_async_factory(self):
        """Test that create_tool awaits async factories."""
        mock_tool = MagicMock()
        mock_factory = AsyncMock(return_value=mock_tool)

        with patch.dict(
            TOOL_REGISTRY,
            {"test_tool": ToolRegistration(factory=mock_factory, config_class=SearchWebConfig)},
        ):
            result = await create_tool({"tool_type": "test_tool"}, MagicMock())

        assert result is mock_tool
        mock_factory.assert_awaited_once()

    async def test_create_tool_returns_none_from_factory(self):
        """Test that create_tool returns None when factory returns None."""
        mock_factory = AsyncMock(return_value=None)

        with patch.dict(
            TOOL_REGISTRY,
            {"test_tool": ToolRegistration(factory=mock_factory, config_class=SearchWebConfig)},
        ):
            result = await create_tool({"tool_type": "test_tool"}, MagicMock())

        assert result is None

    async def test_create_tool_invalid_config_params(self):
        """Test that invalid config parameters raise TypeError."""
        with patch.dict(
            TOOL_REGISTRY,
            {"test_tool": ToolRegistration(factory=MagicMock(), config_class=SearchWebConfig)},
        ):
            with pytest.raises(TypeError):
                await create_tool({"tool_type": "test_tool", "nonexistent_param": 42}, MagicMock())

    async def test_create_tool_uses_config_defaults(self):
        """Omitted config fields use dataclass defaults."""
        mock_factory = MagicMock(return_value=MagicMock())

        with patch.dict(
            TOOL_REGISTRY,
            {"test_tool": ToolRegistration(factory=mock_factory, config_class=SearchWebConfig)},
        ):
            await create_tool({"tool_type": "test_tool"}, MagicMock())

        config = mock_factory.call_args[0][0]
        assert isinstance(config, SearchWebConfig)
        # Defaults from SearchWebConfig dataclass
        assert config.max_results == 3
        assert config.timeout == 10
        assert config.max_content_length == 3000

    async def test_create_tool_does_not_mutate_input(self):
        """Test that create_tool does not modify the input dict."""
        mock_factory = MagicMock(return_value=MagicMock())
        original = {"tool_type": "test_tool", "max_results": 5}
        input_copy = dict(original)

        with patch.dict(
            TOOL_REGISTRY,
            {"test_tool": ToolRegistration(factory=mock_factory, config_class=SearchWebConfig)},
        ):
            await create_tool(input_copy, MagicMock())

        assert input_copy == original


class TestCreateToolLifespans:
    """Tests for the create_tool_lifespans helper."""

    async def test_returns_only_lifespans_for_tools_with_lifespan(self):
        """Only configured tools that declare a lifespan show up in the result."""
        result = await create_tool_lifespans(
            {
                "web": {"tool_type": "search_web"},
                "lean": {
                    "tool_type": "search_lean_search",
                    "server_url": "http://example.com",
                },
            }
        )
        assert set(result.keys()) == {"search_lean_search"}

    async def test_returns_empty_when_no_tools_configured(self):
        """No tools configured means no lifespans."""
        result = await create_tool_lifespans({})
        assert result == {}

    async def test_returns_empty_when_only_lifespan_less_tools_configured(self):
        """Tools without a registered lifespan don't produce entries."""
        result = await create_tool_lifespans({"web": {"tool_type": "search_web"}})
        assert result == {}
