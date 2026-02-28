"""Tool registry with auto-registration via decorator."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ToolRegistration:
    """A registered tool factory with its config class."""

    factory: Callable
    config_class: type


# Registry populated by @register_tool decorator
TOOL_REGISTRY: dict[str, ToolRegistration] = {}


def tool_name_from_type(tool_type: str) -> str:
    """Derive the canonical tool name from a tool type identifier."""
    return f"{tool_type}_tool"


def register_tool(tool_type: str, config_class: type):
    """Decorator to register a tool factory with its config class.
    The factory must take a single argument, the config object it is registered with.

    Usage:
        @register_tool("search_web", SearchWebConfig)
        def create_search_web_tool(config: SearchWebConfig) -> StructuredTool:
            ...
    """

    def decorator(factory: Callable) -> Callable:
        if tool_type in TOOL_REGISTRY:
            raise ValueError(f"Duplicate tool registration: {tool_type}")
        TOOL_REGISTRY[tool_type] = ToolRegistration(factory=factory, config_class=config_class)
        return factory

    return decorator


async def create_tool(
    tool_config: dict[str, Any],
) -> BaseTool | None:
    """Create a tool from a config dict with a tool_type discriminator.

    Args:
        tool_config: Dict with 'tool_type' key plus tool-specific parameters.

    Returns:
        BaseTool instance, or None if creation failed (e.g., warmup failed).

    Raises:
        ValueError: If tool_type is missing or unknown.
        TypeError: If config parameters don't match the tool's config class.
    """
    tool_config = dict(tool_config)  # Make a copy to avoid modifying the original
    tool_type = tool_config.pop("tool_type", None)
    if not tool_type:
        raise ValueError(f"Tool config missing 'tool_type': {tool_config}")

    registration = TOOL_REGISTRY.get(tool_type)
    if registration is None:
        raise ValueError(
            f"Unknown tool_type: '{tool_type}'. Available: {sorted(TOOL_REGISTRY.keys())}"
        )

    config = registration.config_class(**tool_config)

    # Call factory (handle both sync and async)
    tool = registration.factory(config)
    if inspect.iscoroutine(tool):
        tool = await tool

    if tool is not None:
        tool.name = tool_name_from_type(tool_type)  # Ensure the tool name follows the convention

    return tool
