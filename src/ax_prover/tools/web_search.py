"""Web search tool using Tavily."""

import os
from dataclasses import dataclass

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from ..utils import get_logger
from .registry import register_tool, tool_name_from_type

logger = get_logger(__name__)

WEB_SEARCH_TOOL_TYPE = "search_web"


@dataclass
class SearchWebConfig:
    """Configuration for web search tool."""

    max_results: int = 3
    timeout: int = 10
    max_content_length: int = 3000


def search_web(query: str, config: SearchWebConfig) -> str:
    """
    Search the web and return formatted results for LLM consumption.

    Args:
        query: Search query string
        config: Web search configuration

    Returns:
        Formatted string with search results or error message
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        error_msg = "TAVILY_API_KEY not found in environment variables"
        logger.error(error_msg)
        return f"Error: {error_msg}"

    try:
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            max_results=config.max_results,
            include_answer=True,
            search_depth="advanced",
            timeout=config.timeout,
        )

        results = response.get("results", [])
        logger.debug(f"Searched for '{query}', found {len(results)} results")

        parts = []

        if answer := response.get("answer"):
            parts.append(f"Summary: {answer}\n")

        if results:
            parts.append("Results:")
            for i, result in enumerate(results, 1):
                if result.get("title"):  # Skip answer-only entries
                    parts.append(f"\n{i}. {result['title']}")
                    parts.append(f"   URL: {result.get('url', '')}")
                    if content := result.get("content"):
                        # Truncate long content
                        if len(content) > config.max_content_length:
                            content = content[: config.max_content_length] + "..."
                        parts.append(f"   {content}")

        return "\n".join(parts) if parts else "No results found"

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {str(e)}"


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query string")


@register_tool(WEB_SEARCH_TOOL_TYPE, SearchWebConfig)
def create_search_web_tool(config: SearchWebConfig) -> StructuredTool:
    """Create a web search tool with the given configuration."""
    return StructuredTool(
        name=tool_name_from_type(WEB_SEARCH_TOOL_TYPE),
        description="""Search the web for mathematical concepts, definitions, or examples.

Use this when you need:
- Real-world context or applications
- Mathematical definitions not in Lean yet
- Examples or counterexamples
- Background on unfamiliar concepts
""",
        func=lambda query: search_web(query, config),
        args_schema=SearchInput,
    )
