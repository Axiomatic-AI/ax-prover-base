"""LangChain tool wrapping for the cross-iteration tool log.

When ``ProverConfig.tool_log.enabled`` is true, the agent passes its ``BaseTool``
list through :func:`wrap_tools`, which produces a parallel list of tools that
delegate to the originals and record each invocation into a :class:`ToolLog`.

Each tool is responsible for compressing its raw output into a prompt-ready
string via a ``summarize_for_log`` callable attached to its
``StructuredTool.metadata``. If absent, a plain string fallback is used.

Wrapping is the only place this module is invoked; nothing in the wrapped
tool's interface (name, description, args_schema) is altered, so the LLM-facing
behavior is identical to the unwrapped tool.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from ..models.tool_log import ToolLog, ToolLogEntry

_DEFAULT_SUMMARY_LIMIT = 800


def _default_summarize(result: Any) -> str:
    text = result if isinstance(result, str) else str(result)
    if len(text) > _DEFAULT_SUMMARY_LIMIT:
        text = text[:_DEFAULT_SUMMARY_LIMIT].rstrip() + "…"
    return text


def _resolve_summarizer(tool: BaseTool) -> Callable[[Any], str]:
    metadata = getattr(tool, "metadata", None) or {}
    fn = metadata.get("summarize_for_log")
    return fn if callable(fn) else _default_summarize


def wrap_tool(
    inner: BaseTool,
    sink: ToolLog,
    iter_ref: Callable[[], int],
) -> BaseTool:
    """Return a tool that delegates to ``inner`` and appends to ``sink``.

    ``iter_ref`` is read at invocation time so the wrapper can stamp entries
    with the proposer's current iteration number without holding a stale value.
    All proposer tools take a single ``query: str`` argument; we rely on that.
    """
    summarize = _resolve_summarizer(inner)
    tool_name = inner.name
    inner_coroutine = getattr(inner, "coroutine", None)
    inner_func = getattr(inner, "func", None)

    def _record(query: str, result: Any) -> None:
        sink.add(
            ToolLogEntry(
                iter=iter_ref(),
                tool=tool_name,
                query=query,
                summary=summarize(result),
            )
        )

    async def wrapped_coroutine(query: str) -> Any:
        result = await inner_coroutine(query)
        _record(query, result)
        return result

    def wrapped_func(query: str) -> Any:
        result = inner_func(query)
        _record(query, result)
        return result

    return StructuredTool(
        name=inner.name,
        description=inner.description,
        args_schema=inner.args_schema,
        coroutine=wrapped_coroutine if inner_coroutine is not None else None,
        func=wrapped_func if inner_func is not None else None,
        metadata=inner.metadata,
    )


def wrap_tools(
    tools: list[BaseTool],
    sink: ToolLog,
    iter_ref: Callable[[], int],
) -> list[BaseTool]:
    """Wrap each tool to record into ``sink``."""
    return [wrap_tool(t, sink, iter_ref) for t in tools]
