"""Unit tests for the cross-iteration proposer tool log."""

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ax_prover.models.tool_log import ToolLog, ToolLogEntry
from ax_prover.prover.tool_log import _default_summarize, wrap_tools
from ax_prover.tools.lean_explore import summarize_for_log as lean_explore_sum
from ax_prover.tools.lean_search import summarize_for_log as lean_search_sum
from ax_prover.tools.loogle import summarize_for_log as loogle_sum
from ax_prover.tools.web_search import summarize_for_log as web_sum


def _entry(iter: int, tool: str, query: str, summary: str = "ok") -> ToolLogEntry:
    return ToolLogEntry(iter=iter, tool=tool, query=query, summary=summary)


class TestToolLog:
    def test_empty_render(self):
        assert ToolLog().render() == ""

    def test_basic_render(self):
        log = ToolLog()
        log.add(_entry(1, "search_lean_search_tool", "monotone Cauchy", "• Foo\n  : Bar"))
        out = log.render()
        assert "<prior-tool-calls>" in out
        assert "[iter 1] search_lean_search_tool('monotone Cauchy')" in out
        assert "• Foo" in out
        assert out.endswith("</prior-tool-calls>")

    def test_duplicate_collapses_to_repeat_iters(self):
        log = ToolLog()
        log.add(_entry(1, "t", "q", "first body"))
        log.add(_entry(3, "t", "q", "later body"))
        log.add(_entry(7, "t", "q", "even later"))
        assert len(log.entries) == 1
        assert log.entries[0].summary == "first body"
        assert log.entries[0].repeat_iters == [3, 7]
        rendered = log.render()
        assert "re-issued in iters [3, 7]" in rendered
        assert "later body" not in rendered

    def test_max_total_drops_oldest(self):
        log = ToolLog(max_total=2)
        log.add(_entry(1, "t", "q1"))
        log.add(_entry(2, "t", "q2"))
        log.add(_entry(3, "t", "q3"))
        assert len(log.entries) == 2
        assert log.entries[0].iter == 2
        assert log.entries[1].iter == 3


class _SearchInput(BaseModel):
    query: str = Field(...)


class TestWrapTools:
    @pytest.mark.asyncio
    async def test_records_async_call(self):
        captured: list[str] = []

        async def _coro(query: str) -> str:
            captured.append(query)
            return f"result for {query}"

        inner = StructuredTool(
            name="my_tool",
            description="x",
            coroutine=_coro,
            args_schema=_SearchInput,
            metadata={"summarize_for_log": lambda r: r.upper()},
        )

        sink = ToolLog()
        current_iter = 7
        [wrapped] = wrap_tools([inner], sink, lambda: current_iter)

        out = await wrapped.coroutine("hello")

        assert out == "result for hello"
        assert captured == ["hello"]
        assert len(sink.entries) == 1
        e = sink.entries[0]
        assert e.iter == 7
        assert e.tool == "my_tool"
        assert e.query == "hello"
        assert e.summary == "RESULT FOR HELLO"

    @pytest.mark.asyncio
    async def test_iter_ref_read_at_call_time(self):
        async def _coro(query: str) -> str:
            return query

        inner = StructuredTool(
            name="t",
            description="x",
            coroutine=_coro,
            args_schema=_SearchInput,
        )

        sink = ToolLog()
        iter_holder = {"v": 1}
        [wrapped] = wrap_tools([inner], sink, lambda: iter_holder["v"])

        await wrapped.coroutine("a")
        iter_holder["v"] = 5
        await wrapped.coroutine("b")

        assert [e.iter for e in sink.entries] == [1, 5]

    @pytest.mark.asyncio
    async def test_default_summarizer_when_metadata_missing(self):
        async def _coro(query: str) -> str:
            return "x" * 1000

        inner = StructuredTool(
            name="t",
            description="x",
            coroutine=_coro,
            args_schema=_SearchInput,
        )

        sink = ToolLog()
        [wrapped] = wrap_tools([inner], sink, lambda: 1)
        await wrapped.coroutine("q")

        assert sink.entries[0].summary.endswith("…")

    def test_default_summarize_short_passthrough(self):
        assert _default_summarize("hello") == "hello"

    def test_default_summarize_truncates_long(self):
        out = _default_summarize("a" * 5000)
        assert out.endswith("…")
        assert len(out) < 5000


class TestPerToolSummarizers:
    def test_lean_search_keeps_top_k_bullets(self):
        raw = (
            "=== query (5 matches) ===\n\n"
            "• Foo.bar [theorem]\n"
            "  Foo.bar : Nat → Nat\n"
            "  Doc: long docstring\n\n"
            "• Foo.baz [theorem]\n"
            "  Foo.baz : Nat → Nat\n"
            "  Doc: more doc\n\n"
            "• Foo.qux [theorem]\n"
            "  Foo.qux : Bool → Bool\n\n"
            "• Foo.quux [theorem]\n"
            "  Foo.quux : Int → Int\n"
        )
        out = lean_search_sum(raw, top_k=3)
        assert "Foo.bar" in out
        assert "Foo.baz" in out
        assert "Foo.qux" in out
        assert "Foo.quux" not in out
        assert "Doc:" not in out

    def test_lean_search_no_results_passthrough(self):
        out = lean_search_sum("No results found for: foo")
        assert out.startswith("No results")

    def test_loogle_keeps_top_k_bullets(self):
        raw = (
            "=== q (10 total, showing 5) ===\n"
            "• Nat.add  [Nat.Basic]\n"
            "  : Nat → Nat → Nat\n"
            "• Nat.mul  [Nat.Basic]\n"
            "  : Nat → Nat → Nat\n"
            "  Doc: skip me\n"
            "• Nat.sub  [Nat.Basic]\n"
            "  : Nat → Nat → Nat\n"
            "• Nat.div  [Nat.Basic]\n"
        )
        out = loogle_sum(raw, top_k=2)
        assert "Nat.add" in out
        assert "Nat.mul" in out
        assert "Nat.sub" not in out
        assert "Doc:" not in out

    def test_lean_explore_drops_source_blocks(self):
        raw = (
            "=== q (3 matches) ===\n\n"
            "• Foo.bar  [Mathlib.Foo]\n"
            "  Source:\n    very long source text\n"
            "  Doc: docstring\n"
            "  Informal: prose\n\n"
            "• Foo.baz  [Mathlib.Foo]\n"
            "  Source:\n    more source\n"
        )
        out = lean_explore_sum(raw, top_k=2)
        assert "Foo.bar" in out
        assert "Foo.baz" in out
        assert "Source:" not in out
        assert "Doc:" not in out

    def test_web_keeps_summary_and_titles(self):
        raw = (
            "Summary: short answer\n\n"
            "Results:\n\n"
            "1. First Title\n"
            "   URL: https://example.com/1\n"
            "   long content body\n\n"
            "2. Second Title\n"
            "   URL: https://example.com/2\n"
            "   more body\n"
        )
        out = web_sum(raw, top_k=2)
        assert "Summary:" in out
        assert "First Title" in out
        assert "Second Title" in out
        assert "long content body" not in out
