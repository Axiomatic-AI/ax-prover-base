"""Cross-iteration log of proposer tool calls.

Records (tool, query, summarized result) entries produced by the proposer's
tool calls so that subsequent iterations can see what has already been searched
and what was returned, without re-issuing identical queries.

The log is opt-in via ``ProverConfig.tool_log.enabled``. When disabled the
agent never constructs a ToolLog and the proposer prompt is unchanged.
"""

from pydantic import BaseModel, Field


class ToolLogEntry(BaseModel):
    """One observed tool call."""

    iter: int = Field(description="Proposer iteration in which the call was made (1-indexed)")
    tool: str = Field(description="Tool name, e.g. 'search_lean_search_tool'")
    query: str = Field(description="Tool input as the LLM provided it")
    summary: str = Field(description="Compact rendering of the result for prompt re-injection")
    repeat_iters: list[int] = Field(
        default_factory=list,
        description="Iterations in which this (tool, query) pair was re-issued by the proposer",
    )


class ToolLog(BaseModel):
    """Append-only log with simple compaction on insert.

    Each (tool, query) pair lives as a single entry. Re-issuing the same query
    in a later iteration is recorded on the existing entry's ``repeat_iters``
    rather than appended as a new line, so the rendered block grows in the
    distinct-queries dimension only. A global ``max_total`` cap evicts the
    oldest distinct entries.
    """

    entries: list[ToolLogEntry] = Field(default_factory=list)
    max_total: int = Field(default=50)

    def add(self, entry: ToolLogEntry) -> None:
        for e in self.entries:
            if e.tool == entry.tool and e.query == entry.query:
                e.repeat_iters.append(entry.iter)
                return

        self.entries.append(entry)
        overflow = len(self.entries) - self.max_total
        if overflow > 0:
            self.entries = self.entries[overflow:]

    def render(self) -> str:
        """Render as a `<prior-tool-calls>` block, or empty string if no entries."""
        if not self.entries:
            return ""
        lines = [
            "<prior-tool-calls>",
            "The following library/web searches were already issued in earlier",
            "iterations of this problem. Consult these before re-issuing similar",
            "queries; if you must repeat one, expect the same answer.",
            "",
        ]
        for e in self.entries:
            suffix = f"  (also re-issued in iters {e.repeat_iters})" if e.repeat_iters else ""
            lines.append(f"[iter {e.iter}] {e.tool}({e.query!r}):{suffix}")
            for line in e.summary.splitlines() or [""]:
                lines.append(f"  {line}")
        lines.append("</prior-tool-calls>")
        return "\n".join(lines)
