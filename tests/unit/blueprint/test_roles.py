"""Token accounting and the transcript-preserving tool loop."""

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ax_prover.blueprint.roles import TokenBudget, parse_proposal, run_turn


class Answer(BaseModel):
    value: str


class EchoInput(BaseModel):
    text: str


def usage(total: int) -> dict:
    return {"input_tokens": total // 2, "output_tokens": total - total // 2, "total_tokens": total}


def echo_tool(calls: list[str]) -> StructuredTool:
    async def echo(text: str) -> str:
        calls.append(text)
        return f"echoed {text}"

    return StructuredTool(
        name="echo", description="echo text", coroutine=echo, args_schema=EchoInput
    )


class ScriptedClient:
    def __init__(self, *responses: AIMessage):
        self.responses = list(responses)
        self.seen: list[list] = []
        self.kwargs: list[dict] = []

    async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
        self.seen.append(list(messages))
        self.kwargs.append({"tools": tools, "output_schema": output_schema})
        return self.responses.pop(0)


def test_budget_accumulates_reported_usage():
    budget = TokenBudget(limit=100)

    budget.record(AIMessage(content="a", usage_metadata=usage(40)))
    budget.record(AIMessage(content="b", usage_metadata=usage(40)))

    assert budget.spent == 80
    assert budget.calls == 2
    assert not budget.exhausted

    budget.record(AIMessage(content="c", usage_metadata=usage(40)))

    assert budget.exhausted


def test_a_missing_usage_record_costs_nothing():
    budget = TokenBudget(limit=10)

    budget.record(AIMessage(content="a", usage_metadata=None))

    assert budget.spent == 0
    assert budget.calls == 1


def test_a_zero_limit_means_unlimited():
    budget = TokenBudget(limit=0)

    budget.record(AIMessage(content="a", usage_metadata=usage(10**9)))

    assert not budget.exhausted


async def test_run_turn_executes_tool_calls_then_asks_for_the_proposal():
    calls: list[str] = []
    client = ScriptedClient(
        AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "hello"}, "id": "1"}],
            usage_metadata=usage(10),
        ),
        AIMessage(content="I have what I need.", usage_metadata=usage(5)),
        AIMessage(content=json.dumps({"value": "done"}), usage_metadata=usage(20)),
    )
    budget = TokenBudget(limit=0)

    turn = await run_turn(
        client, [HumanMessage(content="go")], [echo_tool(calls)], Answer, 3, budget
    )

    assert calls == ["hello"]
    assert parse_proposal(turn, Answer).value == "done"
    assert budget.spent == 35


async def test_the_schema_is_never_requested_alongside_tools():
    """Binding both suppresses tool calls on OpenAI-compatible providers."""
    calls: list[str] = []
    client = ScriptedClient(
        AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "x"}, "id": "1"}],
            usage_metadata=usage(1),
        ),
        AIMessage(content="done thinking", usage_metadata=usage(1)),
        AIMessage(content=json.dumps({"value": "ok"}), usage_metadata=usage(1)),
    )

    await run_turn(
        client, [HumanMessage(content="go")], [echo_tool(calls)], Answer, 3, TokenBudget(limit=0)
    )

    for kwargs in client.kwargs:
        assert not (kwargs.get("tools") and kwargs.get("output_schema")), kwargs

    assert any(kwargs.get("tools") for kwargs in client.kwargs), "tool phase must bind tools"
    assert client.kwargs[-1].get("output_schema") is Answer


async def test_a_turn_without_tools_makes_a_single_request():
    client = ScriptedClient(AIMessage(content=json.dumps({"value": "x"}), usage_metadata=usage(3)))

    turn = await run_turn(client, [HumanMessage(content="go")], [], Answer, 0, TokenBudget(limit=0))

    assert len(client.seen) == 1
    assert client.kwargs[0].get("output_schema") is Answer
    assert parse_proposal(turn, Answer).value == "x"


async def test_run_turn_stops_calling_tools_when_the_budget_runs_out():
    calls: list[str] = []
    tool_call = [{"name": "echo", "args": {"text": "x"}, "id": "1"}]
    client = ScriptedClient(
        AIMessage(content="", tool_calls=tool_call, usage_metadata=usage(100)),
        AIMessage(content=json.dumps({"value": "cut short"}), usage_metadata=usage(1)),
    )
    budget = TokenBudget(limit=50)

    turn = await run_turn(
        client, [HumanMessage(content="go")], [echo_tool(calls)], Answer, 5, budget
    )

    assert calls == ["x"], "the in-flight tool call still runs"
    assert budget.exhausted
    # Only the tool phase stops early; the proposal is still requested.
    assert parse_proposal(turn, Answer).value == "cut short"


async def test_the_last_tool_iteration_forbids_further_calls():
    calls: list[str] = []
    tool_call = [{"name": "echo", "args": {"text": "x"}, "id": "1"}]
    client = ScriptedClient(
        AIMessage(content="", tool_calls=tool_call, usage_metadata=usage(1)),
        AIMessage(content=json.dumps({"value": "forced"}), usage_metadata=usage(1)),
    )

    await run_turn(
        client, [HumanMessage(content="go")], [echo_tool(calls)], Answer, 1, TokenBudget(limit=0)
    )

    final_prompt = "\n".join(str(m.content) for m in client.seen[-1])
    assert "NO MORE TOOL CALLS ALLOWED" in final_prompt


async def test_an_unknown_tool_is_reported_back_to_the_model():
    from ax_prover.blueprint.roles import execute_tool_calls

    results, executed = await execute_tool_calls(
        [echo_tool([])], [{"name": "ghost", "args": {}, "id": "1"}]
    )

    assert "Unknown tool 'ghost'" in results[0].content
    assert "echo" in results[0].content
    assert executed == {}, "a refused call is not usage"


async def test_a_raising_tool_is_reported_back_to_the_model():
    from ax_prover.blueprint.roles import execute_tool_calls

    async def boom(text: str) -> str:
        raise RuntimeError("the compiler died")

    exploding = StructuredTool(name="echo", description="d", coroutine=boom, args_schema=EchoInput)

    results, executed = await execute_tool_calls(
        [exploding], [{"name": "echo", "args": {"text": "x"}, "id": "1"}]
    )

    assert "the compiler died" in results[0].content
    assert executed == {}, "a raising call is not usage"


def test_parse_proposal_returns_none_for_malformed_output():
    from ax_prover.blueprint.roles import RoleTurn

    turn = RoleTurn(response=AIMessage(content="not json"))

    assert parse_proposal(turn, Answer) is None


def search_tool(calls: list[str]) -> StructuredTool:
    async def search(query: str) -> str:
        calls.append(query)
        return "6 matches"

    return StructuredTool(
        name="mathlib_search", description="search", coroutine=search, args_schema=SearchInput
    )


class SearchInput(BaseModel):
    query: str


class BatchingClient:
    """Emits several search calls per turn, the way the real model does."""

    def __init__(self, searches_per_turn: int, turns: int):
        self.searches_per_turn = searches_per_turn
        self.turns = turns
        self.offered: list[list[str]] = []

    async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
        self.offered.append(sorted(t.name for t in (tools or [])))
        if output_schema is not None:
            return AIMessage(content=json.dumps({"value": "done"}), usage_metadata=None)
        if self.turns <= 0:
            return AIMessage(content="no more tools", usage_metadata=None)
        self.turns -= 1
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "mathlib_search", "args": {"query": f"q{i}"}, "id": str(i)}
                for i in range(self.searches_per_turn)
            ],
            usage_metadata=None,
        )


async def test_search_budget_withdraws_the_tool_so_turns_can_compile():
    """A turn batches many calls, so the iteration cap alone cannot stop over-searching."""
    searches: list[str] = []
    client = BatchingClient(searches_per_turn=4, turns=6)
    tools = [echo_tool([]), search_tool(searches)]

    turn = await run_turn(
        client, [HumanMessage(content="go")], tools, Answer, 6, TokenBudget(0), search_budget=6
    )

    assert len(searches) <= 8, "the cap bounds searching within a turn's batch"
    assert turn.tools_exhausted
    # Once spent, later turns are offered lean_compile only.
    assert any("mathlib_search" not in offered for offered in client.offered[1:])
    # Counts cover executed calls only, so refused ones do not inflate usage.
    assert turn.tool_calls["mathlib_search"] == len(searches)


async def test_hitting_the_iteration_cap_marks_the_answer_as_forced():
    client = BatchingClient(searches_per_turn=1, turns=10)
    tools = [echo_tool([]), search_tool([])]

    turn = await run_turn(client, [HumanMessage(content="go")], tools, Answer, 3, TokenBudget(0))

    assert turn.iterations_exhausted


async def test_stopping_early_is_not_a_forced_answer():
    client = BatchingClient(searches_per_turn=1, turns=1)
    tools = [echo_tool([]), search_tool([])]

    turn = await run_turn(client, [HumanMessage(content="go")], tools, Answer, 4, TokenBudget(0))

    assert not turn.iterations_exhausted


async def test_zero_search_budget_leaves_the_tool_available():
    searches: list[str] = []
    client = BatchingClient(searches_per_turn=3, turns=2)
    tools = [echo_tool([]), search_tool(searches)]

    turn = await run_turn(
        client, [HumanMessage(content="go")], tools, Answer, 4, TokenBudget(0), search_budget=0
    )

    assert not turn.tools_exhausted
    assert all("mathlib_search" in o for o in client.offered if o)


async def test_turn_reports_its_tool_usage():
    calls: list[str] = []
    client = ScriptedClient(
        AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "x"}, "id": "1"}],
            usage_metadata=None,
        ),
        AIMessage(content="done", usage_metadata=None),
        AIMessage(content=json.dumps({"value": "ok"}), usage_metadata=None),
    )

    turn = await run_turn(
        client, [HumanMessage(content="go")], [echo_tool(calls)], Answer, 4, TokenBudget(0)
    )

    assert turn.tool_calls == {"echo": 1}
    assert turn.turns == 2


async def test_a_withdrawn_tool_explains_itself_rather_than_reading_as_unknown():
    from ax_prover.blueprint.roles import execute_tool_calls

    results, executed = await execute_tool_calls(
        [echo_tool([])],
        [{"name": "mathlib_search", "args": {"query": "x"}, "id": "1"}],
        {"mathlib_search": "Search budget spent (6 queries). Use `lean_compile`."},
    )

    assert "Search budget spent" in results[0].content
    assert "Unknown tool" not in results[0].content
    assert executed == {}
