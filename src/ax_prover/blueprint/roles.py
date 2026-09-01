"""Shared LLM plumbing for the three blueprint roles.

The architect, node prover, and refiner all run the same shape of loop: a tool-calling
turn that ends in a structured proposal, repeated until the harness accepts the proposal
or a budget runs out. This module owns token accounting and the transcript-preserving
tool loop; the roles own their prompts and acceptance checks.
"""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..utils.llm import LLMClient
from ..utils.logging import get_logger

logger = get_logger(__name__)

#: Tool that is withdrawn once its per-turn budget is spent.
MATHLIB_SEARCH = "mathlib_search"

#: Closes the tool phase and asks for the structured proposal. Sent on its own request so
#: the schema never competes with tool bindings.
_ANSWER_REQUEST = HumanMessage(
    content="Now return your final answer in the required structured format."
)


@dataclass
class TokenBudget:
    """Accumulated LLM token usage against a per-role ceiling."""

    limit: int
    spent: int = 0
    calls: int = 0

    def record(self, message: BaseMessage) -> None:
        """Add one response's reported usage to the running total."""
        usage = getattr(message, "usage_metadata", None) or {}
        self.spent += usage.get("total_tokens", 0)
        self.calls += 1

    @property
    def exhausted(self) -> bool:
        """True once the ceiling is reached. A limit of 0 or less means unlimited."""
        return self.limit > 0 and self.spent >= self.limit


@dataclass
class RoleTurn:
    """One tool-calling turn: the final response plus everything it produced."""

    response: AIMessage
    messages: list[BaseMessage] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    tools_exhausted: bool = False
    iterations_exhausted: bool = False

    @property
    def text(self) -> str:
        """Final response text, which carries the structured proposal."""
        return self.response.text


async def execute_tool_calls(
    tools: list[BaseTool],
    tool_calls: list[dict],
    withdrawn: dict[str, str] | None = None,
) -> tuple[list[ToolMessage], dict[str, int]]:
    """Run a response's tool calls, returning their messages and what actually ran.

    `withdrawn` maps a deliberately removed tool name to the explanation the model should
    see, so a spent budget reads as a budget rather than as `Unknown tool`.

    The returned counts cover executed calls only: a refused call is not usage.

    Tools are dispatched directly rather than through LangGraph's `ToolNode`, which needs a
    compiled graph's runtime in its config; the blueprint orchestrator is plain async code.
    A tool error is reported back to the model instead of aborting the turn, since a bad
    argument is something the model can correct on its next call.
    """
    by_name = {tool.name: tool for tool in tools}
    withdrawn = withdrawn or {}
    results: list[ToolMessage] = []
    executed: dict[str, int] = {}

    for call in tool_calls:
        name = call.get("name", "")
        tool = by_name.get(name)

        if name in withdrawn:
            content = withdrawn[name]
        elif tool is None:
            content = f"Unknown tool {name!r}. Available tools: {', '.join(sorted(by_name))}."
        else:
            try:
                content = str(await tool.ainvoke(call.get("args", {})))
                executed[name] = executed.get(name, 0) + 1
            except Exception as e:  # noqa: BLE001 - surfaced to the model, not swallowed
                logger.warning(f"Tool {name!r} raised: {e}")
                content = f"Tool {name!r} failed: {e}"

        results.append(ToolMessage(content=content, tool_call_id=call.get("id", ""), name=name))

    return results, executed


async def run_turn(
    client: LLMClient,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    output_schema: type[BaseModel],
    max_tool_iterations: int,
    budget: TokenBudget,
    search_budget: int = 0,
) -> RoleTurn:
    """Run one turn: let the model use its tools, then extract a structured proposal.

    `search_budget` caps `mathlib_search` calls within the turn. A single response may batch
    many tool calls, so `max_tool_iterations` bounds turns rather than calls and cannot on
    its own stop a model from spending every turn searching and never compiling.

    The two phases are deliberately separate calls. Requesting a forced JSON schema and
    binding tools in the same request is mutually exclusive on OpenAI-compatible providers:
    the schema constrains the whole response, so the model cannot emit a tool call and every
    tool silently goes unused. So the tool phase runs unconstrained, and only the final
    answer request carries the schema.
    """
    transcript: list[BaseMessage] = []
    response: AIMessage | None = None
    used: dict[str, int] = {}
    turns = 0
    exhausted = False
    ran_out = False

    for iteration in range(max_tool_iterations):
        turns = iteration + 1
        # A turn may batch several tool calls, so the iteration cap bounds turns, not calls.
        # Once the search budget is spent, `mathlib_search` is withdrawn so the remaining
        # turns can only compile: the compiler is a stronger signal source than search.
        offered = tools
        withdrawn: dict[str, str] = {}
        if search_budget > 0 and used.get(MATHLIB_SEARCH, 0) >= search_budget:
            offered = [t for t in tools if t.name != MATHLIB_SEARCH]
            if len(offered) < len(tools):
                exhausted = True
                withdrawn[MATHLIB_SEARCH] = (
                    f"Search budget spent ({search_budget} queries). Mathlib does not contain "
                    "the solution directly; work from what you have and use `lean_compile`."
                )

        response = await client.ainvoke(messages + transcript, tools=offered)
        budget.record(response)
        transcript.append(response)

        if not response.tool_calls:
            break

        tool_messages, executed = await execute_tool_calls(offered, response.tool_calls, withdrawn)
        for name, count in executed.items():
            used[name] = used.get(name, 0) + count
        transcript += tool_messages

        if budget.exhausted:
            logger.debug(f"Token budget exhausted mid-turn ({budget.spent}/{budget.limit})")
            break

        if iteration == max_tool_iterations - 1:
            # The model still wanted tools when the cap cut it off, so the answer below
            # is forced rather than chosen.
            ran_out = True
            transcript.append(
                HumanMessage(content="NO MORE TOOL CALLS ALLOWED. Return your answer now.")
            )

    answer = await client.ainvoke(
        messages + transcript + [_ANSWER_REQUEST], output_schema=output_schema
    )
    budget.record(answer)
    transcript.append(answer)

    return RoleTurn(
        response=answer,
        messages=transcript,
        tool_calls=used,
        turns=turns,
        tools_exhausted=exhausted,
        iterations_exhausted=ran_out,
    )


def parse_proposal(turn: RoleTurn, schema: type[BaseModel]) -> BaseModel | None:
    """Parse a structured proposal out of a turn, returning None when it is malformed."""
    try:
        return schema.model_validate_json(turn.text)
    except ValueError as e:
        logger.warning(f"Structured output parsing failed: {e}")
        return None
