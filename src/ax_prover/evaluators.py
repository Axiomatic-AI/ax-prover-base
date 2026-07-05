"""Evaluators for LangSmith experiments."""

import time
from dataclasses import fields

from langsmith import Client, traceable
from langsmith.schemas import Run

from .config import ProverConfig
from .tools.registry import tool_name_from_type
from .utils import get_logger

logger = get_logger(__name__)

# LangSmith uploads child runs (LLM calls, tool calls) from a background thread,
# so they may not be queryable the instant an evaluator fires. Retry listing the
# trace until it looks populated (an LLM run is always present in a proof run).
_TRACE_LIST_RETRIES = 4
_TRACE_LIST_RETRY_WAIT_S = 1.5


def _list_trace_runs(client: Client, trace_id) -> list[Run]:
    """List a trace's runs, retrying until the trace looks populated.

    Tolerates LangSmith's asynchronous run upload: returns as soon as an LLM run
    appears (every proof run makes at least one LLM call), otherwise falls back
    to the last attempt's result.
    """
    runs: list[Run] = []
    for attempt in range(_TRACE_LIST_RETRIES):
        runs = list(client.list_runs(trace_id=trace_id))
        if any(r.run_type == "llm" for r in runs):
            return runs
        if attempt < _TRACE_LIST_RETRIES - 1:
            time.sleep(_TRACE_LIST_RETRY_WAIT_S)
    return runs


@traceable
def is_proven(outputs: dict) -> bool:
    """LangSmith evaluator that checks if a theorem was proven."""
    logger.debug(f"OUTPUTS: {outputs}")

    if "error" in outputs:
        logger.warning(f"Experiment failed with error: {outputs.get('error')}")
        return False

    return outputs.get("item", {}).get("is_proven", False)


@traceable
def tool_usage(run: Run, config: ProverConfig) -> dict[str, int]:
    """LangSmith evaluator that counts the number of tool calls for each tool."""

    available_tools = []
    for field in fields(config):
        if "tool" in field.name:
            available_tools.extend(
                [
                    tool_config.get("tool_type")
                    for tool_config in getattr(config, field.name).values()
                ]
            )

    if not available_tools:
        logger.warning("The experiment runs without tools")
        return {"key": "tool_usage", "score": 0}

    tool_calls = {tool_name_from_type(tool_type): 0 for tool_type in available_tools}

    # Since we wrap our run function and the root does not populate child runs, we need to list
    # all the runs in the same trace and filter for the tool calls. Guard against transient
    # LangSmith errors so a network hiccup reports zero rather than failing the whole evaluator.
    try:
        client = Client()
        for r in _list_trace_runs(client, run.trace_id):
            if r.run_type == "tool":
                tool_calls[r.name] = tool_calls.get(r.name, 0) + 1
    except Exception as e:
        logger.warning(f"tool_usage evaluator could not list trace runs: {e}")

    tool_usage = {"key": "tool_usage", "score": sum(tool_calls.values())}
    return [tool_usage] + [{"key": k, "score": v} for k, v in tool_calls.items()]


@traceable
def number_of_iterations(outputs: dict) -> int:
    """LangSmith evaluator that counts the number of times the prover agent went over the main theorem."""
    return outputs.get("metrics", {}).get("number_of_iterations", 0)


@traceable
def reviewer_rejections(outputs: dict) -> int:
    """LangSmith evaluator that counts the number of times the prover agent rejected the proof."""
    return outputs.get("metrics", {}).get("reviewer_rejections", 0)


@traceable
def compilation_error_count(outputs: dict) -> int:
    """LangSmith evaluator that counts the number of compilation errors during proving."""
    return outputs.get("metrics", {}).get("compilation_error_count", 0)


@traceable
def build_timeout_count(outputs: dict) -> int:
    """LangSmith evaluator that counts the number of build timeouts during proving."""
    return outputs.get("metrics", {}).get("build_timeout_count", 0)


@traceable
def max_iterations_reached(outputs: dict) -> bool:
    """LangSmith evaluator that checks if max iterations has been reached."""
    return outputs.get("metrics", {}).get("max_iterations_reached", False)


def _compute_costs(
    input_tokens: int,
    output_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> dict[str, float]:
    """Cost in USD from token counts and per-1M-token prices."""
    input_cost = input_tokens / 1_000_000 * input_price_per_m
    output_cost = output_tokens / 1_000_000 * output_price_per_m
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def _trace_token_usage(run: Run, client: Client) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) for the whole trace.

    Prefers the aggregated counts LangSmith stores on the root run; falls back to
    summing the trace's LLM runs when the root lacks them.
    """
    if run.prompt_tokens is not None or run.completion_tokens is not None:
        return run.prompt_tokens or 0, run.completion_tokens or 0

    input_tokens = 0
    output_tokens = 0
    for r in _list_trace_runs(client, run.trace_id):
        if r.run_type == "llm":
            input_tokens += r.prompt_tokens or 0
            output_tokens += r.completion_tokens or 0
    return input_tokens, output_tokens


@traceable
def token_cost(run: Run, config: ProverConfig) -> list[dict]:
    """LangSmith evaluator that reports input/output/total cost in USD.

    LangSmith's native Cost columns require the model to be in its price map;
    custom endpoints like DeepSeek's are not, so cost is derived here from the
    configured per-1M-token prices. Returns no feedback when prices are unset.
    """
    llm_config = getattr(config, "prover_llm", None)
    input_price = getattr(llm_config, "input_token_price", None) if llm_config else None
    output_price = getattr(llm_config, "output_token_price", None) if llm_config else None
    if input_price is None or output_price is None:
        logger.warning(
            "No token prices configured (prover_llm.input_token_price / output_token_price); "
            "skipping cost evaluator"
        )
        return []

    try:
        client = Client()
        input_tokens, output_tokens = _trace_token_usage(run, client)
    except Exception as e:
        logger.warning(f"token_cost evaluator could not read token usage: {e}")
        return []

    costs = _compute_costs(input_tokens, output_tokens, input_price, output_price)
    return [{"key": k, "score": v} for k, v in costs.items()]
