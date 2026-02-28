"""Evaluators for LangSmith experiments."""

from dataclasses import fields

from langsmith import Client, traceable
from langsmith.schemas import Run

from .config import ProverConfig
from .tools.registry import tool_name_from_type
from .utils import get_logger

logger = get_logger(__name__)


@traceable
def is_proven(outputs: dict) -> bool:
    """LangSmith evaluator that checks if a theorem was proven."""
    logger.debug(f"OUTPUTS: {outputs}")

    if "error" in outputs:
        logger.warning(f"Experiment failed with error: {outputs.get('error')}")
        return False

    return outputs.get("item", {}).get("proven", False)


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
    # all the runs in the same trace and filter for the tool calls.
    client = Client()
    for r in client.list_runs(trace_id=run.trace_id):
        if r.run_type == "tool":
            tool_calls[r.name] = tool_calls.get(r.name, 0) + 1

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
