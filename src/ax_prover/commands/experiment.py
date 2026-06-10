"""Experiment command for ax-prover CLI."""

from pathlib import Path

from langsmith import Client, traceable
from langsmith.schemas import Run
from omegaconf import OmegaConf

from ..config import Config
from ..evaluators import (
    build_timeout_count,
    compilation_error_count,
    is_proven,
    max_iterations_reached,
    number_of_iterations,
    reviewer_rejections,
    tool_usage,
)
from ..models import ProverOutput, TargetItem
from ..models.proving import ProverAgentState
from ..prover.agent import ProverAgent
from ..runtime import Runtime
from ..tools import create_tool_lifespans
from ..utils import get_logger, parse_prove_target, prove_single_item, write_json_output
from ..utils.git import get_git_hash, is_git_dirty

logger = get_logger(__name__)


async def experiment(
    folder: str,
    dataset: str,
    config: Config,
    max_concurrency: int = 4,
    experiment_prefix: str | None = None,
    output_file: str | None = None,
) -> int:
    """
    Run a prover experiment on a LangSmith dataset asynchronously.

    Args:
        folder: Base folder path for the Lean project
        dataset: Dataset name or dataset ID to run experiments on
        config: Configuration object to pass to the experiment
        max_concurrency: Max number of concurrent experiments
        experiment_prefix: Prefix for experiment name
        output_file: File path to write JSON output

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    if experiment_prefix is None:
        experiment_prefix = "experiment"

    base_path = str(Path(folder).resolve())

    logger.info(f"Running experiment on dataset: {dataset}")
    logger.debug(f"Max concurrency: {max_concurrency}")
    logger.debug(f"Experiment prefix: {experiment_prefix}")

    try:
        client = Client()

        def _tool_usage(run: Run) -> dict[str, int]:
            # Wrapper to pass the config to the tool_usage evaluator preserving the signature
            return tool_usage(run, config.prover)

        try:
            config_dict = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
        except Exception as e:
            # Fallback to basic info if serialization fails
            logger.warning(f"Failed to serialize config: {e}")
            config_dict = {"error": str(e)}

        experiment_metadata = {
            "config": config_dict,
            "git_hash": get_git_hash(),
            "git_dirty": is_git_dirty(),
        }

        tool_lifespans = await create_tool_lifespans(config.prover.proposer_tools)
        async with Runtime.open(config.runtime, base_path, tool_lifespans) as rt:
            # Create a wrapper function that includes the config and runtime. We need to use a
            # lambda instead of partial to avoid LangSmith's internal config parameter collision.
            @traceable
            async def experiment_func(inputs: dict[str, str]) -> dict:
                return await _run_experiment_sample(inputs, config, rt)

            results = await client.aevaluate(
                experiment_func,
                data=dataset,
                evaluators=[
                    build_timeout_count,
                    compilation_error_count,
                    is_proven,
                    number_of_iterations,
                    _tool_usage,
                    max_iterations_reached,
                    reviewer_rejections,
                ],
                max_concurrency=max_concurrency,
                experiment_prefix=experiment_prefix,
                metadata=experiment_metadata,
            )

            await results.wait()

            error_count = 0
            for result in results._results:
                outputs = result["run"].outputs
                if outputs and outputs.get("error") == "exception":
                    error_count += 1
                    logger.error(
                        f"Experiment failed for {outputs.get('path')}: {outputs.get('message')}"
                    )

            if output_file:
                prover_outputs = {}
                for result in results._results:
                    out = result["run"].outputs
                    if out and out.get("error") == "exception":
                        path = out.get("path", "unknown")
                        prover_outputs[path] = ProverOutput(success=False, error=out.get("message"))
                    else:
                        state = ProverAgentState.model_validate(out)
                        key = (
                            state.item.location.formatted_context
                            if state.item.location
                            else state.item.name
                        )
                        prover_outputs[key] = ProverOutput.from_prover_state(state)
                write_json_output(prover_outputs, output_file)

            if error_count > 0:
                logger.error(
                    f"Experiment completed with {error_count} unhandled error(s). Marking as failed."
                )
                return 1

            logger.info("Experiment completed successfully")
            return 0

    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        logger.exception("Full traceback:")
        return 1


@traceable
async def _run_experiment_sample(inputs: dict[str, str], config: Config, runtime: Runtime) -> dict:
    """Run prover on a single item for a LangSmith experiment."""
    target = inputs["path"]
    logger.info(f"Running experiment for: {target}")

    try:
        items = await parse_prove_target(runtime.lean_interact_server, runtime.base_folder, target)

        if not items:
            logger.warning(f"No unproven functions found in: {target}")
            item = TargetItem(proven=True)
            return ProverAgentState(item=item).model_dump()

        if len(items) > 1:
            logger.warning(
                f"Multiple items found ({len(items)}), "
                f"using first: {items[0].name}. "
                f"Use location string (Module:theorem) for specific theorem."
            )
        item = items[0]

        logger.info(f"Running prover experiment on: {item.location.formatted_context}")
        prover = await ProverAgent.create(config=config.prover, runtime=runtime)
        result = await prove_single_item(prover, item)
        logger.info("Experiment completed successfully")
        return result.model_dump()

    except Exception as e:
        logger.error(f"Experiment failed with exception: {e}")
        logger.exception("Full traceback:")
        return {"error": "exception", "message": str(e), "path": target}
