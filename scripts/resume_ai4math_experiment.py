#!/usr/bin/env python3
"""
Resume an interrupted ax-prover experiment on AI4Math_TCS_Challenge.

Re-runs only the 4 problems that were interrupted mid-way, appending the
new runs to the SAME LangSmith experiment (so summary stats end up in one
place — no merging step needed).

Mirrors `ax_prover.commands.experiment.experiment()` but:
  1. Filters `data=` to the 4 missing examples (fetched by path from the
     dataset).
  2. Passes `experiment=<existing_experiment>` to `client.aevaluate(...)`,
     which extends the existing project instead of creating a new one.
  3. Deletes any partial/in-progress runs for those 4 paths first so they
     don't pollute the summary.

Usage (run inside the ax-prover venv, with cwd = AI4Math/challenges so the
Lean repo and .env.secrets resolve correctly):

    cd /path/to/AI4Math/challenges
    python /path/to/ax-prover-base/scripts/resume_ai4math_experiment.py \
        --experiment "<experiment name OR uuid from LangSmith>"

Or pass --folder explicitly from anywhere:

    python ax-prover-base/scripts/resume_ai4math_experiment.py \
        --experiment "<experiment name OR uuid>" \
        --folder /path/to/AI4Math/challenges

Optional flags:
    --dry-run          List what would be deleted / re-run, then exit.
    --max-concurrency  Default 4. Match what you used originally.
    --no-skip-build    Force `lake exe cache get` + `lake build` first
                       (default: skip, since the repo is presumably built).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from asyncio import Semaphore
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, traceable
from langsmith.schemas import Run

from ax_prover.commands.experiment import run_experiment
from ax_prover.config import Config
from ax_prover.evaluators import (
    build_timeout_count,
    compilation_error_count,
    is_proven,
    max_iterations_reached,
    number_of_iterations,
    reviewer_rejections,
    tool_usage,
)
from ax_prover.tools.lean_search import lean_search_session_manager
from ax_prover.utils import (
    get_logger,
    load_env_secrets,
    merge_configs,
    reconfigure_log_level,
)
from ax_prover.utils.build import build_lean_repo
from ax_prover.utils.lean_interact import lean_interact_session_manager

logger = get_logger(__name__)

DATASET_NAME = "AI4Math_TCS_Challenge"

# The 4 problems that were interrupted mid-run.
MISSING_PATHS: list[str] = [
    "Challenges/Treap/Challenge_Treap_7.lean",
    "Challenges/Segment_Tree/Challenge_CoverageIntervalDefs.lean",
    "Challenges/BinaryHeap_Dijkstra/Challenge_BinaryHeap_2.lean",
    "Challenges/BinaryHeap_Dijkstra/Challenge_BinaryHeap_3.lean",
]


def _resolve_experiment(client: Client, experiment_arg: str):
    """Accept either an experiment name or a UUID; return the TracerSession."""
    try:
        exp_uuid = uuid.UUID(experiment_arg)
        return client.read_project(project_id=exp_uuid)
    except ValueError:
        return client.read_project(project_name=experiment_arg)


def _fetch_missing_examples(client: Client, dataset_name: str, paths: list[str]):
    """Return the dataset examples whose inputs.path is in `paths`."""
    all_examples = list(client.list_examples(dataset_name=dataset_name))
    by_path = {ex.inputs.get("path"): ex for ex in all_examples}

    missing = []
    unknown = []
    for p in paths:
        ex = by_path.get(p)
        if ex is None:
            unknown.append(p)
        else:
            missing.append(ex)

    if unknown:
        raise SystemExit(
            f"These paths were not found in dataset '{dataset_name}':\n  - "
            + "\n  - ".join(unknown)
            + f"\n\nDataset paths available: {sorted(by_path)}"
        )
    return missing


def _delete_partial_runs(client: Client, project, paths: list[str], dry_run: bool) -> int:
    """Delete any existing runs in `project` whose inputs.path is one of `paths`."""
    target_paths = set(paths)
    runs = client.list_runs(project_id=project.id, is_root=True)
    to_delete = []
    for run in runs:
        if run.inputs and run.inputs.get("path") in target_paths:
            to_delete.append(run)

    logger.info(
        f"Found {len(to_delete)} existing run(s) for the 4 paths in experiment '{project.name}'."
    )
    for run in to_delete:
        status = "(would delete)" if dry_run else "(deleting)"
        logger.info(
            f"  {status} run_id={run.id}  path={run.inputs.get('path')}  "
            f"status={getattr(run, 'status', '?')}"
        )
        if not dry_run:
            client.delete_run(run_id=run.id)
    return len(to_delete)


async def _resume(
    folder: str,
    config: Config,
    experiment_arg: str,
    max_concurrency: int,
    dry_run: bool,
) -> int:
    client = Client()

    project = _resolve_experiment(client, experiment_arg)
    logger.info(f"Target experiment: name={project.name!r}  id={project.id}")

    examples = _fetch_missing_examples(client, DATASET_NAME, MISSING_PATHS)
    logger.info(f"Resolved {len(examples)} examples to re-run:")
    for ex in examples:
        logger.info(f"  example_id={ex.id}  path={ex.inputs.get('path')}")

    _delete_partial_runs(client, project, MISSING_PATHS, dry_run=dry_run)

    if dry_run:
        logger.info("Dry run — exiting before re-running anything.")
        return 0

    lean_semaphore = Semaphore(config.runtime.lean.max_concurrent_builds)

    @traceable
    async def experiment_func(inputs: dict[str, str]) -> dict:
        return await run_experiment(inputs, config, lean_semaphore, folder)

    def _tool_usage(run: Run) -> dict[str, int]:
        return tool_usage(run, config.prover)

    async with lean_search_session_manager():
        async with lean_interact_session_manager():
            results = await client.aevaluate(
                experiment_func,
                data=examples,
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
                # KEY LINE: extend the existing experiment instead of creating a new one.
                experiment=project,
            )

            await results.wait()

            error_count = 0
            for result in results._results:
                out = result["run"].outputs
                if out and out.get("error") == "exception":
                    error_count += 1
                    logger.error(f"Resume failed for {out.get('path')}: {out.get('message')}")

            if error_count > 0:
                logger.error(f"Resume completed with {error_count} unhandled error(s).")
                return 1

    logger.info(
        f"Resume completed successfully. New runs appended to "
        f"experiment {project.id} ({project.name})."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume interrupted ax-prover experiment on AI4Math_TCS_Challenge.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Existing experiment name OR UUID (from LangSmith URL).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max concurrent provers (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted / re-run, then exit.",
    )
    parser.add_argument(
        "--no-skip-build",
        action="store_true",
        help="Force `lake exe cache get` + `lake build` before re-running.",
    )
    parser.add_argument(
        "--folder",
        default=os.getcwd(),
        help="Base folder for the Lean project (default: cwd).",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=["default.yaml"],
        help="Config YAML file(s), merged in order. Pass the same one(s) you used "
        "in the original run so metadata stays consistent.",
    )
    args, unknown_args = parser.parse_known_args()

    folder = args.folder
    load_env_secrets(folder)
    load_dotenv(os.path.join(folder, ".env.secrets"))  # belt + braces

    config_sources = [Config(), *args.config]
    if unknown_args:
        config_sources.append(unknown_args)
    config = merge_configs(config_sources, folder=folder)
    reconfigure_log_level(config.runtime.log_level)

    if args.no_skip_build:
        logger.info("Prebuilding Lean4 repo...")
        success, output = build_lean_repo(folder, config.runtime.lean)
        logger.debug(output)
        if not success:
            logger.error("Build failed.")
            sys.exit(1)
    else:
        logger.info(
            "Skipping Lean4 repo build (default for resume; pass --no-skip-build to force)."
        )

    exit_code = asyncio.run(
        _resume(
            folder=folder,
            config=config,
            experiment_arg=args.experiment,
            max_concurrency=args.max_concurrency,
            dry_run=args.dry_run,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
