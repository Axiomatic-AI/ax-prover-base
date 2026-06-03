#!/usr/bin/env python3
"""
Replay the 25 good runs from the broken `experiment-5078521c` into a NEW
LangSmith experiment that's properly linked to the current 25-example
`AI4Math_TCS_Challenge` dataset.

Why: the old experiment's `reference_dataset_id` points to a dataset UUID
that no longer exists (the dataset was deleted+recreated by an upload
script run). LangSmith can't re-point an experiment, so the UI view is
broken. But the run outputs themselves are intact.

This script does NOT call the prover. It builds a `path -> cached output`
map from the old project's runs, then calls `client.aevaluate(...)` with a
cached-lookup target. Result: a fresh experiment with 25 runs, same
outputs, freshly-computed evaluator feedback, properly linked to the
current dataset.

Usage (in the ax-prover venv, from anywhere — paths are absolute):
    python /Users/krystian/Documents/Axiomatic/Baku/ax-prover-base/scripts/replay_25_to_new_experiment.py \
        --folder /Users/krystian/Documents/Axiomatic/Baku/AI4Math/challenges
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from langsmith import Client, traceable
from langsmith.schemas import Run

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
from ax_prover.utils import (
    get_logger,
    load_env_secrets,
    merge_configs,
    reconfigure_log_level,
)

logger = get_logger(__name__)

DEFAULT_OLD_EXPERIMENT_ID = "4c8ede95-b8ca-4855-a982-d463c2a5fa36"
TARGET_DATASET = "AI4Math_TCS_Challenge"


def _extract_path(inputs) -> str | None:
    """Pull a `path` field out of a run's inputs.

    LangSmith stores the @traceable wrapper's input under a nested
    'inputs' key, so the real shape is `{"inputs": {"path": "..."}}`.
    Also tolerate the flat shape just in case.
    """
    if not isinstance(inputs, dict):
        return None
    if isinstance(inputs.get("path"), str):
        return inputs["path"]
    nested = inputs.get("inputs")
    if isinstance(nested, dict) and isinstance(nested.get("path"), str):
        return nested["path"]
    return None


def _build_cache(client: Client, old_project_id: str) -> dict[str, dict]:
    """Build {path -> best cached output} from the old project's runs.

    'Best' = status=success + outputs look like a real ProverAgentState
    (has an 'item' key with sub-keys), not just {"output": null} or
    {"error": "exception", ...}.
    """
    runs = list(client.list_runs(project_id=old_project_id, is_root=True))
    logger.info(f"Pulled {len(runs)} root runs from old project {old_project_id}.")

    by_path: dict[str, list[Run]] = {}
    no_path = 0
    for r in runs:
        path = _extract_path(r.inputs)
        if not path:
            no_path += 1
            continue
        by_path.setdefault(path, []).append(r)

    logger.info(
        f"Grouped runs across {len(by_path)} unique paths "
        f"({no_path} run(s) had no extractable path)."
    )

    def _quality(run: Run) -> tuple[int, int, int, int]:
        out = run.outputs or {}
        if not isinstance(out, dict):
            return (0, 0, 0, 0)
        has_item = isinstance(out.get("item"), dict)
        # Prefer outputs whose item.proven == True (actually solved).
        proven = bool(has_item and out["item"].get("proven"))
        is_success = getattr(run, "status", None) == "success"
        not_error_dict = out.get("error") != "exception"
        return (int(has_item), int(proven), int(is_success), int(not_error_dict))

    cache: dict[str, dict] = {}
    skipped = []
    for path, candidates in by_path.items():
        best = max(candidates, key=_quality)
        q = _quality(best)
        if q[0] == 0:  # no candidate has a real `item` — nothing usable
            skipped.append((path, q, [getattr(r, "status", None) for r in candidates]))
            continue
        cache[path] = best.outputs

    logger.info(f"Built cache for {len(cache)} unique paths.")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} path(s) with no usable cached output:")
        for p, q, statuses in skipped:
            logger.warning(f"  - {p}  quality={q}  candidate_statuses={statuses}")
    return cache


async def _main(
    folder: str, old_experiment: str, experiment_prefix: str, config: Config, max_concurrency: int
) -> int:
    client = Client()

    try:
        old_project = client.read_project(project_id=uuid.UUID(old_experiment))
    except ValueError:
        old_project = client.read_project(project_name=old_experiment)
    logger.info(f"Old (broken) experiment: name={old_project.name!r}  id={old_project.id}")

    cache = _build_cache(client, str(old_project.id))

    # Sanity: every example in the current dataset must have a cached output.
    examples = list(client.list_examples(dataset_name=TARGET_DATASET))
    logger.info(f"Target dataset '{TARGET_DATASET}' has {len(examples)} examples.")
    missing = [ex.inputs.get("path") for ex in examples if ex.inputs.get("path") not in cache]
    if missing:
        logger.error(f"No cached output found for {len(missing)} example path(s):")
        for p in missing:
            logger.error(f"  - {p}")
        logger.error("Aborting before creating a half-populated experiment.")
        return 1

    @traceable
    async def cached_target(inputs: dict[str, str]) -> dict:
        path = inputs.get("path")
        if path in cache:
            return cache[path]
        return {"error": "missing_cached_output", "path": path}

    def _tool_usage(run: Run) -> dict[str, int]:
        return tool_usage(run, config.prover)

    logger.info(
        f"Replaying {len(cache)} cached runs into a new experiment "
        f"(prefix={experiment_prefix!r}, dataset={TARGET_DATASET!r})..."
    )

    results = await client.aevaluate(
        cached_target,
        data=TARGET_DATASET,
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
        metadata={
            "replayed_from_experiment_id": str(old_project.id),
            "replayed_from_experiment_name": old_project.name,
            "note": "Cached-output replay; prover was not re-run.",
        },
    )
    await results.wait()

    logger.info(f"Replay done. New experiment name: {results.experiment_name}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-experiment",
        default=DEFAULT_OLD_EXPERIMENT_ID,
        help=f"Old experiment name or UUID to pull cached outputs from "
        f"(default: {DEFAULT_OLD_EXPERIMENT_ID}).",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="ai4math_25_replay",
        help="Prefix for the new experiment (default: ai4math_25_replay).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent evaluator runs (default: 8). Cached lookups "
        "are cheap, so higher is fine.",
    )
    parser.add_argument(
        "--folder",
        default=os.getcwd(),
        help="Base folder for the project (default: cwd). Used for .env.secrets + Config loading.",
    )
    parser.add_argument(
        "-c", "--config", action="append", default=["default.yaml"], help="Config YAML file(s)."
    )
    args, unknown_args = parser.parse_known_args()

    folder = args.folder
    load_env_secrets(folder)
    load_dotenv(os.path.join(folder, ".env.secrets"))

    config_sources = [Config(), *args.config]
    if unknown_args:
        config_sources.append(unknown_args)
    config = merge_configs(config_sources, folder=folder)
    reconfigure_log_level(config.runtime.log_level)

    exit_code = asyncio.run(
        _main(
            folder=folder,
            old_experiment=args.old_experiment,
            experiment_prefix=args.experiment_prefix,
            config=config,
            max_concurrency=args.max_concurrency,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
