#!/usr/bin/env python
"""Command line interface for ax-prover."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .commands import experiment, prove
from .config import Config
from .utils import get_logger, load_env_secrets, merge_configs, reconfigure_log_level, save_config
from .utils.build import build_lean_repo

logger = get_logger(__name__)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Ax-Prover — Lean 4 theorem proving agent",
        prog="ax-prover",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Use --config to specify YAML config files (must come BEFORE subcommand)
  Use dot-notation to override config values: key.subkey=value
  Example: ax-prover --config custom.yaml experiment experiment-name

Examples:
  # Prove a specific theorem by location (module path)
  ax-prover prove QuantumLib.Operators:my_theorem

  # Prove a specific theorem by location (file path)
  ax-prover prove MyProject/Algebra/Ring.lean:my_theorem

  # Prove all unproven functions in a file
  ax-prover prove MyProject/Algebra/Ring.lean

  # Prove a theorem in a specific project
  ax-prover prove MyProject.Algebra:ring_lemma --folder /path/to/project

  # Force re-proving even if already proven
  ax-prover prove MyProject/Algebra/Ring.lean:my_theorem --overwrite

  # Skip the lake build step before proving
  ax-prover prove example.md --skip-build

  # Prove the theorem at a specific line in a file
  ax-prover prove MyProject/Algebra/Ring.lean#L42

  # Run an experiment on a dataset
  ax-prover experiment dataset_name

  # Run an experiment on a dataset (by ID)
  ax-prover experiment 8c0f8560-34b2-411c-8cc1-cc8aaa624e05

  # Run an experiment with custom concurrency and prefix
  ax-prover experiment dataset_name --max-concurrency 8 --experiment-prefix my_experiment
""",
    )

    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=["default.yaml"],
        help=(
            "Configuration YAML file (can be used multiple times, merged in order). "
            "Must be specified BEFORE the subcommand. "
            "Example: ax-prover --config my_config.yaml plan example.md"
        ),
    )
    parser.add_argument(
        "--save-config",
        type=str,
        metavar="NAME",
        default=None,
        help=(
            "Save the merged configuration to .axiomatic/NAME.yaml (relative to --folder). "
            "Must be specified BEFORE the subcommand. "
            "Example: ax-prover --save-config my_config plan example.md"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("configure", help="Interactively set up API keys")

    prove_parser = subparsers.add_parser(
        "prove", help="Prove theorems by location, or all unproven in a file"
    )
    prove_parser.add_argument(
        "target",
        help=(
            "Target to prove. Formats:\n"
            "  - Theorem location: Module.Path:theorem_name\n"
            "  - File/module: Module.Path or path/to/file.lean (all unproven)\n"
            "  - Line number: path/to/file.lean#L42 (theorem at line)"
        ),
    )
    prove_parser.add_argument(
        "--folder",
        default=os.getcwd(),
        help="Base folder for the project (default: current directory)",
    )
    prove_parser.add_argument(
        "--overwrite", action="store_true", help="Re-prove even if already proven"
    )
    prove_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip running 'lake exe cache get' and 'lake build' before proving",
    )
    prove_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Write JSON output to file",
    )

    experiment_parser = subparsers.add_parser(
        "experiment", help="Run prover experiments on a LangSmith dataset"
    )
    experiment_parser.add_argument(
        "dataset",
        help="Dataset name (e.g., quantum_v0) or dataset ID (UUID)",
    )
    experiment_parser.add_argument(
        "--folder",
        default=os.getcwd(),
        help="Base folder for the project (default: current directory)",
    )
    experiment_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent experiments (default: 4)",
    )
    experiment_parser.add_argument(
        "--experiment-prefix",
        type=str,
        default=None,
        help="Prefix for the experiment name (default: <dataset_name>_experiment)",
    )
    experiment_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip running 'lake exe cache get' and 'lake build' before proving",
    )
    experiment_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Write JSON output to file",
    )

    # Parse known args to allow dot-notation overrides as unknown args
    args, unknown_args = parser.parse_known_args()

    # Handle configure command before loading configs (it doesn't need them)
    if args.command == "configure":
        from .commands.configure import configure

        configure()
        sys.exit(0)

    folder = getattr(args, "folder", None)
    load_env_secrets(folder)

    config_sources = [Config(), *args.config]
    if unknown_args:
        config_sources.append(unknown_args)
    config = merge_configs(config_sources, folder=folder)
    reconfigure_log_level(config.runtime.log_level)

    if args.save_config:
        folder = getattr(args, "folder", os.getcwd())
        config_path = Path(folder) / ".axiomatic" / f"{args.save_config}.yaml"
        save_config(config, config_path)
        logger.info(f"Saved configuration to: {config_path}")

    skip_build = getattr(args, "skip_build", False)
    if args.command and not skip_build:
        folder = getattr(args, "folder", os.getcwd())
        logger.info("Prebuilding Lean4 repo...")
        success, output = build_lean_repo(folder, config.runtime.lean)
        logger.debug(output)
        if not success:
            logger.error("Build failed. Check your Lean4 installation and repo config.")
            sys.exit(1)
    else:
        logger.info("Skipping Lean4 repo build (--skip-build flag set)")

    if args.command == "prove":
        folder = args.folder
        target = args.target
        overwrite = args.overwrite
        output_file = args.output
        exit_code = asyncio.run(
            prove(
                folder,
                target,
                config,
                overwrite=overwrite,
                output_file=output_file,
            )
        )
        sys.exit(exit_code)
    elif args.command == "experiment":
        folder = args.folder
        dataset = args.dataset
        max_concurrency = args.max_concurrency
        experiment_prefix = args.experiment_prefix

        output_file = args.output

        exit_code = asyncio.run(
            experiment(
                folder,
                dataset,
                config,
                max_concurrency=max_concurrency,
                experiment_prefix=experiment_prefix,
                output_file=output_file,
            )
        )
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
