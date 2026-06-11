"""Create a LangSmith dataset from a whitespace-separated list of Putnam problem names.

Each problem name (e.g. `putnam_1962_a1`) becomes one example with input
`{"path": "putnam_1962_a1:putnam_1962_a1"}` — the format expected by
`ax_prover.commands.experiment.run_experiment`.

Usage:
    python scripts/create_putnam_dataset.py NAMES_FILE [--dataset-name NAME]
"""

import argparse
import re
import sys
from pathlib import Path

from langsmith import Client


PUTNAM_NAME_RE = re.compile(r"\bputnam_\d{4}_[ab]\d+\b")


def load_names(path: Path) -> list[str]:
    text = path.read_text()
    return list(dict.fromkeys(PUTNAM_NAME_RE.findall(text)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names_file", type=Path, help="File containing Putnam problem names")
    parser.add_argument(
        "--dataset-name",
        default="putnam_100_tool_ablation",
        help="LangSmith dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N problems from the file (default: all)",
    )
    args = parser.parse_args()

    names = load_names(args.names_file)
    if not names:
        print(f"No `putnam_YYYY_xN` names found in {args.names_file}", file=sys.stderr)
        return 1
    if args.limit is not None:
        names = names[: args.limit]
    print(f"Loaded {len(names)} problems from {args.names_file}")

    client = Client()

    existing = list(client.list_datasets(dataset_name=args.dataset_name))
    if existing:
        print(f"Dataset '{args.dataset_name}' already exists (id={existing[0].id}). Aborting.")
        return 1

    dataset = client.create_dataset(
        dataset_name=args.dataset_name,
        description=f"Putnam tool ablation set ({len(names)} problems)",
    )
    client.create_examples(
        inputs=[{"path": f"src/{n}:{n}"} for n in names],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{args.dataset_name}' (id={dataset.id}) with {len(names)} examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
