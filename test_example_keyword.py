"""Smoke test: ax-prover prove on a Lean `example` declaration."""

import subprocess
import sys

FOLDER = "/private/tmp/test_example_lean"
TARGET = "TestExample.Basic:example"

result = subprocess.run(
    ["ax-prover", "prove", TARGET, "--folder", FOLDER, "--skip-build"],
    cwd="/Users/leopoldo/local/ax-prover-base",
)
sys.exit(result.returncode)
