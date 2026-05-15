"""Shared fixtures for regression tests."""

import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURE_SRC = Path(__file__).parent / "fixtures" / "lean_minimal"


@pytest.fixture(scope="session")
def lean_minimal_project(tmp_path_factory):
    """Copy the minimal Lean fixture to a tmp dir, build it once, return its path.

    Skips the test if `lake` is not on PATH or the build fails so the suite stays
    usable on machines without Lean installed (CI gates this separately).
    """
    if shutil.which("lake") is None:
        pytest.skip("lake not on PATH")

    project = tmp_path_factory.mktemp("lean_minimal")
    for item in FIXTURE_SRC.iterdir():
        if item.name in {".lake", "build", "lake-manifest.json"}:
            continue
        dest = project / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    result = subprocess.run(
        ["lake", "build"], cwd=project, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        pytest.skip(f"`lake build` failed in fixture project:\n{result.stderr}")

    return str(project)
