"""Shared fixtures for regression tests."""

import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_fixture_project(tmp_path_factory, name: str) -> str:
    """Copy a Lean fixture to a tmp dir, build it once, and return its path.

    Skips the test if `lake` is not on PATH or the build fails so the suite stays
    usable on machines without Lean installed (CI gates this separately).
    """
    if shutil.which("lake") is None:
        pytest.skip("lake not on PATH")

    project = tmp_path_factory.mktemp(name)
    for item in (FIXTURES / name).iterdir():
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


@pytest.fixture(scope="session")
def lean_minimal_project(tmp_path_factory):
    """A built copy of the minimal Lean fixture."""
    return _build_fixture_project(tmp_path_factory, "lean_minimal")


@pytest.fixture(scope="session")
def lean_blueprint_project(tmp_path_factory):
    """A built copy of the blueprint Lean fixture, whose target still has a `sorry`."""
    return _build_fixture_project(tmp_path_factory, "lean_blueprint")


#: Session-scoped project fixtures whose Lean sources a test may legitimately rewrite.
_MUTABLE_PROJECT_FIXTURES = ("lean_blueprint_project", "lean_minimal_project")


@pytest.fixture(autouse=True)
def restore_lean_sources(request):
    """Snapshot and restore a fixture project's Lean sources around each test.

    A successful blueprint run rewrites its target file by design, and the built project is
    session-scoped because `lake build` is slow. Restoring here keeps tests independent of
    each other and of ordering, including when one fails partway through.
    """
    in_use = [name for name in _MUTABLE_PROJECT_FIXTURES if name in request.fixturenames]
    if not in_use:
        yield
        return

    roots = [Path(request.getfixturevalue(name)) for name in in_use]
    snapshot = {
        path: path.read_bytes()
        for root in roots
        for path in root.rglob("*.lean")
        if ".lake" not in path.parts
    }

    yield

    for path, content in snapshot.items():
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
    # Remove any .lean file a test created that the snapshot does not know about.
    for root in roots:
        for path in root.rglob("*.lean"):
            if ".lake" not in path.parts and path not in snapshot:
                path.unlink()
