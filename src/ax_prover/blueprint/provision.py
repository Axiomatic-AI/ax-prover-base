"""Provision Comparator's toolchain-matched binaries on demand.

`lean4export` must match the *target project's* Lean version, and its repo tags one
release per toolchain (v4.15.0 through current), named exactly like the toolchain. So
instead of requiring a hand-installed binary per project, the right version is cloned and
built once into a per-version cache directory and reused; a second project on a different
toolchain gets its own build. The `comparator` binary is toolchain-independent (it builds
on its own pinned toolchain) and is cached once.

`landrun` stays a system prerequisite: it is a Go binary with no Lean version coupling.
"""

import asyncio
from pathlib import Path

from platformdirs import user_cache_dir

from ..utils.build import run_lean_subprocess
from ..utils.logging import get_logger

logger = get_logger(__name__)

LEAN4EXPORT_REPO = "https://github.com/leanprover/lean4export"
COMPARATOR_REPO = "https://github.com/leanprover/comparator"

#: Builds land here, one directory per tool+version, and are reused across runs.
CACHE_ROOT = Path(user_cache_dir("ax-prover")) / "provision"

#: First build fetches a toolchain and compiles; generous by design.
_BUILD_TIMEOUT_SECONDS = 1800.0

#: Serializes builds within one process; concurrent experiment samples share the result.
_build_lock = asyncio.Lock()


class ProvisionError(RuntimeError):
    """A binary could not be provisioned; the caller reports it, never crashes the run."""


def read_project_toolchain(base_folder: str) -> str:
    """The target project's toolchain version tag, e.g. 'v4.24.0'.

    Raises:
        ProvisionError: The project has no readable `lean-toolchain` file.
    """
    path = Path(base_folder) / "lean-toolchain"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ProvisionError(f"cannot read {path}: {e}") from e
    if not text:
        raise ProvisionError(f"{path} is empty")
    # "leanprover/lean4:v4.24.0" -> "v4.24.0"; a bare version passes through.
    return text.split(":", 1)[-1]


async def _run(command: list[str], cwd: Path) -> None:
    returncode, stdout, stderr = await run_lean_subprocess(
        command, cwd=str(cwd), timeout=_BUILD_TIMEOUT_SECONDS
    )
    if returncode != 0:
        detail = (stderr or stdout).strip()[-2000:]
        raise ProvisionError(f"`{' '.join(command)}` exited with {returncode}:\n{detail}")


async def _clone_and_build(repo: str, ref: str | None, dest: Path, target: str) -> Path:
    """Clone `repo` (at `ref` if given) into `dest`, `lake build target`, return the binary."""
    binary = dest / ".lake" / "build" / "bin" / target

    async with _build_lock:
        if binary.exists():
            return binary

        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            branch = ["--branch", ref] if ref else []
            logger.info(f"Provisioning {target}: cloning {repo}" + (f" at {ref}" if ref else ""))
            await _run(["git", "clone", "--depth", "1", *branch, repo, str(dest)], cwd=dest.parent)

        logger.info(f"Provisioning {target}: building in {dest} (first build fetches a toolchain)")
        await _run(["lake", "build", target], cwd=dest)

        if not binary.exists():
            raise ProvisionError(f"build succeeded but {binary} does not exist")
        return binary


async def ensure_lean4export(toolchain_version: str) -> Path:
    """The lean4export binary matching `toolchain_version`, building it on first use.

    Raises:
        ProvisionError: No release tag matches the toolchain, or the build failed.
    """
    dest = CACHE_ROOT / f"lean4export-{toolchain_version}"
    try:
        return await _clone_and_build(LEAN4EXPORT_REPO, toolchain_version, dest, "lean4export")
    except ProvisionError as e:
        if "Remote branch" in str(e) or "not found" in str(e):
            raise ProvisionError(
                f"lean4export has no release for toolchain {toolchain_version}: {e}"
            ) from e
        raise


async def ensure_comparator() -> Path:
    """The comparator binary, building it on first use.

    Raises:
        ProvisionError: The clone or build failed.
    """
    return await _clone_and_build(COMPARATOR_REPO, None, CACHE_ROOT / "comparator", "comparator")
