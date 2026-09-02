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
import re
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


#: Comparator invokes `lean4export <module> -- <decls>` under landrun, and landrun
#: swallows the bare `--` (measured: `landrun ... echo a -- b` prints `a b`), so the
#: declaration names arrive as extra module arguments and the export dies with "unknown
#: module prefix". The shim reinserts the separator, and passes through untouched if a
#: fixed landrun ever preserves it.
_LEAN4EXPORT_SHIM = """\
#!/bin/sh
if [ "$2" = "--" ]; then
  exec "{real}" "$@"
fi
first="$1"
shift
exec "{real}" "$first" -- "$@"
"""


def write_lean4export_shim(real: Path, directory: Path) -> Path:
    """Write the `--`-restoring wrapper for `real` into `directory` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "lean4export"
    shim.write_text(_LEAN4EXPORT_SHIM.format(real=real), encoding="utf-8")
    shim.chmod(0o755)
    return shim


async def ensure_lean4export(toolchain_version: str) -> Path:
    """A lean4export for `toolchain_version`, built on first use, wrapped in the shim.

    Raises:
        ProvisionError: No release tag matches the toolchain, or the build failed.
    """
    dest = CACHE_ROOT / f"lean4export-{toolchain_version}"
    try:
        binary = await _clone_and_build(LEAN4EXPORT_REPO, toolchain_version, dest, "lean4export")
    except ProvisionError as e:
        if "Remote branch" in str(e) or "not found" in str(e):
            raise ProvisionError(
                f"lean4export has no release for toolchain {toolchain_version}: {e}"
            ) from e
        raise
    return write_lean4export_shim(binary, dest / "shim")


_VERSION_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-rc(\d+))?$")


def _version_key(tag: str) -> tuple | None:
    """Sortable key for a `vX.Y.Z[-rcN]` tag; an rc orders below its final release."""
    match = _VERSION_TAG.match(tag)
    if match is None:
        return None
    major, minor, patch, rc = match.groups()
    return (int(major), int(minor), int(patch), (0, int(rc)) if rc else (1, 0))


def select_comparator_tag(tags: list[str], toolchain_version: str) -> str:
    """The comparator release matched to the target toolchain.

    Comparator's requested export constants grow with Lean core (its current Main.lean
    asks for `String.ofList`, which a v4.24.0 environment lacks, and lean4export panics),
    so the newest tag at or below the toolchain is the safe pick. A project older than
    every release gets the oldest one, the closest era available.

    Raises:
        ProvisionError: No version-shaped tags exist, or the toolchain is unparseable.
    """
    want = _version_key(toolchain_version)
    if want is None:
        raise ProvisionError(f"unrecognized toolchain version {toolchain_version!r}")

    versioned = sorted((key, tag) for tag in tags if (key := _version_key(tag)) is not None)
    if not versioned:
        raise ProvisionError(f"comparator has no version tags among {tags!r}")

    at_or_below = [tag for key, tag in versioned if key <= want]
    return at_or_below[-1] if at_or_below else versioned[0][1]


async def _list_remote_tags(repo: str) -> list[str]:
    returncode, stdout, stderr = await run_lean_subprocess(
        ["git", "ls-remote", "--tags", repo], cwd=str(Path.home()), timeout=120.0
    )
    if returncode != 0:
        raise ProvisionError(f"`git ls-remote --tags {repo}` failed: {stderr.strip()[-500:]}")
    tags = set()
    for line in stdout.splitlines():
        _, _, ref = line.partition("refs/tags/")
        if ref:
            tags.add(ref.removesuffix("^{}"))
    return sorted(tags)


async def ensure_comparator(toolchain_version: str) -> Path:
    """The comparator binary era-matched to `toolchain_version`, building it on first use.

    Raises:
        ProvisionError: No usable release, or the clone or build failed.
    """
    tags = await _list_remote_tags(COMPARATOR_REPO)
    tag = select_comparator_tag(tags, toolchain_version)
    dest = CACHE_ROOT / f"comparator-{tag}"
    return await _clone_and_build(COMPARATOR_REPO, tag, dest, "comparator")
