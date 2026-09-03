"""Final Comparator acceptance gate.

Comparator is a judge, not a node-level tool: it runs once, on the fully assembled proof.
It needs `landrun` (Linux-only), so on macOS a successful full Lean build yields
`comparator_pending` and Linux CI remains authoritative. The `comparator` and
`lean4export` binaries are provisioned on demand when absent from PATH, with lean4export
built at the target project's own toolchain version (see `provision.py`).

The challenge and solution modules both reproduce the target file's trusted prefix, so the
target's statement elaborates identically in each; Comparator then checks that the
solution proves that same statement using no more than the permitted axioms.

This adapter is written against Comparator's documented JSON contract. It has not been
exercised against a real Comparator installation, which requires Linux plus `landrun` and
`lean4export`; any failure is reported as an infrastructure error rather than a rejection.
"""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import ComparatorConfig
from ..utils.build import run_lean_subprocess
from ..utils.logging import get_logger
from .models import Blueprint, ComparatorStatus
from .provision import (
    ProvisionError,
    ensure_comparator,
    ensure_lean4export,
    read_project_toolchain,
)
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)


@dataclass(frozen=True)
class ComparatorReport:
    """Outcome of the final Comparator gate."""

    status: ComparatorStatus
    detail: str = ""
    output: str = ""


def unavailable_reason(config: ComparatorConfig) -> str | None:
    """Why Comparator cannot run here, or None when it can.

    Only the true system prerequisites are checked: Linux and `landrun` (respected via the
    documented `COMPARATOR_LANDRUN` override too, so PATH absence is best-effort). The
    `comparator` and `lean4export` binaries are provisioned on demand, matched to the
    target project's toolchain, when they are not already on PATH.
    """
    if not sys.platform.startswith("linux"):
        return f"Comparator requires Linux (landrun); this host is {sys.platform}"

    if shutil.which("landrun") is None and not os.environ.get("COMPARATOR_LANDRUN"):
        return "landrun not found on PATH (and COMPARATOR_LANDRUN is unset)"

    return None


async def _resolve_binaries(
    config: ComparatorConfig, base_folder: str
) -> tuple[str, dict[str, str] | None]:
    """The comparator binary to invoke, plus the env override when lean4export was built.

    A PATH installation always wins; otherwise the binaries are cloned and built into the
    per-version provision cache (see `provision.py`).

    Raises:
        ProvisionError: A needed binary is neither installed nor buildable.
    """
    binary = shutil.which(config.binary)
    if binary is None:
        toolchain = read_project_toolchain(base_folder)
        binary = str(await ensure_comparator(toolchain))

    env = None
    if shutil.which("lean4export") is None and not os.environ.get("COMPARATOR_LEAN4EXPORT"):
        toolchain = read_project_toolchain(base_folder)
        lean4export = await ensure_lean4export(toolchain)
        # Older comparator releases resolve lean4export from PATH only (the
        # COMPARATOR_LEAN4EXPORT override is newer), so provide both.
        env = {
            **os.environ,
            "COMPARATOR_LEAN4EXPORT": str(lean4export),
            "PATH": f"{lean4export.parent}{os.pathsep}{os.environ.get('PATH', '')}",
        }

    return binary, env


def render_challenge(workspace: BlueprintWorkspace) -> str:
    """Render the challenge module: the trusted prefix plus the sorried target."""
    blocks = [
        workspace.prefix.rstrip(),
        f"{workspace.target_statement.rstrip()} := sorry",
    ]
    return "\n\n".join(block for block in blocks if block.strip()) + "\n"


def render_solution(workspace: BlueprintWorkspace, helper_block: str, target_body: str) -> str:
    """Render the solution module: the same prefix, the helpers, and the proven target."""
    blocks = [
        workspace.prefix.rstrip(),
        helper_block.strip(),
        f"{workspace.target_statement.rstrip()} := {target_body.strip()}",
    ]
    return "\n\n".join(block for block in blocks if block.strip()) + "\n"


async def run_comparator(
    workspace: BlueprintWorkspace,
    blueprint: Blueprint,
    config: ComparatorConfig,
    helper_block: str,
    target_body: str,
) -> ComparatorReport:
    """Run Comparator on the assembled proof and report its verdict.

    Returns:
        `PASSED` when Comparator accepts the target, `REJECTED` when it does not, and
        `PENDING` when Comparator is unavailable on this host.
    """
    reason = unavailable_reason(config)
    if reason is not None:
        logger.warning(f"Skipping Comparator: {reason}")
        return ComparatorReport(status=ComparatorStatus.PENDING, detail=reason)

    try:
        binary, env = await _resolve_binaries(config, workspace.base_folder)
    except ProvisionError as e:
        logger.warning(f"Skipping Comparator: {e}")
        return ComparatorReport(status=ComparatorStatus.PENDING, detail=str(e))

    # The modules live beside the target file so Lake resolves them under the same
    # library as the target itself: `lake build` addresses any module inside a declared
    # lib's tree, while a root-level module belongs to no target at all (measured:
    # "error: unknown target `AxProverComparatorChallenge`").
    #
    # The names carry the target's generated-namespace digest because concurrent
    # experiment samples share one project directory: a fixed name had ten samples
    # writing, building, and deleting the same two modules, which surfaced as "Constant
    # <target> not found in environment" and "no such file or directory" - and those were
    # reported as proof rejections.
    directory = workspace.file_path.parent
    rel = directory.relative_to(Path(workspace.base_folder))
    stem = f"{config.module_prefix}_{workspace.namespace}"
    dotted = ".".join((*rel.parts, stem))
    challenge_module = f"{dotted}Challenge"
    solution_module = f"{dotted}Solution"

    paths = {
        directory / f"{stem}Challenge.lean": render_challenge(workspace),
        directory / f"{stem}Solution.lean": render_solution(workspace, helper_block, target_body),
    }
    config_path = directory / f"{stem}Config.json"

    payload = {
        "challenge_module": challenge_module,
        "solution_module": solution_module,
        "theorem_names": [blueprint.target.lean_name],
        "permitted_axioms": list(config.permitted_axioms),
    }

    try:
        for path, content in paths.items():
            path.write_text(content, encoding="utf-8")
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info(f"Running Comparator on {blueprint.target.lean_name}")
        returncode, stdout, stderr = await run_lean_subprocess(
            ["lake", "env", binary, str(config_path)],
            cwd=workspace.base_folder,
            timeout=config.timeout,
            env=env,
        )
        output = (stdout + stderr).strip()
    except Exception as e:  # noqa: BLE001 - any failure here is infrastructure, not a verdict
        logger.error(f"Comparator invocation failed: {e}")
        return ComparatorReport(status=ComparatorStatus.PENDING, detail=str(e))
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)

    if returncode == 0:
        return ComparatorReport(status=ComparatorStatus.PASSED, output=output)

    # A rejection is a verdict about the proof; a sandbox or binary-resolution failure is
    # not. A run where landrun could not exec lean4export was reported as REJECTED and
    # failed a sample whose every node had been proven.
    if "landrun:error" in output or "executable file not found" in output:
        return ComparatorReport(
            status=ComparatorStatus.PENDING,
            detail=f"Comparator infrastructure failure (exit {returncode})",
            output=output,
        )

    return ComparatorReport(
        status=ComparatorStatus.REJECTED,
        detail=f"Comparator exited with code {returncode}",
        output=output,
    )
