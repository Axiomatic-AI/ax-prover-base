"""Final Comparator acceptance gate.

Comparator is a judge, not a node-level tool: it runs once, on the fully assembled proof.
It needs `landrun` (Linux-only) and `lean4export` on PATH, so on macOS a successful full
Lean build yields `comparator_pending` and Linux CI remains authoritative.

The challenge and solution modules both reproduce the target file's trusted prefix, so the
target's statement elaborates identically in each; Comparator then checks that the
solution proves that same statement using no more than the permitted axioms.

This adapter is written against Comparator's documented JSON contract. It has not been
exercised against a real Comparator installation, which requires Linux plus `landrun` and
`lean4export`; any failure is reported as an infrastructure error rather than a rejection.
"""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import ComparatorConfig
from ..utils.build import run_lean_subprocess
from ..utils.logging import get_logger
from .models import Blueprint, ComparatorStatus
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)

#: Binaries Comparator itself requires. Their paths may be overridden with the documented
#: COMPARATOR_* environment variables, which is why a missing binary is only a hint.
REQUIRED_BINARIES = ("landrun", "lean4export")


@dataclass(frozen=True)
class ComparatorReport:
    """Outcome of the final Comparator gate."""

    status: ComparatorStatus
    detail: str = ""
    output: str = ""


def unavailable_reason(config: ComparatorConfig) -> str | None:
    """Why Comparator cannot run here, or None when it can.

    Missing binaries are only checked on PATH; Comparator also accepts the
    `COMPARATOR_LANDRUN` / `COMPARATOR_LEAN4EXPORT` overrides, so this is a best-effort
    precheck rather than a guarantee.
    """
    if not sys.platform.startswith("linux"):
        return f"Comparator requires Linux (landrun); this host is {sys.platform}"

    if shutil.which(config.binary) is None:
        return f"Comparator binary {config.binary!r} not found on PATH"

    missing = [name for name in REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        return f"Comparator dependencies not found on PATH: {', '.join(missing)}"

    return None


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

    challenge_module = f"{config.module_prefix}Challenge"
    solution_module = f"{config.module_prefix}Solution"

    # Root-level modules so `import`-free, self-contained challenge and solution files
    # resolve under the project's default Lake target.
    root = Path(workspace.base_folder)
    paths = {
        root / f"{challenge_module}.lean": render_challenge(workspace),
        root / f"{solution_module}.lean": render_solution(workspace, helper_block, target_body),
    }
    config_path = root / f"{config.module_prefix}Config.json"

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
            ["lake", "env", config.binary, str(config_path)],
            cwd=workspace.base_folder,
            timeout=config.timeout,
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

    return ComparatorReport(
        status=ComparatorStatus.REJECTED,
        detail=f"Comparator exited with code {returncode}",
        output=output,
    )
