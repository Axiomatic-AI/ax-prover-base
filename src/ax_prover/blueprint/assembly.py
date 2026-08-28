"""Deterministic same-file assembly and the single atomic source edit.

Helpers are rendered in topological order inside the run's unique namespace, placed
immediately before the original target, which keeps its own docstring and statement
verbatim. Harness-only blueprint metadata never reaches the user's file.
"""

import os
import re
from pathlib import Path

from ..utils.logging import get_logger
from .graph import required_nodes, topological_order
from .models import Blueprint, BlueprintError, BlueprintNode
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)

#: Placeholders and cheating tactics that must never survive into the user's source.
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?<![\w.])sorry(?![\w])", "sorry"),
    (r"(?<![\w.])admit(?![\w])", "admit"),
    (r"(?<![\w.])sorryAx(?![\w])", "sorryAx"),
    (r"^\s*axiom\s", "axiom declaration"),
    (r"(?<![\w.])native_decide(?![\w])", "native_decide"),
    (r"tmp_bp_", "scratch file placeholder"),
)


class AssemblyError(BlueprintError):
    """The assembled proof failed a pre-commit safety check."""


def render_helper(node: BlueprintNode, proof_body: str) -> str:
    """Render one helper with its stored proof, keeping only human-readable prose."""
    declaration = f"{node.statement_source_no_doc.rstrip()} := {proof_body.strip()}"
    if not node.doc_text.strip():
        return declaration
    return f"/--\n{node.doc_text.strip()}\n-/\n{declaration}"


def render_helper_block(
    workspace: BlueprintWorkspace, blueprint: Blueprint, proofs: dict[str, str]
) -> str:
    """Render every helper, parents first, inside the deterministic namespace.

    Raises:
        AssemblyError: A helper has no stored proof.
    """
    required = required_nodes(blueprint)
    helpers = [
        node for node in topological_order(blueprint) if not node.is_target and node.id in required
    ]
    missing = [node.id for node in helpers if node.id not in proofs]
    if missing:
        raise AssemblyError(f"missing proofs for helpers: {', '.join(sorted(missing))}")

    rendered = "\n\n".join(render_helper(node, proofs[node.id]) for node in helpers)
    return workspace.render_helper_block(rendered)


def render_assembly(
    workspace: BlueprintWorkspace, blueprint: Blueprint, proofs: dict[str, str]
) -> str:
    """Render the complete candidate file with helpers and the proven target.

    Raises:
        AssemblyError: A required proof is missing.
    """
    target = blueprint.target
    if target.id not in proofs:
        raise AssemblyError("missing proof for the target node")

    helper_block = render_helper_block(workspace, blueprint, proofs)
    target_block = (
        f"{workspace.target_statement_with_doc.rstrip()} := {proofs[target.id].strip()}\n"
    )
    return workspace.render_final_file(helper_block, target_block)


def check_generated_region(workspace: BlueprintWorkspace, assembled: str) -> list[str]:
    """Find forbidden patterns in the generated region of an assembled file.

    Only the region the harness wrote is checked; unrelated `sorry`s elsewhere in the
    user's file are none of this run's business.
    """
    start = len(workspace.prefix.rstrip())
    end = len(assembled) - len(workspace.suffix.strip()) if workspace.suffix.strip() else None
    region = assembled[start:end]

    problems = []
    for pattern, label in FORBIDDEN_PATTERNS:
        if re.search(pattern, region, re.MULTILINE):
            problems.append(f"assembled proof still contains {label}")
    return problems


def commit_source(workspace: BlueprintWorkspace, assembled: str) -> None:
    """Replace the target file with `assembled` in one atomic write.

    Raises:
        AssemblyError: The file changed since run start, so a concurrent edit would be
            overwritten.
    """
    if not workspace.source_unchanged():
        raise AssemblyError(
            f"{workspace.file_path} changed during the run; refusing to overwrite it. "
            "Completed artifacts are preserved in the checkpoint."
        )

    path = workspace.file_path
    temporary = path.with_name(f".{path.name}.axprover.tmp")
    temporary.write_text(assembled, encoding="utf-8")
    os.replace(temporary, path)
    logger.info(f"Wrote assembled proof to {path}")


def artifact_path(artifacts_dir: str | Path, name: str) -> Path:
    """Path for a run artifact, creating the directory on demand."""
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name
