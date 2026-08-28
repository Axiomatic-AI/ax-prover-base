"""Blueprint refinement: turn node diagnoses into a revised helper skeleton.

The refiner sees the current skeleton, which helpers are already proven, and structured
diagnoses for the failures. It returns a whole revised skeleton rather than a patch, which
the harness recompiles, re-extracts, and reconciles against the proof store.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import BlueprintRoleConfig
from ..utils.lean_interact import LeanInteractServer
from ..utils.llm import LLMClient
from ..utils.logging import get_logger
from .generation import SkeletonCandidate, run_skeleton_loop, truncate_context
from .models import Blueprint, NodeRecord, NodeStatus
from .prompts import BLUEPRINT_PROTOCOL, REFINER_SYSTEM_PROMPT, REFINER_USER_PROMPT
from .tools import make_skeleton_compile_tool
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)


def format_solved(blueprint: Blueprint, records: dict[str, NodeRecord]) -> str:
    """List solved helpers with their exact statements, so the refiner can preserve them."""
    solved = [
        node
        for node in blueprint.helpers
        if records.get(node.id) and records[node.id].status is NodeStatus.SOLVED
    ]
    if not solved:
        return "(none yet)"

    return "\n\n".join(
        f"- `{node.id}` is proven:\n```lean\n{node.statement_source_no_doc.rstrip()}\n```"
        for node in solved
    )


def format_failures(blueprint: Blueprint, records: dict[str, NodeRecord]) -> str:
    """Report each unsolved node's diagnosis and its last compiler output."""
    by_id = blueprint.by_id
    sections = []

    for node_id, record in records.items():
        if record.status is NodeStatus.SOLVED or node_id not in by_id:
            continue

        node = by_id[node_id]
        diagnosis = record.diagnosis
        outcome = diagnosis.outcome if diagnosis else "NOT ATTEMPTED"
        detail = diagnosis.detail if diagnosis else "blocked: a parent was never solved"
        errors = diagnosis.last_error if diagnosis else ""

        section = (
            f"## `{node_id}` - {outcome}\n\n"
            f"```lean\n{node.statement_source_no_doc.rstrip()}\n```\n\n"
            f"Diagnosis: {detail}\n"
            f"Attempts: {record.attempts}"
        )
        if errors:
            section += f"\n\nLast compiler output:\n```\n{errors}\n```"
        sections.append(section)

    return "\n\n".join(sections) if sections else "(none)"


async def refine_blueprint(
    workspace: BlueprintWorkspace,
    server: LeanInteractServer,
    client: LLMClient,
    role: BlueprintRoleConfig,
    blueprint: Blueprint,
    records: dict[str, NodeRecord],
    round_number: int,
) -> SkeletonCandidate:
    """Produce a revised, validated blueprint from the current run state.

    Raises:
        BlueprintError: No valid revision was produced within budget.
    """
    system = REFINER_SYSTEM_PROMPT.format(protocol=BLUEPRINT_PROTOCOL)
    user = REFINER_USER_PROMPT.format(
        target_statement=workspace.target_statement_with_doc.rstrip() + " := by sorry",
        target_signature=workspace.target_signature,
        namespace_full=workspace.namespace_full,
        helpers=blueprint.skeleton.strip() or "(unavailable)",
        solved=format_solved(blueprint, records),
        failures=format_failures(blueprint, records),
        file_context=truncate_context(workspace.prefix),
    )

    logger.info(f"Refinement round {round_number}: requesting a revised blueprint")
    return await run_skeleton_loop(
        workspace,
        server,
        client,
        role,
        base_messages=[SystemMessage(content=system), HumanMessage(content=user)],
        tools=[make_skeleton_compile_tool(workspace)],
        label=f"refiner round {round_number}",
    )
