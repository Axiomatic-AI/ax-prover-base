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
from .graph import topological_order
from .models import Blueprint, NodeDiagnosis, NodeRecord, NodeStatus
from .prompts import BLUEPRINT_PROTOCOL, REFINER_SYSTEM_PROMPT, REFINER_USER_PROMPT
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)


#: Verdict markers the refiner reads. Input-only: the prompt tells it not to copy them back.
PROVED_MARKER = "-- PROVED"
UNPROVED_MARKER = "-- UNPROVED"
NOT_ATTEMPTED_MARKER = "-- NOT ATTEMPTED"


def render_review(diagnosis: NodeDiagnosis | None) -> str:
    """Render a failed node's three-part review block."""
    if diagnosis is None:
        return ""

    sections = [f"## Diagnosis\n{diagnosis.outcome}"]
    analysis = diagnosis.analysis or diagnosis.detail or "(no analysis recorded)"
    sections.append(f"## Analysis\n{analysis.strip()}")
    if diagnosis.suggested_fix.strip():
        sections.append(f"## Suggested Fix\n{diagnosis.suggested_fix.strip()}")
    if diagnosis.last_error.strip():
        sections.append(f"## Last compiler output\n{diagnosis.last_error.strip()}")

    return "/- Diagnosis\n" + "\n\n".join(sections) + "\n-/"


def annotate_skeleton(blueprint: Blueprint, records: dict[str, NodeRecord]) -> str:
    """Render the graph with each node's verdict and review attached in-band.

    The paper's refiner reads verdicts as comments inside the skeleton rather than as a
    separate report, which keeps each diagnosis adjacent to the statement it concerns.
    """
    blocks = []

    for node in topological_order(blueprint):
        if node.is_target:
            continue

        record = records.get(node.id)
        statement = node.statement_source.rstrip()
        declaration = f"{statement} := by sorry"

        undeclared = node.undeclared_statement_parents
        if undeclared:
            # The harness scheduled it correctly anyway, but an undeclared statement
            # dependency usually means the decomposition is not what was intended.
            declaration = (
                f"-- NOTE: this statement depends on {', '.join(undeclared)}, which "
                f"`parents` did not declare. Declare it, or restate the lemma.\n"
                f"{declaration}"
            )

        if record is None:
            blocks.append(f"{NOT_ATTEMPTED_MARKER}\n{declaration}")
            continue

        if record.status is NodeStatus.SOLVED:
            blocks.append(f"{PROVED_MARKER}\n{declaration}")
            continue

        if record.diagnosis is None:
            blocks.append(f"{NOT_ATTEMPTED_MARKER}\n{declaration}")
            continue

        review = render_review(record.diagnosis)
        blocks.append(f"{UNPROVED_MARKER}\n{declaration}\n\n{review}")

    return "\n\n".join(blocks) if blocks else "(the graph has no helpers yet)"


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
        annotated_skeleton=annotate_skeleton(blueprint, records),
        file_context=truncate_context(workspace.prefix),
    )

    logger.info(f"Refinement round {round_number}: requesting a revised blueprint")
    return await run_skeleton_loop(
        workspace,
        server,
        client,
        role,
        base_messages=[SystemMessage(content=system), HumanMessage(content=user)],
        label=f"refiner round {round_number}",
    )
