"""Frontier scheduling: prove only nodes whose parents are already solved.

Each round takes the whole ready frontier and proves it concurrently, up to the configured
limit. Every outcome is persisted immediately, so an interrupted run resumes without
reproving anything. This exposes less parallelism than attempting nodes under conditional
assumptions, but it makes proof reuse and failure attribution simple.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool

from ..config import BlueprintRoleConfig
from ..utils.llm import LLMClient
from ..utils.logging import get_logger
from .graph import ready_frontier, required_nodes
from .lean_service import CompileCancelled
from .models import Blueprint, NodeDiagnosis, NodeOutcome, NodeStatus
from .node_prover import NodeAttemptResult, prove_node
from .proof_store import ProofStore
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)


def dispatchable(
    blueprint: Blueprint,
    statuses: dict[str, NodeStatus],
    required: set[str],
    speculative: bool,
) -> tuple[str, ...]:
    """Node ids to prove now.

    Speculatively, every unsolved required node: a scratch module renders parents as
    `axiom` declarations, so a node's proof never needs its parents' proofs and waiting for
    them only serializes the graph. Otherwise the classic ready frontier, where a node waits
    until every declared and statement parent is solved.
    """
    if not speculative:
        return tuple(
            node_id for node_id in ready_frontier(blueprint, statuses) if node_id in required
        )

    return tuple(
        node.id
        for node in blueprint.nodes
        if node.id in required and statuses.get(node.id, NodeStatus.PENDING) is NodeStatus.PENDING
    )


@dataclass
class ScheduleReport:
    """What one scheduling pass over a blueprint achieved."""

    rounds: int = 0
    solved: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    infrastructure_error: str = ""


async def run_schedule(
    workspace: BlueprintWorkspace,
    blueprint: Blueprint,
    store: ProofStore,
    client: LLMClient,
    role: BlueprintRoleConfig,
    search_tool: BaseTool | None = None,
    max_node_agents: int = 12,
    speculative: bool = True,
) -> ScheduleReport:
    """Prove the target's required nodes in dependency order until the frontier dries up.

    Returns:
        A report of what was solved and failed. An infrastructure error stops the pass
        immediately and is recorded on the report.
    """
    required = required_nodes(blueprint)
    by_id = blueprint.by_id
    service = getattr(workspace, "compile_service", None)

    def service_cancel(node_id: str) -> None:
        """Drop a finished node's queued compiles.

        The server lease is deliberately not released here: it belongs to the target, and
        dropping it mid-run would move later nodes to a server with a cold prefix.
        """
        if service is not None:
            service.cancel_node(workspace.node_key(node_id))

    # Bounds concurrent model agents only; their Lean compilations are limited separately
    # by the compile service. Zero or less means unbounded, which is what the paper does:
    # it "dispatches each lemma to a Lean prover in parallel" with no stated cap, so the
    # whole ready frontier runs at once. A positive value throttles, which is worth setting
    # when a provider rate limit, rather than the frontier, is the real constraint.
    limiter = (
        asyncio.Semaphore(max_node_agents) if max_node_agents > 0 else contextlib.nullcontext()
    )
    report = ScheduleReport()

    while True:
        frontier = dispatchable(blueprint, store.statuses, required, speculative)
        if not frontier:
            break

        report.rounds += 1
        logger.info(
            f"Round {report.rounds}: proving {len(frontier)} node(s): {', '.join(frontier)}"
        )

        async def attempt(node_id: str) -> None:
            """Prove one node and persist its outcome the moment it is known.

            Persisting here rather than after the round's gather matters: a run killed
            mid-round would otherwise discard every proof the round had already found,
            since the gather never returns.
            """
            async with limiter:
                try:
                    result = await prove_node(
                        workspace, blueprint, by_id[node_id], client, role, search_tool
                    )
                except asyncio.CancelledError:
                    # CancelledError is a BaseException, so the handler below cannot see it.
                    # The node stays pending in the checkpoint and is retried on resume, so
                    # nothing is lost; cancellation must keep propagating.
                    logger.warning(f"Node {node_id!r} cancelled; it stays pending")
                    raise
                except CompileCancelled as e:
                    # The node's compiles were dropped because it already finished, or a
                    # cancellation leaked from an earlier round. Neither is an infrastructure
                    # fault, and reporting it as one would abandon the whole graph over one
                    # node. Record it as an ordinary failure so refinement can address it:
                    # leaving it pending would have `dispatchable` re-select it every round
                    # and never terminate.
                    logger.warning(f"Node {node_id!r} had its compiles cancelled: {e}")
                    result = NodeAttemptResult(
                        outcome=NodeOutcome.PROOF_TOO_HARD,
                        diagnosis=NodeDiagnosis(
                            outcome=NodeOutcome.PROOF_TOO_HARD,
                            detail=f"its queued compilations were cancelled ({e})",
                        ),
                    )
                except Exception as e:  # noqa: BLE001 - reported as an infrastructure error
                    logger.exception(f"Node {node_id!r} raised an unexpected error")
                    result = NodeAttemptResult(
                        outcome=NodeOutcome.INFRASTRUCTURE_ERROR,
                        diagnosis=NodeDiagnosis(
                            outcome=NodeOutcome.INFRASTRUCTURE_ERROR, detail=str(e)
                        ),
                    )

            if result.outcome is NodeOutcome.SOLVED:
                store.mark_solved(node_id, result.proof_body, result.attempts)
                report.solved.append(node_id)
                # Nothing else needs to compile for this node.
                service_cancel(node_id)
            elif result.outcome is NodeOutcome.INFRASTRUCTURE_ERROR:
                store.mark_failed(node_id, result.diagnosis, result.attempts)
                report.infrastructure_error = (
                    result.diagnosis.detail if result.diagnosis else "unknown infrastructure error"
                )
            else:
                store.mark_failed(node_id, result.diagnosis, result.attempts)
                report.failed.append(node_id)

        await asyncio.gather(*(attempt(n) for n in frontier))

        if report.infrastructure_error:
            logger.error(f"Stopping schedule: {report.infrastructure_error}")
            break

    return report


def is_complete(blueprint: Blueprint, store: ProofStore) -> bool:
    """True when every node the target needs, including the target, is solved."""
    statuses = store.statuses
    return all(statuses.get(node_id) is NodeStatus.SOLVED for node_id in required_nodes(blueprint))


def blocked_nodes(blueprint: Blueprint, store: ProofStore) -> list[str]:
    """Required nodes that are unsolved, so the refiner knows what to address."""
    statuses = store.statuses
    return sorted(
        node_id
        for node_id in required_nodes(blueprint)
        if statuses.get(node_id) is not NodeStatus.SOLVED
    )
