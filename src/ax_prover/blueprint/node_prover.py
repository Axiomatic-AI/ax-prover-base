"""Isolated, proof-body-only node proving.

A node prover sees one statement, its direct parents' signatures, the trusted file
context, and its own tool transcript. It never sees unrelated generated siblings or any
parent's proof body, and it can only ever return a proof body: the harness constructs
every module it compiles.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..config import BlueprintRoleConfig
from ..utils.build import LeanBuildError
from ..utils.llm import LLMClient
from ..utils.logging import get_logger
from .models import Blueprint, BlueprintNode, NodeDiagnosis, NodeOutcome
from .prompts import (
    NODE_DIAGNOSIS_SYSTEM_PROMPT,
    NODE_DIAGNOSIS_USER_PROMPT,
    NODE_PARENTS_SECTION,
    NODE_PLAN_SECTION,
    NODE_PROVER_SYSTEM_PROMPT,
    NODE_PROVER_USER_PROMPT,
)
from .roles import TokenBudget, parse_proposal, run_turn
from .tools import make_node_compile_tool, normalize_proof_body
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)

#: Placeholders and cheating tactics a candidate proof body may never contain.
FORBIDDEN_IN_BODY = re.compile(r"(?<![\w.])(sorry|admit|sorryAx|native_decide)(?![\w])")

#: How many previous attempts are replayed to the model. Bounds context growth.
ATTEMPT_HISTORY_DEPTH = 2


class NodeProposal(BaseModel):
    """Structured output of one node proving attempt."""

    reasoning: str = Field(default="", description="Brief justification for this proof")
    proof_body: str = Field(description="The proof body only: the text following `:=`")


class NodeTriage(BaseModel):
    """Three-part review of a node that could not be proven."""

    outcome: Literal["PROOF_TOO_HARD", "STATEMENT_WRONG"]
    detail: str = Field(description="One or two sentences explaining the verdict")
    analysis: str = Field(
        default="",
        description=(
            "Forensic account: what was tried, what compiled, what errors remained, and "
            "where the gap is"
        ),
    )
    suggested_fix: str = Field(
        default="",
        description=(
            "For STATEMENT_WRONG, why it is false and how to repair it. For "
            "PROOF_TOO_HARD, a helper-lemma decomposition that bridges the gap."
        ),
    )


@dataclass
class _Attempt:
    """One independent attempt's outcome, before the wave picks a winner."""

    index: int
    body: str = ""
    error: str = ""
    solved: bool = False
    infrastructure: str = ""


@dataclass
class NodeAttemptResult:
    """Outcome of a full node proving attempt sequence."""

    outcome: NodeOutcome
    proof_body: str | None = None
    attempts: int = 0
    diagnosis: NodeDiagnosis | None = None
    transcript: list[str] = field(default_factory=list)


def format_parents(parents: tuple[BlueprintNode, ...]) -> str:
    """Render parent interfaces: fully qualified names and signatures, never proofs."""
    return "\n".join(f"theorem {parent.lean_name} {parent.signature}" for parent in parents)


def _build_messages(
    node: BlueprintNode,
    parents: tuple[BlueprintNode, ...],
    history: list[str],
) -> list[BaseMessage]:
    """Build the node prover's prompt, replaying only the most recent attempts."""
    parents_section = (
        NODE_PARENTS_SECTION.format(parents=format_parents(parents)) if parents else ""
    )
    plan_section = NODE_PLAN_SECTION.format(plan=node.doc_text) if node.doc_text.strip() else ""

    user = NODE_PROVER_USER_PROMPT.format(
        statement=f"{node.statement_source_no_doc.rstrip()} := <your proof body here>",
        plan_section=plan_section,
        parents_section=parents_section,
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=NODE_PROVER_SYSTEM_PROMPT),
        HumanMessage(content=user),
    ]

    recent = history[-ATTEMPT_HISTORY_DEPTH:]
    if recent:
        messages.append(
            HumanMessage(
                content="# Your previous attempts failed\n\n"
                + "\n\n".join(recent)
                + "\n\nDo not repeat them. Try a different approach."
            )
        )

    return messages


def _format_attempt(index: int, body: str, error: str) -> str:
    return f"## Attempt {index}\n\n```lean\n{body}\n```\n\nCompiler output:\n```\n{error}\n```"


async def prove_node(
    workspace: BlueprintWorkspace,
    blueprint: Blueprint,
    node: BlueprintNode,
    client: LLMClient,
    role: BlueprintRoleConfig,
    search_tool: BaseTool | None = None,
) -> NodeAttemptResult:
    """Prove one node in isolation and return its outcome.

    The harness recompiles every returned proof body itself, so a model claiming success
    without compiling cannot produce a `SOLVED` outcome.
    """
    by_id = blueprint.by_id
    parents = tuple(by_id[parent_id] for parent_id in node.parents if parent_id in by_id)

    budget = TokenBudget(role.max_total_tokens)
    tools: list[BaseTool] = [make_node_compile_tool(workspace, node, parents)]
    if search_tool is not None:
        tools.append(search_tool)

    history: list[str] = []
    attempts = 0
    last_error = ""
    starved = False
    infrastructure: NodeDiagnosis | None = None

    async def one_attempt(index: int, replay: list[str]) -> _Attempt:
        """Run a single independent attempt against the node's statement."""
        messages = _build_messages(node, parents, replay)
        try:
            turn = await run_turn(
                client,
                messages,
                tools,
                NodeProposal,
                role.max_tool_iterations,
                budget,
                search_budget=role.max_searches,
            )
        except LeanBuildError as e:
            return _Attempt(index=index, infrastructure=str(e))

        compiles = turn.tool_calls.get("lean_compile", 0)
        searches = turn.tool_calls.get("mathlib_search", 0)
        logger.debug(
            f"Node {node.id!r} attempt {index}: {turn.turns} turn(s), "
            f"{compiles} compile(s), {searches} search(es)"
            + (", search budget spent" if turn.tools_exhausted else "")
        )
        if compiles == 0:
            # The harness verifies every returned body anyway, so this is wasted effort
            # rather than a soundness risk, but it is the cheapest attempt to lose.
            logger.warning(
                f"Node {node.id!r} attempt {index} answered without compiling "
                f"({searches} search(es) in {turn.turns} turn(s))"
            )

        proposal = parse_proposal(turn, NodeProposal)
        if proposal is None:
            return _Attempt(
                index,
                error="the response was not valid structured output",
                body="(unparseable response)",
            )

        body, notes = normalize_proof_body(proposal.proof_body)
        if notes:
            logger.debug(f"Node {node.id!r}: normalized proposal ({'; '.join(notes)})")
        if not body:
            return _Attempt(index, error="the proof body was empty", body=proposal.proof_body)

        forbidden = FORBIDDEN_IN_BODY.search(body)
        if forbidden:
            return _Attempt(
                index,
                error=f"the proof body uses {forbidden.group(1)!r}, which is not allowed",
                body=body,
            )

        source = workspace.render_node_module(node, parents, body)
        try:
            # `check_axioms_of` rejects a proof that reaches a `sorry` by any route, which a
            # textual scan of the body cannot guarantee on its own.
            result = await workspace.compile_candidate(
                source,
                node_id=node.id,
                check_axioms_of=node.lean_name,
                allowed_axioms=workspace.allowed_axioms(parents),
                label=f"verify_{node.id}",
            )
        except LeanBuildError as e:
            return _Attempt(index=index, infrastructure=str(e))

        if result.success:
            return _Attempt(index, body=body, solved=True)
        return _Attempt(index, error=result.output, body=body)

    # Attempts run in waves: parallel within a wave, so a node needing several tries does
    # not pay a full round-trip for each, and sequential across waves, so every later
    # attempt still sees the earlier failures.
    wave_size = max(1, role.attempt_concurrency)

    while attempts < role.max_attempts:
        if budget.exhausted:
            logger.warning(
                f"Node {node.id!r}: token budget exhausted after {attempts}/"
                f"{role.max_attempts} attempts ({budget.spent} tokens). Raise "
                "blueprint.prover.max_total_tokens; this is starvation, not difficulty."
            )
            starved = attempts < role.max_attempts
            break

        size = min(wave_size, role.max_attempts - attempts)
        replay = list(history)
        wave = await asyncio.gather(*(one_attempt(attempts + i + 1, replay) for i in range(size)))
        attempts += size

        solved = next((a for a in wave if a.solved), None)
        if solved is not None:
            logger.info(f"Node {node.id!r}: solved in {attempts} attempt(s)")
            return NodeAttemptResult(
                outcome=NodeOutcome.SOLVED,
                proof_body=solved.body,
                attempts=attempts,
                transcript=history,
            )

        broken = next((a for a in wave if a.infrastructure), None)
        if broken is not None:
            logger.error(f"Node {node.id!r}: Lean infrastructure failure: {broken.infrastructure}")
            infrastructure = NodeDiagnosis(
                outcome=NodeOutcome.INFRASTRUCTURE_ERROR,
                detail=broken.infrastructure,
                last_error=last_error,
            )
            break

        for a in wave:
            last_error = a.error or last_error
            history.append(_format_attempt(a.index, a.body, a.error))

    if infrastructure is not None:
        return NodeAttemptResult(
            outcome=NodeOutcome.INFRASTRUCTURE_ERROR,
            attempts=attempts,
            diagnosis=infrastructure,
            transcript=history,
        )

    if starved:
        # Triage here would be both misleading and self-defeating: the statement was never
        # properly attempted, and the call needs budget this node no longer has.
        return NodeAttemptResult(
            outcome=NodeOutcome.BUDGET_EXHAUSTED,
            attempts=attempts,
            diagnosis=NodeDiagnosis(
                outcome=NodeOutcome.BUDGET_EXHAUSTED,
                detail=(
                    f"exhausted {budget.spent} tokens after {attempts} of "
                    f"{role.max_attempts} attempts; the statement was not fully tested"
                ),
                last_error=last_error,
            ),
            transcript=history,
        )

    diagnosis = await diagnose_node(node, client, role, history, last_error, attempts)
    logger.info(f"Node {node.id!r}: {diagnosis.outcome} after {attempts} attempt(s)")
    return NodeAttemptResult(
        outcome=diagnosis.outcome,
        attempts=attempts,
        diagnosis=diagnosis,
        transcript=history,
    )


async def diagnose_node(
    node: BlueprintNode,
    client: LLMClient,
    role: BlueprintRoleConfig,
    history: list[str],
    last_error: str,
    attempts: int,
) -> NodeDiagnosis:
    """Classify why a node failed, so the refiner knows whether to split or restate it.

    Falls back to `PROOF_TOO_HARD` when triage itself fails: assuming the statement is
    fine is the conservative call, since restating a correct lemma discards its subtree.
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=NODE_DIAGNOSIS_SYSTEM_PROMPT),
        HumanMessage(
            content=NODE_DIAGNOSIS_USER_PROMPT.format(
                statement=node.statement_source_no_doc.rstrip(),
                plan=node.doc_text or "(none given)",
                errors=last_error or "(no compiler output)",
                attempts="\n\n".join(history[-ATTEMPT_HISTORY_DEPTH:]) or "(none)",
            )
        ),
    ]

    budget = TokenBudget(role.max_total_tokens)
    try:
        turn = await run_turn(client, messages, [], NodeTriage, 0, budget)
        triage = parse_proposal(turn, NodeTriage)
    except Exception as e:
        logger.warning(f"Node {node.id!r}: triage call failed: {e}")
        triage = None

    if triage is None:
        return NodeDiagnosis(
            outcome=NodeOutcome.PROOF_TOO_HARD,
            detail=f"exhausted {attempts} attempt(s); triage unavailable",
            last_error=last_error,
        )

    return NodeDiagnosis(
        outcome=NodeOutcome(triage.outcome),
        detail=triage.detail,
        analysis=triage.analysis,
        suggested_fix=triage.suggested_fix,
        last_error=last_error,
    )
