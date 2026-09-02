"""The blueprint architect: generate a compiling helper skeleton, then canonicalize it.

The architect's only tool is `lean_compile`. It never writes the target, the imports, or
the namespace; the harness assembles every module, so the target is immutable by
construction rather than by post-hoc checking.
"""

from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..config import BlueprintRoleConfig
from ..utils.lean_interact import LeanInteractServer
from ..utils.llm import LLMClient, was_truncated
from ..utils.logging import get_logger
from .extraction import extract_nodes
from .graph import validate_blueprint
from .lean_service import CompilePriority
from .metadata import FENCE_LANGUAGE
from .models import Blueprint, BlueprintError, BlueprintValidationError
from .prompts import (
    ARCHITECT_REPAIR_PROMPT,
    ARCHITECT_REPAIR_SOURCE,
    ARCHITECT_SYSTEM_PROMPT,
    ARCHITECT_USER_PROMPT,
    BLUEPRINT_PROTOCOL,
    INFORMAL_PROOF_GUIDE,
    INFORMAL_PROOF_SYSTEM_PROMPT,
    INFORMAL_PROOF_USER_PROMPT,
)
from .roles import TokenBudget, parse_proposal, run_turn
from .tools import make_skeleton_compile_tool, strip_outer_fence
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)

#: Characters of trusted file context shown to the architect and refiner. The tail of the
#: file matters most: it holds the declarations closest to the target.
FILE_CONTEXT_LIMIT = 20000


class ArchitectProposal(BaseModel):
    """Structured output of one architect or refiner round."""

    reasoning: str = Field(default="", description="Why this decomposition")
    helpers: str = Field(description="Lean source of the docstringed helper lemmas")
    target_parents: list[str] = Field(
        default_factory=list, description="Helper ids the target's own proof uses directly"
    )
    target_proof_plan: str = Field(
        default="", description="How the target follows from those helpers"
    )


@dataclass
class SkeletonCandidate:
    """An architect proposal that survived compilation, extraction, and validation."""

    blueprint: Blueprint
    helpers: str
    target_parents: tuple[str, ...]
    target_proof_plan: str


def truncate_context(text: str, limit: int = FILE_CONTEXT_LIMIT) -> str:
    """Keep the tail of a long file context, flagging what was dropped."""
    if len(text) <= limit:
        return text
    return (
        f"-- ... {len(text) - limit} characters of earlier file content omitted ...\n"
        + text[-limit:]
    )


async def build_blueprint(
    workspace: BlueprintWorkspace,
    server: LeanInteractServer,
    helpers: str,
    target_parents: tuple[str, ...],
    target_proof_plan: str,
) -> Blueprint:
    """Compile a helper source, extract it, and validate the canonical graph.

    Raises:
        BlueprintValidationError: The skeleton does not compile, or the extracted graph
            violates the protocol.
    """
    source = workspace.render_skeleton(helpers, target_parents, target_proof_plan)

    if workspace.compile_service is not None:
        # The warm REPL returns diagnostics and declarations in one response, so the
        # separate extraction pass is unnecessary.
        result = await workspace.compile_candidate(
            source, priority=CompilePriority.STRUCTURAL, label="skeleton"
        )
        declarations = list(result.declarations)
    else:
        result, declarations = await workspace.compile_and_extract(source, server)

    if not result.success:
        raise BlueprintValidationError(
            [f"the assembled skeleton does not compile:\n{result.output}"]
        )

    nodes = extract_nodes(
        declarations,
        source,
        workspace.namespace_full,
        workspace.target_lean_name,
        workspace.trusted_names,
    )

    # The REPL can return an incomplete declaration list for a cleanly compiling module
    # (a measured run omitted one helper), and validation then blames the model for a
    # "missing" helper that sits in its file - one run burned 8 repair rounds on that
    # lie. A metadata-block count mismatch is infrastructure, never model fault.
    declared = source.count(f"```{FENCE_LANGUAGE}")
    extracted = sum(1 for node in nodes if not node.is_target)
    if extracted < declared:
        found = sorted(node.id for node in nodes if not node.is_target)
        raise BlueprintError(
            f"the REPL reported {extracted} generated helper(s) but the compiled source "
            f"declares {declared}; declarations were lost in extraction, not by the "
            f"model. Extracted ids: {found}"
        )
    return validate_blueprint(
        nodes,
        namespace=workspace.namespace_full,
        target_lean_name=workspace.target_lean_name,
        target_signature=workspace.target_signature,
        skeleton=helpers,
    )


async def run_skeleton_loop(
    workspace: BlueprintWorkspace,
    server: LeanInteractServer,
    client: LLMClient,
    role: BlueprintRoleConfig,
    base_messages: list[BaseMessage],
    label: str,
) -> SkeletonCandidate:
    """Propose skeletons until one validates, feeding every problem back each round.

    Shared by the architect and the refiner, which differ only in their prompts.

    The accepted source is what the compile tool verified, not what the model re-types.
    The final structured answer re-emits the helpers from memory, and a measured run
    reintroduced errors that way twice: six clean tool compiles in a round, then a
    rejected submission. So the turn ends at the first successful tool compile, and a
    submission that fails the gate falls back to the tool-verified source.

    Raises:
        BlueprintError: The budget ran out before a skeleton validated.
    """
    budget = TokenBudget(role.max_total_tokens)
    problems = ""
    rejected = ""
    verified: list[str] = []

    def record(helpers: str, success: bool, _output: str) -> None:
        if success:
            verified.clear()
            verified.append(helpers)

    tools: list[BaseTool] = [make_skeleton_compile_tool(workspace, on_result=record)]

    for attempt in range(1, role.max_attempts + 1):
        if budget.exhausted:
            logger.info(f"{label}: token budget exhausted after {attempt - 1} round(s)")
            break

        messages = list(base_messages)
        if problems:
            repair = ARCHITECT_REPAIR_PROMPT.format(problems=problems)
            if rejected:
                # Without the rejected source the model cannot see what it wrote, so it
                # regenerates the same skeleton and earns the same rejection.
                repair += ARCHITECT_REPAIR_SOURCE.format(rejected=rejected)
            messages.append(HumanMessage(content=repair))

        verified.clear()
        turn = await run_turn(
            client,
            messages,
            tools,
            ArchitectProposal,
            role.max_tool_iterations,
            budget,
            stop=lambda: bool(verified),
        )
        proposal = parse_proposal(turn, ArchitectProposal)

        if proposal is None:
            if was_truncated(turn.response):
                # Retrying blind would truncate again; say what went wrong.
                problems = (
                    "- your previous response was cut off at the output token limit before "
                    "it produced a complete answer. Reason more briefly and keep the helper "
                    "skeleton compact."
                )
            else:
                problems = "- the response was not valid structured output"
            continue

        helpers, notes = strip_outer_fence(proposal.helpers)
        if notes:
            logger.debug(f"{label}: {'; '.join(notes)}")

        if not helpers:
            problems = "- `helpers` was empty; the blueprint needs at least one helper lemma"
            rejected = ""
            continue

        target_parents = tuple(dict.fromkeys(proposal.target_parents))

        # The submitted helpers are a from-memory re-typing; the tool-verified source is
        # the ground truth. Try the submission first in case the model improved it.
        candidates = [helpers]
        if verified and verified[0] != helpers:
            candidates.append(verified[0])

        blueprint: Blueprint | None = None
        accepted = helpers
        error: BlueprintValidationError | None = None
        for candidate in candidates:
            try:
                blueprint = await build_blueprint(
                    workspace, server, candidate, target_parents, proposal.target_proof_plan
                )
                accepted = candidate
                break
            except BlueprintValidationError as e:
                # Feedback quotes the submitted version, so keep its report.
                error = error or e

        if blueprint is None:
            forced = turn.iterations_exhausted
            logger.info(
                f"{label} round {attempt} rejected"
                + (" (submission forced by the tool-iteration cap)" if forced else "")
                + f": {error}"
            )
            problems = error.report
            if forced:
                problems += (
                    "\n- you hit the tool-call limit and had to submit before `lean_compile` "
                    "ever reported success; fix the errors above and confirm a clean compile "
                    "before answering"
                )
            rejected = helpers
            continue

        if accepted != helpers:
            logger.info(
                f"{label}: the submitted helpers failed the gate; accepted this round's "
                "tool-verified skeleton instead"
            )
        logger.info(
            f"{label}: accepted a blueprint with {len(blueprint.helpers)} helper(s) "
            f"in round {attempt}"
        )
        return SkeletonCandidate(
            blueprint=blueprint,
            helpers=accepted,
            target_parents=target_parents,
            target_proof_plan=proposal.target_proof_plan,
        )

    raise BlueprintError(
        f"{label} did not produce a valid blueprint within its budget. Last problems:\n{problems}"
    )


async def generate_informal_proof(client: LLMClient, workspace: BlueprintWorkspace) -> str:
    """Produce a natural-language proof of the target to seed the architect.

    The guide gives the architect real mathematical content to formalize; without one, an
    architect that cannot elaborate its statements degenerates into compiling placeholders
    (a measured run stubbed every helper to `(1 : ℝ) = (1 : ℝ)`).
    """
    response = await client.ainvoke(
        [
            SystemMessage(content=INFORMAL_PROOF_SYSTEM_PROMPT),
            HumanMessage(
                content=INFORMAL_PROOF_USER_PROMPT.format(
                    target_statement=workspace.target_statement_with_doc
                )
            ),
        ]
    )
    return response.text


async def generate_blueprint(
    workspace: BlueprintWorkspace,
    server: LeanInteractServer,
    client: LLMClient,
    role: BlueprintRoleConfig,
    extra_context: str = "",
) -> SkeletonCandidate:
    """Generate and validate the initial blueprint for a target.

    Without caller-supplied context, an informal proof is generated first and passed as a
    structural guide, the remedy arXiv:2606.06468 measures at +13 points on PutnamBench.

    Raises:
        BlueprintError: No valid blueprint was produced within budget.
    """
    if not extra_context:
        try:
            informal = await generate_informal_proof(client, workspace)
        except Exception as e:  # noqa: BLE001 - the guide is optional; proceed unguided
            logger.warning(f"Informal proof generation failed, proceeding unguided: {e}")
            informal = ""
        if informal:
            logger.info(f"architect: seeded with an informal proof ({len(informal)} chars)")
            extra_context = INFORMAL_PROOF_GUIDE.format(informal_proof=informal)

    system = ARCHITECT_SYSTEM_PROMPT.format(
        protocol=BLUEPRINT_PROTOCOL, namespace=workspace.namespace
    )
    user = ARCHITECT_USER_PROMPT.format(
        target_statement=workspace.target_statement_with_doc.rstrip() + " := by sorry",
        target_signature=workspace.target_signature,
        namespace_full=workspace.namespace_full,
        file_context=truncate_context(workspace.prefix),
        extra_context=f"\n# Additional context\n\n{extra_context}\n" if extra_context else "",
    )

    return await run_skeleton_loop(
        workspace,
        server,
        client,
        role,
        base_messages=[SystemMessage(content=system), HumanMessage(content=user)],
        label="architect",
    )
