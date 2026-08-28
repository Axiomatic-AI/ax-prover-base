"""Immutable graph models and mutable runtime results for blueprint proving.

The canonical blueprint (`Blueprint`, `BlueprintNode`) is derived from a compiled Lean
skeleton and never carries proving state. Mutable state lives in `NodeRecord` /
`ProofStoreState`, which the proof store persists separately (see plan section 4.3).
"""

from enum import StrEnum

from pydantic import BaseModel, Field

BLUEPRINT_METADATA_VERSION = 1

#: Reserved blueprint id of the immutable original target. Helpers may not use it.
TARGET_NODE_ID = "target"


class BlueprintError(Exception):
    """Base class for blueprint-mode failures."""


class BlueprintValidationError(BlueprintError):
    """A skeleton failed canonical validation.

    Carries every individual problem so the architect/refiner can repair all of them
    in one round instead of one per round.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))

    @property
    def report(self) -> str:
        """Model-facing bullet list of the validation problems."""
        return "\n".join(f"- {problem}" for problem in self.problems)


class NodeMetadata(BaseModel, frozen=True):
    """The `ax-blueprint` fenced JSON block of a node's docstring."""

    version: int = Field(description="Metadata protocol version")
    id: str = Field(description="Stable logical node identifier")
    parents: tuple[str, ...] = Field(default=(), description="Direct parent node ids")


class BlueprintNode(BaseModel, frozen=True):
    """A canonical node: blueprint metadata plus extracted Lean facts.

    `statement_source` is the verbatim skeleton text from the start of the declaration
    up to (but excluding) `:=`, so `statement_source + ":= " + proof_body` reproduces a
    complete declaration. This is why the harness never needs the model to restate a
    theorem: it owns every statement it compiles.
    """

    id: str
    parents: tuple[str, ...] = ()
    lean_name: str = Field(description="Fully qualified Lean declaration name")
    kind: str = Field(description="Lean declaration kind, e.g. 'theorem'")
    signature: str = Field(description="Pretty-printed elaborated signature")
    type_pp: str = Field(default="", description="Pretty-printed elaborated conclusion")
    statement_source: str = Field(description="Declaration text up to, excluding, ':='")
    statement_source_no_doc: str = Field(description="`statement_source` without its docstring")
    doc_text: str = Field(default="", description="Docstring content with metadata fence removed")
    type_deps: tuple[str, ...] = ()
    value_deps: tuple[str, ...] = ()
    start_line: int = 0
    is_target: bool = False

    @property
    def short_name(self) -> str:
        """Last component of the fully qualified Lean name."""
        return self.lean_name.rsplit(".", 1)[-1]

    def render(self, proof_body: str, include_doc: bool = True) -> str:
        """Render the declaration with `proof_body` attached."""
        statement = self.statement_source if include_doc else self.statement_source_no_doc
        return f"{statement.rstrip()} := {proof_body.strip()}"


class Blueprint(BaseModel, frozen=True):
    """A validated helper DAG plus the immutable target node."""

    namespace: str = Field(description="Deterministic namespace holding every helper")
    nodes: tuple[BlueprintNode, ...]
    skeleton: str = Field(default="", description="The compiling skeleton this was extracted from")

    @property
    def by_id(self) -> dict[str, BlueprintNode]:
        """Nodes keyed by blueprint id."""
        return {node.id: node for node in self.nodes}

    @property
    def target(self) -> BlueprintNode:
        """The immutable target node."""
        return self.by_id[TARGET_NODE_ID]

    @property
    def helpers(self) -> tuple[BlueprintNode, ...]:
        """Every generated helper, excluding the target."""
        return tuple(node for node in self.nodes if not node.is_target)


class NodeStatus(StrEnum):
    """Scheduling state of a node within a run."""

    PENDING = "pending"
    SOLVED = "solved"
    FAILED = "failed"


class NodeOutcome(StrEnum):
    """Result of a single node proving attempt sequence (plan section 7.3)."""

    SOLVED = "SOLVED"
    PROOF_TOO_HARD = "PROOF_TOO_HARD"
    STATEMENT_WRONG = "STATEMENT_WRONG"
    #: Ran out of token budget before using its attempts. Distinct from PROOF_TOO_HARD
    #: because the statement was never really tested, and telling the refiner to split a
    #: lemma that was merely starved sends it after the wrong problem.
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


class NodeDiagnosis(BaseModel):
    """Structured explanation of why a node was not solved.

    `analysis` and `suggested_fix` mirror the paper's three-part review: a one-line verdict
    gives the refiner nothing to act on, whereas a forensic account plus a proposed
    decomposition is what lets it bridge the gap rather than guess.
    """

    outcome: NodeOutcome
    detail: str = ""
    analysis: str = ""
    suggested_fix: str = ""
    last_error: str = ""


class NodeRecord(BaseModel):
    """Mutable per-node runtime state, persisted in the proof store."""

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    proof_body: str | None = None
    attempts: int = 0
    fingerprint: str = ""
    diagnosis: NodeDiagnosis | None = None
    reused: bool = False


class ComparatorStatus(StrEnum):
    """Outcome of the final Comparator gate."""

    PASSED = "passed"
    REJECTED = "rejected"
    PENDING = "pending"
    SKIPPED = "skipped"


class RunStatus(StrEnum):
    """Terminal status of a blueprint run."""

    SOLVED = "solved"
    FAILED = "failed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    COMPARATOR_PENDING = "comparator_pending"


class BlueprintRunResult(BaseModel):
    """Everything a blueprint run reports back to the CLI and experiment runner."""

    status: RunStatus
    target: str
    namespace: str = ""
    graph_size: int = 0
    refinement_rounds: int = 0
    reused_proofs: int = 0
    node_records: list[NodeRecord] = Field(default_factory=list)
    comparator_status: ComparatorStatus = ComparatorStatus.SKIPPED
    source_modified: bool = False
    error: str = ""
    compile_stats: dict = Field(
        default_factory=dict,
        description="Lean compile queue volume, latency, and warm-up cost for the run",
    )

    @property
    def is_success(self) -> bool:
        """True when the source was updated with a verified proof."""
        return self.status in (RunStatus.SOLVED, RunStatus.COMPARATOR_PENDING)
