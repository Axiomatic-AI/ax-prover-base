"""Data models for the prover agent."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from .files import Location
from .messages import (
    FEEDBACK_CONSTRUCTORS,
    FeedbackMessage,
    ProposalMessage,
    ReviewApprovedFeedback,
)


class TargetItem(BaseModel):
    """A single item to be proven (definition, theorem, lemma, etc.)."""

    title: str = Field(
        description="Unique identifier for this item (e.g., 'group_def', 'cayley_theorem')"
    )
    location: Location | None = Field(
        default=None,
        description="Location where the formalization is stored (includes path and Lean name)",
    )
    proven: bool = Field(
        default=False,
        description="Whether this item has been proven",
    )


class ProverMetrics(BaseModel, validate_assignment=True):
    """Metrics for prover workflow execution."""

    build_timeout_count: int = Field(default=0, description="Number of times build timed out")
    compilation_error_count: int = Field(
        default=0, description="Number of compilation errors (non-timeout)"
    )

    number_of_iterations: int = Field(
        default=0, description="Number of iterations for main theorem only"
    )
    reviewer_rejections: int = Field(
        default=0,
        description="Total number of times the proof was rejected by the reviewer agent",
    )

    max_iterations_reached: bool = Field(
        default=False, description="Whether max iterations limit was reached"
    )


class ProverAgentState(BaseModel):
    """State model for the prover agent workflow."""

    item: TargetItem = Field(description="The item to prove")

    messages: Annotated[list[SerializeAsAny[BaseMessage]], add_messages] = Field(
        default_factory=list,
        description="History of all messages in the formalization workflow",
    )

    metrics: ProverMetrics = Field(
        default_factory=ProverMetrics,
        description="Execution metrics for the prover workflow.",
    )

    experience: str = Field(
        default="",
        description="Context with the key lessons learned from the prover's previous attempts in natural language",
    )

    summary: str = Field(default="", description="LLM-generated summary of the prover run")

    @model_validator(mode="before")
    @classmethod
    def normalize_messages(cls, data: dict) -> dict:
        """
        Normalize messages from deserialization.

        LangGraph's JsonPlusSerializer may deserialize Pydantic models as dicts.
        This validator reconstructs the proper message types based on the 'type' discriminator.
        """
        if "messages" not in data or not data["messages"]:
            return data

        normalized_messages = []
        for msg in data["messages"]:
            if isinstance(msg, ProposalMessage | FeedbackMessage):
                normalized_messages.append(msg)
            elif isinstance(msg, dict):
                msg_type = msg.get("type")
                if msg_type == "proposal":
                    normalized_messages.append(ProposalMessage(**msg))
                elif msg_type == "feedback":
                    feedback_type = msg.get("feedback_type")
                    feedback_cls = FEEDBACK_CONSTRUCTORS.get(feedback_type)
                    if feedback_cls:
                        normalized_messages.append(feedback_cls(**msg))
                    else:
                        normalized_messages.append(msg)
                else:
                    normalized_messages.append(msg)
            else:
                normalized_messages.append(msg)

        data["messages"] = normalized_messages
        return data

    @property
    def iteration_count(self) -> int:
        """Count the number of proposal messages (iterations)."""
        return sum(1 for msg in self.messages if msg.type == "proposal")

    @property
    def last_proposal(self) -> ProposalMessage | None:
        """Get the most recent proposal message."""
        for msg in reversed(self.messages):
            if msg.type == "proposal":
                return msg
        return None

    @property
    def last_feedback(self) -> FeedbackMessage | None:
        """Get the most recent feedback message."""
        for msg in reversed(self.messages):
            if msg.type == "feedback":
                return msg
        return None

    @property
    def approved(self) -> bool:
        """Check if the formalization has been approved by the reviewer."""
        for msg in reversed(self.messages):
            if isinstance(msg, ReviewApprovedFeedback):
                return True
        return False


class ProverResult(BaseModel):
    """Output schema for prover proposer agent."""

    imports: list[str] = Field(
        default_factory=list,
        description="List of import statements needed for this proof that are not yet imported in the file (e.g., ['Mathlib.Topology.Basic', 'MyProject.Algebra.Ring'])",
    )
    opens: list[str] = Field(
        default_factory=list,
        description="List of namespace opens needed for this proof that are not yet opened in the file (e.g., ['Algebra', 'Topology.Basic'])",
    )
    updated_theorem: str = Field(
        description=(
            "Theorem item with proof body. Include the full theorem statement and proof.\n"
            "Example:\n"
            "theorem foo (n : Nat) : n + 0 = n := by\n"
            "  have h1 : n = n := rfl  -- TODO: Intermediate step\n"
            "  omega"
        ),
    )


# Define structured output with explicit boolean checks
# The LLM can hallucinate in 'check3', 'approved', and 'reasoning', but we derive
# the real approval from check1 and check2 ONLY
class ReviewDecision(BaseModel):
    check_1: bool = Field(description="True if statements are identical, False otherwise")
    check_2: bool = Field(description="True if NEW PROOF contains no sorry/admit, False otherwise")
    check_3: bool = Field(
        description="True if no other issues found (undefined refs, syntax errors, etc.), False if issues found"
    )
    approved: bool = Field(description="True if all checks pass (ignored, derived from checks)")
