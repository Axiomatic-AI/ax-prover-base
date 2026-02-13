"""Message types for the formalization workflow."""

from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ConfigDict, Field

from .files import Location


class ProposalMessage(AIMessage):
    """A formalization proposal from the proposer node."""

    type: Literal["proposal"] = "proposal"
    reasoning: str
    location: Location | None
    imports: list[str] = Field(default_factory=list)
    opens: list[str] = Field(default_factory=list)
    code: str

    def __init__(
        self,
        reasoning: str,
        code: str,
        location: Location | None = None,
        imports: list[str] | None = None,
        opens: list[str] | None = None,
        **kwargs,
    ):
        """Initialize with auto-formatted content from structured fields."""
        kwargs.pop("content", None)
        if isinstance(location, dict):
            location = Location(**location)
        imports_list = imports or []
        opens_list = opens or []
        imports_str = f"\n\nImports: {', '.join(imports_list)}" if imports_list else ""
        opens_str = f"\n\nOpens: {', '.join(opens_list)}" if opens_list else ""
        location_str = location.formatted_context if location else "No location"
        content = f"Reasoning: {reasoning}\n\nLocation: {location_str}{imports_str}{opens_str}\n\nCode:\n```lean\n{code}\n```"

        super().__init__(
            content=content.rstrip(),
            reasoning=reasoning,
            location=location,
            imports=imports_list,
            opens=opens_list,
            code=code,
            **kwargs,
        )

    @property
    def has_changes(self) -> bool:
        """Check if this proposal contains any actual changes."""
        return bool(self.imports or self.opens or self.code)


class FeedbackMessage(HumanMessage):
    """
    Abstract base class for all feedback messages.

    Feedback messages represent the results of builder and reviewer nodes,
    providing information for routing decisions and proposer context.
    """

    model_config = ConfigDict(validate_default=True)

    type: Literal["feedback"] = "feedback"  # For compatibility with message unions
    feedback_type: str  # Discriminator for Pydantic
    timestamp: datetime = Field(default_factory=datetime.now)
    is_success: bool
    is_terminal: bool = False


class BuildSuccessFeedback(FeedbackMessage):
    """Build succeeded - continue to reviewer or extractor."""

    feedback_type: Literal["build_success"] = "build_success"
    is_success: bool = True

    def __init__(self, **kwargs):
        """Initialize with empty content (success not shown to LLM)."""
        kwargs.pop("content", None)
        super().__init__(content="", **kwargs)


class BuildFailedFeedback(FeedbackMessage):
    """Build failed - recoverable, should retry proposer."""

    feedback_type: Literal["build_failed"] = "build_failed"
    error_output: str
    is_success: bool = False

    def __init__(self, error_output: str, **kwargs):
        """Initialize with formatted error message."""
        kwargs.pop("content", None)
        content = f"BUILD FAILED:\n\n{error_output}"
        super().__init__(content=content, error_output=error_output, **kwargs)


class ReviewApprovedFeedback(FeedbackMessage):
    """Review approved - workflow complete or continue to dependencies."""

    feedback_type: Literal["review_approved"] = "review_approved"
    comments: str = ""
    is_success: bool = True

    def __init__(self, comments: str = "", **kwargs):
        """Initialize with empty content (success not shown to LLM)."""
        kwargs.pop("content", None)
        super().__init__(content="", comments=comments, **kwargs)


class ReviewRejectedFeedback(FeedbackMessage):
    """Review rejected - recoverable, should retry proposer."""

    feedback_type: Literal["review_rejected"] = "review_rejected"
    feedback: str
    is_success: bool = False

    def __init__(self, feedback: str, **kwargs):
        """Initialize with formatted rejection message."""
        kwargs.pop("content", None)
        content = f"REVIEW NEEDS REVISION:\n\n{feedback}"
        super().__init__(content=content, feedback=feedback, **kwargs)


class MaxIterationsFeedback(FeedbackMessage):
    """Terminal failure - iteration limit exceeded."""

    feedback_type: Literal["max_iterations"] = "max_iterations"
    max_iterations: int
    is_success: bool = False
    is_terminal: bool = True

    def __init__(self, max_iterations: int, **kwargs):
        """Initialize with formatted max iterations message."""
        kwargs.pop("content", None)
        content = f"Maximum iteration limit ({max_iterations}) reached."
        super().__init__(content=content, max_iterations=max_iterations, **kwargs)


class MissingTargetTheoremFeedback(FeedbackMessage):
    """Target theorem not found in proposed code - validation failure."""

    feedback_type: Literal["missing_target_theorem"] = "missing_target_theorem"
    theorem_name: str
    is_success: bool = False

    def __init__(self, theorem_name: str, **kwargs):
        """Initialize with formatted missing theorem message."""
        kwargs.pop("content", None)
        content = (
            f"MISSING TARGET THEOREM: The proposed code does not contain the target theorem '{theorem_name}'. "
            "Please ensure your code includes the complete definition/theorem that was requested."
        )
        super().__init__(content=content, theorem_name=theorem_name, **kwargs)


class SorriesGoalStateFeedback(FeedbackMessage):
    """Proposed code contains sorries."""

    feedback_type: Literal["sorries_goal_state"] = "sorries_goal_state"
    sorry_count: int
    goal_state_at_sorries: str
    is_success: bool = False

    def __init__(self, sorry_count: int, goal_state_at_sorries: str, **kwargs):
        """Initialize with formatted sorries goal state message."""
        kwargs.pop("content", None)
        content = (
            f"SORRIES DETECTED: The proposed code contains {sorry_count} sorry/admit. "
            f"Work on closing these sorries. Here is the Goal State at these sorries {goal_state_at_sorries} ."
        )
        super().__init__(
            content=content,
            sorry_count=sorry_count,
            goal_state_at_sorries=goal_state_at_sorries,
            **kwargs,
        )


class StructuredOutputParsingFailedFeedback(FeedbackMessage):
    """Structured output parsing failed - LLM couldn't produce valid output."""

    feedback_type: Literal["structured_output_parsing_failed"] = "structured_output_parsing_failed"
    error_message: str
    is_success: bool = False

    def __init__(self, error_message: str, **kwargs):
        """Initialize with formatted parsing failed message."""
        kwargs.pop("content", None)
        content = (
            f"STRUCTURED OUTPUT PARSING FAILED: The system could not parse the structured output.\n\nError: {error_message}\n\n"
            "Please ensure your output follows the required schema format."
        )
        super().__init__(content=content, error_message=error_message, **kwargs)


# Auto-derived mapping from feedback_type to subclass (for deserialization)
FEEDBACK_CONSTRUCTORS: dict[str, type[FeedbackMessage]] = {
    cls.model_fields["feedback_type"].default: cls for cls in FeedbackMessage.__subclasses__()
}

FormalizationMessage = ProposalMessage | FeedbackMessage
