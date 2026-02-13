"""Memory processing implementations for the prover agent."""

from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.retry import RunnableRetry

from ..config import LLMConfig
from ..models import ProverAgentState
from ..models.messages import FeedbackMessage, ProposalMessage
from ..utils import get_logger
from ..utils.llm import create_llm
from .prompts import ATTEMPT_TEMPLATE


class BaseMemory(ABC):
    """Base class for prover memory processing strategies."""

    llm: RunnableRetry | None = None

    def __init__(self, llm_config: LLMConfig | None = None):
        if llm_config:
            self.llm = create_llm(LLMConfig(**llm_config))

        self.logger = get_logger(__name__)

    @abstractmethod
    async def process(self, state: ProverAgentState) -> dict:
        """Process the state and return experience for the next iteration."""
        pass


class MemorylessProcessor(BaseMemory):
    """Memory processor that provides no historical context."""

    async def process(self, state: ProverAgentState) -> dict:
        """Return empty experience."""
        return {"experience": ""}


PROPOSER_PAST_K_USER_PROMPT = """
These are your most recent attempts preceding the current one (right above) in reverse chronological order (most recent first):

<previous-attempts>
{previous_attempts}
</previous-attempts>
"""


class PreviousKProcessor(BaseMemory):
    """Provide K-1 most recent attempts as full context."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        k: int = 1,
    ):
        super().__init__(llm_config=llm_config)
        self.k = k

    def _find_previous_proposal(
        self, messages: list, feedback_msg: FeedbackMessage
    ) -> ProposalMessage | None:
        """Find the proposal message that preceded a given feedback message."""
        feedback_index = messages.index(feedback_msg)
        for prev_msg in reversed(messages[:feedback_index]):
            if isinstance(prev_msg, ProposalMessage):
                return prev_msg
        return None

    def _find_last_k_attempts(
        self, state: ProverAgentState, k: int
    ) -> list[tuple[ProposalMessage, FeedbackMessage]]:
        """Find last k-1 attempts in reverse chronological order (most recent first).

        Excludes the most recent attempt which is already passed by _proposer_node.

        """
        if k <= 1:
            return []

        attempts = []
        for msg in state.messages:
            if isinstance(msg, FeedbackMessage):
                proposal = self._find_previous_proposal(state.messages, msg)
                if proposal:
                    attempts.append((proposal, msg))

        last_k_minus_one = attempts[-k:-1] if len(attempts) >= k else attempts[:-1]
        return last_k_minus_one[::-1]

    async def process(self, state: ProverAgentState) -> dict:
        """Provide K-1 most recent attempts as full context.

        The most recent attempt is already passed by _proposer_node, so we provide
        K-1 older attempts to give the prover more historical context.
        """
        previous_attempts = self._find_last_k_attempts(state, self.k)

        if not previous_attempts:
            return {"experience": ""}

        formatted = []
        for proposal, feedback in previous_attempts:
            formatted.append(
                ATTEMPT_TEMPLATE.format(
                    reasoning=proposal.reasoning,
                    code=proposal.code,
                    feedback=feedback.content,
                )
            )

        formatted_attempts = "\n\n".join(formatted)
        experience = PROPOSER_PAST_K_USER_PROMPT.format(previous_attempts=formatted_attempts)

        return {"experience": experience}


PROPOSER_EXPERIENCE_USER_PROMPT = """
What follows is additional context with relevant lessons learned from your previous attempts at proving this theorem.

<experience>
{experience}
</experience>
"""


EXPERIENCE_PROCESSOR_SYSTEM_PROMPT = f"""
An LLM agent is trying to prove a theorem in Lean 4 and it failed to successfully complete the full proof.
This prover agent can work iteratively and take multiple attempts to prove the theorem, but it does not have access to its own history of previous attempts.
Instead, it is your task to provide it with meaningful context condensing the experience gained from its previous attempts, which will help it succeed in its next one at proving the theorem.
Therefore, when the prover takes another attempt at proving the theorem, it will be given both the current state of the proof (code and feedback) plus the context that you prepare.

You will recieve a message containing the last attempt's information: reasoning that the LLM output before generating the code, the Lean 4 code itself, and the feedback that comes either from the `lake build` output or a reviewer agent.
Additionally, the message will contain the previous context that you had prepared for the prover agent synthesizing its past experience to help it generate this code.
If the previous context was empty, it means this was the first attempt at proving the theorem.

Reflect on the current attempt in light of the previous context and experience, and extract the key lessons to avoid repeating the same mistakes and any of the previous ones in the future.
Write the context that will be given to the prover agent in the next iteration.
Ensure that the important information from the experience captured in the previous context is never lost when writing the new context, as we want to prevent the prover agent from repeating the same mistakes even after multiple future iterations.
You should stick to the facts and refrain from proposing new actions or strategies, as the prover agent will draw the necessary conclusions by itself with the information that you provide.
Keep in mind that this context should not be excessively long, and it should be concise and to the point with the goal of maximizing the performance of the prover agent.

Your output will be given to the prover agent in another message, where it will be the content embedded within the <experience> tag in the following template:
<message-template>
{PROPOSER_EXPERIENCE_USER_PROMPT}
</message-template>"""

EXPERIENCE_PROCESSOR_USER_PROMPT = (
    ATTEMPT_TEMPLATE
    + """

<previous-context>
{previous_context}
</previous-context>
"""
)


class ExperienceProcessor(BaseMemory):
    """Process current attempt to extract key lessons and insights."""

    async def process(self, state: ProverAgentState) -> dict:
        """Extract and summarize key lessons from the current attempt."""
        self.logger.debug("Processing the current attempt and extracting key lessons and insights.")

        # If there's no proposal yet (proposer failed before creating one), skip experience processing
        if not state.last_proposal:
            self.logger.warning(
                "No proposal to process - proposer likely failed with structured output "
                "parsing error. Skipping experience processing."
            )
            return {}

        context_summary_prompt = EXPERIENCE_PROCESSOR_USER_PROMPT.format(
            reasoning=state.last_proposal.reasoning,
            code=state.last_proposal.code,
            feedback=state.last_feedback.content,
            previous_context=state.experience,
        )

        experience_response = await self.llm.ainvoke(
            [
                SystemMessage(content=EXPERIENCE_PROCESSOR_SYSTEM_PROMPT),
                HumanMessage(content=context_summary_prompt),
            ]
        )

        experience = PROPOSER_EXPERIENCE_USER_PROMPT.format(experience=experience_response.text)
        return {"experience": experience}
