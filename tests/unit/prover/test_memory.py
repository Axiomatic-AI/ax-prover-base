"""Unit tests for prover memory processors."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ax_prover.models import ProverAgentState, TargetItem
from ax_prover.models.files import Location
from ax_prover.models.messages import (
    BuildFailedFeedback,
    BuildSuccessFeedback,
    ProposalMessage,
)
from ax_prover.prover.memory import (
    BaseMemory,
    ExperienceProcessor,
    MemorylessProcessor,
    PreviousKProcessor,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock()
    return llm


def _make_processor(cls, mock_llm=None, **kwargs):
    """Create a processor and optionally inject a mock LLM."""
    processor = cls(**kwargs)
    if mock_llm is not None:
        processor.llm = mock_llm
    return processor


@pytest.fixture
def sample_location():
    """Create a sample location for testing."""
    return Location(name="sample_theorem", module_path="Test.Module", is_external=False)


@pytest.fixture
def sample_item(sample_location):
    """Create a sample formalization item for testing."""
    return TargetItem(
        title="Sample Theorem",
        reference="Let P be a theorem. Then P holds.",
        description="A test theorem",
        location=sample_location,
    )


@pytest.fixture
def empty_state(sample_item):
    """Create an empty prover state for testing."""
    return ProverAgentState(item=sample_item)


@pytest.fixture
def state_with_proposal(sample_item, sample_location):
    """Create a state with a proposal and feedback."""
    proposal = ProposalMessage(
        reasoning="Test reasoning",
        location=sample_location,
        imports=[],
        opens=[],
        code="theorem sample_theorem : True := trivial",
    )
    feedback = BuildFailedFeedback(error_output="Test error")
    return ProverAgentState(
        item=sample_item,
        messages=[proposal, feedback],
    )


@pytest.fixture
def state_with_multiple_attempts(sample_item, sample_location):
    """Create a state with multiple proposal/feedback pairs."""
    messages = []
    for i in range(3):
        proposal = ProposalMessage(
            reasoning=f"Attempt {i + 1} reasoning",
            location=sample_location,
            imports=[],
            opens=[],
            code=f"theorem sample_theorem : True := by\n  -- Attempt {i + 1}\n  trivial",
        )
        feedback = (
            BuildSuccessFeedback() if i == 2 else BuildFailedFeedback(error_output=f"Error {i + 1}")
        )
        messages.extend([proposal, feedback])

    return ProverAgentState(
        item=sample_item,
        messages=messages,
    )


class TestBaseMemory:
    """Test the abstract BaseMemory class."""

    def test_base_memory_is_abstract(self):
        """BaseMemory cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMemory()

    def test_base_memory_requires_process_implementation(self):
        """Subclasses must implement the process method."""

        class IncompleteMemory(BaseMemory):
            pass

        with pytest.raises(TypeError):
            IncompleteMemory()


class TestMemorylessProcessor:
    """Test MemorylessProcessor implementation."""

    def test_initialization_without_llm(self):
        """MemorylessProcessor can be initialized without LLM."""
        processor = MemorylessProcessor()
        assert processor.llm is None

    def test_initialization_with_llm_config(self):
        """MemorylessProcessor can be initialized with llm_config (though unused without valid config)."""
        processor = MemorylessProcessor(llm_config=None)
        assert processor.llm is None

    @pytest.mark.asyncio
    async def test_process_returns_empty_experience(self, empty_state):
        """MemorylessProcessor always returns empty experience."""
        processor = MemorylessProcessor()
        result = await processor.process(empty_state)
        assert result == {"experience": ""}

    @pytest.mark.asyncio
    async def test_process_with_proposal_returns_empty(self, state_with_proposal):
        """MemorylessProcessor returns empty even with proposals."""
        processor = MemorylessProcessor()
        result = await processor.process(state_with_proposal)
        assert result == {"experience": ""}

    @pytest.mark.asyncio
    async def test_process_with_multiple_attempts_returns_empty(self, state_with_multiple_attempts):
        """MemorylessProcessor returns empty even with multiple attempts."""
        processor = MemorylessProcessor()
        result = await processor.process(state_with_multiple_attempts)
        assert result == {"experience": ""}


class TestExperienceProcessor:
    """Test ExperienceProcessor implementation."""

    def test_initialization_requires_llm(self):
        """ExperienceProcessor can be initialized without LLM."""
        processor = ExperienceProcessor()
        assert processor.llm is None  # Can be None initially

    def test_initialization_with_llm(self, mock_llm):
        """ExperienceProcessor stores the provided LLM."""
        processor = _make_processor(ExperienceProcessor, mock_llm=mock_llm)
        assert processor.llm == mock_llm

    @pytest.mark.asyncio
    async def test_process_without_proposal_returns_empty(self, empty_state, mock_llm):
        """ExperienceProcessor returns empty dict when no proposal exists."""
        processor = _make_processor(ExperienceProcessor, mock_llm=mock_llm)
        result = await processor.process(empty_state)
        assert result == {}
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_with_proposal_calls_llm(self, state_with_proposal, mock_llm):
        """ExperienceProcessor calls LLM to summarize experience."""
        mock_response = MagicMock()
        mock_response.text = "Key insight: The proof attempt failed due to X."
        mock_llm.ainvoke.return_value = mock_response

        processor = _make_processor(ExperienceProcessor, mock_llm=mock_llm)
        result = await processor.process(state_with_proposal)

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # System and user messages
        assert "System" in str(type(call_args[0]))
        assert "Human" in str(type(call_args[1]))

        assert "experience" in result
        assert "Key insight" in result["experience"]

    @pytest.mark.asyncio
    async def test_process_includes_previous_context(self, state_with_proposal, mock_llm):
        """ExperienceProcessor includes previous experience in prompt."""
        mock_response = MagicMock()
        mock_response.text = "Updated insight."
        mock_llm.ainvoke.return_value = mock_response

        state_with_proposal.experience = "Previous insight: X failed."

        processor = _make_processor(ExperienceProcessor, mock_llm=mock_llm)
        await processor.process(state_with_proposal)

        call_args = mock_llm.ainvoke.call_args[0][0]
        user_message = call_args[1].content
        assert "Previous insight" in user_message

    @pytest.mark.asyncio
    async def test_process_formats_experience_for_proposer(self, state_with_proposal, mock_llm):
        """ExperienceProcessor formats experience for use in proposer."""
        mock_response = MagicMock()
        mock_response.text = "Summarized lesson"
        mock_llm.ainvoke.return_value = mock_response

        processor = _make_processor(ExperienceProcessor, mock_llm=mock_llm)
        result = await processor.process(state_with_proposal)

        assert "experience" in result
        assert "Summarized lesson" in result["experience"]
        assert (
            "## Past Insights" in result["experience"]
            or "experience" in result["experience"].lower()
        )


class TestPreviousKProcessor:
    """Test PreviousKProcessor implementation."""

    def test_initialization_with_k(self):
        """PreviousKProcessor stores k parameter."""
        processor = PreviousKProcessor(k=3)
        assert processor.k == 3

    def test_initialization_default_k(self):
        """PreviousKProcessor defaults to k=1."""
        processor = PreviousKProcessor()
        assert processor.k == 1

    @pytest.mark.asyncio
    async def test_process_with_k_1_returns_empty(self, state_with_multiple_attempts):
        """PreviousKProcessor with k=1 returns empty (most recent handled by proposer)."""
        processor = PreviousKProcessor(k=1)
        result = await processor.process(state_with_multiple_attempts)
        assert result == {"experience": ""}

    @pytest.mark.asyncio
    async def test_process_without_attempts_returns_empty(self, empty_state):
        """PreviousKProcessor returns empty when no attempts exist."""
        processor = PreviousKProcessor(k=5)
        result = await processor.process(empty_state)
        assert result == {"experience": ""}

    @pytest.mark.asyncio
    async def test_process_formats_previous_attempts(self, state_with_multiple_attempts):
        """PreviousKProcessor formats K-1 previous attempts."""
        processor = PreviousKProcessor(k=3)
        result = await processor.process(state_with_multiple_attempts)

        assert "experience" in result
        experience = result["experience"]
        assert "Attempt 1" in experience
        assert "Attempt 2" in experience
        assert "Attempt 3" not in experience or experience.count("Attempt") == 2

    @pytest.mark.asyncio
    async def test_process_with_k_larger_than_attempts(self, state_with_multiple_attempts):
        """PreviousKProcessor handles k larger than available attempts."""
        processor = PreviousKProcessor(k=10)
        result = await processor.process(state_with_multiple_attempts)

        assert "experience" in result
        experience = result["experience"]
        assert "Attempt 1" in experience
        assert "Attempt 2" in experience

    @pytest.mark.asyncio
    async def test_process_includes_reasoning_and_feedback(self, state_with_multiple_attempts):
        """PreviousKProcessor includes both reasoning and feedback."""
        processor = PreviousKProcessor(k=3)
        result = await processor.process(state_with_multiple_attempts)

        experience = result["experience"]
        assert "reasoning" in experience.lower()
        assert "Error 1" in experience or "error" in experience.lower()

    @pytest.mark.asyncio
    async def test_find_previous_proposal(self, state_with_multiple_attempts):
        """Test internal _find_previous_proposal method."""
        processor = PreviousKProcessor(k=3)

        feedback = state_with_multiple_attempts.messages[3]  # Index 3 is feedback for attempt 2
        proposal = processor._find_previous_proposal(
            state_with_multiple_attempts.messages, feedback
        )

        assert proposal is not None
        assert isinstance(proposal, ProposalMessage)
        assert "Attempt 2" in proposal.reasoning

    @pytest.mark.asyncio
    async def test_attempts_in_reverse_chronological_order(self, state_with_multiple_attempts):
        """PreviousKProcessor provides attempts in reverse chronological order."""
        processor = PreviousKProcessor(k=3)
        result = await processor.process(state_with_multiple_attempts)

        experience = result["experience"]
        attempt_2_pos = experience.find("Attempt 2")
        attempt_1_pos = experience.find("Attempt 1")
        assert attempt_2_pos < attempt_1_pos


class TestMemoryIntegration:
    """Integration tests for memory processors."""

    @pytest.mark.asyncio
    async def test_all_processors_implement_base_interface(self, mock_llm):
        """All memory processors implement BaseMemory interface."""
        processors = [
            MemorylessProcessor(),
            _make_processor(ExperienceProcessor, mock_llm=mock_llm),
            PreviousKProcessor(k=3),
        ]

        for processor in processors:
            assert isinstance(processor, BaseMemory)
            assert hasattr(processor, "process")
            assert callable(processor.process)

    @pytest.mark.asyncio
    async def test_all_processors_return_dict(self, empty_state, state_with_proposal, mock_llm):
        """All memory processors return dict with experience key or empty dict."""
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_llm.ainvoke.return_value = mock_response

        processors = [
            MemorylessProcessor(),
            _make_processor(ExperienceProcessor, mock_llm=mock_llm),
            PreviousKProcessor(k=1),
        ]

        for processor in processors:
            result = await processor.process(state_with_proposal)
            assert isinstance(result, dict)
            # Should either have experience key or be empty
            assert "experience" in result or result == {}
