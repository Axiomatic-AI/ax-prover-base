"""Tests for normalized reasoning extraction."""

from langchain_core.messages import AIMessage

from ax_prover.utils.llm import LLMClient, get_reasoning


def test_get_reasoning_prefers_openai_reasoning_content() -> None:
    response = AIMessage(
        content="final",
        additional_kwargs={"reasoning_content": "private trace"},
    )

    assert get_reasoning(response) == "private trace"


def test_get_reasoning_supports_content_blocks() -> None:
    response = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "first"},
            {"type": "text", "text": "final"},
        ]
    )

    assert get_reasoning(response) == "first"


def test_llm_client_normalizes_none_profile() -> None:
    client = LLMClient.__new__(LLMClient)
    client._base_llm = type("ModelWithNoneProfile", (), {"profile": None})()

    assert client.profile == {}
