"""Truncation detection: a cut-off response arrives as a normal message, not an exception."""

import logging

from langchain_core.messages import AIMessage

from ax_prover.utils.llm import _warn_if_truncated, was_truncated


def _message(finish_reason: str | None = None, stop_reason: str | None = None) -> AIMessage:
    metadata = {}
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    if stop_reason is not None:
        metadata["stop_reason"] = stop_reason
    return AIMessage(content="partial", response_metadata=metadata)


def test_openai_length_is_truncation():
    assert was_truncated(_message(finish_reason="length"))


def test_anthropic_max_tokens_is_truncation():
    assert was_truncated(_message(stop_reason="max_tokens"))


def test_google_upper_case_max_tokens_is_truncation():
    assert was_truncated(_message(finish_reason="MAX_TOKENS"))


def test_normal_stop_is_not_truncation():
    assert not was_truncated(_message(finish_reason="stop"))


def test_tool_calls_finish_is_not_truncation():
    assert not was_truncated(_message(finish_reason="tool_calls"))


def test_missing_metadata_is_not_truncation():
    assert not was_truncated(AIMessage(content="fine"))


def test_truncation_warns_with_token_count(caplog):
    message = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={"input_tokens": 10, "output_tokens": 64000, "total_tokens": 64010},
    )
    with caplog.at_level(logging.WARNING):
        _warn_if_truncated(message)

    assert "truncated" in caplog.text
    assert "64000" in caplog.text


def test_complete_response_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_truncated(_message(finish_reason="stop"))

    assert caplog.text == ""
