"""Tests for lossless reasoning alignment and entropy calculations."""

import gzip
import json
import math

import pytest
from langchain_core.messages import AIMessage

from ax_prover.config import ReasoningTraceConfig
from ax_prover.utils.reasoning_trace import (
    LLMTraceCapture,
    LLMTraceContext,
    RawHTTPExchange,
    ReasoningAlignmentError,
    ReasoningTraceWriter,
    analyze_provider_payload,
)


def _entry(token: str, probability: float, alternatives: list[tuple[str, float]]) -> dict:
    return {
        "token": token,
        "bytes": list(token.encode()),
        "logprob": math.log(probability),
        "top_logprobs": [
            {
                "token": alternative,
                "bytes": list(alternative.encode()),
                "logprob": math.log(alternative_probability),
            }
            for alternative, alternative_probability in alternatives
        ],
    }


def _payload() -> dict:
    return {
        "id": "completion-1",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "think",
                    "content": '{"updated_theorem":"theorem t : True := by trivial"}',
                },
                "logprobs": {
                    "content": [
                        _entry("think", 0.5, [("think", 0.5), ("prove", 0.25)]),
                        _entry(
                            '{"updated_theorem":"theorem t : True := by trivial"}',
                            0.75,
                            [
                                (
                                    '{"updated_theorem":"theorem t : True := by trivial"}',
                                    0.75,
                                ),
                                ("[]", 0.1),
                            ],
                        ),
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 2},
    }


def test_aligns_only_reasoning_tokens_and_computes_entropy() -> None:
    analysis = analyze_provider_payload(_payload(), vocabulary_size=4)

    assert analysis["alignment_status"] == "aligned"
    assert analysis["reasoning_token_count"] == 1
    assert analysis["reasoning_token_start"] == 0
    assert analysis["reasoning_token_end"] == 1
    assert analysis["final_alignment_status"] == "aligned"
    assert analysis["final_token_count"] == 1
    token = analysis["tokens"][0]
    assert token["alternative_count"] == 2  # sampled token is de-duplicated
    assert token["sampled_surprisal"] == pytest.approx(-math.log(0.5))
    assert token["returned_mass"] == pytest.approx(0.75)
    assert token["partial_entropy"] == pytest.approx(-(0.5 * math.log(0.5) + 0.25 * math.log(0.25)))
    assert token["entropy_lower_bound"] == pytest.approx(
        token["partial_entropy"] - 0.25 * math.log(0.25)
    )
    assert token["entropy_upper_bound"] == pytest.approx(
        token["partial_entropy"] - 0.25 * math.log(0.25 / 2)
    )


def test_zero_tail_has_equal_entropy_bounds() -> None:
    payload = _payload()
    payload["choices"][0]["logprobs"]["content"][0] = _entry(
        "think", 0.5, [("think", 0.5), ("prove", 0.5)]
    )

    token = analyze_provider_payload(payload, vocabulary_size=2)["tokens"][0]

    assert token["tail_mass"] == pytest.approx(0.0)
    assert token["entropy_lower_bound"] == pytest.approx(math.log(2))
    assert token["entropy_upper_bound"] == pytest.approx(math.log(2))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda payload: payload["choices"][0].pop("logprobs"), "logprobs_missing"),
        (
            lambda payload: payload["choices"][0]["message"].update(reasoning_content="absent"),
            "reasoning_not_found_in_logprob_tokens",
        ),
    ],
)
def test_reports_missing_alignment(mutate, expected: str) -> None:
    payload = _payload()
    mutate(payload)

    assert analyze_provider_payload(payload, vocabulary_size=None)["alignment_status"] == expected


def test_aligns_reasoning_between_special_tokens() -> None:
    payload = _payload()
    entries = payload["choices"][0]["logprobs"]["content"]
    payload["choices"][0]["logprobs"]["content"] = [
        _entry("<think>", 0.9, [("<think>", 0.9)]),
        entries[0],
        _entry("</think>", 0.9, [("</think>", 0.9)]),
        entries[1],
    ]

    analysis = analyze_provider_payload(payload, vocabulary_size=151936)

    assert analysis["alignment_status"] == "aligned"
    assert analysis["reasoning_token_start"] == 1
    assert analysis["reasoning_token_end"] == 2
    assert analysis["final_token_start"] == 3
    assert analysis["alignment_basis"] == "qwen3_delimiter_token_span"


def test_aligns_unclosed_reasoning_after_length_stop() -> None:
    payload = _payload()
    payload["choices"][0]["finish_reason"] = "length"
    payload["choices"][0]["message"] = {"content": "<think>truncated thought"}
    payload["choices"][0]["logprobs"]["content"] = [
        _entry("<think>", 0.9, [("<think>", 0.9)]),
        _entry("truncated thought", 0.8, [("truncated thought", 0.8)]),
    ]

    analysis = analyze_provider_payload(payload, vocabulary_size=151936)

    assert analysis["alignment_status"] == "aligned"
    assert analysis["reasoning_field"] == "content_unclosed_think_truncated"
    assert analysis["reasoning"] == "truncated thought"
    assert analysis["reasoning_token_count"] == 1
    assert analysis["final_content"] == ""
    assert analysis["final_alignment_status"] == "empty"
    assert analysis["finish_reason"] == "length"


def test_truncated_region_uses_token_boundaries_when_provider_bytes_replace_unicode() -> None:
    payload = _payload()
    payload["choices"][0]["finish_reason"] = "length"
    payload["choices"][0]["message"] = {"content": "<think>∑ thought"}
    unicode_entry = _entry("∑ thought", 0.8, [("∑ thought", 0.8)])
    unicode_entry["bytes"] = [239, 191, 189, 32, 116, 104, 111, 117, 103, 104, 116]
    payload["choices"][0]["logprobs"]["content"] = [
        _entry("<think>", 0.9, [("<think>", 0.9)]),
        unicode_entry,
    ]

    analysis = analyze_provider_payload(payload, vocabulary_size=151936)

    assert analysis["alignment_status"] == "aligned"
    assert analysis["alignment_basis"] == "opening_think_delimiter_to_length_stop"
    assert analysis["reasoning_token_count"] == 1
    assert analysis["reasoning_decode_matches_provider"] is False


def test_closed_delimiters_align_unicode_despite_replacement_provider_bytes() -> None:
    payload = _payload()
    payload["choices"][0]["message"]["reasoning_content"] = "∑ thought"
    payload["choices"][0]["message"]["content"] = '{"updated_theorem":"theorem ∑"}'
    reasoning_entry = _entry("∑ thought", 0.8, [("∑ thought", 0.8)])
    reasoning_entry["bytes"] = [239, 191, 189, 32, 116, 104, 111, 117, 103, 104, 116]
    final_entry = _entry('{"updated_theorem":"theorem ∑"}', 0.8, [])
    final_entry["bytes"] = list('{"updated_theorem":"theorem �"}'.encode())
    payload["choices"][0]["logprobs"]["content"] = [
        _entry("<think>", 0.9, [("<think>", 0.9)]),
        reasoning_entry,
        _entry("</think>", 0.9, [("</think>", 0.9)]),
        final_entry,
        _entry("<|im_end|>", 0.9, [("<|im_end|>", 0.9)]),
    ]

    analysis = analyze_provider_payload(payload, vocabulary_size=151936)

    assert analysis["alignment_status"] == "aligned"
    assert analysis["alignment_basis"] == "qwen3_delimiter_token_span"
    assert analysis["reasoning_token_count"] == 1
    assert analysis["reasoning_decode_matches_provider"] is False
    assert analysis["final_alignment_status"] == "aligned"
    assert analysis["final_token_count"] == 1
    assert analysis["final_decode_matches_provider"] is False


def test_first_closing_delimiter_aligns_reasoning_when_later_closings_repeat() -> None:
    payload = _payload()
    payload["choices"][0]["message"]["reasoning_content"] = "∑ thought"
    payload["choices"][0]["message"]["content"] = "finalfinal"
    reasoning_entry = _entry("∑ thought", 0.8, [("∑ thought", 0.8)])
    reasoning_entry["bytes"] = [239, 191, 189, 32, 116, 104, 111, 117, 103, 104, 116]
    payload["choices"][0]["logprobs"]["content"] = [
        _entry("<think>", 0.9, [("<think>", 0.9)]),
        reasoning_entry,
        _entry("</think>", 0.9, [("</think>", 0.9)]),
        _entry("final", 0.8, [("final", 0.8)]),
        _entry("</think>", 0.9, [("</think>", 0.9)]),
        _entry("final", 0.8, [("final", 0.8)]),
        _entry("<|im_end|>", 0.9, [("<|im_end|>", 0.9)]),
    ]

    analysis = analyze_provider_payload(payload, vocabulary_size=151936)

    assert analysis["alignment_status"] == "aligned"
    assert analysis["alignment_basis"] == "qwen3_first_closing_delimiter_token_span"
    assert analysis["reasoning_token_start"] == 1
    assert analysis["reasoning_token_end"] == 2
    assert analysis["reasoning_token_count"] == 1
    assert analysis["reasoning_decode_matches_provider"] is False


def test_rejects_reasoning_boundary_that_splits_a_token() -> None:
    payload = _payload()
    payload["choices"][0]["logprobs"]["content"][0] = _entry(
        "prefixthink", 0.5, [("prefixthink", 0.5)]
    )

    analysis = analyze_provider_payload(payload, vocabulary_size=None)

    assert analysis["alignment_status"] == "reasoning_boundary_splits_token"


def test_rejects_ambiguous_reasoning_alignment() -> None:
    payload = _payload()
    payload["choices"][0]["logprobs"]["content"] = [
        _entry("think", 0.5, [("think", 0.5)]),
        _entry("think", 0.5, [("think", 0.5)]),
        payload["choices"][0]["logprobs"]["content"][1],
    ]

    analysis = analyze_provider_payload(payload, vocabulary_size=None)

    assert analysis["alignment_status"] == "reasoning_alignment_ambiguous"


def test_writer_persists_lossless_call_tokens_and_lean_link(tmp_path) -> None:
    context = LLMTraceContext(
        run_id="run-1",
        problem_uuid="problem-1",
        theorem_name="t",
        iteration=1,
        call_index=1,
        role="proposer",
    )
    payload = _payload()
    capture = LLMTraceCapture(
        context=context,
        exchanges=[
            RawHTTPExchange(
                captured_at="2026-08-20T00:00:00+00:00",
                method="POST",
                url="http://127.0.0.1:8000/v1/chat/completions",
                status_code=200,
                request_headers={"content-type": "application/json"},
                response_headers={"content-type": "application/json"},
                request_body_utf8=json.dumps({"messages": [{"role": "user", "content": "p"}]}),
                response_body_utf8=json.dumps(payload),
            )
        ],
    )
    writer = ReasoningTraceWriter(
        ReasoningTraceConfig(
            enabled=True,
            output_dir=str(tmp_path),
            run_id="run-1",
            problem_uuid="problem-1",
            vocabulary_size=4,
        )
    )

    summary = writer.record_proposer(
        capture=capture,
        normalized_response=AIMessage(
            content=payload["choices"][0]["message"]["content"],
            additional_kwargs={"reasoning_content": "think"},
        ),
    )
    check_id = writer.record_lean_check(
        call_id=summary["call_id"],
        problem_uuid="problem-1",
        theorem_name="t",
        iteration=1,
        outcome="build_success",
        success=True,
        feedback_type="build_success",
        diagnostics="",
        duration_seconds=0.25,
    )

    call = json.loads((tmp_path / "llm_calls.jsonl").read_text().splitlines()[0])
    with gzip.open(tmp_path / "reasoning_tokens.jsonl.gz", "rt") as handle:
        token = json.loads(handle.readline())
    with gzip.open(tmp_path / "final_tokens.jsonl.gz", "rt") as handle:
        final_token = json.loads(handle.readline())
    check = json.loads((tmp_path / "lean_checks.jsonl").read_text().splitlines()[0])
    assert call["provider_response"] == payload
    assert call["reasoning"] == "think"
    assert token["call_id"] == call["call_id"]
    assert final_token["region"] == "structured_final_output"
    assert final_token["call_id"] == call["call_id"]
    assert check["call_id"] == token["call_id"]
    assert check_id == f"{call['call_id']}:lean"


def test_multiple_problem_writers_append_readable_trace_members(tmp_path) -> None:
    for index in range(1, 5):
        problem_uuid = f"problem-{index}"
        context = LLMTraceContext("run", problem_uuid, f"t{index}", 1, 1, "proposer")
        payload = _payload()
        capture = LLMTraceCapture(
            context=context,
            exchanges=[
                RawHTTPExchange(
                    captured_at="now",
                    method="POST",
                    url="http://local",
                    status_code=200,
                    request_headers={},
                    response_headers={},
                    request_body_utf8="{}",
                    response_body_utf8=json.dumps(payload),
                )
            ],
        )
        writer = ReasoningTraceWriter(
            ReasoningTraceConfig(
                enabled=True,
                output_dir=str(tmp_path),
                run_id="run",
                problem_uuid=problem_uuid,
                vocabulary_size=4,
            )
        )
        writer.record_proposer(
            capture=capture,
            normalized_response=AIMessage(
                content=payload["choices"][0]["message"]["content"],
                additional_kwargs={"reasoning_content": "think"},
            ),
        )

    calls = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    with gzip.open(tmp_path / "reasoning_tokens.jsonl.gz", "rt") as handle:
        reasoning_tokens = [json.loads(line) for line in handle]
    with gzip.open(tmp_path / "final_tokens.jsonl.gz", "rt") as handle:
        final_tokens = [json.loads(line) for line in handle]

    expected = {f"problem-{index}" for index in range(1, 5)}
    assert {call["problem_uuid"] for call in calls} == expected
    assert {token["problem_uuid"] for token in reasoning_tokens} == expected
    assert {token["problem_uuid"] for token in final_tokens} == expected


def test_writer_labels_tool_continuation_as_a_separate_model_call(tmp_path) -> None:
    context = LLMTraceContext("run", "problem", "t", 1, 1, "proposer")
    exchanges = []
    for index in range(2):
        payload = _payload()
        payload["id"] = f"completion-{index}"
        exchanges.append(
            RawHTTPExchange(
                captured_at="now",
                method="POST",
                url="http://local",
                status_code=200,
                request_headers={},
                response_headers={},
                request_body_utf8=json.dumps({"messages": []}),
                response_body_utf8=json.dumps(payload),
            )
        )
    writer = ReasoningTraceWriter(
        ReasoningTraceConfig(
            enabled=True,
            output_dir=str(tmp_path),
            vocabulary_size=4,
        )
    )

    summary = writer.record_proposer(
        capture=LLMTraceCapture(context=context, exchanges=exchanges),
        normalized_response=AIMessage(content="final"),
    )

    calls = [
        json.loads(line)
        for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [call["role"] for call in calls] == ["proposer", "tool_continuation"]
    assert calls[1]["parent_call_id"] == context.call_id
    assert summary["call_id"] == f"{context.call_id}:tool:1"


def test_writer_fails_closed_after_recording_unaligned_call(tmp_path) -> None:
    context = LLMTraceContext("run", "problem", "t", 1, 1, "proposer")
    payload = _payload()
    payload["choices"][0].pop("logprobs")
    capture = LLMTraceCapture(
        context=context,
        exchanges=[
            RawHTTPExchange(
                captured_at="now",
                method="POST",
                url="http://local",
                status_code=200,
                request_headers={},
                response_headers={},
                request_body_utf8="{}",
                response_body_utf8=json.dumps(payload),
            )
        ],
    )
    writer = ReasoningTraceWriter(ReasoningTraceConfig(enabled=True, output_dir=str(tmp_path)))

    with pytest.raises(ReasoningAlignmentError, match="logprobs_missing"):
        writer.record_proposer(capture=capture, normalized_response=AIMessage(content="{}"))

    record = json.loads((tmp_path / "llm_calls.jsonl").read_text())
    assert record["alignment_status"] == "logprobs_missing"


def test_writer_preserves_retry_exchanges_and_parser_failure_link(tmp_path) -> None:
    context = LLMTraceContext("run", "problem", "t", 1, 1, "proposer")
    payload = _payload()
    capture = LLMTraceCapture(
        context=context,
        exchanges=[
            RawHTTPExchange(
                captured_at="first",
                method="POST",
                url="http://local",
                status_code=500,
                request_headers={},
                response_headers={},
                request_body_utf8="{}",
                response_body_utf8='{ "error": "retry" }',
            ),
            RawHTTPExchange(
                captured_at="second",
                method="POST",
                url="http://local",
                status_code=200,
                request_headers={},
                response_headers={},
                request_body_utf8="{}",
                response_body_utf8=json.dumps(payload),
            ),
        ],
    )
    writer = ReasoningTraceWriter(
        ReasoningTraceConfig(enabled=True, output_dir=str(tmp_path), vocabulary_size=151936)
    )

    summary = writer.record_proposer(
        capture=capture,
        normalized_response=AIMessage(content="malformed final response"),
    )
    writer.record_lean_check(
        call_id=summary["call_id"],
        problem_uuid="problem",
        theorem_name="t",
        iteration=1,
        outcome="not_run_structured_output_parse_failed",
        success=False,
        feedback_type="structured_output_parsing_failed",
        diagnostics="invalid JSON",
        duration_seconds=0,
    )

    call = json.loads((tmp_path / "llm_calls.jsonl").read_text())
    check = json.loads((tmp_path / "lean_checks.jsonl").read_text())
    assert len(call["raw_http_exchanges"]) == 2
    assert call["provider_response"] == payload
    assert check["call_id"] == call["call_id"]
    assert check["outcome"] == "not_run_structured_output_parse_failed"


def test_writer_recovers_truncated_reasoning_when_client_normalization_fails(tmp_path) -> None:
    context = LLMTraceContext("run", "problem", "t", 1, 1, "proposer")
    payload = _payload()
    payload["choices"][0]["finish_reason"] = "length"
    payload["choices"][0]["message"] = {"content": "<think>truncated thought"}
    payload["choices"][0]["logprobs"]["content"] = [
        _entry("<think>", 0.9, [("<think>", 0.9)]),
        _entry("truncated thought", 0.8, [("truncated thought", 0.8)]),
    ]
    capture = LLMTraceCapture(
        context=context,
        exchanges=[
            RawHTTPExchange(
                captured_at="now",
                method="POST",
                url="http://local",
                status_code=200,
                request_headers={},
                response_headers={},
                request_body_utf8="{}",
                response_body_utf8=json.dumps(payload),
            )
        ],
    )
    writer = ReasoningTraceWriter(
        ReasoningTraceConfig(enabled=True, output_dir=str(tmp_path), vocabulary_size=151936)
    )

    summary = writer.record_transport_failure(capture=capture, error=RuntimeError("length"))

    call = json.loads((tmp_path / "llm_calls.jsonl").read_text())
    with gzip.open(tmp_path / "reasoning_tokens.jsonl.gz", "rt") as handle:
        token = json.loads(handle.readline())
    assert summary["alignment_status"] == "aligned"
    assert summary["finish_reason"] == "length"
    assert call["reasoning_token_count"] == 1
    assert call["error_type"] == "RuntimeError"
    assert token["token"] == "truncated thought"


def test_writer_normalizes_every_successful_attempt_before_terminal_failure(tmp_path) -> None:
    context = LLMTraceContext("run", "problem", "t", 1, 1, "proposer")
    exchanges = []
    for index in range(3):
        payload = _payload()
        payload["id"] = f"completion-{index}"
        payload["choices"][0]["message"]["reasoning_content"] = "think"
        payload["choices"][0]["logprobs"]["content"][0] = _entry(
            "think", 0.5, [("think", 0.5), ("prove", 0.25)]
        )
        if index:
            payload["choices"][0]["finish_reason"] = "length"
            payload["choices"][0]["message"]["content"] = ""
            payload["choices"][0]["logprobs"]["content"] = [
                payload["choices"][0]["logprobs"]["content"][0]
            ]
        exchanges.append(
            RawHTTPExchange(
                captured_at=f"attempt-{index}",
                method="POST",
                url="http://local",
                status_code=200,
                request_headers={},
                response_headers={},
                request_body_utf8=json.dumps({"messages": []}),
                response_body_utf8=json.dumps(payload),
            )
        )
    writer = ReasoningTraceWriter(
        ReasoningTraceConfig(enabled=True, output_dir=str(tmp_path), vocabulary_size=4)
    )

    summary = writer.record_transport_failure(
        capture=LLMTraceCapture(context=context, exchanges=exchanges),
        error=RuntimeError("length"),
    )

    calls = [
        json.loads(line)
        for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with gzip.open(tmp_path / "reasoning_tokens.jsonl.gz", "rt") as handle:
        tokens = [json.loads(line) for line in handle]
    assert [call["call_id"] for call in calls] == [
        context.call_id,
        f"{context.call_id}:tool:1",
        f"{context.call_id}:tool:2",
    ]
    assert [call["role"] for call in calls] == [
        "proposer",
        "tool_continuation",
        "tool_continuation",
    ]
    assert [call["finish_reason"] for call in calls] == ["stop", "length", "length"]
    assert [token["call_id"] for token in tokens] == [
        context.call_id,
        f"{context.call_id}:tool:1",
        f"{context.call_id}:tool:2",
    ]
    assert "error_type" not in calls[0]
    assert "error_type" not in calls[1]
    assert calls[2]["error_type"] == "RuntimeError"
    assert summary["call_id"] == f"{context.call_id}:tool:2"
