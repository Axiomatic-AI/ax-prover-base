"""Lossless vLLM reasoning traces and top-K entropy metrics."""

from __future__ import annotations

import contextvars
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ..config import ReasoningTraceConfig


class ReasoningTraceError(RuntimeError):
    """Base error for reasoning-trace contract failures."""


class ReasoningAlignmentError(ReasoningTraceError):
    """Raised when reasoning cannot be aligned to provider logprob tokens."""


@dataclass(frozen=True)
class LLMTraceContext:
    """Stable identity for one model call."""

    run_id: str
    problem_uuid: str
    theorem_name: str
    iteration: int
    call_index: int
    role: str

    @property
    def call_id(self) -> str:
        return f"{self.run_id}:{self.problem_uuid}:{self.role}:{self.iteration}:{self.call_index}"


@dataclass
class RawHTTPExchange:
    """One HTTP response captured before LangChain normalizes it."""

    captured_at: str
    method: str
    url: str
    status_code: int
    request_headers: dict[str, str]
    response_headers: dict[str, str]
    request_body_utf8: str
    response_body_utf8: str

    def request_json(self) -> Any:
        return _load_json(self.request_body_utf8)

    def response_json(self) -> Any:
        return _load_json(self.response_body_utf8)


@dataclass
class LLMTraceCapture:
    """Mutable capture populated by the HTTP response hook during one call."""

    context: LLMTraceContext
    exchanges: list[RawHTTPExchange] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_monotonic: float = field(default_factory=time.monotonic)

    @property
    def successful_exchange(self) -> RawHTTPExchange | None:
        for exchange in reversed(self.exchanges):
            payload = exchange.response_json()
            if 200 <= exchange.status_code < 300 and isinstance(payload, dict):
                if isinstance(payload.get("choices"), list):
                    return exchange
        return None


_ACTIVE_CAPTURE: contextvars.ContextVar[LLMTraceCapture | None] = contextvars.ContextVar(
    "ax_prover_reasoning_trace_capture", default=None
)


def current_trace_capture() -> LLMTraceCapture | None:
    """Return the call-local trace identity without changing capture state."""

    return _ACTIVE_CAPTURE.get()


@contextmanager
def activate_trace_capture(capture: LLMTraceCapture | None) -> Iterator[None]:
    """Expose a call-local capture to the async HTTP response hook."""

    if capture is None:
        yield
        return
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield
    finally:
        _ACTIVE_CAPTURE.reset(token)


async def capture_httpx_response(response: httpx.Response) -> None:
    """Capture an OpenAI-compatible response before model-specific parsing."""

    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        return
    body = await response.aread()
    request = response.request
    capture.exchanges.append(
        RawHTTPExchange(
            captured_at=_now(),
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            request_headers=_safe_headers(request.headers),
            response_headers=_safe_headers(response.headers),
            request_body_utf8=request.content.decode("utf-8", errors="replace"),
            response_body_utf8=body.decode("utf-8", errors="replace"),
        )
    )


def analyze_provider_payload(
    payload: dict[str, Any],
    *,
    vocabulary_size: int | None,
) -> dict[str, Any]:
    """Align reasoning with provider logprobs and compute token metrics."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return _analysis_failure("provider_choices_missing")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return _analysis_failure("provider_message_missing")

    reasoning, reasoning_field = _provider_reasoning(message)
    final_content = message.get("content")
    if not isinstance(final_content, str):
        final_content = ""
    truncated_open_think = (
        not reasoning
        and choice.get("finish_reason") == "length"
        and final_content.startswith("<think>")
        and "</think>" not in final_content
    )
    if truncated_open_think:
        # qwen3 cannot close its reasoning delimiter after a length stop. vLLM
        # therefore leaves the truncated trace in content instead of exposing a
        # reasoning field. The bytes after the opening delimiter are still an
        # exact, auditable reasoning region; there is intentionally no final output.
        reasoning = final_content[len("<think>") :]
        reasoning_field = "content_unclosed_think_truncated"
        final_content = ""
    if not reasoning:
        return {
            **_analysis_failure("reasoning_missing"),
            "reasoning": reasoning,
            "reasoning_field": reasoning_field,
            "final_content": final_content,
        }

    logprobs = choice.get("logprobs")
    entries = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(entries, list) or not entries:
        return {
            **_analysis_failure("logprobs_missing"),
            "reasoning": reasoning,
            "reasoning_field": reasoning_field,
            "final_content": final_content,
        }

    try:
        token_bytes = [_entry_bytes(entry) for entry in entries]
    except ReasoningAlignmentError as error:
        return {
            **_analysis_failure("invalid_logprob_token_entry"),
            "reasoning": reasoning,
            "reasoning_field": reasoning_field,
            "final_content": final_content,
            "alignment_error": str(error),
        }
    generated = b"".join(token_bytes)
    boundaries = [0]
    for item in token_bytes:
        boundaries.append(boundaries[-1] + len(item))
    boundary_set = set(boundaries)
    opening_delimiters = [
        index for index, entry in enumerate(entries) if entry.get("token") == "<think>"
    ]
    closing_delimiters = [
        index for index, entry in enumerate(entries) if entry.get("token") == "</think>"
    ]
    reasoning_delimiter_span = (
        len(opening_delimiters) == 1
        and bool(closing_delimiters)
        and opening_delimiters[0] < closing_delimiters[0]
    )
    closed_delimiter_span = (
        len(opening_delimiters) == 1
        and len(closing_delimiters) == 1
        and opening_delimiters[0] < closing_delimiters[0]
    )
    alignment_basis = "provider_reasoning_content_span"
    if reasoning and reasoning_delimiter_span:
        token_start = opening_delimiters[0] + 1
        token_end = closing_delimiters[0]
        byte_start = boundaries[token_start]
        byte_end = boundaries[token_end]
        alignment_basis = (
            "qwen3_delimiter_token_span"
            if closed_delimiter_span
            else "qwen3_first_closing_delimiter_token_span"
        )
    elif truncated_open_think:
        opening_end = len(b"<think>")
        if not generated.startswith(b"<think>"):
            return {
                **_analysis_failure("truncated_reasoning_opening_delimiter_missing"),
                "reasoning": reasoning,
                "reasoning_field": reasoning_field,
                "final_content": final_content,
            }
        if opening_end not in boundary_set:
            return {
                **_analysis_failure("truncated_reasoning_delimiter_splits_token"),
                "reasoning": reasoning,
                "reasoning_field": reasoning_field,
                "final_content": final_content,
            }
        byte_start = opening_end
        byte_end = len(generated)
        alignment_basis = "opening_think_delimiter_to_length_stop"
    else:
        reasoning_bytes = reasoning.encode("utf-8")
        occurrence_starts: list[int] = []
        search_from = 0
        while True:
            occurrence = generated.find(reasoning_bytes, search_from)
            if occurrence < 0:
                break
            occurrence_starts.append(occurrence)
            search_from = occurrence + 1
        if not occurrence_starts:
            return {
                **_analysis_failure("reasoning_not_found_in_logprob_tokens"),
                "reasoning": reasoning,
                "reasoning_field": reasoning_field,
                "final_content": final_content,
                "generated_text": generated.decode("utf-8", errors="replace"),
            }
        aligned_starts = [
            start
            for start in occurrence_starts
            if start in boundary_set and start + len(reasoning_bytes) in boundary_set
        ]
        if not aligned_starts:
            return {
                **_analysis_failure("reasoning_boundary_splits_token"),
                "reasoning": reasoning,
                "reasoning_field": reasoning_field,
                "final_content": final_content,
                "candidate_byte_starts": occurrence_starts,
            }
        if len(aligned_starts) > 1:
            return {
                **_analysis_failure("reasoning_alignment_ambiguous"),
                "reasoning": reasoning,
                "reasoning_field": reasoning_field,
                "final_content": final_content,
            }
        byte_start = aligned_starts[0]
        byte_end = byte_start + len(reasoning_bytes)
    if not reasoning_delimiter_span:
        try:
            token_start = boundaries.index(byte_start)
            token_end = boundaries.index(byte_end)
        except ValueError:
            return {
                **_analysis_failure("reasoning_boundary_splits_token"),
                "reasoning": reasoning,
                "reasoning_field": reasoning_field,
                "final_content": final_content,
                "reasoning_byte_start": byte_start,
                "reasoning_byte_end": byte_end,
            }

    reasoning_entries = entries[token_start:token_end]
    token_records: list[dict[str, Any]] = []
    for reasoning_position, entry in enumerate(reasoning_entries):
        generated_position = token_start + reasoning_position
        token_records.append(
            {
                "generated_position": generated_position,
                "reasoning_position": reasoning_position,
                "position_fraction": (reasoning_position / max(1, len(reasoning_entries) - 1)),
                "token": str(entry.get("token") or ""),
                "bytes": entry.get("bytes"),
                "sampled_logprob": _finite_float(entry.get("logprob")),
                "top_logprobs": entry.get("top_logprobs") or [],
                **_token_entropy_metrics(entry, vocabulary_size=vocabulary_size),
            }
        )

    decoded_reasoning = b"".join(token_bytes[token_start:token_end]).decode(
        "utf-8", errors="strict"
    )
    reasoning_decode_matches_provider = decoded_reasoning == reasoning
    if (
        not truncated_open_think
        and not reasoning_delimiter_span
        and not reasoning_decode_matches_provider
    ):
        return {
            **_analysis_failure("reasoning_decode_mismatch"),
            "reasoning": reasoning,
            "reasoning_field": reasoning_field,
            "final_content": final_content,
            "decoded_reasoning": decoded_reasoning,
        }

    final_alignment_status = "empty"
    final_token_start = None
    final_token_end = None
    final_token_records: list[dict[str, Any]] = []
    final_decode_matches_provider = None
    if final_content and closed_delimiter_span:
        final_token_start = closing_delimiters[0] + 1
        final_token_end = len(entries)
        if final_token_end > final_token_start and entries[-1].get("token") == "<|im_end|>":
            final_token_end -= 1
        final_alignment_status = "aligned"
        for final_position, entry in enumerate(entries[final_token_start:final_token_end]):
            final_token_records.append(
                {
                    "generated_position": final_token_start + final_position,
                    "final_position": final_position,
                    "token": str(entry.get("token") or ""),
                    "bytes": entry.get("bytes"),
                    "sampled_logprob": _finite_float(entry.get("logprob")),
                    "top_logprobs": entry.get("top_logprobs") or [],
                }
            )
        decoded_final = b"".join(token_bytes[final_token_start:final_token_end]).decode(
            "utf-8", errors="strict"
        )
        final_decode_matches_provider = decoded_final == final_content
    elif final_content:
        final_bytes = final_content.encode("utf-8")
        final_byte_start = generated.find(final_bytes, byte_end)
        if final_byte_start < 0:
            final_alignment_status = "final_content_not_found_in_logprob_tokens"
        else:
            final_byte_end = final_byte_start + len(final_bytes)
            try:
                final_token_start = boundaries.index(final_byte_start)
                final_token_end = boundaries.index(final_byte_end)
            except ValueError:
                final_alignment_status = "final_content_boundary_splits_token"
            else:
                final_alignment_status = "aligned"
                for final_position, entry in enumerate(entries[final_token_start:final_token_end]):
                    final_token_records.append(
                        {
                            "generated_position": final_token_start + final_position,
                            "final_position": final_position,
                            "token": str(entry.get("token") or ""),
                            "bytes": entry.get("bytes"),
                            "sampled_logprob": _finite_float(entry.get("logprob")),
                            "top_logprobs": entry.get("top_logprobs") or [],
                        }
                    )
                decoded_final = b"".join(token_bytes[final_token_start:final_token_end]).decode(
                    "utf-8", errors="strict"
                )
                final_decode_matches_provider = decoded_final == final_content

    return {
        "alignment_status": "aligned",
        "alignment_basis": alignment_basis,
        "reasoning_decode_matches_provider": reasoning_decode_matches_provider,
        "reasoning": reasoning,
        "reasoning_field": reasoning_field,
        "final_content": final_content,
        "final_content_field": "content",
        "reasoning_byte_start": byte_start,
        "reasoning_byte_end": byte_end,
        "reasoning_token_start": token_start,
        "reasoning_token_end": token_end,
        "generated_token_count": len(entries),
        "reasoning_token_count": len(token_records),
        "final_alignment_status": final_alignment_status,
        "final_decode_matches_provider": final_decode_matches_provider,
        "final_token_start": final_token_start,
        "final_token_end": final_token_end,
        "final_token_count": len(final_token_records),
        "final_tokens": final_token_records,
        "tokens": token_records,
        "aggregates": _aggregate_tokens(token_records),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
    }


class ReasoningTraceWriter:
    """Append reasoning calls, token metrics, and Lean outcomes under scratch."""

    def __init__(self, config: ReasoningTraceConfig):
        self.config = config
        if config.enabled and not config.output_dir:
            raise ValueError("reasoning_trace.output_dir is required when tracing is enabled")
        self.output_dir = (
            Path(config.output_dir).expanduser().resolve() if config.output_dir else None
        )
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def record_proposer(
        self,
        *,
        capture: LLMTraceCapture,
        normalized_response: Any,
    ) -> dict[str, Any]:
        """Persist every provider call in a node and return the final call summary."""

        if not self.enabled or self.output_dir is None:
            return {"call_id": capture.context.call_id, "alignment_status": "disabled"}

        exchange_groups: list[list[RawHTTPExchange]] = []
        pending: list[RawHTTPExchange] = []
        for exchange in capture.exchanges:
            pending.append(exchange)
            payload = exchange.response_json()
            if (
                200 <= exchange.status_code < 300
                and isinstance(payload, dict)
                and isinstance(payload.get("choices"), list)
            ):
                exchange_groups.append(pending)
                pending = []
        if not exchange_groups:
            exchange_groups = [[]]
        summaries = []
        for exchange_index, exchange_group in enumerate(exchange_groups):
            summaries.append(
                self._record_exchange(
                    capture=capture,
                    exchange_group=exchange_group,
                    exchange_index=exchange_index,
                    exchange_count=len(exchange_groups),
                    normalized_response=(
                        normalized_response if exchange_index == len(exchange_groups) - 1 else None
                    ),
                )
            )
        return summaries[-1]

    def _record_exchange(
        self,
        *,
        capture: LLMTraceCapture,
        exchange_group: list[RawHTTPExchange],
        exchange_index: int,
        exchange_count: int,
        normalized_response: Any,
        terminal_error: BaseException | None = None,
        enforce_alignment: bool = True,
    ) -> dict[str, Any]:
        exchange = exchange_group[-1] if exchange_group else None
        provider_payload = exchange.response_json() if exchange else None
        if isinstance(provider_payload, dict):
            analysis = analyze_provider_payload(
                provider_payload,
                vocabulary_size=self.config.vocabulary_size,
            )
        else:
            analysis = {
                **_analysis_failure("successful_provider_response_missing"),
                "reasoning": _normalized_reasoning(normalized_response),
                "reasoning_field": "langchain_fallback",
                "final_content": _normalized_text(normalized_response),
            }

        context = asdict(capture.context)
        role = context["role"]
        call_id = capture.context.call_id
        if exchange_index:
            role = "tool_continuation"
            call_id = f"{capture.context.call_id}:tool:{exchange_index}"
        context.update(
            role=role,
            exchange_index=exchange_index,
            exchange_count=exchange_count,
            parent_call_id=capture.context.call_id if exchange_index else None,
        )
        request_payload = exchange.request_json() if exchange else None
        call_record = {
            "schema_version": 1,
            "entropy_version": self.config.entropy_version,
            "started_at": capture.started_at,
            "captured_at": _now(),
            "latency_seconds": round(time.monotonic() - capture.started_monotonic, 6),
            "call_id": call_id,
            **context,
            "prompt_hash": _exchange_request_hash(exchange),
            "response_hash": _exchange_response_hash(exchange),
            "request": request_payload,
            "decoder_settings": _decoder_settings(request_payload),
            "provider_response": provider_payload,
            "raw_http_exchanges": [asdict(item) for item in exchange_group],
            "langchain_response": _normalized_response_dump(normalized_response),
            "alignment_status": analysis["alignment_status"],
            "alignment_basis": analysis.get("alignment_basis"),
            "reasoning_decode_matches_provider": analysis.get("reasoning_decode_matches_provider"),
            "reasoning_field": analysis.get("reasoning_field"),
            "reasoning": analysis.get("reasoning", ""),
            "final_content": analysis.get("final_content", ""),
            "final_content_field": analysis.get("final_content_field", "content"),
            "reasoning_token_start": analysis.get("reasoning_token_start"),
            "reasoning_token_end": analysis.get("reasoning_token_end"),
            "reasoning_token_count": analysis.get("reasoning_token_count", 0),
            "generated_token_count": analysis.get("generated_token_count"),
            "final_alignment_status": analysis.get("final_alignment_status"),
            "final_decode_matches_provider": analysis.get("final_decode_matches_provider"),
            "final_token_start": analysis.get("final_token_start"),
            "final_token_end": analysis.get("final_token_end"),
            "final_token_count": analysis.get("final_token_count", 0),
            "finish_reason": analysis.get("finish_reason"),
            "usage": analysis.get("usage"),
            "reasoning_aggregates": analysis.get("aggregates", {}),
        }
        if terminal_error is not None:
            call_record.update(
                error_type=type(terminal_error).__name__,
                error=str(terminal_error),
            )
        _append_jsonl(self.output_dir / "llm_calls.jsonl", call_record)

        reasoning_records = [
            {
                "schema_version": 1,
                "entropy_version": self.config.entropy_version,
                "call_id": call_id,
                **context,
                **token_record,
            }
            for token_record in analysis.get("tokens", [])
        ]
        _append_gzip_jsonl_many(self.output_dir / "reasoning_tokens.jsonl.gz", reasoning_records)

        final_records = [
            {
                "schema_version": 1,
                "region": "structured_final_output",
                "call_id": call_id,
                **context,
                **token_record,
            }
            for token_record in analysis.get("final_tokens", [])
        ]
        _append_gzip_jsonl_many(self.output_dir / "final_tokens.jsonl.gz", final_records)

        if (
            enforce_alignment
            and analysis["alignment_status"] != "aligned"
            and self.config.require_alignment
        ):
            raise ReasoningAlignmentError(f"{call_id}: {analysis['alignment_status']}")
        if (
            enforce_alignment
            and analysis.get("final_content")
            and analysis.get("final_alignment_status") != "aligned"
            and self.config.require_alignment
        ):
            raise ReasoningAlignmentError(f"{call_id}: {analysis['final_alignment_status']}")
        return {
            "call_id": call_id,
            "alignment_status": analysis["alignment_status"],
            "reasoning": analysis.get("reasoning", ""),
            "reasoning_token_count": analysis.get("reasoning_token_count", 0),
            "finish_reason": analysis.get("finish_reason"),
            "aggregates": analysis.get("aggregates", {}),
        }

    def record_transport_failure(
        self,
        *,
        capture: LLMTraceCapture,
        error: BaseException,
    ) -> dict[str, Any]:
        """Persist every provider attempt and recover metrics when normalization raises."""

        if not self.enabled or self.output_dir is None:
            return {"call_id": capture.context.call_id, "alignment_status": "disabled"}
        exchange_groups: list[list[RawHTTPExchange]] = []
        pending: list[RawHTTPExchange] = []
        for exchange in capture.exchanges:
            pending.append(exchange)
            payload = exchange.response_json()
            if (
                200 <= exchange.status_code < 300
                and isinstance(payload, dict)
                and isinstance(payload.get("choices"), list)
            ):
                exchange_groups.append(pending)
                pending = []
        if pending:
            exchange_groups.append(pending)
        if not exchange_groups:
            exchange_groups = [[]]

        summaries = []
        for exchange_index, exchange_group in enumerate(exchange_groups):
            summaries.append(
                self._record_exchange(
                    capture=capture,
                    exchange_group=exchange_group,
                    exchange_index=exchange_index,
                    exchange_count=len(exchange_groups),
                    normalized_response=None,
                    terminal_error=(error if exchange_index == len(exchange_groups) - 1 else None),
                    enforce_alignment=False,
                )
            )
        return summaries[-1]

    def record_lean_check(
        self,
        *,
        call_id: str | None,
        problem_uuid: str,
        theorem_name: str,
        iteration: int,
        outcome: str,
        success: bool,
        feedback_type: str,
        diagnostics: str,
        duration_seconds: float,
    ) -> str | None:
        """Persist the verifier outcome following a proposer call."""

        if not self.enabled or self.output_dir is None or not call_id:
            return None
        check_id = f"{call_id}:lean"
        _append_jsonl(
            self.output_dir / "lean_checks.jsonl",
            {
                "schema_version": 1,
                "recorded_at": _now(),
                "check_id": check_id,
                "call_id": call_id,
                "run_id": self.config.run_id,
                "problem_uuid": problem_uuid,
                "theorem_name": theorem_name,
                "iteration": iteration,
                "outcome": outcome,
                "success": success,
                "feedback_type": feedback_type,
                "diagnostics": diagnostics,
                "duration_seconds": round(duration_seconds, 6),
            },
        )
        return check_id


def _provider_reasoning(message: dict[str, Any]) -> tuple[str, str | None]:
    for field_name in ("reasoning", "reasoning_content"):
        value = message.get(field_name)
        if isinstance(value, str) and value:
            return value, field_name
    return "", None


def _entry_bytes(entry: Any) -> bytes:
    if not isinstance(entry, dict):
        raise ReasoningAlignmentError("logprob token entry is not a mapping")
    raw_bytes = entry.get("bytes")
    if isinstance(raw_bytes, list) and all(isinstance(item, int) for item in raw_bytes):
        return bytes(raw_bytes)
    token = entry.get("token")
    if isinstance(token, str):
        return token.encode("utf-8")
    raise ReasoningAlignmentError("logprob token entry has neither bytes nor token text")


def _token_entropy_metrics(
    entry: dict[str, Any],
    *,
    vocabulary_size: int | None,
) -> dict[str, float | int | None]:
    alternatives: dict[tuple[Any, ...], float] = {}
    sampled_key = _token_key(entry)
    sampled_logprob = _finite_float(entry.get("logprob"))
    if sampled_logprob is not None:
        alternatives[sampled_key] = sampled_logprob
    raw_top = entry.get("top_logprobs")
    if isinstance(raw_top, list):
        for item in raw_top:
            if not isinstance(item, dict):
                continue
            logprob = _finite_float(item.get("logprob"))
            if logprob is None:
                continue
            key = _token_key(item)
            previous = alternatives.get(key)
            if previous is None or logprob > previous:
                alternatives[key] = logprob

    probabilities = [math.exp(value) for value in alternatives.values()]
    returned_mass_raw = sum(probabilities)
    returned_mass = min(1.0, max(0.0, returned_mass_raw))
    partial_entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    normalized_entropy = None
    if returned_mass_raw > 0:
        normalized = [value / returned_mass_raw for value in probabilities]
        normalized_entropy = -sum(value * math.log(value) for value in normalized if value > 0)
    tail_mass = max(0.0, 1.0 - returned_mass)
    lower_bound = partial_entropy + _entropy_term(tail_mass)
    upper_bound = None
    if vocabulary_size is not None:
        remaining = vocabulary_size - len(alternatives)
        if remaining < 0:
            raise ValueError("vocabulary_size is smaller than returned alternative count")
        upper_bound = (
            partial_entropy
            if tail_mass == 0
            else partial_entropy - tail_mass * math.log(tail_mass / max(1, remaining))
        )
    return {
        "alternative_count": len(alternatives),
        "sampled_surprisal": -sampled_logprob if sampled_logprob is not None else None,
        "returned_mass": returned_mass,
        "returned_mass_raw": returned_mass_raw,
        "partial_entropy": partial_entropy,
        "normalized_topk_entropy": normalized_entropy,
        "tail_mass": tail_mass,
        "entropy_lower_bound": lower_bound,
        "entropy_upper_bound": upper_bound,
    }


def _token_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    raw_bytes = entry.get("bytes")
    if isinstance(raw_bytes, list) and all(isinstance(item, int) for item in raw_bytes):
        return ("bytes", *raw_bytes)
    return ("token", str(entry.get("token") or ""))


def _entropy_term(probability: float) -> float:
    return 0.0 if probability <= 0 else -probability * math.log(probability)


def _aggregate_tokens(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"reasoning_token_count": len(tokens)}
    for field_name in (
        "sampled_surprisal",
        "partial_entropy",
        "normalized_topk_entropy",
        "returned_mass",
        "entropy_lower_bound",
        "entropy_upper_bound",
    ):
        values = [
            float(token[field_name])
            for token in tokens
            if isinstance(token.get(field_name), int | float)
        ]
        result[f"mean_{field_name}"] = statistics.fmean(values) if values else None
        result[f"median_{field_name}"] = statistics.median(values) if values else None
        result[f"max_{field_name}"] = max(values) if values else None
    return result


def _analysis_failure(status: str) -> dict[str, Any]:
    return {
        "alignment_status": status,
        "tokens": [],
        "aggregates": {},
        "reasoning_token_count": 0,
    }


def _normalized_reasoning(response: Any) -> str:
    additional = getattr(response, "additional_kwargs", {})
    if isinstance(additional, dict):
        for field_name in ("reasoning", "reasoning_content"):
            value = additional.get(field_name)
            if isinstance(value, str) and value:
                return value
    try:
        blocks = response.content_blocks
    except (AttributeError, TypeError, ValueError):
        blocks = []
    if isinstance(blocks, list):
        values = [
            str(block.get("reasoning") or block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "reasoning"
        ]
        return "\n\n".join(value for value in values if value)
    return ""


def _normalized_text(response: Any) -> str:
    value = getattr(response, "text", "")
    return value() if callable(value) else str(value or "")


def _normalized_response_dump(response: Any) -> Any:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return {"text": _normalized_text(response), "reasoning": _normalized_reasoning(response)}


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    allowed = {"content-type", "content-length", "x-request-id", "request-id"}
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowed}


def _load_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _decoder_settings(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        return {}
    return {
        key: request.get(key)
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "stream",
            "include_reasoning",
        )
        if key in request
    }


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _exchange_request_hash(exchange: RawHTTPExchange | None) -> str | None:
    return _sha256(exchange.request_body_utf8) if exchange else None


def _exchange_response_hash(exchange: RawHTTPExchange | None) -> str | None:
    return _sha256(exchange.response_body_utf8) if exchange else None


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    _append_bytes(path, data)


def _append_gzip_jsonl_many(path: Path, payloads: list[dict[str, Any]]) -> None:
    if not payloads:
        return
    lines = "".join(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n" for payload in payloads
    ).encode("utf-8")
    _append_bytes(path, gzip.compress(lines, mtime=0))


def _append_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
    finally:
        os.close(descriptor)
