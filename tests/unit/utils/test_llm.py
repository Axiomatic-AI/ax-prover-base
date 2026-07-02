"""Tests for LLM factory helpers and DeepSeek-specific structured-output handling."""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ax_prover.config import LLMConfig
from ax_prover.utils.llm import LLMClient, _is_deepseek_model, get_reasoning


class SamplePerson(BaseModel):
    name: str
    age: int


def test_is_deepseek_true_by_model_name():
    llm = ChatOpenAI(model="deepseek-v4-pro", api_key="test-key", base_url="https://api.deepseek.com")
    assert _is_deepseek_model(llm) is True


def test_is_deepseek_true_by_base_url():
    llm = ChatOpenAI(model="some-proxy-model", api_key="test-key", base_url="https://api.deepseek.com/v1")
    assert _is_deepseek_model(llm) is True


def test_is_deepseek_false_for_plain_openai():
    llm = ChatOpenAI(model="gpt-4o", api_key="test-key")
    assert _is_deepseek_model(llm) is False


def _deepseek_client() -> LLMClient:
    config = LLMConfig(
        model="deepseek-v4-pro",
        provider_config={
            "model_provider": "openai",
            "base_url": "https://api.deepseek.com",
            "api_key": "test-key",
            "temperature": None,
            "max_tokens": None,
            "reasoning_effort": "high",
        },
    )
    return LLMClient(config)


def _openai_client(monkeypatch) -> LLMClient:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return LLMClient(LLMConfig(model="openai:gpt-4o", provider_config={"api_key": "test-key"}))


def test_structured_kwargs_deepseek_uses_json_object():
    client = _deepseek_client()
    kwargs = client._structured_output_bind_kwargs(SamplePerson)
    assert kwargs == {"response_format": {"type": "json_object"}}


def test_reasoning_effort_reaches_chat_openai():
    # Confirms the config's reasoning_effort flows through init_chat_model to ChatOpenAI.
    client = _deepseek_client()
    assert getattr(client._base_llm, "reasoning_effort", None) == "high"


def test_deepseek_does_not_use_strict_tools():
    client = _deepseek_client()
    assert client._use_strict_tools(SamplePerson) is False


def test_openai_uses_strict_tools_with_schema(monkeypatch):
    client = _openai_client(monkeypatch)
    assert client._use_strict_tools(SamplePerson) is True


def test_openai_no_strict_without_schema(monkeypatch):
    client = _openai_client(monkeypatch)
    assert client._use_strict_tools(None) is False


def test_schema_injection_appends_json_instruction_for_deepseek():
    client = _deepseek_client()
    messages = [SystemMessage(content="sys"), HumanMessage(content="prove it")]
    out = client._maybe_inject_schema(messages, SamplePerson)

    assert len(out) == 3
    assert isinstance(out[-1], HumanMessage)
    assert "JSON" in out[-1].content
    assert "properties" in out[-1].content  # the schema itself is embedded
    # original list is not mutated
    assert len(messages) == 2


def test_no_schema_injection_without_schema():
    client = _deepseek_client()
    messages = [HumanMessage(content="hi")]
    assert client._maybe_inject_schema(messages, None) == messages


def test_no_schema_injection_for_non_deepseek(monkeypatch):
    client = _openai_client(monkeypatch)
    messages = [HumanMessage(content="hi")]
    assert client._maybe_inject_schema(messages, SamplePerson) == messages


def test_get_reasoning_falls_back_to_deepseek_reasoning_content():
    response = AIMessage(
        content="final answer",
        additional_kwargs={"reasoning_content": "step-by-step thinking"},
        response_metadata={"model_provider": "openai"},
    )
    assert get_reasoning(response) == "step-by-step thinking"


def test_get_reasoning_empty_when_no_reasoning():
    response = AIMessage(content="hi")
    assert get_reasoning(response) == ""
