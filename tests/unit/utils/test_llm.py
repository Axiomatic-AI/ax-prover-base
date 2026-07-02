"""Tests for LLM factory helpers and DeepSeek-specific structured-output handling."""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ax_prover.config import LLMConfig
from ax_prover.utils.llm import LLMClient, _is_deepseek_model


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


def test_structured_kwargs_deepseek_uses_json_object():
    client = _deepseek_client()
    kwargs = client._structured_output_bind_kwargs(SamplePerson)
    assert kwargs == {"response_format": {"type": "json_object"}}


def test_reasoning_effort_reaches_chat_openai():
    # Confirms the config's reasoning_effort flows through init_chat_model to ChatOpenAI.
    client = _deepseek_client()
    assert getattr(client._base_llm, "reasoning_effort", None) == "high"
