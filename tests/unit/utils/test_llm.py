"""Tests for LLM factory helpers and DeepSeek-specific structured-output handling."""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ax_prover.utils.llm import _is_deepseek_model


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
