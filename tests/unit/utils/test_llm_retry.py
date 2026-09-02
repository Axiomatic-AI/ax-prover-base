"""Retry policy: transient errors retry, permanent errors fail fast."""

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ax_prover.utils.llm import LLMClient


def _status_error(cls, status: int):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


def client_raising(errors: list[BaseException]) -> tuple[LLMClient, dict]:
    """An LLMClient whose model raises each error in turn, then answers."""
    calls = {"n": 0}

    def fake_model(_messages):
        calls["n"] += 1
        if calls["n"] <= len(errors):
            raise errors[calls["n"] - 1]
        return AIMessage(content="ok")

    client = LLMClient.__new__(LLMClient)
    client._base_llm = RunnableLambda(fake_model)
    client._retry_config = {"stop_after_attempt": 5}
    return client, calls


async def test_a_permanent_client_error_fails_fast():
    """A routing 404 was retried invisibly for 90 minutes; it fails every time."""
    error = _status_error(openai.NotFoundError, 404)
    client, calls = client_raising([error, error, error])

    with pytest.raises(openai.NotFoundError):
        await client.ainvoke([])

    assert calls["n"] == 1


async def test_a_rate_limit_is_retried():
    client, calls = client_raising([_status_error(openai.RateLimitError, 429)])

    response = await client.ainvoke([])

    assert response.content == "ok"
    assert calls["n"] == 2


async def test_a_transient_provider_error_in_a_200_body_is_retried():
    """OpenRouter reports provider failures inside HTTP 200 bodies; langchain-openai
    raises them as bare ValueError. A 504 'Provider timed out' killed a run."""
    error = ValueError({"message": "Provider timed out after 14939ms", "code": 504})
    client, calls = client_raising([error])

    response = await client.ainvoke([])

    assert response.content == "ok"
    assert calls["n"] == 2


async def test_a_permanent_provider_error_in_a_200_body_fails_fast():
    error = ValueError({"message": "No endpoints available", "code": 404})
    client, calls = client_raising([error, error])

    with pytest.raises(ValueError):
        await client.ainvoke([])

    assert calls["n"] == 1


async def test_a_malformed_response_crash_is_retried():
    """A `choices: null` response crashes langchain-openai with a bare TypeError; it is
    transient provider garbage, not a permanent error, so it retries."""
    client, calls = client_raising([TypeError("'NoneType' object is not iterable")])

    response = await client.ainvoke([])

    assert response.content == "ok"
    assert calls["n"] == 2


async def test_an_auth_error_fails_fast():
    error = _status_error(openai.AuthenticationError, 401)
    client, calls = client_raising([error, error])

    with pytest.raises(openai.AuthenticationError):
        await client.ainvoke([])

    assert calls["n"] == 1
