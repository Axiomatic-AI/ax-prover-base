"""Tests for LLM factory helpers and DeepSeek-specific structured-output handling."""

import asyncio
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from omegaconf import OmegaConf
from pydantic import BaseModel

from ax_prover.config import LLMConfig
from ax_prover.utils.config import load_env_secrets, merge_configs
from ax_prover.utils.llm import (
    LLMClient,
    _DeepSeekChatOpenAI,
    _is_deepseek_config,
    _is_deepseek_model,
    create_llm,
    get_reasoning,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


class SamplePerson(BaseModel):
    name: str
    age: int


def test_is_deepseek_true_by_model_name():
    llm = ChatOpenAI(
        model="deepseek-v4-pro", api_key="test-key", base_url="https://api.deepseek.com"
    )
    assert _is_deepseek_model(llm) is True


def test_is_deepseek_true_by_base_url():
    llm = ChatOpenAI(
        model="some-proxy-model", api_key="test-key", base_url="https://api.deepseek.com/v1"
    )
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


@tool
def _sample_tool(query: str) -> str:
    """A sample tool for binding tests."""
    return "result"


def test_deepseek_skips_structured_output_bind_with_tools():
    # DeepSeek's json_object response_format forces langchain_openai through OpenAI's
    # .parse() path, which rejects non-strict tools client-side. DeepSeek can't use
    # strict tools, so response_format must be skipped when tools are bound.
    client = _deepseek_client()
    assert client._should_bind_structured_output([_sample_tool]) is False


def test_deepseek_binds_structured_output_without_tools():
    client = _deepseek_client()
    assert client._should_bind_structured_output(None) is True


def test_openai_binds_structured_output_with_tools(monkeypatch):
    # Real OpenAI supports strict tools + response_format together, so the bind stays.
    client = _openai_client(monkeypatch)
    assert client._should_bind_structured_output([_sample_tool]) is True


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


def test_schema_injection_omits_tool_language_without_tools():
    client = _deepseek_client()
    out = client._maybe_inject_schema([HumanMessage(content="q")], SamplePerson, has_tools=False)
    text = out[-1].content
    assert "JSON" in text
    assert "tool" not in text.lower()  # no tool-permission language when no tools are bound


def test_schema_injection_includes_tool_language_with_tools():
    client = _deepseek_client()
    out = client._maybe_inject_schema([HumanMessage(content="q")], SamplePerson, has_tools=True)
    text = out[-1].content
    assert "JSON" in text
    assert "tool" in text.lower()


def test_no_schema_injection_without_schema():
    client = _deepseek_client()
    messages = [HumanMessage(content="hi")]
    assert client._maybe_inject_schema(messages, None) == messages


def test_no_schema_injection_for_non_deepseek(monkeypatch):
    client = _openai_client(monkeypatch)
    messages = [HumanMessage(content="hi")]
    assert client._maybe_inject_schema(messages, SamplePerson) == messages


def test_is_deepseek_config_true_by_model_name():
    cfg = LLMConfig(model="deepseek-v4-pro", provider_config={"model_provider": "openai"})
    assert _is_deepseek_config(cfg) is True


def test_is_deepseek_config_true_by_base_url():
    cfg = LLMConfig(
        model="proxy",
        provider_config={"model_provider": "openai", "base_url": "https://api.deepseek.com"},
    )
    assert _is_deepseek_config(cfg) is True


def test_is_deepseek_config_false_for_plain_openai():
    cfg = LLMConfig(model="gpt-4o", provider_config={"model_provider": "openai"})
    assert _is_deepseek_config(cfg) is False


def test_create_llm_uses_deepseek_subclass():
    cfg = LLMConfig(
        model="deepseek-v4-pro",
        provider_config={
            "model_provider": "openai",
            "base_url": "https://api.deepseek.com",
            "api_key": "test-key",
            "reasoning_effort": "high",
        },
    )
    llm = create_llm(cfg)
    assert isinstance(llm, _DeepSeekChatOpenAI)
    # subclass must still be detected as DeepSeek by the instance check
    assert _is_deepseek_model(llm) is True


def _deepseek_subclass() -> _DeepSeekChatOpenAI:
    return _DeepSeekChatOpenAI(
        model="deepseek-v4-pro", api_key="test-key", base_url="https://api.deepseek.com"
    )


def test_deepseek_subclass_injects_reasoning_content():
    llm = _deepseek_subclass()
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "the answer",
                    "reasoning_content": "step-by-step thinking",
                },
                "finish_reason": "stop",
            }
        ],
        "model": "deepseek-v4-pro",
    }
    result = llm._create_chat_result(response)
    msg = result.generations[0].message
    assert msg.additional_kwargs.get("reasoning_content") == "step-by-step thinking"
    # and it round-trips through get_reasoning
    assert get_reasoning(msg) == "step-by-step thinking"


def test_deepseek_subclass_no_reasoning_key_when_absent():
    llm = _deepseek_subclass()
    response = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "model": "deepseek-v4-pro",
    }
    result = llm._create_chat_result(response)
    assert "reasoning_content" not in result.generations[0].message.additional_kwargs


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


def test_deepseek_llms_config_entry(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-key")
    cfg = OmegaConf.load(_REPO_ROOT / "configs" / "llms.yaml")
    ds = cfg.llm_configs.deepseek_v4_pro
    assert ds.model == "deepseek-v4-pro"
    assert ds.provider_config.model_provider == "openai"
    assert ds.provider_config.base_url == "https://api.deepseek.com"
    assert ds.provider_config.reasoning_effort == "high"
    # api_key resolves from DEEPSEEK_API_KEY via the oc.env resolver
    assert ds.provider_config.api_key == "dummy-key"


def test_deepseek_local_run_config_selects_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-key")
    cfg = merge_configs(["configs/deepseek_local.yaml"], folder=_REPO_ROOT)

    assert cfg.prover.prover_llm.model == "deepseek-v4-pro"

    tools = cfg.prover.proposer_tools
    assert "search_lean" in tools
    assert "search_web" in tools
    # Reading resolved values proves interpolation succeeded end-to-end;
    # a dangling ${tool_configs.*} reference would raise here instead of silently passing.
    assert tools["search_lean"]["tool_type"] == "search_lean_search"
    assert tools["search_web"]["tool_type"] == "search_web"


def test_llms_import_resolves_without_deepseek_key(monkeypatch):
    # merge_configs eagerly resolves the whole tree, including the deepseek_v4_pro
    # entry, even for configs that select a different model. The api_key resolver
    # must therefore tolerate a missing DEEPSEEK_API_KEY (default null) so unrelated
    # runs (Opus/qwen/Gemini) do not crash. Regression guard.
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = merge_configs(["configs/default.yaml"], folder=_REPO_ROOT)
    assert cfg.prover.prover_llm.model == "anthropic:claude-opus-4-5"


def _run_live_tests() -> bool:
    load_env_secrets()  # loads .env.secrets into os.environ if present
    return os.environ.get("AX_PROVER_LIVE_TESTS") == "1" and bool(
        os.environ.get("DEEPSEEK_API_KEY")
    )


@pytest.mark.skipif(
    not _run_live_tests(),
    reason="opt-in live DeepSeek test; set AX_PROVER_LIVE_TESTS=1 (requires DEEPSEEK_API_KEY)",
)
def test_live_deepseek_structured_output_parallel():
    load_env_secrets()
    config = LLMConfig(
        model="deepseek-v4-pro",
        provider_config={
            "model_provider": "openai",
            "base_url": "https://api.deepseek.com",
            "api_key": os.environ["DEEPSEEK_API_KEY"],
            "reasoning_effort": "high",
        },
        retry_config={"stop_after_attempt": 3},
    )
    client = LLMClient(config)

    async def one(person_desc: str) -> SamplePerson:
        response = await client.ainvoke(
            [HumanMessage(content=f"Extract the person: {person_desc}")],
            output_schema=SamplePerson,
        )
        return SamplePerson.model_validate_json(response.text)

    async def run_all():
        return await asyncio.gather(
            one("Alice is 30 years old"),
            one("Bob is 42 years old"),
            one("Carol is 25 years old"),
        )

    results = asyncio.run(run_all())
    assert len(results) == 3
    assert all(isinstance(r, SamplePerson) for r in results)
    assert {r.name for r in results} == {"Alice", "Bob", "Carol"}
