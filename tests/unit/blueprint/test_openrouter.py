"""OpenRouter provider mapping and blueprint role configuration."""

import pytest
from langchain_openai import ChatOpenAI

from ax_prover.config import BlueprintConfig, BlueprintRoleConfig, LLMConfig
from ax_prover.utils.llm import DEFAULT_OPENROUTER_BASE_URL, create_llm

DEEPSEEK = "openrouter:deepseek/deepseek-v4-flash-0731"


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_openrouter_model_becomes_an_openai_compatible_client(api_key):
    llm = create_llm(LLMConfig(model=DEEPSEEK, retry_config={}))

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "deepseek/deepseek-v4-flash-0731"
    assert str(llm.openai_api_base) == DEFAULT_OPENROUTER_BASE_URL


def test_provider_sort_is_lifted_into_openrouter_routing(api_key):
    llm = create_llm(
        LLMConfig(
            model=DEEPSEEK,
            provider_config={"provider_sort": "exacto", "reasoning_effort": "xhigh"},
            retry_config={},
        )
    )

    assert llm.extra_body == {"provider": {"sort": "exacto"}}
    assert llm.reasoning_effort == "xhigh"


def test_provider_sort_merges_with_an_existing_extra_body(api_key):
    llm = create_llm(
        LLMConfig(
            model=DEEPSEEK,
            provider_config={
                "provider_sort": "throughput",
                "extra_body": {"provider": {"order": ["a"]}, "transforms": []},
            },
            retry_config={},
        )
    )

    assert llm.extra_body == {
        "provider": {"order": ["a"], "sort": "throughput"},
        "transforms": [],
    }


def test_base_url_can_be_overridden(api_key):
    llm = create_llm(
        LLMConfig(
            model=DEEPSEEK,
            provider_config={"base_url": "https://proxy.example/v1"},
            retry_config={},
        )
    )

    assert str(llm.openai_api_base) == "https://proxy.example/v1"


def test_a_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(OSError, match="OPENROUTER_API_KEY is not set"):
        create_llm(LLMConfig(model=DEEPSEEK, retry_config={}))


def test_roles_fall_back_to_the_shared_blueprint_llm():
    shared = LLMConfig(model=DEEPSEEK)
    config = BlueprintConfig(llm=shared)

    assert config.role("architect").llm is shared
    assert config.role("prover").llm is shared
    assert config.role("refiner").llm is shared


def test_a_role_can_override_the_shared_llm():
    shared = LLMConfig(model=DEEPSEEK)
    architect = LLMConfig(model="anthropic:claude-opus-4-5")
    config = BlueprintConfig(llm=shared, architect=BlueprintRoleConfig(llm=architect))

    assert config.role("architect").llm is architect
    assert config.role("prover").llm is shared


def test_a_role_without_any_llm_is_an_error():
    with pytest.raises(ValueError, match="blueprint.prover.llm is unset"):
        BlueprintConfig().role("prover")


def test_role_fallback_preserves_the_roles_budgets():
    config = BlueprintConfig(
        llm=LLMConfig(model=DEEPSEEK),
        prover=BlueprintRoleConfig(max_total_tokens=65536, max_attempts=4),
    )

    role = config.role("prover")

    assert role.max_total_tokens == 65536
    assert role.max_attempts == 4


def test_default_budgets_are_generous_enough_for_the_attempt_counts():
    """The paper's 65536 starved node provers after 1-2 of 4 attempts for under a cent."""
    config = BlueprintConfig()

    assert config.architect.max_attempts == 8
    assert config.prover.max_attempts == 4
    assert config.max_refinement_rounds == 8
    assert config.comparator.permitted_axioms == ["propext", "Quot.sound", "Classical.choice"]

    # Generous by design: token spend is not the binding constraint on a cheap model.
    assert config.prover.max_total_tokens >= 1_000_000
    assert config.architect.max_total_tokens >= config.prover.max_total_tokens
    assert config.refiner.max_total_tokens >= config.prover.max_total_tokens
