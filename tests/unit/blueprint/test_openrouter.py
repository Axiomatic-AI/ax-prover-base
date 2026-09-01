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


def test_extra_body_passes_through_verbatim(api_key):
    """OpenRouter's controls live at the body root, so extra_body must not be rewritten."""
    extra = {
        "reasoning": {"effort": "high"},
        "session_id": "ax-prover-blueprint",
        "provider": {"only": ["deepseek/fp8", "coreweave/fp8"]},
    }
    llm = create_llm(
        LLMConfig(model=DEEPSEEK, provider_config={"extra_body": extra}, retry_config={})
    )

    assert llm.extra_body == extra


def test_reasoning_effort_is_not_sent_as_a_kwarg(api_key):
    """A `reasoning_effort` kwarg does not reach OpenRouter's unified reasoning control.

    Effort belongs in `extra_body.reasoning.effort`; a kwarg leaves v4-flash at its HIGH
    default, which silently invalidated a high-versus-xhigh comparison.
    """
    from ax_prover.config import Config
    from ax_prover.utils import merge_configs

    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])
    provider_config = config.blueprint.llm.provider_config

    assert "reasoning_effort" not in provider_config
    assert provider_config["extra_body"]["reasoning"]["effort"] == "high"


def test_routing_is_pinned_to_fp8_or_better(api_key):
    """Mixing fp4, fp8 and unquantized upstreams puts a confound under every measurement."""
    from ax_prover.config import Config
    from ax_prover.utils import merge_configs

    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])
    extra = config.blueprint.llm.provider_config["extra_body"]

    only = extra["provider"]["only"]
    assert only, "routing must be restricted"
    # "or better" is the point: bf16 is higher precision than fp8, fp4 is lower. The
    # unsuffixed first-party upstream is allowed because it is the reference deployment.
    permitted = ("/fp8", "/bf16", "/fp16")
    assert all(p == "deepseek" or p.endswith(permitted) for p in only), only
    assert not any(p.endswith("/fp4") for p in only), only
    # `order` would disable sticky routing, and a narrow list with fallbacks off 404s.
    assert "order" not in extra["provider"]
    assert extra["provider"].get("allow_fallbacks") is not False
    # A stable session_id is what keeps the prompt cache warm across node attempts.
    assert extra["session_id"]


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
