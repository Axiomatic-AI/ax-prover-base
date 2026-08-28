"""Blueprint YAML config resolution and default-mode isolation."""

from ax_prover.config import Config
from ax_prover.utils import merge_configs


def test_default_config_leaves_blueprint_mode_off():
    config = merge_configs([Config(), "default.yaml"])

    assert config.blueprint.enabled is False
    assert config.prover.prover_llm is not None
    assert config.prover.max_iterations == 50


def test_blueprint_yaml_enables_the_mode_and_resolves_every_role():
    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])

    assert config.blueprint.enabled is True
    assert config.blueprint.llm.model == "openrouter:deepseek/deepseek-v4-flash-0731"
    assert config.blueprint.llm.provider_config["provider_sort"] == "exacto"

    for role in ("architect", "prover", "refiner"):
        assert config.blueprint.role(role).llm is not None


def test_blueprint_yaml_budgets_allow_every_attempt_to_run():
    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])

    assert config.blueprint.architect.max_attempts == 8
    assert config.blueprint.prover.max_attempts == 4
    assert config.blueprint.max_refinement_rounds == 8
    # Generous on purpose: the paper's 65536 starved provers for under a cent.
    assert config.blueprint.prover.max_total_tokens >= 1_000_000
    assert config.blueprint.architect.max_total_tokens >= 2_000_000


def test_blueprint_yaml_exposes_only_mathlib_search():
    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])

    assert set(config.blueprint.prover_tools) == {"mathlib_search"}
    assert config.blueprint.prover_tools["mathlib_search"]["tool_type"] == "search_lean_search"


def test_blueprint_yaml_keeps_the_direct_prover_configured():
    """Blueprint mode is additive: the existing prover config is untouched."""
    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])

    assert config.prover.prover_llm.model.startswith("anthropic:")
    assert set(config.prover.proposer_tools) == {"search_lean", "search_web"}


def test_cli_overrides_fold_into_the_blueprint_config():
    from ax_prover.commands.prove import BlueprintOverrides

    config = merge_configs([Config(), "default.yaml"])
    BlueprintOverrides(
        enabled=True,
        max_refinements=16,
        max_node_agents=8,
        max_lean_compiles=2,
        checkpoint_dir="/tmp/checkpoints",
        require_comparator=True,
    ).apply(config)

    assert config.blueprint.enabled is True
    assert config.blueprint.max_refinement_rounds == 16
    assert config.blueprint.max_node_agents == 8
    assert config.blueprint.max_lean_compiles == 2
    assert config.blueprint.checkpoint_dir == "/tmp/checkpoints"
    assert config.blueprint.require_comparator is True


def test_no_overrides_leave_the_config_alone():
    from ax_prover.commands.prove import BlueprintOverrides

    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])
    BlueprintOverrides().apply(config)

    assert config.blueprint.enabled is True
    assert config.blueprint.max_refinement_rounds == 8
    assert config.blueprint.require_comparator is False


def test_putnam_config_raises_the_refinement_budget():
    config = merge_configs([Config(), "default.yaml", "configs/blueprint_putnam.yaml"])

    assert config.blueprint.enabled is True
    assert config.blueprint.max_refinement_rounds == 16
