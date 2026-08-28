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


def test_experiment_subcommand_accepts_the_blueprint_flags():
    """These were only on `prove`, so `experiment --max-lean-compiles 8` was ignored."""
    import contextlib
    import io

    from ax_prover.main import main

    argv = [
        "ax-prover",
        "experiment",
        "ds",
        "--max-lean-compiles",
        "8",
        "--max-node-agents",
        "6",
        "--max-refinements",
        "3",
    ]
    with contextlib.suppress(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        import sys

        original, sys.argv = sys.argv, argv
        try:
            # Fails later for lack of a dataset; argument parsing is what matters here.
            main()
        finally:
            sys.argv = original


def test_a_mistyped_flag_is_rejected_not_swallowed(capsys):
    """`parse_known_args` folded unknown flags into config overrides, hiding typos."""
    import sys

    import pytest

    from ax_prover.main import main

    original, sys.argv = sys.argv, ["ax-prover", "experiment", "ds", "--no-such-flag", "8"]
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = original

    assert "unrecognized argument" in capsys.readouterr().err


def test_blueprint_overrides_tolerate_absent_mode_flags():
    """`experiment` has no --blueprint/--restart, so the collector must not require them."""
    import argparse

    from ax_prover.commands.prove import BlueprintOverrides
    from ax_prover.main import _blueprint_overrides

    args = argparse.Namespace(
        max_refinements=3,
        max_node_agents=6,
        max_lean_compiles=8,
        checkpoint_dir=None,
        require_comparator=False,
    )
    overrides = _blueprint_overrides(args)

    assert isinstance(overrides, BlueprintOverrides)
    assert overrides.max_lean_compiles == 8
    assert overrides.enabled is False

    config = merge_configs([Config(), "default.yaml", "blueprint.yaml"])
    overrides.apply(config)
    assert config.blueprint.max_lean_compiles == 8
    assert config.blueprint.max_node_agents == 6
    assert config.blueprint.max_refinement_rounds == 3
