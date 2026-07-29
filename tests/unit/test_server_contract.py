"""Guards the surface ax-prover-server depends on from ax-prover-base.

ax-prover-server consumes ax-prover-base as an external tool, so refactors here can silently
break the bot with no signal in this repo (it happened once, in #28, which removed
`get_unproven`). This module is the single place that enumerates each dependency the server
relies on. Each test names the exact server usage site so the contract is auditable from here.

These tests assert only that ax-prover-base still *exposes* the contract; they cannot verify
the server still calls it correctly. A true end-to-end check belongs in ax-prover-server.
"""

import inspect
from pathlib import Path

import ax_prover
from ax_prover.utils.lean_parsing import get_unproven


def test_get_unproven_is_sync_two_arg_callable():
    """Sorry detection imports get_unproven and calls it as get_unproven(folder, file_path).

    Server usage: src/ax_prover_server/scripts/detect_unproven.sh.template
        from ax_prover.utils.lean_parsing import get_unproven
        get_unproven(".", "<lean_file>")
    """
    assert not inspect.iscoroutinefunction(get_unproven)
    assert list(inspect.signature(get_unproven).parameters) == ["folder", "file_path"]


def test_default_config_ships_with_package():
    """Prover jobs run `ax-prover --config configs/default.yaml prove ...`, so the bundled
    default config must exist inside the installed package.

    Server usage: src/ax_prover_server/scripts/run_prover.sh.template
    """
    default_config = Path(ax_prover.__file__).parent / "configs" / "default.yaml"
    assert default_config.is_file()
