"""End-to-end smoke test for Runtime against a real (minimal) Lean project.

Validates that Runtime.open wires up LeanInteractServer correctly: the server is
lazily started, can run a real Lean command, and is cleaned up on exit.
"""

from lean_interact import Command
from lean_interact.interface import LeanError

from ax_prover.config import RuntimeConfig
from ax_prover.runtime import Runtime


async def test_runtime_lean_interact_server_runs_trivial_command(lean_minimal_project):
    """Opening a Runtime and running a trivial `decide` proof returns a non-error response."""
    async with Runtime.open(RuntimeConfig(), lean_minimal_project) as rt:
        response = await rt.lean_interact_server.run(
            Command(cmd="example : 1 + 1 = 2 := by decide")
        )

    assert not isinstance(response, LeanError), f"unexpected LeanError: {response}"
    assert not response.sorries
