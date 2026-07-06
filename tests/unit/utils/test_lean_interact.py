"""Unit tests for LeanInteractServer.

These tests intentionally do not spin up a real Lean process. The third test mocks
AutoLeanServer/LocalProject/LeanREPLConfig at the module path to validate that the
class wires `run()` to the underlying server. End-to-end behavior against a real
Lean project is covered in tests/regression.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lean_interact import Command

from ax_prover.config import LeanInteractConfig
from ax_prover.utils.lean_interact import LeanInteractServer


class TestLeanInteractServer:
    """Tests for LeanInteractServer."""

    async def test_aclose_on_unused_server_is_safe(self, tmp_path):
        """Closing a server that was never started must not raise."""
        server = LeanInteractServer(str(tmp_path), LeanInteractConfig())
        await server.aclose()  # no AttributeError

    async def test_async_context_manager_does_not_start_server(self, tmp_path):
        """Entering and exiting the context manager without calling run() is a no-op on the server."""
        async with LeanInteractServer(str(tmp_path), LeanInteractConfig()) as server:
            assert server._server is None
        assert server._server is None

    async def test_run_invokes_underlying_server(self, tmp_path, monkeypatch):
        """run() lazily starts AutoLeanServer and delegates to async_run."""
        fake_server = MagicMock()
        dummy_result = "ok"
        fake_server.async_run = AsyncMock(return_value=dummy_result)

        monkeypatch.setattr(
            "ax_prover.utils.lean_interact.AutoLeanServer", lambda *_a, **_k: fake_server
        )
        monkeypatch.setattr("ax_prover.utils.lean_interact.LocalProject", lambda **_k: None)
        monkeypatch.setattr("ax_prover.utils.lean_interact.LeanREPLConfig", lambda **_k: None)

        async with LeanInteractServer(str(tmp_path), LeanInteractConfig()) as server:
            result = await server.run(Command(cmd="example : 1 = 1 := rfl"))

        assert result == dummy_result
        fake_server.async_run.assert_awaited_once()
        fake_server.kill.assert_called_once()

    async def test_run_serializes_concurrent_commands(self, tmp_path, monkeypatch):
        """Concurrent run() calls never overlap on the single-subprocess REPL."""
        active = 0
        max_active = 0

        async def fake_async_run(_command):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)  # yield: overlap would surface here if unlocked
            active -= 1
            return "ok"

        fake_server = MagicMock()
        fake_server.async_run = AsyncMock(side_effect=fake_async_run)

        monkeypatch.setattr(
            "ax_prover.utils.lean_interact.AutoLeanServer", lambda *_a, **_k: fake_server
        )
        monkeypatch.setattr("ax_prover.utils.lean_interact.LocalProject", lambda **_k: None)
        monkeypatch.setattr("ax_prover.utils.lean_interact.LeanREPLConfig", lambda **_k: None)

        async with LeanInteractServer(str(tmp_path), LeanInteractConfig()) as server:
            await asyncio.gather(
                *(server.run(Command(cmd="example : 1 = 1 := rfl")) for _ in range(10))
            )

        assert max_active == 1
