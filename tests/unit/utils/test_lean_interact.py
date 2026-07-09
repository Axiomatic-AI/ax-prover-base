"""Unit tests for LeanInteractServer.

These tests intentionally do not spin up a real Lean process. They mock
AutoLeanServer/LocalProject/LeanREPLConfig/get_project_lean_version at the module
path (and the hard-coded REPL_SOURCES list) to validate that the class wires `run()`
to the underlying server and selects a REPL fork based on the project's Lean version.
End-to-end behavior against a real Lean project is covered in tests/regression.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from lean_interact import Command

from ax_prover.config import LeanInteractConfig
from ax_prover.utils.lean_interact import LeanInteractServer, ReplSource

DEFAULT_GIT = "https://github.com/augustepoiroux/repl"
DEFAULT_REV = "v1.3.17"
NEWER_GIT = "https://github.com/leanprover-community/repl"
NEWER_REV = "master"


def _install_mocks(monkeypatch, *, project_version, sources, supported_by_git):
    """Wire module-level mocks and return (fake_server, repl_config_calls).

    `sources` replaces the hard-coded REPL_SOURCES list. `supported_by_git` maps a fork
    git URL to the versions its probe advertises. Each `LeanREPLConfig(...)` call is
    recorded so tests can assert which fork the real (project-carrying) config used.
    """
    fake_server = MagicMock()
    fake_server.async_run = AsyncMock(return_value="ok")

    repl_config_calls: list[dict] = []

    def fake_repl_config(**kwargs):
        repl_config_calls.append(kwargs)
        cfg = MagicMock()
        git = kwargs.get("repl_git", DEFAULT_GIT)
        cfg.get_available_lean_versions.return_value = supported_by_git.get(git, [])
        return cfg

    monkeypatch.setattr(
        "ax_prover.utils.lean_interact.AutoLeanServer", lambda *_a, **_k: fake_server
    )
    monkeypatch.setattr("ax_prover.utils.lean_interact.LocalProject", lambda **_k: "project")
    monkeypatch.setattr("ax_prover.utils.lean_interact.LeanREPLConfig", fake_repl_config)
    monkeypatch.setattr(
        "ax_prover.utils.lean_interact.get_project_lean_version",
        lambda _folder: project_version,
    )
    monkeypatch.setattr("ax_prover.utils.lean_interact.REPL_SOURCES", tuple(sources))
    return fake_server, repl_config_calls


def _real_config_call(calls: list[dict]) -> dict:
    """Return the kwargs of the config built with a project (the server config)."""
    real = [c for c in calls if "project" in c]
    assert len(real) == 1, f"expected exactly one project config, got {real}"
    return real[0]


DEFAULT_SOURCES = [
    ReplSource(git=DEFAULT_GIT, rev=DEFAULT_REV),
    ReplSource(git=NEWER_GIT, rev=NEWER_REV),
]


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
        fake_server, calls = _install_mocks(
            monkeypatch,
            project_version="v4.20.0",
            sources=DEFAULT_SOURCES,
            supported_by_git={DEFAULT_GIT: ["v4.19.0", "v4.20.0"]},
        )

        async with LeanInteractServer(str(tmp_path), LeanInteractConfig()) as server:
            result = await server.run(Command(cmd="example : 1 = 1 := rfl"))

        assert result == "ok"
        fake_server.async_run.assert_awaited_once()
        fake_server.kill.assert_called_once()
        # First source supports the version, so the server config uses it.
        assert _real_config_call(calls)["repl_git"] == DEFAULT_GIT

    async def test_uses_later_fork_when_first_unsupported(self, tmp_path, monkeypatch):
        """When the first fork lacks the project's version, a later fork is chosen."""
        _, calls = _install_mocks(
            monkeypatch,
            project_version="v4.32.0-rc1",
            sources=DEFAULT_SOURCES,
            supported_by_git={
                DEFAULT_GIT: ["v4.30.0", "v4.31.0-rc1"],  # backport range, too old
                NEWER_GIT: ["v4.32.0-rc1"],  # newest
            },
        )

        async with LeanInteractServer(str(tmp_path), LeanInteractConfig()) as server:
            await server.run(Command(cmd="example : 1 = 1 := rfl"))

        real = _real_config_call(calls)
        assert real["repl_git"] == NEWER_GIT
        assert real["repl_rev"] == NEWER_REV

    async def test_raises_when_no_fork_supports_version(self, tmp_path, monkeypatch):
        """If no known fork advertises the project's version, server creation raises clearly."""
        _install_mocks(
            monkeypatch,
            project_version="v4.99.0",
            sources=DEFAULT_SOURCES,
            supported_by_git={
                DEFAULT_GIT: ["v4.30.0", "v4.31.0-rc1"],
                NEWER_GIT: ["v4.32.0-rc1"],
            },
        )

        server = LeanInteractServer(str(tmp_path), LeanInteractConfig())
        with pytest.raises(RuntimeError, match="v4.99.0"):
            await server.run(Command(cmd="example : 1 = 1 := rfl"))

    async def test_unknown_project_version_uses_default_fork(self, tmp_path, monkeypatch):
        """An unreadable toolchain falls back to lean_interact's default resolution."""
        _, calls = _install_mocks(
            monkeypatch,
            project_version=None,
            sources=DEFAULT_SOURCES,
            supported_by_git={},
        )

        async with LeanInteractServer(str(tmp_path), LeanInteractConfig()) as server:
            await server.run(Command(cmd="example : 1 = 1 := rfl"))

        # No probing; the single config is built without an explicit fork override.
        real = _real_config_call(calls)
        assert "repl_git" not in real
