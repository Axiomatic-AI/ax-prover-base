"""Unit tests for the Runtime composition root.

These tests do not exercise a real Lean project: LeanInteractServer is constructed
but never started (its `_get_server` is only invoked when `.run()` is called).
End-to-end behavior is covered in tests/regression.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from ax_prover.config import RuntimeConfig
from ax_prover.runtime import Runtime
from ax_prover.utils.lean_interact import LeanInteractServer


@pytest.fixture
def runtime_config():
    return RuntimeConfig()


async def test_open_populates_attributes(runtime_config, tmp_path):
    """Runtime.open exposes the expected attributes and an empty tool_resources dict by default."""
    async with Runtime.open(runtime_config, str(tmp_path)) as rt:
        assert isinstance(rt.lean_interact_server, LeanInteractServer)
        assert isinstance(rt.lean_semaphore, asyncio.Semaphore)
        assert rt._tool_resources == {}
        assert rt.get_tool_resources("missing") is None


def test_direct_construction_leaves_attributes_unset(runtime_config):
    """Constructing a Runtime outside of Runtime.open() leaves resource attributes unset.

    This documents the design intent: the only safe way to obtain a usable Runtime is via
    `async with Runtime.open(...)`. Touching `.lean_interact_server` outside that context
    must fail loudly rather than return a half-initialized object.
    """
    rt = Runtime(runtime_config, "/nonexistent")
    with pytest.raises(AttributeError):
        _ = rt.lean_interact_server
    with pytest.raises(AttributeError):
        _ = rt.lean_semaphore


async def test_tool_lifespans_are_entered_and_resources_exposed(runtime_config, tmp_path):
    """Tool lifespans are entered on open, exposed via get_tool_resources, and exited on close."""
    entered = False
    exited = False

    @asynccontextmanager
    async def fake_lifespan():
        nonlocal entered, exited
        entered = True
        try:
            yield "session-token"
        finally:
            exited = True

    async with Runtime.open(
        runtime_config, str(tmp_path), tool_lifespans={"foo": fake_lifespan()}
    ) as rt:
        assert entered
        assert not exited
        assert rt.get_tool_resources("foo") == "session-token"

    assert exited


async def test_lifespan_cleanup_runs_on_body_exception(runtime_config, tmp_path):
    """Lifespan cleanup must run even if the body of `async with Runtime.open(...)` raises."""
    exited = False

    @asynccontextmanager
    async def fake_lifespan():
        nonlocal exited
        try:
            yield "x"
        finally:
            exited = True

    with pytest.raises(RuntimeError, match="boom"):
        async with Runtime.open(
            runtime_config, str(tmp_path), tool_lifespans={"f": fake_lifespan()}
        ):
            raise RuntimeError("boom")

    assert exited
