"""Tests for the WS1 used-definition caching feature."""

from ax_prover.models.proving import ProverAgentState, TargetItem


def _state() -> ProverAgentState:
    return ProverAgentState(item=TargetItem(title="t"))


def test_used_definitions_defaults_to_empty_dict():
    state = _state()
    assert state.used_definitions == {}


def test_used_definitions_accepts_mapping():
    state = ProverAgentState(
        item=TargetItem(title="t"),
        used_definitions={"A.foo": "-- A.foo — Def.lean:1\ndef foo := 1"},
    )
    assert "A.foo" in state.used_definitions
