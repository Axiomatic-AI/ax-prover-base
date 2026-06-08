"""Tests for the WS1 used-definition caching feature."""

import inspect
from types import SimpleNamespace

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


def test_local_definitions_prompt_wraps_definitions():
    from ax_prover.prover.prompts import LOCAL_DEFINITIONS_USER_PROMPT

    rendered = LOCAL_DEFINITIONS_USER_PROMPT.format(definitions="-- A.foo\ndef foo := 1")
    assert "<local-definitions>" in rendered
    assert "</local-definitions>" in rendered
    assert "def foo := 1" in rendered


def test_iterative_system_prompt_mentions_local_definitions():
    from ax_prover.prover.prompts import PROPOSER_SYSTEM_PROMPT

    assert "<local-definitions>" in PROPOSER_SYSTEM_PROMPT


def test_find_local_searcher_locates_the_searcher(tmp_path):
    from ax_prover.prover.agent import ProverAgent
    from ax_prover.tools.local_lean_search import (
        LocalLeanSearcher,
        SearchLeanLocalConfig,
        create_search_lean_local_tool,
    )

    (tmp_path / "lakefile.toml").write_text('name = "demo"\n', encoding="utf-8")
    tool = create_search_lean_local_tool(SearchLeanLocalConfig(), base_folder=str(tmp_path))
    fake = SimpleNamespace(proposer_tools=[tool])

    searcher = ProverAgent._find_local_searcher(fake)
    assert isinstance(searcher, LocalLeanSearcher)


def test_find_local_searcher_returns_none_when_absent():
    from ax_prover.prover.agent import ProverAgent

    fake = SimpleNamespace(proposer_tools=[])
    assert ProverAgent._find_local_searcher(fake) is None


def test_memory_node_wires_in_accumulation():
    from ax_prover.prover.agent import ProverAgent

    source = inspect.getsource(ProverAgent._memory_processor_node)
    assert "_accumulate_used_definitions(" in source
    assert "used_definitions" in source


def test_proposer_node_injects_local_definitions():
    from ax_prover.prover.agent import ProverAgent

    source = inspect.getsource(ProverAgent._proposer_node)
    assert "state.used_definitions" in source
    assert "LOCAL_DEFINITIONS_USER_PROMPT" in source
