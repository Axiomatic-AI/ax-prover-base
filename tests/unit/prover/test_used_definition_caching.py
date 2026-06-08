"""Tests for the WS1 used-definition caching feature."""

import asyncio
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


def test_memory_node_preserves_experience_and_adds_used_definitions():
    """Behavioral: _memory_processor_node keeps the memory strategy's output and adds the cache."""
    from ax_prover.prover.agent import ProverAgent

    class _StubMemory:
        async def process(self, state):
            return {"experience": "LESSON-X"}

    state = ProverAgentState(item=TargetItem(title="t"))  # used_definitions defaults to {}

    fake = SimpleNamespace(memory=_StubMemory(), _local_searcher=None)
    # Bind the real methods to the fake so the no-op accumulation path runs.
    fake._accumulate_used_definitions = ProverAgent._accumulate_used_definitions.__get__(fake)
    bound_node = ProverAgent._memory_processor_node.__get__(fake)

    result = asyncio.run(bound_node(state))

    assert result["experience"] == "LESSON-X"  # memory strategy output preserved
    assert result["used_definitions"] == {}  # cache added (empty: no searcher / no proposal)


def test_proposer_node_injects_local_definitions():
    from ax_prover.prover.agent import ProverAgent

    source = inspect.getsource(ProverAgent._proposer_node)
    assert "_render_used_definitions" in source
    assert "state.used_definitions" in source


def test_render_used_definitions_drops_trailing_entries_over_budget():
    from ax_prover.prover.agent import ProverAgent

    fake = SimpleNamespace(max_input_tokens=400)  # budget = 100 chars
    render = ProverAgent._render_used_definitions.__get__(fake)
    used = {"A.one": "x" * 60, "A.two": "y" * 60, "A.three": "z" * 60}
    block = render(used)
    assert "x" * 60 in block  # first entry kept
    assert "y" * 60 not in block  # over budget -> dropped
    assert "omitted to fit" in block  # drop is noted


def test_render_used_definitions_keeps_all_when_budget_ample():
    from ax_prover.prover.agent import ProverAgent

    fake = SimpleNamespace(max_input_tokens=1_000_000)
    render = ProverAgent._render_used_definitions.__get__(fake)
    block = render({"A.one": "def one := 1", "A.two": "def two := 2"})
    assert "def one := 1" in block
    assert "def two := 2" in block
    assert "omitted" not in block


def test_render_used_definitions_empty_returns_none():
    from ax_prover.prover.agent import ProverAgent

    fake = SimpleNamespace(max_input_tokens=1000)
    render = ProverAgent._render_used_definitions.__get__(fake)
    assert render({}) is None
