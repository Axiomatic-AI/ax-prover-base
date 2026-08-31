"""Node prover isolation, the proof-body contract, and harness verification."""

import asyncio
import json

from langchain_core.messages import AIMessage

from ax_prover.blueprint.models import NodeOutcome
from ax_prover.blueprint.node_prover import (
    NodeProposal,
    _build_messages,
    format_parents,
    prove_node,
)
from ax_prover.config import BlueprintRoleConfig

from .conftest import make_blueprint, make_node

# attempt_concurrency=1 keeps these tests deterministic: they assert on attempt counts and
# on which canned response was consumed, which a parallel wave would reorder.
ROLE = BlueprintRoleConfig(
    max_total_tokens=0, max_attempts=2, max_tool_iterations=0, attempt_concurrency=1
)


class FakeClient:
    """An LLM client that replays canned responses and records the prompts it saw."""

    def __init__(self, *payloads: dict):
        self.payloads = list(payloads)
        self.prompts: list[list] = []

    async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
        self.prompts.append(messages)
        payload = self.payloads.pop(0) if self.payloads else {}
        return AIMessage(content=json.dumps(payload), usage_metadata=None)


def proposal(body: str) -> dict:
    return NodeProposal(reasoning="because", proof_body=body).model_dump()


def compiler(*results: tuple[bool, str]):
    """A `compile_source` stand-in returning `results` in order."""
    queue = list(results)
    calls: list[str] = []

    async def compile_source(source, label="scratch", **kwargs):
        calls.append(source)
        # The last result repeats, so an extra attempt cannot accidentally succeed.
        success, output = queue.pop(0) if len(queue) > 1 else queue[0]
        return type("Result", (), {"success": success, "output": output, "source": source})()

    compile_source.calls = calls
    return compile_source


def test_prompt_shows_only_direct_parent_interfaces():
    parent = make_node("parent", signature=": True")
    node = make_node("child", ("parent",), signature=": True", doc_text="## Proof\n\nUse parent.")

    messages = _build_messages(node, (parent,), [])
    user_prompt = str(messages[1].content)

    assert "theorem AxProverGenerated_target_deadbeef.parent : True" in user_prompt
    assert "Use parent." in user_prompt
    assert "your proof body here" in user_prompt
    # Parents appear as signatures only: no proof body, no placeholder.
    assert "sorry" not in user_prompt
    assert user_prompt.count(":=") == 1


def test_prompt_replays_only_the_two_most_recent_attempts():
    node = make_node("child")
    history = [f"## Attempt {i}" for i in range(1, 5)]

    text = "\n".join(str(m.content) for m in _build_messages(node, (), history))

    assert "## Attempt 3" in text and "## Attempt 4" in text
    assert "## Attempt 1" not in text


def test_format_parents_uses_fully_qualified_names():
    rendered = format_parents((make_node("parent", signature=": True"),))

    assert rendered == "theorem AxProverGenerated_target_deadbeef.parent : True"


async def test_a_compiling_proof_body_is_solved(workspace, monkeypatch):
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    monkeypatch.setattr(workspace, "compile_source", compiler((True, "")))

    result = await prove_node(workspace, blueprint, node, FakeClient(proposal("by trivial")), ROLE)

    assert result.outcome is NodeOutcome.SOLVED
    assert result.proof_body == "by trivial"
    assert result.attempts == 1


async def test_the_harness_recompiles_rather_than_trusting_the_model(workspace, monkeypatch):
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    compile_source = compiler((False, "unsolved goals"), (False, "unsolved goals"))
    monkeypatch.setattr(workspace, "compile_source", compile_source)

    client = FakeClient(
        proposal("by trivial"), proposal("by simp"), {"outcome": "PROOF_TOO_HARD", "detail": "no"}
    )
    result = await prove_node(workspace, blueprint, node, client, ROLE)

    assert result.outcome is NodeOutcome.PROOF_TOO_HARD
    assert result.proof_body is None
    assert len(compile_source.calls) == 2


async def test_a_sorry_in_the_proposal_is_rejected_without_compiling(workspace, monkeypatch):
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    compile_source = compiler((True, ""))
    monkeypatch.setattr(workspace, "compile_source", compile_source)

    client = FakeClient(
        proposal("by sorry"),
        proposal("by admit"),
        {"outcome": "PROOF_TOO_HARD", "detail": "placeholders only"},
    )
    result = await prove_node(workspace, blueprint, node, client, ROLE)

    assert result.outcome is NodeOutcome.PROOF_TOO_HARD
    assert compile_source.calls == []


async def test_a_restated_theorem_is_reduced_to_its_body(workspace, monkeypatch):
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    compile_source = compiler((True, ""))
    monkeypatch.setattr(workspace, "compile_source", compile_source)

    client = FakeClient(proposal("theorem helper : True := trivial"))
    result = await prove_node(workspace, blueprint, node, client, ROLE)

    assert result.outcome is NodeOutcome.SOLVED
    assert result.proof_body == "trivial"


async def test_a_wrong_statement_is_diagnosed_as_such(workspace, monkeypatch):
    node = make_node("helper", signature=": False")
    blueprint = make_blueprint(node)
    monkeypatch.setattr(workspace, "compile_source", compiler((False, "type mismatch")))

    client = FakeClient(
        proposal("by trivial"),
        proposal("by simp"),
        {"outcome": "STATEMENT_WRONG", "detail": "the goal is false as stated"},
    )
    result = await prove_node(workspace, blueprint, node, client, ROLE)

    assert result.outcome is NodeOutcome.STATEMENT_WRONG
    assert result.diagnosis.detail == "the goal is false as stated"
    assert result.diagnosis.last_error == "type mismatch"


async def test_triage_failure_falls_back_to_proof_too_hard(workspace, monkeypatch):
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    monkeypatch.setattr(workspace, "compile_source", compiler((False, "boom")))

    client = FakeClient(proposal("by trivial"), proposal("by simp"), {"nonsense": True})
    result = await prove_node(workspace, blueprint, node, client, ROLE)

    assert result.outcome is NodeOutcome.PROOF_TOO_HARD


async def test_an_unparseable_response_costs_an_attempt(workspace, monkeypatch):
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    compile_source = compiler((True, ""))
    monkeypatch.setattr(workspace, "compile_source", compile_source)

    client = FakeClient({"not": "a proposal"}, proposal("by trivial"))
    result = await prove_node(workspace, blueprint, node, client, ROLE)

    assert result.outcome is NodeOutcome.SOLVED
    assert result.attempts == 2


async def test_starvation_reports_budget_exhausted_not_proof_too_hard(workspace, monkeypatch):
    """A starved node must not be labelled hard: it misdirects the refiner into splitting."""
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    monkeypatch.setattr(workspace, "compile_source", compiler((False, "unsolved goals")))
    monkeypatch.setattr(workspace, "compile_candidate", compiler((False, "unsolved goals")))

    # A tiny ceiling with usage reported, so the budget dies before the attempts run out.
    role = BlueprintRoleConfig(max_total_tokens=10, max_attempts=4, max_tool_iterations=0)

    class MeteredClient(FakeClient):
        async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
            self.prompts.append(messages)
            payload = self.payloads.pop(0) if self.payloads else {}
            return AIMessage(
                content=json.dumps(payload),
                usage_metadata={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
            )

    client = MeteredClient(proposal("by trivial"), proposal("by simp"))
    result = await prove_node(workspace, blueprint, node, client, role)

    assert result.outcome is NodeOutcome.BUDGET_EXHAUSTED
    assert result.attempts < role.max_attempts
    assert "exhausted" in result.diagnosis.detail
    assert result.diagnosis.outcome is NodeOutcome.BUDGET_EXHAUSTED


async def test_using_every_attempt_still_reports_a_real_diagnosis(workspace, monkeypatch):
    """Exhausting attempts rather than budget must still triage normally."""
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    monkeypatch.setattr(workspace, "compile_candidate", compiler((False, "unsolved goals")))

    client = FakeClient(
        proposal("by trivial"),
        proposal("by simp"),
        {"outcome": "PROOF_TOO_HARD", "detail": "genuinely hard"},
    )
    result = await prove_node(workspace, blueprint, node, client, ROLE)

    assert result.outcome is NodeOutcome.PROOF_TOO_HARD
    assert result.diagnosis.detail == "genuinely hard"


async def test_attempts_run_in_parallel_waves(workspace, monkeypatch):
    """Sequential attempts made a 3-try node pay three full round-trips."""
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    running = 0
    peak = 0

    async def slow_compile(source, label="scratch", **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return type("R", (), {"success": False, "output": "unsolved goals", "source": source})()

    monkeypatch.setattr(workspace, "compile_candidate", slow_compile)
    role = BlueprintRoleConfig(
        max_total_tokens=0, max_attempts=4, max_tool_iterations=0, attempt_concurrency=2
    )
    client = FakeClient(*[proposal("by simp")] * 4, {"outcome": "PROOF_TOO_HARD", "detail": "x"})

    result = await prove_node(workspace, blueprint, node, client, role)

    assert peak == 2, "two attempts should be in flight per wave"
    assert result.attempts == 4
    assert result.outcome is NodeOutcome.PROOF_TOO_HARD


async def test_a_wave_returns_the_first_success_and_stops(workspace, monkeypatch):
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    calls = {"n": 0}

    async def compile_second_ok(source, label="scratch", **kwargs):
        calls["n"] += 1
        ok = calls["n"] == 2
        return type("R", (), {"success": ok, "output": "" if ok else "no", "source": source})()

    monkeypatch.setattr(workspace, "compile_candidate", compile_second_ok)
    role = BlueprintRoleConfig(
        max_total_tokens=0, max_attempts=4, max_tool_iterations=0, attempt_concurrency=2
    )
    client = FakeClient(*[proposal("by simp")] * 4)

    result = await prove_node(workspace, blueprint, node, client, role)

    assert result.outcome is NodeOutcome.SOLVED
    assert result.attempts == 2, "the wave that succeeded ends the node"


async def test_later_waves_see_every_earlier_failure(workspace, monkeypatch):
    """Parallelism must not cost the feedback loop."""
    node = make_node("helper", signature=": True")
    blueprint = make_blueprint(node)
    prompts: list[str] = []

    async def failing(source, label="scratch", **kwargs):
        return type("R", (), {"success": False, "output": "unsolved goals", "source": source})()

    monkeypatch.setattr(workspace, "compile_candidate", failing)

    class Recorder(FakeClient):
        async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
            prompts.append("\n".join(str(getattr(m, "content", m)) for m in messages))
            return await super().ainvoke(messages, tools, output_schema, retry_config)

    role = BlueprintRoleConfig(
        max_total_tokens=0, max_attempts=4, max_tool_iterations=0, attempt_concurrency=2
    )
    client = Recorder(*[proposal("by simp")] * 4, {"outcome": "PROOF_TOO_HARD", "detail": "x"})

    await prove_node(workspace, blueprint, node, client, role)

    # The second wave's prompts replay the first wave's failures.
    assert any("previous attempts failed" in p for p in prompts[2:])
