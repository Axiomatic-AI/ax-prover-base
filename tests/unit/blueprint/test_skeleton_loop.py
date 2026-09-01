"""The architect/refiner repair loop: a rejected skeleton must come back to the model."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ax_prover.blueprint import generation
from ax_prover.blueprint.generation import ArchitectProposal, run_skeleton_loop
from ax_prover.blueprint.models import BlueprintValidationError
from ax_prover.blueprint.roles import RoleTurn
from ax_prover.config import BlueprintRoleConfig

from .conftest import make_blueprint, make_node

BASE = [SystemMessage(content="system"), HumanMessage(content="user")]

FIRST = "theorem bad : True := by sorry"
SECOND = "theorem good : True := by sorry"


def stub_turns(monkeypatch, sources, seen):
    """Return each source in turn, recording the messages the model was handed."""
    proposals = iter(sources)

    async def fake_run_turn(client, messages, *args, **kwargs):
        seen.append(messages)
        return RoleTurn(response=AIMessage(content="{}"))

    def fake_parse(turn, schema):
        return ArchitectProposal(helpers=next(proposals))

    monkeypatch.setattr(generation, "run_turn", fake_run_turn)
    monkeypatch.setattr(generation, "parse_proposal", fake_parse)


async def test_a_rejected_skeleton_is_shown_back_on_the_repair_round(workspace, monkeypatch):
    """The bug: the loop rebuilt the prompt from scratch, so the model never saw its own
    skeleton and reproduced the identical error until the budget ran out."""
    seen: list[list] = []
    stub_turns(monkeypatch, [FIRST, SECOND], seen)

    calls = {"n": 0}

    async def fake_build(_ws, _server, helpers, *args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BlueprintValidationError(["node 'a' declares unknown parent 'ghost'"])
        return make_blueprint(make_node("a"))

    monkeypatch.setattr(generation, "build_blueprint", fake_build)

    candidate = await run_skeleton_loop(
        workspace, None, None, BlueprintRoleConfig(), BASE, [], "refiner round 7"
    )

    assert candidate.helpers == SECOND
    repair = seen[1][-1].content
    assert "unknown parent 'ghost'" in repair
    assert FIRST in repair, "the rejected source must be quoted back to the model"


async def test_the_first_attempt_carries_no_repair_block(workspace, monkeypatch):
    seen: list[list] = []
    stub_turns(monkeypatch, [SECOND], seen)

    async def fake_build(_ws, _server, helpers, *args):
        return make_blueprint(make_node("a"))

    monkeypatch.setattr(generation, "build_blueprint", fake_build)

    await run_skeleton_loop(workspace, None, None, BlueprintRoleConfig(), BASE, [], "architect")

    assert seen[0] == BASE


async def test_an_unparseable_response_repairs_without_a_source_block(workspace, monkeypatch):
    """There is no rejected skeleton to quote when the response did not parse at all."""
    seen: list[list] = []
    responses = iter([None, ArchitectProposal(helpers=SECOND)])

    async def fake_run_turn(client, messages, *args, **kwargs):
        seen.append(messages)
        return RoleTurn(response=AIMessage(content="{}"))

    monkeypatch.setattr(generation, "run_turn", fake_run_turn)
    monkeypatch.setattr(generation, "parse_proposal", lambda turn, schema: next(responses))

    async def fake_build(_ws, _server, helpers, *args):
        return make_blueprint(make_node("a"))

    monkeypatch.setattr(generation, "build_blueprint", fake_build)

    await run_skeleton_loop(workspace, None, None, BlueprintRoleConfig(), BASE, [], "architect")

    repair = seen[1][-1].content
    assert "not valid structured output" in repair
    assert "```lean" not in repair


class InformalClient:
    """Answers the informal-proof call; run_turn is stubbed so no other call happens."""

    def __init__(self, proof="Step 1: n + 0 = n by the definition of addition.", fail=False):
        self.proof = proof
        self.fail = fail
        self.calls = 0

    async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return AIMessage(content=self.proof)


def stub_loop(monkeypatch, seen):
    async def fake_run_turn(client, messages, *args, **kwargs):
        seen.append(messages)
        return RoleTurn(response=AIMessage(content="{}"))

    async def fake_build(_ws, _server, helpers, *args):
        return make_blueprint(make_node("a"))

    monkeypatch.setattr(generation, "run_turn", fake_run_turn)
    monkeypatch.setattr(
        generation, "parse_proposal", lambda t, s: ArchitectProposal(helpers=SECOND)
    )
    monkeypatch.setattr(generation, "build_blueprint", fake_build)


async def test_the_architect_is_seeded_with_an_informal_proof(workspace, monkeypatch):
    seen: list[list] = []
    stub_loop(monkeypatch, seen)
    client = InformalClient()

    await generation.generate_blueprint(workspace, None, client, BlueprintRoleConfig())

    assert client.calls == 1
    user_prompt = seen[0][-1].content
    assert "informal proof" in user_prompt
    assert "Step 1: n + 0 = n" in user_prompt


async def test_caller_supplied_context_skips_the_informal_proof(workspace, monkeypatch):
    seen: list[list] = []
    stub_loop(monkeypatch, seen)
    client = InformalClient()

    await generation.generate_blueprint(
        workspace, None, client, BlueprintRoleConfig(), extra_context="use the official solution"
    )

    assert client.calls == 0
    assert "use the official solution" in seen[0][-1].content


async def test_a_failed_informal_proof_proceeds_unguided(workspace, monkeypatch):
    seen: list[list] = []
    stub_loop(monkeypatch, seen)

    await generation.generate_blueprint(
        workspace, None, InformalClient(fail=True), BlueprintRoleConfig()
    )

    assert "Additional context" not in seen[0][-1].content
