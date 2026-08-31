"""Frontier scheduling, concurrency, persistence, and completion."""

import asyncio

import pytest

from ax_prover.blueprint import scheduler
from ax_prover.blueprint.lean_service import CompileCancelled
from ax_prover.blueprint.models import NodeDiagnosis, NodeOutcome, NodeStatus
from ax_prover.blueprint.node_prover import NodeAttemptResult
from ax_prover.blueprint.proof_store import ProofStore
from ax_prover.blueprint.scheduler import blocked_nodes, is_complete, run_schedule
from ax_prover.config import BlueprintRoleConfig

from .conftest import make_blueprint, make_node

ROLE = BlueprintRoleConfig()
ENVIRONMENT = "env"


@pytest.fixture
def store(tmp_path):
    return ProofStore.open(tmp_path, "Mod:my_target")


def cancelled_set(values):
    """Identity helper, so the released/cancelled assertion reads symmetrically."""
    return set(values)


def fake_prover(outcomes: dict[str, NodeOutcome], observed: list[str] | None = None):
    """A `prove_node` stand-in driven by a per-node outcome table."""

    async def prove_node(workspace, blueprint, node, client, role, search_tool=None):
        if observed is not None:
            observed.append(node.id)
        outcome = outcomes.get(node.id, NodeOutcome.SOLVED)
        if outcome is NodeOutcome.SOLVED:
            return NodeAttemptResult(outcome=outcome, proof_body=f"by {node.id}", attempts=1)
        return NodeAttemptResult(
            outcome=outcome,
            attempts=1,
            diagnosis=NodeDiagnosis(outcome=outcome, detail=f"{node.id} failed"),
        )

    return prove_node


async def test_nodes_are_proven_in_dependency_order(
    workspace, store, linear_blueprint, monkeypatch
):
    observed: list[str] = []
    monkeypatch.setattr(scheduler, "prove_node", fake_prover({}, observed))
    store.reconcile(linear_blueprint, ENVIRONMENT)

    report = await run_schedule(workspace, linear_blueprint, store, None, ROLE, speculative=False)

    assert observed == ["base", "middle", "target"]
    assert report.rounds == 3
    assert is_complete(linear_blueprint, store)


async def test_independent_nodes_share_one_round(workspace, store, wide_blueprint, monkeypatch):
    monkeypatch.setattr(scheduler, "prove_node", fake_prover({}))
    store.reconcile(wide_blueprint, ENVIRONMENT)

    report = await run_schedule(workspace, wide_blueprint, store, None, ROLE, speculative=False)

    assert report.rounds == 2
    assert set(report.solved) == {"left", "right", "target"}


async def test_node_agent_limit_is_respected(workspace, store, monkeypatch):
    """A positive limit throttles concurrent model agents, for provider rate limits."""
    blueprint = make_blueprint(
        *(make_node(f"h{i}") for i in range(6)),
        make_node(
            "target", tuple(f"h{i}" for i in range(6)), is_target=True, lean_name="my_target"
        ),
    )
    running = 0
    peak = 0

    async def prove_node(workspace, blueprint, node, client, role, search_tool=None):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        running -= 1
        return NodeAttemptResult(outcome=NodeOutcome.SOLVED, proof_body="by x", attempts=1)

    monkeypatch.setattr(scheduler, "prove_node", prove_node)
    store.reconcile(blueprint, ENVIRONMENT)

    await run_schedule(workspace, blueprint, store, None, ROLE, max_node_agents=2)

    assert peak <= 2


async def test_a_failed_parent_blocks_its_children(workspace, store, linear_blueprint, monkeypatch):
    observed: list[str] = []
    monkeypatch.setattr(
        scheduler, "prove_node", fake_prover({"base": NodeOutcome.PROOF_TOO_HARD}, observed)
    )
    store.reconcile(linear_blueprint, ENVIRONMENT)

    report = await run_schedule(workspace, linear_blueprint, store, None, ROLE, speculative=False)

    assert observed == ["base"]
    assert report.failed == ["base"]
    assert not is_complete(linear_blueprint, store)
    assert blocked_nodes(linear_blueprint, store) == ["base", "middle", "target"]


async def test_each_outcome_is_persisted_immediately(
    workspace, store, linear_blueprint, monkeypatch
):
    monkeypatch.setattr(
        scheduler, "prove_node", fake_prover({"middle": NodeOutcome.STATEMENT_WRONG})
    )
    store.reconcile(linear_blueprint, ENVIRONMENT)

    await run_schedule(workspace, linear_blueprint, store, None, ROLE)

    resumed = ProofStore.open(store.path.parent, "Mod:my_target", resume=True)
    assert resumed.records["base"].status is NodeStatus.SOLVED
    assert resumed.records["base"].proof_body == "by base"
    assert resumed.records["middle"].status is NodeStatus.FAILED
    assert resumed.records["middle"].diagnosis.outcome is NodeOutcome.STATEMENT_WRONG


async def test_an_infrastructure_error_stops_the_pass(
    workspace, store, wide_blueprint, monkeypatch
):
    monkeypatch.setattr(
        scheduler, "prove_node", fake_prover({"left": NodeOutcome.INFRASTRUCTURE_ERROR})
    )
    store.reconcile(wide_blueprint, ENVIRONMENT)

    report = await run_schedule(workspace, wide_blueprint, store, None, ROLE)

    assert report.infrastructure_error == "left failed"
    assert report.rounds == 1


async def test_an_unexpected_exception_becomes_an_infrastructure_error(
    workspace, store, linear_blueprint, monkeypatch
):
    async def exploding(*args, **kwargs):
        raise RuntimeError("the REPL died")

    monkeypatch.setattr(scheduler, "prove_node", exploding)
    store.reconcile(linear_blueprint, ENVIRONMENT)

    report = await run_schedule(workspace, linear_blueprint, store, None, ROLE)

    assert "the REPL died" in report.infrastructure_error


async def test_already_solved_nodes_are_not_reproven(
    workspace, store, linear_blueprint, monkeypatch
):
    observed: list[str] = []
    monkeypatch.setattr(scheduler, "prove_node", fake_prover({}, observed))
    store.reconcile(linear_blueprint, ENVIRONMENT)
    store.mark_solved("base", "by base", attempts=1)

    await run_schedule(workspace, linear_blueprint, store, None, ROLE)

    assert observed == ["middle", "target"]


async def test_dead_helpers_are_never_scheduled(workspace, store, monkeypatch):
    blueprint = make_blueprint(
        make_node("used"),
        make_node("orphan"),
        make_node("target", ("used",), is_target=True, lean_name="my_target"),
    )
    observed: list[str] = []
    monkeypatch.setattr(scheduler, "prove_node", fake_prover({}, observed))
    store.reconcile(blueprint, ENVIRONMENT)

    await run_schedule(workspace, blueprint, store, None, ROLE)

    assert observed == ["used", "target"]
    assert is_complete(blueprint, store)


async def test_a_solved_node_is_persisted_before_the_round_finishes(
    workspace, store, wide_blueprint, monkeypatch
):
    """A run killed mid-round must not lose proofs the round already found.

    Persisting only after the round's gather meant an interrupted run discarded every
    solved node, because the gather never returned.
    """
    fast_done = asyncio.Event()
    release_slow = asyncio.Event()

    async def prove_node(workspace, blueprint, node, client, role, search_tool=None):
        if node.id == "left":
            return NodeAttemptResult(outcome=NodeOutcome.SOLVED, proof_body="by left", attempts=1)
        # 'right' blocks, standing in for a node still running when the process dies.
        fast_done.set()
        await release_slow.wait()
        return NodeAttemptResult(outcome=NodeOutcome.SOLVED, proof_body="by right", attempts=1)

    monkeypatch.setattr(scheduler, "prove_node", prove_node)
    store.reconcile(wide_blueprint, ENVIRONMENT)

    task = asyncio.create_task(
        run_schedule(workspace, wide_blueprint, store, None, ROLE, max_node_agents=2)
    )
    await fast_done.wait()
    await asyncio.sleep(0)

    # Mid-round: the fast node's proof must already be on disk.
    resumed = ProofStore.open(store.path.parent, "Mod:my_target", resume=True)
    assert resumed.records["left"].status is NodeStatus.SOLVED
    assert resumed.records["left"].proof_body == "by left"

    release_slow.set()
    await task


async def test_solving_a_node_cancels_its_queued_compiles(
    workspace, store, wide_blueprint, monkeypatch
):
    cancelled: list[str] = []

    released: list[str] = []

    class FakeService:
        def cancel_node(self, node_id):
            cancelled.append(node_id)

        def release(self, node_id):
            released.append(node_id)

    workspace.compile_service = FakeService()
    monkeypatch.setattr(scheduler, "prove_node", fake_prover({}))
    store.reconcile(wide_blueprint, ENVIRONMENT)
    try:
        await run_schedule(workspace, wide_blueprint, store, None, ROLE)
    finally:
        workspace.compile_service = None

    # Keys are target-qualified so unrelated targets cannot share a lease.
    assert set(cancelled) == {"my_target:left", "my_target:right", "my_target:target"}
    # The lease belongs to the target, so the scheduler must not release it mid-run.
    assert released == []


async def test_zero_means_unbounded_so_the_whole_frontier_runs_at_once(
    workspace, store, monkeypatch
):
    """The paper dispatches every ready lemma in parallel with no stated cap."""
    width = 9
    blueprint = make_blueprint(
        *(make_node(f"h{i}") for i in range(width)),
        make_node(
            "target", tuple(f"h{i}" for i in range(width)), is_target=True, lean_name="my_target"
        ),
    )
    running = 0
    peak = 0
    released = asyncio.Event()

    async def prove_node(workspace, blueprint, node, client, role, search_tool=None):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if peak >= width:
            released.set()
        await released.wait()
        running -= 1
        return NodeAttemptResult(outcome=NodeOutcome.SOLVED, proof_body="by x", attempts=1)

    monkeypatch.setattr(scheduler, "prove_node", prove_node)
    store.reconcile(blueprint, ENVIRONMENT)

    await run_schedule(workspace, blueprint, store, None, ROLE, max_node_agents=0)

    assert peak == width, "every ready node should run concurrently"


async def test_a_positive_limit_still_caps_a_wide_frontier(workspace, store, monkeypatch):
    blueprint = make_blueprint(
        *(make_node(f"h{i}") for i in range(9)),
        make_node(
            "target", tuple(f"h{i}" for i in range(9)), is_target=True, lean_name="my_target"
        ),
    )
    running = 0
    peak = 0

    async def prove_node(workspace, blueprint, node, client, role, search_tool=None):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        running -= 1
        return NodeAttemptResult(outcome=NodeOutcome.SOLVED, proof_body="by x", attempts=1)

    monkeypatch.setattr(scheduler, "prove_node", prove_node)
    store.reconcile(blueprint, ENVIRONMENT)

    await run_schedule(workspace, blueprint, store, None, ROLE, max_node_agents=3)

    assert peak <= 3


async def test_a_cancelled_node_stays_pending_and_propagates(
    workspace, store, wide_blueprint, monkeypatch
):
    """Cancellation is a BaseException, so the broad handler cannot absorb it."""

    async def cancelled(workspace, blueprint, node, client, role, search_tool=None):
        raise asyncio.CancelledError()

    monkeypatch.setattr(scheduler, "prove_node", cancelled)
    store.reconcile(wide_blueprint, ENVIRONMENT)

    with pytest.raises(asyncio.CancelledError):
        await run_schedule(workspace, wide_blueprint, store, None, ROLE)

    # Nothing is lost: the checkpoint still has the node pending, so resume retries it.
    resumed = ProofStore.open(store.path.parent, "Mod:my_target", resume=True)
    assert resumed.records["left"].status is NodeStatus.PENDING


async def test_speculative_dispatch_sends_a_whole_chain_at_once(workspace, store, monkeypatch):
    """A chain was proven one node per round, leaving the run at concurrency 1."""
    blueprint = make_blueprint(
        make_node("a"),
        make_node("b", ("a",)),
        make_node("c", ("b",)),
        make_node("target", ("c",), is_target=True, lean_name="my_target"),
    )
    running = 0
    peak = 0
    seen: list[str] = []

    async def prove_node(workspace, blueprint, node, client, role, search_tool=None):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        seen.append(node.id)
        await asyncio.sleep(0.01)
        running -= 1
        return NodeAttemptResult(outcome=NodeOutcome.SOLVED, proof_body=f"by {node.id}", attempts=1)

    monkeypatch.setattr(scheduler, "prove_node", prove_node)
    store.reconcile(blueprint, ENVIRONMENT)

    report = await run_schedule(
        workspace, blueprint, store, None, ROLE, max_node_agents=12, speculative=True
    )

    assert peak == 4, "the whole chain should be in flight at once"
    assert report.rounds == 1
    assert set(seen) == {"a", "b", "c", "target"}
    assert is_complete(blueprint, store)


async def test_speculative_dispatch_still_skips_solved_and_dead_nodes(
    workspace, store, monkeypatch
):
    blueprint = make_blueprint(
        make_node("used"),
        make_node("orphan"),
        make_node("target", ("used",), is_target=True, lean_name="my_target"),
    )
    seen: list[str] = []
    monkeypatch.setattr(scheduler, "prove_node", fake_prover({}, seen))
    store.reconcile(blueprint, ENVIRONMENT)
    store.mark_solved("used", "by used", attempts=1)

    await run_schedule(workspace, blueprint, store, None, ROLE, speculative=True)

    assert seen == ["target"], "solved and unreachable nodes stay out of the dispatch"


async def test_dispatchable_selects_by_mode(wide_blueprint):
    from ax_prover.blueprint.graph import required_nodes
    from ax_prover.blueprint.scheduler import dispatchable

    required = required_nodes(wide_blueprint)
    statuses = dict.fromkeys(["left", "right", "target"], NodeStatus.PENDING)

    frontier = dispatchable(wide_blueprint, statuses, required, speculative=False)
    assert set(frontier) == {"left", "right"}, "target waits for its parents"

    everything = dispatchable(wide_blueprint, statuses, required, speculative=True)
    assert set(everything) == {"left", "right", "target"}


async def test_a_leaked_compile_cancellation_does_not_abort_the_graph(
    workspace, store, linear_blueprint, monkeypatch
):
    """A stale cancellation must cost one node, not the whole run.

    Reported from a real 19-node run: a node solved in one round was invalidated by
    refinement, its qualified cancellation key was never cleared, and every compile then
    raised `CompileCancelled`. The scheduler reported that as an infrastructure error and
    stopped, discarding 18 proven nodes.
    """

    async def cancelled(*args, **kwargs):
        raise CompileCancelled("node 'my_target:base' was cancelled")

    monkeypatch.setattr(scheduler, "prove_node", cancelled)
    store.reconcile(linear_blueprint, ENVIRONMENT)

    report = await run_schedule(workspace, linear_blueprint, store, None, ROLE)

    # Recorded as an ordinary failure, so refinement gets a shot and the round terminates
    # instead of re-dispatching a pending node forever.
    assert report.infrastructure_error == ""
    assert sorted(report.failed) == ["base", "middle", "target"]
    assert all(store.statuses.get(node.id) is NodeStatus.FAILED for node in linear_blueprint.nodes)
