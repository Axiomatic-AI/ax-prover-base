"""Serialized compile queue: ordering, fairness, cancellation, axioms, observability."""

import asyncio

import pytest
from lean_interact.interface import CommandResponse, Message, Pos

from ax_prover.blueprint.lean_service import (
    CompileCancelled,
    CompilePriority,
    LeanCompileService,
    format_diagnostics,
    parse_axioms,
    split_messages,
)


def message(severity: str, data: str) -> Message:
    return Message(start_pos=Pos(line=1, column=0), end_pos=None, severity=severity, data=data)


def response(*messages: Message, declarations=None, sorries=None) -> CommandResponse:
    return CommandResponse(
        messages=list(messages),
        sorries=list(sorries or []),
        env=0,
        tactics=[],
        declarations=list(declarations or []),
    )


class FakeServer:
    """Records the commands it receives and replays scripted responses."""

    def __init__(self, *responses, delay: float = 0.0):
        self.responses = list(responses)
        self.delay = delay
        self.commands: list[str] = []
        self.session_cached: list[str] = []
        self.concurrent = 0
        self.peak_concurrent = 0

    async def run(self, command, add_to_session_cache=False, timeout=None):
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            self.commands.append(command.cmd)
            if add_to_session_cache:
                self.session_cached.append(command.cmd)
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        finally:
            self.concurrent -= 1


@pytest.fixture
def ok_response():
    return response(message("info", "'foo' depends on axioms: [propext]"))


async def service_for(server, **kwargs):
    service = LeanCompileService(server, **kwargs)
    await service.start()
    return service


def test_parse_axioms_reads_the_print_axioms_report():
    assert parse_axioms(
        response(message("info", "'foo' depends on axioms: [propext, Quot.sound]"))
    ) == ("propext", "Quot.sound")


def test_parse_axioms_returns_empty_without_a_report():
    assert parse_axioms(response(message("info", "unrelated"))) == ()


def test_split_messages_separates_severities():
    errors, warnings = split_messages(
        response(message("error", "boom"), message("warning", "meh"), message("info", "fyi"))
    )

    assert errors == ("line 1: boom",)
    assert warnings == ("line 1: meh",)


def test_split_messages_quotes_the_offending_source_line():
    source = "import Mathlib\n\ntheorem foo : (1 : Nat) = 1) := rfl\n"
    bad = Message(
        start_pos=Pos(line=3, column=27), end_pos=None, severity="error", data="unexpected token"
    )

    errors, _ = split_messages(response(bad), source)

    assert errors == ("line 3: unexpected token\n  | theorem foo : (1 : Nat) = 1) := rfl",)


def test_split_messages_survives_a_position_past_the_source():
    bad = Message(start_pos=Pos(line=99, column=0), end_pos=None, severity="error", data="boom")

    errors, _ = split_messages(response(bad), "one line only")

    assert errors == ("line 99: boom",)


def test_diagnostics_explain_a_sorry_tainted_proof():
    text = format_diagnostics((), (), ("sorryAx",), "Ns.helper")

    assert "sorryAx" in text
    assert "Ns.helper" in text
    assert "does not actually prove" in text


def test_diagnostics_name_other_disallowed_axioms():
    text = format_diagnostics((), (), ("myCheat",), "Ns.helper")

    assert "disallowed axioms" in text
    assert "myCheat" in text


def test_diagnostics_keep_compiler_errors_verbatim():
    assert format_diagnostics(("unknown identifier `foo`",), (), False, None) == (
        "unknown identifier `foo`"
    )


async def test_a_clean_compile_succeeds(ok_response):
    service = await service_for(FakeServer(ok_response))

    outcome = await service.compile(
        "theorem a : True := trivial",
        check_axioms_of="a",
        allowed_axioms=frozenset({"propext"}),
    )

    assert outcome.success
    assert outcome.axioms == ("propext",)
    assert not outcome.depends_on_sorry
    await service.aclose()


async def test_an_error_fails_the_compile():
    service = await service_for(FakeServer(response(message("error", "unknown identifier"))))

    outcome = await service.compile("bad")

    assert not outcome.success
    assert outcome.errors == ("line 1: unknown identifier\n  | bad",)
    await service.aclose()


async def test_a_sorry_tainted_proof_is_rejected_despite_no_errors():
    """The axiom gate catches a `sorry` reached by any route, not just a textual one."""
    server = FakeServer(response(message("info", "'foo' depends on axioms: [sorryAx]")))
    service = await service_for(server)

    outcome = await service.compile(
        "theorem foo : True := by tricky", check_axioms_of="foo", allowed_axioms=frozenset()
    )

    assert not outcome.success
    assert outcome.depends_on_sorry
    assert "sorryAx" in outcome.output
    assert "#print axioms foo" in server.commands[0]
    await service.aclose()


async def test_warnings_alone_still_compile():
    service = await service_for(FakeServer(response(message("warning", "unused variable"))))

    outcome = await service.compile("theorem a : True := trivial")

    assert outcome.success
    assert outcome.warnings == ("line 1: unused variable\n  | theorem a : True := trivial",)
    await service.aclose()


async def test_a_lean_server_error_is_reported_not_raised(ok_response):
    class Broken:
        async def run(self, command, add_to_session_cache=False, timeout=None):
            return object()

    service = await service_for(Broken())

    outcome = await service.compile("x")

    assert not outcome.success
    assert "Lean server error" in outcome.output
    await service.aclose()


async def test_only_the_stable_prefix_enters_the_session_cache(ok_response):
    """Candidates must stay out of the replay cache or recovery replays obsolete work."""
    server = FakeServer(ok_response)
    service = await service_for(server)

    await service.warm("import Mathlib\n")
    await service.compile("import Mathlib\n\ntheorem a : True := trivial")

    assert server.session_cached == ["import Mathlib\n"]
    await service.aclose()


async def test_warming_the_same_prefix_twice_is_a_no_op(ok_response):
    server = FakeServer(ok_response)
    service = await service_for(server)

    await service.warm("import Mathlib\n")
    await service.warm("import Mathlib\n")

    assert service.stats.warmups == 1
    await service.aclose()


async def test_invalidation_forces_a_rewarm(ok_response):
    server = FakeServer(ok_response)
    service = await service_for(server)

    await service.warm("import Mathlib\n")
    service.invalidate("imports changed")
    await service.warm("import Mathlib\n")

    assert service.stats.warmups == 2
    await service.aclose()


async def test_compiles_are_serialized_on_a_single_server(ok_response):
    server = FakeServer(ok_response, delay=0.01)
    service = await service_for(server, max_lean_compiles=1)

    await asyncio.gather(*(service.compile(f"c{i}", node_id=f"n{i}") for i in range(6)))

    assert server.peak_concurrent == 1
    await service.aclose()


async def test_several_servers_compile_in_parallel(ok_response):
    """Parallelism comes from more servers, not more workers on one server."""
    servers = [FakeServer(ok_response, delay=0.02) for _ in range(3)]
    service = await service_for(servers)

    # Distinct groups, so leases spread across the pool.
    await asyncio.gather(
        *(service.compile(f"c{i}", node_id=f"n{i}", group=f"g{i}") for i in range(6))
    )

    assert service.max_lean_compiles == 3
    assert sum(1 for s in servers if s.commands) > 1, "work should span several servers"
    for s in servers:
        assert s.peak_concurrent <= 1, "each server still runs one compile at a time"
    await service.aclose()


async def test_a_targets_whole_graph_stays_on_one_server(ok_response):
    """Sticky leasing keeps the target's warm prefix and node branches reusable."""
    servers = [FakeServer(ok_response) for _ in range(3)]
    service = await service_for(servers)

    for i in range(4):
        await service.compile("body", node_id=f"Mod:t:n{i}", group="Mod:t")

    busy = [i for i, s in enumerate(servers) if s.commands]
    assert len(busy) == 1
    assert len(servers[busy[0]].commands) == 4
    await service.aclose()


async def test_groups_spread_round_robin_across_servers(ok_response):
    """Least-in-flight sent everything to server 0: warm compiles finish too fast."""
    servers = [FakeServer(ok_response) for _ in range(4)]
    service = await service_for(servers)

    # Sequential, so every lease is taken while nothing is in flight.
    for t in range(4):
        await service.compile("body", node_id=f"t{t}:n", group=f"target_{t}")

    assert all(s.commands for s in servers), "each server should get one target"
    assert len(set(service._leases.values())) == 4
    await service.aclose()


async def test_two_targets_never_share_a_prefix_on_one_server(ok_response):
    """Sharing a server makes it thrash between the two targets' prefixes."""
    servers = [FakeServer(ok_response) for _ in range(4)]
    service = await service_for(servers)

    a = service.lease("Mod:target_a")
    b = service.lease("Mod:target_b")

    assert a != b
    await service.aclose()


async def test_releasing_a_lease_frees_the_group(ok_response):
    servers = [FakeServer(ok_response) for _ in range(2)]
    service = await service_for(servers)

    first = service.lease("Mod:t")
    assert service.lease("Mod:t") == first

    service.release("Mod:t")
    assert "Mod:t" not in service._leases
    await service.aclose()


async def test_extra_workers_on_one_server_are_not_created(ok_response):
    """One server serializes internally, so the pool size caps the worker count."""
    service = await service_for(FakeServer(ok_response), max_lean_compiles=8)

    assert service.max_lean_compiles == 1
    await service.aclose()


async def test_warm_all_warms_every_server(ok_response):
    servers = [FakeServer(ok_response) for _ in range(3)]
    service = await service_for(servers)

    await service.warm_all("import Mathlib\n")

    assert all(s.session_cached == ["import Mathlib\n"] for s in servers)
    assert service.stats.warmups == 3
    await service.aclose()


async def test_invalidation_clears_leases_and_warm_state(ok_response):
    servers = [FakeServer(ok_response) for _ in range(2)]
    service = await service_for(servers)
    await service.warm_all("import Mathlib\n")
    service.lease("n1")

    service.invalidate("imports changed")
    await service.warm_all("import Mathlib\n")

    assert service.stats.warmups == 4
    await service.aclose()


async def test_structural_work_outranks_node_candidates(ok_response):
    server = FakeServer(ok_response, delay=0.01)
    service = LeanCompileService(server, max_lean_compiles=1)

    # Queue everything before starting the worker, so ordering is decided by priority.
    pending = [
        asyncio.create_task(service.compile(f"node{i}", priority=CompilePriority.NODE))
        for i in range(3)
    ]
    pending.append(
        asyncio.create_task(service.compile("skeleton", priority=CompilePriority.STRUCTURAL))
    )
    await asyncio.sleep(0)
    await asyncio.gather(*pending)

    assert "skeleton" in server.commands[:2]
    await service.aclose()


async def test_a_cancelled_node_is_not_compiled(ok_response):
    server = FakeServer(ok_response)
    service = await service_for(server)
    service.cancel_node("solved_node")

    with pytest.raises(CompileCancelled):
        await service.compile("body", node_id="solved_node")

    assert server.commands == []
    await service.aclose()


async def test_a_resumed_node_can_queue_again(ok_response):
    service = await service_for(FakeServer(ok_response))
    service.cancel_node("n1")
    service.resume_node("n1")

    outcome = await service.compile("body", node_id="n1")

    assert outcome.success
    await service.aclose()


async def test_stats_track_volume_and_per_node_usage(ok_response):
    service = await service_for(FakeServer(ok_response))

    await service.warm("import Mathlib\n")
    await service.compile("a", node_id="n1")
    await service.compile("b", node_id="n1")
    await service.compile("c", node_id="n2")

    stats = service.stats.as_dict()
    assert stats["submitted"] == 3
    assert stats["completed"] == 3
    assert stats["per_node"] == {"n1": 2, "n2": 1}
    assert stats["warmups"] == 1
    await service.aclose()


async def test_full_source_is_submitted_every_time(ok_response):
    """LeanInteract finds the reuse point itself; splitting the source defeats it."""
    server = FakeServer(ok_response)
    service = await service_for(server)
    source = "import Mathlib\n\ntheorem a : True := trivial"

    await service.compile(source)

    assert server.commands[0].startswith("import Mathlib")


async def test_a_placeholder_parent_axiom_is_allowed():
    """Using a proven parent, standing in as a named axiom, must not fail the gate."""
    server = FakeServer(
        response(message("info", "'child' depends on axioms: [propext, Ns.parent]"))
    )
    service = await service_for(server)

    outcome = await service.compile(
        "theorem child : True := Ns.parent",
        check_axioms_of="child",
        allowed_axioms=frozenset({"propext", "Ns.parent"}),
    )

    assert outcome.success
    assert outcome.disallowed_axioms == ()
    await service.aclose()


async def test_an_unexpected_axiom_fails_the_gate():
    server = FakeServer(response(message("info", "'child' depends on axioms: [propext, myCheat]")))
    service = await service_for(server)

    outcome = await service.compile(
        "theorem child : True := myCheat",
        check_axioms_of="child",
        allowed_axioms=frozenset({"propext"}),
    )

    assert not outcome.success
    assert outcome.disallowed_axioms == ("myCheat",)
    await service.aclose()


async def test_axioms_are_not_gated_without_a_declaration_to_check():
    """Skeleton compiles are full of sorries by design, so they are not gated."""
    server = FakeServer(response(message("info", "'x' depends on axioms: [sorryAx]")))
    service = await service_for(server)

    outcome = await service.compile("skeleton source")

    assert outcome.success
    await service.aclose()


async def test_qualified_keys_keep_targets_on_separate_servers(ok_response):
    """Every graph has a node called `target`; unqualified keys would collide."""
    servers = [FakeServer(ok_response) for _ in range(4)]
    service = await service_for(servers)

    a = service.lease("putnam_2005_a5:target")
    b = service.lease("putnam_1986_a6:target")

    assert a != b
    assert service.lease("putnam_2005_a5:target") == a
    assert service.lease("putnam_1986_a6:target") == b
    await service.aclose()


async def test_two_groups_do_not_evict_each_others_warm_prefix(ok_response):
    """Warming the whole pool per target made concurrent targets thrash each other."""
    servers = [FakeServer(ok_response) for _ in range(2)]
    service = await service_for(servers)

    a, b = "Mod:a", "Mod:b"
    for _ in range(3):
        for group, prefix in ((a, "import A\n"), (b, "import B\n")):
            await service.warm(prefix, index=service.lease(group))
            await service.compile("body", node_id=f"{group}:n", group=group)

    # One warm-up per group, not one per alternation.
    assert service.stats.warmups == 2
    assert len(set(service._leases.values())) == 2
    await service.aclose()


async def test_clear_cancellations_frees_every_node(ok_response):
    """A refinement round clears cancellations without rebuilding qualified node keys."""
    service = await service_for(FakeServer(ok_response))
    service.cancel_node("target_a:n1")
    service.cancel_node("target_a:n2")

    service.clear_cancellations()

    assert (await service.compile("body", node_id="target_a:n1")).success
    assert (await service.compile("body", node_id="target_a:n2")).success
    await service.aclose()


async def test_resume_with_an_unqualified_key_does_not_free_a_qualified_one(ok_response):
    """The bug that aborted a 19-node run: cancel qualified, resume bare, key leaks.

    `scheduler` cancels with `workspace.node_key`, so resuming with the bare id silently
    discards a key that was never added and the node can never compile again.
    """
    service = await service_for(FakeServer(ok_response))
    service.cancel_node("target_a:n1")

    service.resume_node("n1")

    with pytest.raises(CompileCancelled):
        await service.compile("body", node_id="target_a:n1")
    await service.aclose()
