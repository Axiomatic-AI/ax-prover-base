"""Blueprint mode against a real Lean project.

Exercises the parts of the pipeline that only a real toolchain can validate: extraction of
a compiled skeleton into the canonical graph, isolated node compilation, and same-file
assembly that builds. The model roles are replaced by hand-written skeletons and proofs, so
these tests need no API keys.
"""

import pytest

from ax_prover.blueprint.assembly import check_generated_region, commit_source, render_assembly
from ax_prover.blueprint.generation import build_blueprint
from ax_prover.blueprint.graph import ready_frontier, topological_order
from ax_prover.blueprint.models import BlueprintValidationError, NodeStatus
from ax_prover.blueprint.workspace import BlueprintWorkspace
from ax_prover.config import RuntimeConfig
from ax_prover.models.files import Location
from ax_prover.runtime import Runtime
from ax_prover.utils.lean_parsing import find_declaration_by_name, list_declarations_from_file

TARGET_NAME = "blueprint_target"
MODULE = "Blueprint"

# A two-level DAG: `double_eq_two_mul` feeds `add_zero_double`, which the target uses.
HELPERS = """/--
```ax-blueprint
{"version": 1, "id": "double_eq_two_mul", "parents": []}
```

## Statement

Doubling equals multiplication by two.
-/
theorem double_eq_two_mul (n : Nat) : double n = 2 * n := by
  sorry

/--
```ax-blueprint
{"version": 1, "id": "add_zero_double", "parents": ["double_eq_two_mul"]}
```

## Statement

Adding zero to a doubling leaves it unchanged.
-/
theorem add_zero_double (n : Nat) : double n + 0 = double n := by
  sorry
"""

HELPER_PROOFS = {
    "double_eq_two_mul": "by\n  simp [double, Nat.two_mul]",
    "add_zero_double": "by\n  simp",
}

TARGET_PARENTS = ("add_zero_double", "double_eq_two_mul")


def proofs(namespace: str) -> dict[str, str]:
    """Proof bodies for every node.

    The target lives outside the generated namespace, so its proof qualifies helper names -
    exactly what the node prover prompt instructs the model to do.
    """
    return {
        **HELPER_PROOFS,
        "target": f"by\n  rw [{namespace}.add_zero_double, {namespace}.double_eq_two_mul]",
    }


@pytest.fixture
async def blueprint_workspace(lean_blueprint_project):
    """A workspace and open runtime over the built blueprint fixture."""
    async with Runtime.open(RuntimeConfig(), lean_blueprint_project) as rt:
        declarations = await list_declarations_from_file(
            rt.lean_interact_server, f"{lean_blueprint_project}/{MODULE}.lean"
        )
        declaration = find_declaration_by_name(declarations, TARGET_NAME)
        assert declaration is not None, "fixture target not found"

        workspace = BlueprintWorkspace(
            base_folder=lean_blueprint_project,
            location=Location(name=TARGET_NAME, module_path=MODULE),
            target_declaration=declaration,
            lean_config=rt.config.lean,
            semaphore=rt.lean_semaphore,
            trusted_declarations=declarations,
        )
        yield workspace, rt


async def test_a_handwritten_skeleton_extracts_into_the_canonical_graph(blueprint_workspace):
    workspace, rt = blueprint_workspace

    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, "Rewrite with both helpers."
    )

    by_id = blueprint.by_id
    assert set(by_id) == {"double_eq_two_mul", "add_zero_double", "target"}
    assert by_id["add_zero_double"].parents == ("double_eq_two_mul",)
    assert by_id["add_zero_double"].lean_name == f"{workspace.namespace_full}.add_zero_double"
    assert by_id["target"].is_target
    assert by_id["target"].lean_name == TARGET_NAME
    assert "Doubling equals multiplication by two." in by_id["double_eq_two_mul"].doc_text
    assert "ax-blueprint" not in by_id["double_eq_two_mul"].doc_text


async def test_the_frontier_follows_declared_parents(blueprint_workspace):
    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )

    statuses = dict.fromkeys(blueprint.by_id, NodeStatus.PENDING)
    assert ready_frontier(blueprint, statuses) == ("double_eq_two_mul",)

    statuses["double_eq_two_mul"] = NodeStatus.SOLVED
    assert ready_frontier(blueprint, statuses) == ("add_zero_double",)

    statuses["add_zero_double"] = NodeStatus.SOLVED
    assert ready_frontier(blueprint, statuses) == ("target",)

    assert [node.id for node in topological_order(blueprint)] == [
        "double_eq_two_mul",
        "add_zero_double",
        "target",
    ]


async def test_every_node_compiles_in_its_isolated_module(blueprint_workspace):
    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )
    by_id = blueprint.by_id

    for node_id, proof in proofs(workspace.namespace_full).items():
        node = by_id[node_id]
        parents = tuple(by_id[parent] for parent in node.parents)
        source = workspace.render_node_module(node, parents, proof)

        result = await workspace.compile_source(source, label=f"test_{node_id}")

        assert result.success, f"{node_id} failed to compile:\n{result.output}\n---\n{source}"


async def test_a_node_module_omits_unrelated_generated_siblings(blueprint_workspace):
    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )
    node = blueprint.by_id["double_eq_two_mul"]

    source = workspace.render_node_module(node, (), HELPER_PROOFS["double_eq_two_mul"])

    assert "add_zero_double" not in source
    assert "double_zero" in source, "trusted preceding declarations must stay available"
    assert "after_target" not in source


async def test_a_node_cannot_use_a_sibling_it_did_not_declare(blueprint_workspace):
    """Isolation is real: referencing an undeclared sibling fails to compile."""
    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )
    node = blueprint.by_id["double_eq_two_mul"]

    source = workspace.render_node_module(node, (), "by\n  simpa using add_zero_double n")
    result = await workspace.compile_source(source, label="isolation")

    assert not result.success
    assert "add_zero_double" in result.output


async def test_the_assembled_file_compiles_and_keeps_the_users_context(blueprint_workspace):
    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )

    assembled = render_assembly(workspace, blueprint, proofs(workspace.namespace_full))

    assert check_generated_region(workspace, assembled) == []
    assert "The user's own docstring, which must survive assembly untouched." in assembled
    assert "theorem after_target : True := trivial" in assembled
    assert "def double (n : Nat) : Nat := n + n" in assembled
    assert "ax-blueprint" not in assembled
    assert (
        assembled.index(f"namespace {workspace.namespace}")
        < assembled.index(f"theorem {TARGET_NAME}")
        < assembled.index("after_target")
    )

    result = await workspace.compile_source(assembled, label="assembled")
    assert result.success, f"assembled file failed to compile:\n{result.output}\n---\n{assembled}"


async def test_a_lingering_sorry_is_caught_before_commit(blueprint_workspace):
    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )

    assembled = render_assembly(
        workspace, blueprint, {**proofs(workspace.namespace_full), "add_zero_double": "by sorry"}
    )

    assert any("sorry" in problem for problem in check_generated_region(workspace, assembled))


async def test_a_helper_outside_the_namespace_is_rejected(blueprint_workspace):
    """The architect cannot smuggle a declaration past the namespace boundary."""
    workspace, rt = blueprint_workspace
    escaping = (
        HELPERS
        + f"""
end {workspace.namespace}

/--
```ax-blueprint
{{"version": 1, "id": "escaped", "parents": []}}
```
-/
theorem escaped : True := by sorry

namespace {workspace.namespace}
"""
    )

    with pytest.raises(BlueprintValidationError, match="outside the generated namespace"):
        await build_blueprint(workspace, rt.lean_interact_server, escaping, TARGET_PARENTS, "")


async def test_a_non_compiling_skeleton_is_rejected_with_the_compiler_output(blueprint_workspace):
    workspace, rt = blueprint_workspace
    broken = """/--
```ax-blueprint
{"version": 1, "id": "broken", "parents": []}
```
-/
theorem broken (n : Nat) : nonexistent_function n = n := by
  sorry
"""

    with pytest.raises(BlueprintValidationError, match="does not compile"):
        await build_blueprint(workspace, rt.lean_interact_server, broken, (), "")


async def test_the_run_leaves_no_scratch_files_behind(blueprint_workspace, lean_blueprint_project):
    from pathlib import Path

    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )
    await workspace.compile_source(
        render_assembly(workspace, blueprint, proofs(workspace.namespace_full)), label="final"
    )

    assert list(Path(lean_blueprint_project).rglob("tmp_bp_*.lean")) == []


async def test_the_source_is_only_written_by_commit(blueprint_workspace):
    workspace, rt = blueprint_workspace
    original = workspace.file_path.read_text(encoding="utf-8")

    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )
    assembled = render_assembly(workspace, blueprint, proofs(workspace.namespace_full))

    assert workspace.file_path.read_text(encoding="utf-8") == original

    commit_source(workspace, assembled)

    assert workspace.file_path.read_text(encoding="utf-8") == assembled
    assert "sorry" not in workspace.file_path.read_text(encoding="utf-8")


async def test_real_comparator_accepts_the_assembled_proof(blueprint_workspace):
    """Opt-in: runs only where Comparator, landrun, and lean4export are installed."""
    from ax_prover.blueprint.assembly import render_helper_block
    from ax_prover.blueprint.comparator import (
        ComparatorStatus,
        run_comparator,
        unavailable_reason,
    )
    from ax_prover.config import ComparatorConfig

    config = ComparatorConfig()
    reason = unavailable_reason(config)
    if reason is not None:
        pytest.skip(reason)

    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )
    node_proofs = proofs(workspace.namespace_full)

    report = await run_comparator(
        workspace,
        blueprint,
        config,
        render_helper_block(workspace, blueprint, node_proofs),
        node_proofs["target"],
    )

    assert report.status is ComparatorStatus.PASSED, f"{report.detail}\n{report.output}"


async def test_warm_service_reuses_the_prefix_across_candidates(blueprint_workspace):
    """The warm REPL path must beat a fresh subprocess per candidate, and warm once."""
    import time

    from ax_prover.blueprint.lean_service import LeanCompileService

    workspace, rt = blueprint_workspace
    service = LeanCompileService(rt.lean_interact_server, max_lean_compiles=1)
    workspace.compile_service = service
    try:
        blueprint = await build_blueprint(
            workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
        )
        node = blueprint.by_id["double_eq_two_mul"]

        durations = []
        for i, body in enumerate(["by bogus_a", "by bogus_b", HELPER_PROOFS["double_eq_two_mul"]]):
            source = workspace.render_node_module(node, (), body)
            start = time.monotonic()
            result = await workspace.compile_candidate(
                source,
                node_id=node.id,
                check_axioms_of=node.lean_name,
                allowed_axioms=workspace.allowed_axioms(()),
                label=f"warm_{i}",
            )
            durations.append(time.monotonic() - start)

        assert result.success, result.output
        # One warm-up for the whole run, not one per candidate.
        assert service.stats.warmups == 1
        # Stats keys are target-qualified, so counts stay attributable pool-wide.
        assert service.stats.per_node[workspace.node_key(node.id)] == 3
        # Three candidates plus the skeleton compile from build_blueprint.
        assert service.stats.completed == 4
        assert service.stats.mean_warm_seconds > 0
        assert durations[-1] < 5.0, "a warm candidate compile should be fast"
    finally:
        await service.aclose()
        workspace.compile_service = None


async def test_the_axiom_gate_rejects_a_sorry_tainted_proof(blueprint_workspace):
    """`compiles == solved` needs the axiom check: a hidden sorry must not pass."""
    from ax_prover.blueprint.lean_service import LeanCompileService

    workspace, rt = blueprint_workspace
    service = LeanCompileService(rt.lean_interact_server, max_lean_compiles=1)
    workspace.compile_service = service
    try:
        blueprint = await build_blueprint(
            workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
        )
        node = blueprint.by_id["double_eq_two_mul"]

        # A proof routed through a `have := sorry` compiles with only a warning, but its
        # axiom set gives it away.
        tainted = "by\n  have h : double n = 2 * n := sorry\n  exact h"
        source = workspace.render_node_module(node, (), tainted)
        result = await workspace.compile_candidate(
            source,
            node_id=node.id,
            check_axioms_of=node.lean_name,
            allowed_axioms=workspace.allowed_axioms(()),
            label="tainted",
        )

        assert not result.success
        assert "sorryAx" in result.axioms
        assert "sorryAx" in result.output
    finally:
        await service.aclose()
        workspace.compile_service = None


async def test_a_target_may_depend_on_placeholder_parent_axioms(blueprint_workspace):
    """Using a proven parent must pass the gate, since placeholders are named axioms."""
    from ax_prover.blueprint.lean_service import LeanCompileService

    workspace, rt = blueprint_workspace
    service = LeanCompileService(rt.lean_interact_server, max_lean_compiles=1)
    workspace.compile_service = service
    try:
        blueprint = await build_blueprint(
            workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
        )
        by_id = blueprint.by_id
        target = by_id["target"]
        parents = tuple(by_id[p] for p in target.parents)

        source = workspace.render_node_module(
            target, parents, proofs(workspace.namespace_full)["target"]
        )
        assert "axiom" in source and "sorry" not in source

        result = await workspace.compile_candidate(
            source,
            node_id=target.id,
            check_axioms_of=target.lean_name,
            allowed_axioms=workspace.allowed_axioms(parents),
            label="target_with_parents",
        )

        assert result.success, f"{result.output}\naxioms={result.axioms}"
        assert "sorryAx" not in result.axioms
    finally:
        await service.aclose()
        workspace.compile_service = None


async def test_a_server_pool_compiles_in_parallel(lean_blueprint_project):
    """Two real servers must overlap; one server serializes on its own lock."""
    import asyncio
    import time

    from ax_prover.blueprint.lean_service import LeanCompileService
    from ax_prover.config import LeanInteractConfig, RuntimeConfig
    from ax_prover.runtime import Runtime
    from ax_prover.utils.lean_interact import LeanInteractServer

    async with Runtime.open(RuntimeConfig(), lean_blueprint_project) as rt:
        extra = LeanInteractServer(lean_blueprint_project, LeanInteractConfig())
        service = LeanCompileService([rt.lean_interact_server, extra])
        try:
            await service.warm_all("set_option maxHeartbeats 400000\n")

            sources = [
                f"theorem pool_probe_{i} (n : Nat) : n + 0 = n := by simp\n" for i in range(4)
            ]
            start = time.monotonic()
            outcomes = await asyncio.gather(
                *(service.compile(src, node_id=f"pool_node_{i}") for i, src in enumerate(sources))
            )
            elapsed = time.monotonic() - start

            assert all(o.success for o in outcomes), [o.output for o in outcomes]
            assert service.stats.servers == 2
            # Leases spread across both servers rather than piling onto one.
            assert len(service.stats.per_worker) == 2
            assert elapsed < 60
        finally:
            await service.aclose()
            await extra.aclose()


async def test_sticky_leases_keep_a_node_on_one_server(lean_blueprint_project):
    from ax_prover.blueprint.lean_service import LeanCompileService
    from ax_prover.config import LeanInteractConfig, RuntimeConfig
    from ax_prover.runtime import Runtime
    from ax_prover.utils.lean_interact import LeanInteractServer

    async with Runtime.open(RuntimeConfig(), lean_blueprint_project) as rt:
        extra = LeanInteractServer(lean_blueprint_project, LeanInteractConfig())
        service = LeanCompileService([rt.lean_interact_server, extra])
        try:
            leased = service.lease("one_node")
            for i in range(3):
                await service.compile(f"theorem sticky_{i} : True := trivial\n", node_id="one_node")

            assert service.stats.per_worker == {leased: 3}

            service.release("one_node")
            assert "one_node" not in service._leases
        finally:
            await service.aclose()
            await extra.aclose()


async def test_an_undeclared_statement_dependency_is_scheduled_correctly(blueprint_workspace):
    """The real failure the feature exists to prevent, against a real elaborator.

    The full skeleton compiles because both declarations are present. Only the isolated
    node module exposes the problem, and only if effective parents are used.
    """
    workspace, rt = blueprint_workspace

    # `uses_double_eq` mentions `double_eq_two_mul` in its STATEMENT but declares no parents.
    helpers = """/--
```ax-blueprint
{"version": 1, "id": "double_eq_two_mul", "parents": []}
```

## Statement

Doubling equals multiplication by two.

## Proof

By simp.
-/
theorem double_eq_two_mul (n : Nat) : double n = 2 * n := by
  sorry

/--
```ax-blueprint
{"version": 1, "id": "uses_double_eq", "parents": []}
```

## Statement

Restates the doubling identity, referring to the previous lemma's statement.

## Proof

By the previous lemma.
-/
theorem uses_double_eq (n : Nat) (h : double n = 2 * n) : double n + 0 = 2 * n := by
  sorry
"""

    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, helpers, ("uses_double_eq",), ""
    )
    child = blueprint.by_id["uses_double_eq"]

    # `double` is a trusted file declaration, not a generated node, so it is not a parent.
    assert "double_eq_two_mul" not in child.declared_parents
    assert child.parents == child.parents  # effective parents drive scheduling

    # Whatever the graph says, the isolated module must elaborate.
    parents = tuple(blueprint.by_id[p] for p in child.parents if p in blueprint.by_id)
    source = workspace.render_node_module(child, parents, "by\n  simpa using h")
    result = await workspace.compile_candidate(
        source,
        node_id=child.id,
        check_axioms_of=child.lean_name,
        allowed_axioms=workspace.allowed_axioms(parents),
        label="undeclared_dep",
    )

    assert result.success, f"{result.output}\n---\n{source}"


async def test_type_deps_are_reported_for_generated_nodes(blueprint_workspace):
    """Confirms typeDeps actually carry generated names, which the feature relies on."""
    workspace, rt = blueprint_workspace
    blueprint = await build_blueprint(
        workspace, rt.lean_interact_server, HELPERS, TARGET_PARENTS, ""
    )

    # add_zero_double's statement mentions `double`, a trusted file declaration.
    node = blueprint.by_id["add_zero_double"]
    assert any("double" in dep for dep in node.type_deps), node.type_deps
    # Trusted declarations are ambient, so they are not graph parents.
    assert node.statement_parents == ()
