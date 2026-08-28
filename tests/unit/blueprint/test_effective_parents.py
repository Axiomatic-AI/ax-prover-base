"""Statement dependencies from typeDeps join the declared proof parents."""

from ax_prover.blueprint.extraction import resolve_effective_parents
from ax_prover.blueprint.graph import ready_frontier, topological_order, validate_blueprint
from ax_prover.blueprint.models import BlueprintNode, NodeStatus

from .conftest import NAMESPACE, make_blueprint, make_node


def node(node_id: str, declared=(), type_deps=(), is_target=False) -> BlueprintNode:
    lean_name = node_id if is_target else f"{NAMESPACE}.{node_id}"
    statement = f"theorem {lean_name.rsplit('.', 1)[-1]} : True "
    return BlueprintNode(
        id=node_id,
        declared_parents=declared,
        parents=declared,
        lean_name=lean_name,
        kind="theorem",
        signature=": True",
        statement_source=statement,
        statement_source_no_doc=statement,
        type_deps=type_deps,
        is_target=is_target,
    )


def test_an_undeclared_statement_dependency_becomes_an_effective_parent():
    """The skeleton compiles with both present, but the isolated node module would not."""
    a = node("helper_a")
    b = node("helper_b", declared=(), type_deps=(f"{NAMESPACE}.helper_a",))

    resolved = {n.id: n for n in resolve_effective_parents([a, b])}

    assert resolved["helper_b"].declared_parents == ()
    assert resolved["helper_b"].statement_parents == ("helper_a",)
    assert resolved["helper_b"].parents == ("helper_a",)
    assert resolved["helper_b"].undeclared_statement_parents == ("helper_a",)


def test_declared_and_statement_parents_are_unioned_without_duplication():
    a, c = node("helper_a"), node("helper_c")
    b = node("helper_b", declared=("helper_a", "helper_c"), type_deps=(f"{NAMESPACE}.helper_a",))

    resolved = {n.id: n for n in resolve_effective_parents([a, c, b])}

    assert resolved["helper_b"].parents == ("helper_a", "helper_c")
    assert resolved["helper_b"].undeclared_statement_parents == ()


def test_non_generated_type_deps_are_ignored():
    """Mathlib and trusted-context constants are ambient, not graph nodes."""
    a = node("helper_a", type_deps=("Nat", "HAdd.hAdd", "Mathlib.Foo.bar"))

    resolved = resolve_effective_parents([a])

    assert resolved[0].statement_parents == ()
    assert resolved[0].parents == ()


def test_a_node_is_never_its_own_statement_parent():
    a = node("helper_a", type_deps=(f"{NAMESPACE}.helper_a",))

    assert resolve_effective_parents([a])[0].parents == ()


def test_value_deps_are_not_used_as_parents():
    """Bodies are `by sorry`, so intended proof deps are unavailable by construction."""
    a = node("helper_a")
    b = node("helper_b")
    b = b.model_copy(update={"value_deps": (f"{NAMESPACE}.helper_a",)})

    resolved = {n.id: n for n in resolve_effective_parents([a, b])}

    assert resolved["helper_b"].parents == ()


def test_scheduling_respects_an_undeclared_statement_dependency():
    """Without this, helper_b would be dispatched before helper_a and fail to elaborate."""
    blueprint = make_blueprint(
        *resolve_effective_parents(
            [
                node("helper_a"),
                node("helper_b", type_deps=(f"{NAMESPACE}.helper_a",)),
                node("target", declared=("helper_b",), is_target=True),
            ]
        )
    )

    statuses = dict.fromkeys(["helper_a", "helper_b", "target"], NodeStatus.PENDING)
    assert ready_frontier(blueprint, statuses) == ("helper_a",)

    statuses["helper_a"] = NodeStatus.SOLVED
    assert ready_frontier(blueprint, statuses) == ("helper_b",)

    assert [n.id for n in topological_order(blueprint)] == ["helper_a", "helper_b", "target"]


def test_isolation_includes_an_undeclared_statement_parent(workspace):
    """The scratch module must carry it, or the statement itself cannot elaborate."""
    resolved = resolve_effective_parents(
        [node("helper_a"), node("helper_b", type_deps=(f"{NAMESPACE}.helper_a",))]
    )
    by_id = {n.id: n for n in resolved}
    child = by_id["helper_b"]
    parents = tuple(by_id[p] for p in child.parents)

    module = workspace.render_node_module(child, parents, "by trivial")

    assert "axiom helper_a" in module


def test_validation_only_faults_declared_parents():
    """Statement parents are resolved from elaborated types and cannot be unknown."""
    nodes = resolve_effective_parents(
        [
            node("helper_a"),
            node("helper_b", type_deps=(f"{NAMESPACE}.helper_a",)),
            make_node("target", ("helper_b",), is_target=True, lean_name="my_target"),
        ]
    )

    blueprint = validate_blueprint(
        nodes,
        namespace=NAMESPACE,
        target_lean_name="my_target",
        target_signature="(n : Nat) : n = n",
    )

    assert blueprint.by_id["helper_b"].parents == ("helper_a",)


def test_the_refiner_is_told_about_an_undeclared_statement_dependency():
    from ax_prover.blueprint.refinement import annotate_skeleton

    blueprint = make_blueprint(
        *resolve_effective_parents(
            [
                node("helper_a"),
                node("helper_b", type_deps=(f"{NAMESPACE}.helper_a",)),
                make_node("target", ("helper_b",), is_target=True, lean_name="my_target"),
            ]
        )
    )

    annotated = annotate_skeleton(blueprint, {})

    assert "depends on helper_a, which `parents` did not declare" in annotated
