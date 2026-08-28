"""Graph validation, topology, frontier, and reachability."""

import pytest

from ax_prover.blueprint.graph import (
    find_cycle,
    normalize_signature,
    ready_frontier,
    required_nodes,
    topological_order,
    transitive_ancestors,
    transitive_descendants,
    validate_blueprint,
)
from ax_prover.blueprint.models import BlueprintValidationError, NodeStatus

from .conftest import NAMESPACE, make_blueprint, make_node

TARGET_SIGNATURE = "(n : Nat) : n + 0 = n"


def validate(nodes, target_signature: str = TARGET_SIGNATURE):
    return validate_blueprint(
        nodes,
        namespace=NAMESPACE,
        target_lean_name="my_target",
        target_signature=target_signature,
    )


def target(parents=("helper",), signature: str = TARGET_SIGNATURE):
    return make_node("target", parents, signature=signature, is_target=True, lean_name="my_target")


def test_accepts_a_valid_graph():
    blueprint = validate([make_node("helper"), target()])

    assert {node.id for node in blueprint.nodes} == {"helper", "target"}
    assert blueprint.target.lean_name == "my_target"
    assert [node.id for node in blueprint.helpers] == ["helper"]


def test_rejects_duplicate_blueprint_ids():
    with pytest.raises(BlueprintValidationError, match="duplicate blueprint id"):
        validate(
            [make_node("helper"), make_node("helper", lean_name=f"{NAMESPACE}.other"), target()]
        )


def test_rejects_duplicate_lean_names():
    with pytest.raises(BlueprintValidationError, match="duplicate Lean declaration name"):
        validate([make_node("a"), make_node("b", lean_name=f"{NAMESPACE}.a"), target()])


def test_rejects_unknown_parent():
    with pytest.raises(BlueprintValidationError, match="unknown parent 'ghost'"):
        validate([make_node("helper", ("ghost",)), target()])


def test_rejects_self_edge():
    with pytest.raises(BlueprintValidationError, match="declares itself as a parent"):
        validate([make_node("helper", ("helper",)), target()])


def test_rejects_cycles():
    with pytest.raises(BlueprintValidationError, match="cycle"):
        validate([make_node("a", ("b",)), make_node("b", ("a",)), target(parents=("a",))])


def test_rejects_non_lemma_declarations():
    with pytest.raises(BlueprintValidationError, match="only helper lemmas are permitted"):
        validate([make_node("helper", kind="definition"), target()])


def test_rejects_helpers_outside_the_generated_namespace():
    with pytest.raises(BlueprintValidationError, match="outside the generated namespace"):
        validate([make_node("helper", lean_name="Rogue.helper"), target()])


def test_rejects_a_helper_using_the_reserved_target_id():
    rogue = make_node("target", lean_name=f"{NAMESPACE}.target")

    with pytest.raises(BlueprintValidationError, match="reserved blueprint id"):
        validate([rogue, make_node("helper")])


def test_rejects_a_missing_target():
    with pytest.raises(BlueprintValidationError, match="missing target node"):
        validate([make_node("helper")])


def test_rejects_a_changed_target_signature():
    with pytest.raises(BlueprintValidationError, match="may not change"):
        validate([make_node("helper"), target(signature="(n : Nat) : n + 1 = n")])


def test_target_signature_comparison_ignores_whitespace():
    blueprint = validate(
        [make_node("helper"), target(signature="(n : Nat)  :  n + 0\n  = n")],
    )

    assert blueprint.target.is_target


def test_reports_every_problem_at_once():
    with pytest.raises(BlueprintValidationError) as excinfo:
        validate([make_node("helper", ("ghost",), kind="definition")])

    problems = excinfo.value.problems
    assert len(problems) == 3
    assert excinfo.value.report.count("- ") == 3


def test_normalize_signature_collapses_whitespace():
    assert normalize_signature("(n : Nat)\n  : n = n") == "(n : Nat) : n = n"


def test_topological_order_is_parents_first_and_deterministic(linear_blueprint):
    order = [node.id for node in topological_order(linear_blueprint)]

    assert order == ["base", "middle", "target"]


def test_topological_order_breaks_ties_by_id(wide_blueprint):
    order = [node.id for node in topological_order(wide_blueprint)]

    assert order == ["left", "right", "target"]


def test_topological_order_rejects_a_cycle():
    cyclic = make_blueprint(make_node("a", ("b",)), make_node("b", ("a",)))

    with pytest.raises(BlueprintValidationError, match="cycle"):
        topological_order(cyclic)


def test_frontier_only_exposes_nodes_whose_parents_are_solved(linear_blueprint):
    statuses = dict.fromkeys(["base", "middle", "target"], NodeStatus.PENDING)

    assert ready_frontier(linear_blueprint, statuses) == ("base",)

    statuses["base"] = NodeStatus.SOLVED
    assert ready_frontier(linear_blueprint, statuses) == ("middle",)

    statuses["middle"] = NodeStatus.SOLVED
    assert ready_frontier(linear_blueprint, statuses) == ("target",)


def test_frontier_exposes_independent_nodes_together(wide_blueprint):
    statuses = dict.fromkeys(["left", "right", "target"], NodeStatus.PENDING)

    assert set(ready_frontier(wide_blueprint, statuses)) == {"left", "right"}


def test_frontier_excludes_a_failed_parents_child(linear_blueprint):
    statuses = {
        "base": NodeStatus.FAILED,
        "middle": NodeStatus.PENDING,
        "target": NodeStatus.PENDING,
    }

    assert ready_frontier(linear_blueprint, statuses) == ()


def test_find_cycle_returns_the_cycle():
    cycle = find_cycle([make_node("a", ("b",)), make_node("b", ("a",))])

    assert cycle[0] == cycle[-1]
    assert set(cycle) == {"a", "b"}


def test_transitive_descendants_and_ancestors(linear_blueprint):
    assert transitive_descendants(linear_blueprint, "base") == {"middle", "target"}
    assert transitive_ancestors(linear_blueprint, "target") == {"middle", "base"}


def test_required_nodes_excludes_helpers_nothing_depends_on():
    blueprint = make_blueprint(
        make_node("used"),
        make_node("orphan"),
        make_node("target", ("used",), is_target=True, lean_name="my_target"),
    )

    assert required_nodes(blueprint) == {"target", "used"}
