"""Fingerprinting, checkpoint persistence, and proof reconciliation."""

import json

from ax_prover.blueprint.models import NodeDiagnosis, NodeOutcome, NodeStatus
from ax_prover.blueprint.proof_store import (
    ProofStore,
    checkpoint_path,
    node_fingerprint,
)

from .conftest import make_blueprint, make_node

ENVIRONMENT = "env-fingerprint"


def solved_store(tmp_path, blueprint, node_ids, environment=ENVIRONMENT):
    """A store with `node_ids` marked solved and reconciled against `blueprint`."""
    store = ProofStore.open(tmp_path, "Mod:my_target")
    store.reconcile(blueprint, environment)
    for node_id in node_ids:
        store.mark_solved(node_id, f"by {node_id}", attempts=1)
    return store


def test_fingerprint_covers_the_nodes_own_signature(linear_blueprint):
    node = linear_blueprint.by_id["middle"]
    changed = make_blueprint(
        make_node("base"),
        make_node("middle", ("base",), signature="(n : Nat) : n = n + 0"),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )

    original = node_fingerprint(linear_blueprint, node, ENVIRONMENT)
    updated = node_fingerprint(changed, changed.by_id["middle"], ENVIRONMENT)

    assert original != updated


def test_fingerprint_covers_parent_signatures(linear_blueprint):
    changed = make_blueprint(
        make_node("base", signature="(n : Nat) : n = n + 0"),
        make_node("middle", ("base",)),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )

    original = node_fingerprint(linear_blueprint, linear_blueprint.by_id["middle"], ENVIRONMENT)
    updated = node_fingerprint(changed, changed.by_id["middle"], ENVIRONMENT)

    assert original != updated


def test_fingerprint_covers_the_environment(linear_blueprint):
    node = linear_blueprint.by_id["base"]

    assert node_fingerprint(linear_blueprint, node, "a") != node_fingerprint(
        linear_blueprint, node, "b"
    )


def test_fingerprint_ignores_docstring_prose(linear_blueprint):
    documented = make_blueprint(
        make_node("base", doc_text="## Proof\n\nA completely different plan."),
        make_node("middle", ("base",)),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )

    assert node_fingerprint(
        linear_blueprint, linear_blueprint.by_id["base"], ENVIRONMENT
    ) == node_fingerprint(documented, documented.by_id["base"], ENVIRONMENT)


def test_parent_order_does_not_change_the_fingerprint():
    forward = make_blueprint(
        make_node("a"),
        make_node("b"),
        make_node("target", ("a", "b"), is_target=True, lean_name="my_target"),
    )
    reversed_ = make_blueprint(
        make_node("a"),
        make_node("b"),
        make_node("target", ("b", "a"), is_target=True, lean_name="my_target"),
    )

    assert node_fingerprint(forward, forward.target, ENVIRONMENT) == node_fingerprint(
        reversed_, reversed_.target, ENVIRONMENT
    )


def test_docstring_only_refinement_preserves_every_proof(tmp_path, linear_blueprint):
    store = solved_store(tmp_path, linear_blueprint, ["base", "middle", "target"])

    documented = make_blueprint(
        make_node("base", doc_text="new plan"),
        make_node("middle", ("base",), doc_text="also new"),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )
    reused = store.reconcile(documented, ENVIRONMENT)

    assert reused == 3
    assert store.solved_proofs() == {
        "base": "by base",
        "middle": "by middle",
        "target": "by target",
    }


def test_parent_signature_change_invalidates_the_descendant_subtree(tmp_path, linear_blueprint):
    store = solved_store(tmp_path, linear_blueprint, ["base", "middle", "target"])

    changed = make_blueprint(
        make_node("base", signature="(n : Nat) : 0 + n = n"),
        make_node("middle", ("base",)),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )
    reused = store.reconcile(changed, ENVIRONMENT)

    assert reused == 0
    assert store.records["base"].status is NodeStatus.PENDING
    assert store.records["middle"].status is NodeStatus.PENDING
    assert store.records["target"].status is NodeStatus.PENDING


def test_an_unrelated_branch_keeps_its_proof(tmp_path):
    blueprint = make_blueprint(
        make_node("left"),
        make_node("right"),
        make_node("target", ("left", "right"), is_target=True, lean_name="my_target"),
    )
    store = solved_store(tmp_path, blueprint, ["left", "right"])

    changed = make_blueprint(
        make_node("left", signature="(n : Nat) : 0 + n = n"),
        make_node("right"),
        make_node("target", ("left", "right"), is_target=True, lean_name="my_target"),
    )
    store.reconcile(changed, ENVIRONMENT)

    assert store.records["right"].status is NodeStatus.SOLVED
    assert store.records["left"].status is NodeStatus.PENDING


def test_a_removed_helper_invalidates_its_child(tmp_path, linear_blueprint):
    store = solved_store(tmp_path, linear_blueprint, ["base", "middle", "target"])

    without_base = make_blueprint(
        make_node("middle"),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )
    store.reconcile(without_base, ENVIRONMENT)

    assert "base" not in store.records
    assert store.records["middle"].status is NodeStatus.PENDING


def test_an_unchanged_failed_node_keeps_its_diagnosis(tmp_path, linear_blueprint):
    store = ProofStore.open(tmp_path, "Mod:my_target")
    store.reconcile(linear_blueprint, ENVIRONMENT)
    store.mark_failed(
        "base",
        NodeDiagnosis(outcome=NodeOutcome.PROOF_TOO_HARD, detail="too hard", last_error="err"),
        attempts=4,
    )

    store.reconcile(linear_blueprint, ENVIRONMENT)

    record = store.records["base"]
    assert record.status is NodeStatus.FAILED
    assert record.diagnosis.detail == "too hard"
    assert record.attempts == 4


def test_a_restated_failed_node_is_reopened(tmp_path, linear_blueprint):
    store = ProofStore.open(tmp_path, "Mod:my_target")
    store.reconcile(linear_blueprint, ENVIRONMENT)
    store.mark_failed(
        "base", NodeDiagnosis(outcome=NodeOutcome.STATEMENT_WRONG, detail="wrong"), attempts=4
    )

    restated = make_blueprint(
        make_node("base", signature="(n : Nat) : 0 + n = n"),
        make_node("middle", ("base",)),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )
    store.reconcile(restated, ENVIRONMENT)

    assert store.records["base"].status is NodeStatus.PENDING
    assert store.records["base"].diagnosis is None


def test_checkpoint_round_trips_through_disk(tmp_path, linear_blueprint):
    store = solved_store(tmp_path, linear_blueprint, ["base"])
    store.remember_skeleton("theorem base ...", ("base",), "the plan")

    resumed = ProofStore.open(tmp_path, "Mod:my_target", resume=True)

    assert resumed.records["base"].proof_body == "by base"
    assert resumed.state.helpers == "theorem base ..."
    assert resumed.state.target_parents == ["base"]
    assert resumed.state.target_proof_plan == "the plan"


def test_opening_without_resume_ignores_the_checkpoint(tmp_path, linear_blueprint):
    solved_store(tmp_path, linear_blueprint, ["base"])

    fresh = ProofStore.open(tmp_path, "Mod:my_target")

    assert fresh.records == {}


def test_resume_reuses_solved_proofs_after_reconcile(tmp_path, linear_blueprint):
    solved_store(tmp_path, linear_blueprint, ["base", "middle"])

    resumed = ProofStore.open(tmp_path, "Mod:my_target", resume=True)
    reused = resumed.reconcile(linear_blueprint, ENVIRONMENT)

    assert reused == 2
    assert resumed.statuses["target"] is NodeStatus.PENDING


def test_clear_removes_the_checkpoint_file(tmp_path, linear_blueprint):
    store = solved_store(tmp_path, linear_blueprint, ["base"])
    path = store.path
    assert path.exists()

    store.clear()

    assert not path.exists()
    assert store.records == {}


def test_an_unreadable_checkpoint_is_ignored(tmp_path):
    path = checkpoint_path(tmp_path, "Mod:my_target")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    store = ProofStore.open(tmp_path, "Mod:my_target", resume=True)

    assert store.records == {}


def test_a_stale_store_version_is_ignored(tmp_path):
    path = checkpoint_path(tmp_path, "Mod:my_target")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 99, "records": {}}), encoding="utf-8")

    store = ProofStore.open(tmp_path, "Mod:my_target", resume=True)

    assert store.records == {}


def test_saving_leaves_no_temporary_file(tmp_path, linear_blueprint):
    store = solved_store(tmp_path, linear_blueprint, ["base"])

    assert not list(store.path.parent.glob("*.tmp"))


def test_checkpoint_path_sanitizes_the_target_name(tmp_path):
    path = checkpoint_path(tmp_path, "My/Module.Path:thm")

    assert path.parent == tmp_path
    assert "/" not in path.name
