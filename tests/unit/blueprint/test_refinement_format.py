"""In-band refiner input format, mirroring the paper's annotated-skeleton convention."""

from ax_prover.blueprint.models import NodeDiagnosis, NodeOutcome, NodeRecord, NodeStatus
from ax_prover.blueprint.refinement import annotate_skeleton, render_review

from .conftest import make_blueprint, make_node


def records(**statuses: NodeRecord) -> dict[str, NodeRecord]:
    """Node records keyed by id, written as keyword arguments for brevity."""
    return dict(statuses)


def solved(node_id: str) -> NodeRecord:
    return NodeRecord(node_id=node_id, status=NodeStatus.SOLVED, proof_body="by simp")


def failed(node_id: str, outcome: NodeOutcome, **kwargs) -> NodeRecord:
    return NodeRecord(
        node_id=node_id,
        status=NodeStatus.FAILED,
        diagnosis=NodeDiagnosis(outcome=outcome, **kwargs),
    )


def test_review_renders_all_three_sections():
    text = render_review(
        NodeDiagnosis(
            outcome=NodeOutcome.PROOF_TOO_HARD,
            detail="short",
            analysis="tried simp, left an unsolved goal about parity",
            suggested_fix="add a lemma bridging parity to the modulus",
            last_error="unsolved goals",
        )
    )

    assert text.startswith("/- Diagnosis")
    assert text.rstrip().endswith("-/")
    assert "## Diagnosis\nPROOF_TOO_HARD" in text
    assert "## Analysis\ntried simp" in text
    assert "## Suggested Fix\nadd a lemma" in text
    assert "unsolved goals" in text


def test_review_falls_back_to_detail_when_analysis_is_absent():
    text = render_review(NodeDiagnosis(outcome=NodeOutcome.STATEMENT_WRONG, detail="false"))

    assert "## Analysis\nfalse" in text
    assert "## Suggested Fix" not in text


def test_annotated_skeleton_marks_each_node_with_its_verdict():
    blueprint = make_blueprint(
        make_node("base"),
        make_node("middle", ("base",)),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )
    annotated = annotate_skeleton(
        blueprint,
        records(
            base=solved("base"),
            middle=failed(
                "middle", NodeOutcome.PROOF_TOO_HARD, analysis="gap", suggested_fix="split it"
            ),
        ),
    )

    assert "-- PROVED\ntheorem base" in annotated
    assert "-- UNPROVED\ntheorem middle" in annotated
    assert "## Suggested Fix\nsplit it" in annotated
    # The target is immutable, so it is not offered for revision.
    assert "my_target" not in annotated
    # Parents precede dependents, so the refiner reads the graph in order.
    assert annotated.index("theorem base") < annotated.index("theorem middle")


def test_annotated_skeleton_marks_blocked_nodes_not_attempted():
    blueprint = make_blueprint(
        make_node("base"),
        make_node("middle", ("base",)),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )

    annotated = annotate_skeleton(blueprint, records(base=solved("base")))

    assert "-- NOT ATTEMPTED\ntheorem middle" in annotated


def test_annotated_skeleton_reports_budget_exhaustion_distinctly():
    blueprint = make_blueprint(
        make_node("base"),
        make_node("target", ("base",), is_target=True, lean_name="my_target"),
    )

    annotated = annotate_skeleton(
        blueprint, records(base=failed("base", NodeOutcome.BUDGET_EXHAUSTED, detail="starved"))
    )

    assert "## Diagnosis\nBUDGET_EXHAUSTED" in annotated


def test_every_helper_body_stays_a_sorry_placeholder():
    """The refiner must receive a skeleton, never proofs."""
    blueprint = make_blueprint(
        make_node("base"),
        make_node("target", ("base",), is_target=True, lean_name="my_target"),
    )

    annotated = annotate_skeleton(blueprint, records(base=solved("base")))

    assert ":= by sorry" in annotated
    assert "by simp" not in annotated
