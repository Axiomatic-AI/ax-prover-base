"""Deterministic assembly, pre-commit safety checks, and the single atomic edit."""

import pytest

from ax_prover.blueprint.assembly import (
    AssemblyError,
    check_generated_region,
    commit_source,
    render_assembly,
    render_helper,
    render_helper_block,
)

from .conftest import make_blueprint, make_node

PROOFS = {"base": "by simp", "middle": "by simpa using base", "target": "by simp"}


def test_helpers_render_parents_first_inside_the_namespace(workspace, linear_blueprint):
    block = render_helper_block(workspace, linear_blueprint, PROOFS)

    assert block.startswith(f"namespace {workspace.namespace}")
    assert block.rstrip().endswith(f"end {workspace.namespace}")
    assert block.index("theorem base") < block.index("theorem middle")


def test_helper_docstring_keeps_prose_and_drops_the_metadata_fence():
    node = make_node("helper", doc_text="## Statement\n\nA helper.")

    rendered = render_helper(node, "by simp")

    assert rendered.startswith("/--\n## Statement")
    assert "ax-blueprint" not in rendered
    assert rendered.rstrip().endswith(":= by simp")


def test_helper_without_prose_renders_no_docstring():
    assert not render_helper(make_node("helper"), "by simp").startswith("/--")


def test_assembly_preserves_the_users_target_docstring_and_statement(workspace, linear_blueprint):
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)

    assert "The user's own docstring, which must survive assembly." in assembled
    assert "theorem my_target (n : Nat) : n + 0 = n := by simp" in assembled
    assert "ax-blueprint" not in assembled


def test_assembly_places_helpers_immediately_before_the_target(workspace, linear_blueprint):
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)

    assert (
        assembled.index("preceding_lemma")
        < assembled.index(f"namespace {workspace.namespace}")
        < assembled.index("theorem my_target")
        < assembled.index("following_lemma")
    )


def test_assembly_keeps_the_trusted_prefix_and_suffix_verbatim(workspace, linear_blueprint):
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)

    assert "import Mathlib.Tactic" in assembled
    assert "set_option maxHeartbeats 400000" in assembled
    assert "theorem following_lemma : True := trivial" in assembled


def test_assembly_skips_helpers_the_target_does_not_need(workspace):
    blueprint = make_blueprint(
        make_node("used"),
        make_node("orphan"),
        make_node("target", ("used",), is_target=True, lean_name="my_target"),
    )

    assembled = render_assembly(workspace, blueprint, {"used": "by simp", "target": "by simp"})

    assert "theorem used" in assembled
    assert "theorem orphan" not in assembled


def test_assembly_requires_every_needed_helper_proof(workspace, linear_blueprint):
    with pytest.raises(AssemblyError, match="missing proofs for helpers: middle"):
        render_assembly(workspace, linear_blueprint, {"base": "by simp", "target": "by simp"})


def test_assembly_requires_the_target_proof(workspace, linear_blueprint):
    with pytest.raises(AssemblyError, match="missing proof for the target"):
        render_assembly(workspace, linear_blueprint, {"base": "by simp", "middle": "by simp"})


@pytest.mark.parametrize(
    ("body", "label"),
    [
        ("by sorry", "sorry"),
        ("by admit", "admit"),
        ("by native_decide", "native_decide"),
    ],
)
def test_generated_region_check_rejects_placeholders_and_cheats(
    workspace, linear_blueprint, body, label
):
    proofs = {**PROOFS, "middle": body}
    assembled = render_assembly(workspace, linear_blueprint, proofs)

    problems = check_generated_region(workspace, assembled)

    assert any(label in problem for problem in problems)


def test_generated_region_check_rejects_a_temporary_axiom(workspace, linear_blueprint):
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)
    assembled = assembled.replace("theorem my_target", "axiom cheat : True\n\ntheorem my_target", 1)

    assert any("axiom" in problem for problem in check_generated_region(workspace, assembled))


def test_generated_region_check_ignores_sorries_outside_the_generated_region(
    workspace, linear_blueprint
):
    workspace.suffix = "theorem unrelated : True := by sorry\n"
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)

    assert check_generated_region(workspace, assembled) == []


def test_clean_assembly_passes_the_region_check(workspace, linear_blueprint):
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)

    assert check_generated_region(workspace, assembled) == []


def test_commit_writes_the_file_once(workspace, linear_blueprint):
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)

    commit_source(workspace, assembled)

    assert workspace.file_path.read_text(encoding="utf-8") == assembled
    assert not list(workspace.file_path.parent.glob("*.axprover.tmp"))


def test_commit_refuses_to_overwrite_a_concurrent_edit(workspace, linear_blueprint):
    assembled = render_assembly(workspace, linear_blueprint, PROOFS)
    workspace.file_path.write_text("-- edited by someone else\n", encoding="utf-8")

    with pytest.raises(AssemblyError, match="refusing to overwrite"):
        commit_source(workspace, assembled)

    assert workspace.file_path.read_text(encoding="utf-8") == "-- edited by someone else\n"
