"""File context slicing, deterministic namespaces, and node isolation."""

from ax_prover.blueprint.workspace import generated_namespace
from ax_prover.models.files import Location

from .conftest import make_node


def test_namespace_is_deterministic_and_target_specific():
    first = Location(name="my_target", module_path="Mod")
    same = Location(name="my_target", module_path="Mod")
    other = Location(name="other_target", module_path="Mod")
    other_module = Location(name="my_target", module_path="Other")

    assert generated_namespace(first) == generated_namespace(same)
    assert generated_namespace(first) != generated_namespace(other)
    assert generated_namespace(first) != generated_namespace(other_module)
    assert generated_namespace(first).startswith("AxProverGenerated_my_target_")


def test_namespace_sanitizes_non_identifier_characters():
    assert "'" not in generated_namespace(Location(name="my_target'", module_path="Mod"))


def test_prefix_holds_the_trusted_context_and_stops_before_the_target(workspace):
    assert "import Mathlib.Tactic" in workspace.prefix
    assert "set_option maxHeartbeats" in workspace.prefix
    assert "preceding_lemma" in workspace.prefix
    assert "my_target" not in workspace.prefix
    assert "user's own docstring" not in workspace.prefix


def test_suffix_holds_the_declarations_after_the_target(workspace):
    assert "following_lemma" in workspace.suffix
    assert "my_target" not in workspace.suffix


def test_target_statement_slices_are_available_with_and_without_the_docstring(workspace):
    assert workspace.target_statement_with_doc.startswith("/--")
    assert workspace.target_statement.startswith("theorem my_target")
    assert ":=" not in workspace.target_statement


def test_render_target_uses_the_users_statement_not_a_model_response(workspace):
    rendered = workspace.render_target(("helper",), "the plan", "by simp")

    assert "theorem my_target (n : Nat) : n + 0 = n := by simp" in rendered
    assert '"id": "target"' in rendered
    assert '"parents": ["helper"]' in rendered
    assert "the plan" in rendered


def test_skeleton_wraps_helpers_in_the_generated_namespace(workspace):
    skeleton = workspace.render_skeleton("theorem helper : True := by sorry")

    assert f"namespace {workspace.namespace}" in skeleton
    assert f"end {workspace.namespace}" in skeleton
    assert skeleton.index("theorem helper") < skeleton.index("theorem my_target")
    assert "by sorry" in skeleton


def test_skeleton_omits_the_declarations_after_the_target(workspace):
    assert "following_lemma" not in workspace.render_skeleton("theorem helper : True := by sorry")


def test_node_module_hides_unrelated_siblings_and_parent_proofs(workspace):
    parent = make_node("parent", signature=": True")
    unrelated = make_node("unrelated", signature=": False")
    node = make_node("child", ("parent",), signature=": True")

    module = workspace.render_node_module(node, (parent,), "by trivial")

    # Parents appear as named axioms, not sorries, so the axiom gate can tell a used
    # parent apart from an unproven hole.
    assert "axiom parent" in module
    assert "sorry" not in module
    assert "theorem child" in module
    assert "by trivial" in module
    assert unrelated.short_name not in module


def test_node_module_for_the_target_places_parents_in_the_namespace(workspace):
    parent = make_node("parent", signature=": True")
    target = make_node("target", ("parent",), is_target=True, lean_name="my_target")

    module = workspace.render_node_module(target, (parent,), "by simp")

    assert f"namespace {workspace.namespace}" in module
    assert "theorem my_target (n : Nat) : n + 0 = n := by simp" in module
    assert module.index("axiom parent") < module.index("theorem my_target")


def test_source_unchanged_detects_an_external_edit(workspace):
    assert workspace.source_unchanged()

    workspace.file_path.write_text("-- someone else edited this\n", encoding="utf-8")

    assert not workspace.source_unchanged()


def test_environment_fingerprint_changes_with_the_toolchain(workspace, tmp_path):
    before = workspace.environment_fingerprint()

    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.27.0\n", encoding="utf-8")

    assert workspace.environment_fingerprint() != before


def test_parent_placeholders_become_named_axioms(workspace):
    from .conftest import make_node

    assert workspace.render_parent_placeholder(make_node("p", signature=": True")) == (
        "axiom p : True"
    )


def test_allowed_axioms_cover_the_permitted_list_and_parents(workspace):
    from .conftest import make_node

    allowed = workspace.allowed_axioms((make_node("p", signature=": True"),))

    assert "propext" in allowed
    assert "Quot.sound" in allowed
    assert "Classical.choice" in allowed
    assert "p" in allowed
    assert "sorryAx" not in allowed


def test_stable_prefix_excludes_node_specific_parent_signatures(workspace):
    """Only the shared environment is warmed and replay-cached."""
    assert "import Mathlib.Tactic" in workspace.stable_prefix
    assert "preceding_lemma" in workspace.stable_prefix
    assert "axiom" not in workspace.stable_prefix
