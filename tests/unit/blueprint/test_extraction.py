"""Canonicalization of compiled declarations into blueprint nodes."""

import pytest

from ax_prover.blueprint.extraction import extract_nodes, statement_slice, target_signature
from ax_prover.blueprint.models import BlueprintValidationError

from .conftest import NAMESPACE, build_declaration

HELPER_TEXT = """/--
```ax-blueprint
{"version": 1, "id": "helper_one", "parents": []}
```

## Statement

A helper.
-/
theorem helper_one (n : Nat) : n + 0 = n := by
  sorry"""

TARGET_TEXT = """/--
```ax-blueprint
{"version": 1, "id": "target", "parents": ["helper_one"]}
```
-/
theorem my_target (n : Nat) (m : Nat) : n + 0 + m = n + m := by
  sorry"""

SKELETON = f"""import Mathlib.Tactic

theorem trusted_context : True := trivial

namespace {NAMESPACE}

{HELPER_TEXT}

end {NAMESPACE}

{TARGET_TEXT}
"""


def helper_declaration(source: str = SKELETON, text: str = HELPER_TEXT):
    return build_declaration(
        source,
        text,
        name="helper_one",
        full_name=f"{NAMESPACE}.helper_one",
        curr_namespace=NAMESPACE,
        signature="(n : Nat) : n + 0 = n",
        type_pp="n + 0 = n",
        type_deps=("Nat", "HAdd.hAdd"),
    )


def target_declaration(source: str = SKELETON, text: str = TARGET_TEXT):
    return build_declaration(
        source,
        text,
        name="my_target",
        signature="(n : Nat) (m : Nat) : n + 0 + m = n + m",
        type_pp="n + 0 + m = n + m",
    )


def trusted_declaration():
    return build_declaration(
        SKELETON,
        "theorem trusted_context : True := trivial",
        name="trusted_context",
        signature=": True",
        type_pp="True",
    )


def test_statement_slice_ends_before_the_assignment():
    statement, without_doc = statement_slice(SKELETON, helper_declaration().info)

    assert statement.startswith("/--")
    assert without_doc.startswith("theorem helper_one")
    assert ":=" not in without_doc
    assert (
        f"{without_doc.rstrip()} := by simp"
        == "theorem helper_one (n : Nat) : n + 0 = n := by simp"
    )


TRUSTED = frozenset({"trusted_context"})


def test_extracts_generated_nodes_and_the_target():
    nodes = extract_nodes(
        [trusted_declaration(), helper_declaration(), target_declaration()],
        SKELETON,
        NAMESPACE,
        "my_target",
        TRUSTED,
    )

    by_id = {node.id: node for node in nodes}
    assert set(by_id) == {"helper_one", "target"}
    assert by_id["target"].is_target
    assert by_id["target"].parents == ("helper_one",)
    assert by_id["helper_one"].lean_name == f"{NAMESPACE}.helper_one"


def test_ignores_trusted_declarations_outside_the_namespace():
    nodes = extract_nodes([trusted_declaration()], SKELETON, NAMESPACE, "my_target", TRUSTED)

    assert nodes == []


def test_rejects_a_declaration_smuggled_outside_the_namespace():
    with pytest.raises(BlueprintValidationError, match="outside the generated namespace"):
        extract_nodes([trusted_declaration()], SKELETON, NAMESPACE, "my_target")


def test_ignores_lean_generated_companion_declarations():
    text = "theorem helper_two : True := by sorry"
    source = SKELETON.replace(HELPER_TEXT, text)
    companion = build_declaration(
        source,
        text,
        name="match_1",
        full_name=f"{NAMESPACE}.helper_two.match_1",
        curr_namespace=NAMESPACE,
        signature=": True",
    )

    assert extract_nodes([companion], source, NAMESPACE, "my_target") == []


def test_keeps_the_docstring_prose_without_the_metadata_fence():
    nodes = extract_nodes([helper_declaration()], SKELETON, NAMESPACE, "my_target")

    assert nodes[0].doc_text.startswith("## Statement")
    assert "ax-blueprint" not in nodes[0].doc_text


def test_records_elaborated_type_dependencies_for_diagnostics():
    nodes = extract_nodes([helper_declaration()], SKELETON, NAMESPACE, "my_target")

    assert nodes[0].type_deps == ("Nat", "HAdd.hAdd")
    # `by sorry` carries no intended proof dependencies, which is why declared parents,
    # not value deps, drive scheduling.
    assert nodes[0].value_deps == ()


def test_rejects_a_generated_declaration_without_a_docstring():
    text = "theorem helper_two : True := by sorry"
    source = SKELETON.replace(HELPER_TEXT, text)
    declaration = build_declaration(
        source,
        text,
        name="helper_two",
        full_name=f"{NAMESPACE}.helper_two",
        curr_namespace=NAMESPACE,
        signature=": True",
    )

    with pytest.raises(BlueprintValidationError, match="has no docstring"):
        extract_nodes([declaration], source, NAMESPACE, "my_target")


def test_rejects_a_generated_declaration_with_malformed_metadata():
    text = "/-- just prose -/\ntheorem helper_two : True := by sorry"
    source = SKELETON.replace(HELPER_TEXT, text)
    declaration = build_declaration(
        source,
        text,
        name="helper_two",
        full_name=f"{NAMESPACE}.helper_two",
        curr_namespace=NAMESPACE,
        signature=": True",
    )

    with pytest.raises(BlueprintValidationError, match="missing"):
        extract_nodes([declaration], source, NAMESPACE, "my_target")


def test_target_signature_reads_the_original_declaration():
    signature = target_signature([trusted_declaration(), target_declaration()], "my_target")

    assert signature == "(n : Nat) (m : Nat) : n + 0 + m = n + m"


def test_target_signature_reports_a_missing_declaration():
    with pytest.raises(BlueprintValidationError, match="not found"):
        target_signature([trusted_declaration()], "my_target")
