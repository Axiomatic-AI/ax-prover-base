"""Builders for blueprint unit tests.

`build_declaration` fabricates the `lean_interact` declaration shape from plain source
text, so graph, store, and assembly behaviour can be tested without running Lean.
"""

import asyncio

import pytest
from lean_interact.interface import (
    DeclarationInfo,
    DeclModifiers,
    DeclSignature,
    DeclType,
    DeclValue,
    DocString,
    Pos,
    Range,
    ScopeInfo,
)

from ax_prover.blueprint.models import Blueprint, BlueprintNode
from ax_prover.blueprint.workspace import BlueprintWorkspace
from ax_prover.config import LeanConfig
from ax_prover.models.declaration import Declaration
from ax_prover.models.files import Location

NAMESPACE = "AxProverGenerated_target_deadbeef"


def to_pos(source: str, offset: int) -> Pos:
    """Convert a character offset into `lean_interact`'s 1-indexed line / 0-indexed column."""
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    column = offset - (prefix.rfind("\n") + 1)
    return Pos(line=line, column=column)


def _range(source: str, start: int, finish: int) -> Range:
    return Range(synthetic=False, start=to_pos(source, start), finish=to_pos(source, finish))


def build_declaration(
    source: str,
    decl_text: str,
    *,
    name: str,
    full_name: str | None = None,
    kind: str = "theorem",
    curr_namespace: str = "[anonymous]",
    signature: str = "",
    type_pp: str = "",
    type_deps: tuple[str, ...] = (),
    value_deps: tuple[str, ...] = (),
) -> Declaration:
    """Build a `Declaration` for `decl_text` as it appears in `source`."""
    start = source.index(decl_text)
    finish = start + len(decl_text)
    value_start = start + decl_text.index(":=")

    doc_string = None
    if decl_text.lstrip().startswith("/--"):
        doc_start = start + decl_text.index("/--")
        doc_finish = start + decl_text.index("-/") + 2
        doc_string = DocString(
            content=source[doc_start:doc_finish], range=_range(source, doc_start, doc_finish)
        )

    return Declaration(
        info=DeclarationInfo(
            pp=decl_text,
            range=_range(source, start, finish),
            scope=ScopeInfo(
                var_decls=[],
                include_vars=[],
                omit_vars=[],
                level_names=[],
                curr_namespace=curr_namespace,
                open_decl=[],
            ),
            name=name,
            full_name=full_name or name,
            kind=kind,
            modifiers=DeclModifiers(
                doc_string=doc_string,
                visibility="regular",
                compute_kind="regular",
                rec_kind="default",
                is_protected=False,
                is_unsafe=False,
                attributes=[],
            ),
            signature=DeclSignature(
                pp=signature, constants=list(type_deps), range=_range(source, start, value_start)
            ),
            binders=None,
            type=DeclType(
                pp=type_pp, constants=list(type_deps), range=_range(source, start, value_start)
            ),
            value=DeclValue(
                pp=source[value_start:finish],
                constants=list(value_deps),
                range=_range(source, value_start, finish),
            ),
        )
    )


def make_node(
    node_id: str,
    parents: tuple[str, ...] = (),
    *,
    signature: str = "(n : Nat) : n = n",
    is_target: bool = False,
    doc_text: str = "",
    kind: str = "theorem",
    lean_name: str | None = None,
) -> BlueprintNode:
    """Build a canonical node without going through Lean."""
    if lean_name is None:
        lean_name = node_id if is_target else f"{NAMESPACE}.{node_id}"
    statement = f"theorem {lean_name.rsplit('.', 1)[-1]} {signature} "
    return BlueprintNode(
        id=node_id,
        parents=parents,
        lean_name=lean_name,
        kind=kind,
        signature=signature,
        statement_source=statement,
        statement_source_no_doc=statement,
        doc_text=doc_text,
        is_target=is_target,
    )


def make_blueprint(
    *nodes: BlueprintNode, namespace: str = NAMESPACE, skeleton: str = ""
) -> Blueprint:
    """Build a blueprint without validation, for testing downstream behaviour."""
    return Blueprint(namespace=namespace, nodes=tuple(nodes), skeleton=skeleton)


@pytest.fixture
def namespace() -> str:
    """The deterministic generated namespace used across these tests."""
    return NAMESPACE


@pytest.fixture
def linear_blueprint() -> Blueprint:
    """A two-level chain: `base` -> `middle` -> target."""
    return make_blueprint(
        make_node("base"),
        make_node("middle", ("base",)),
        make_node("target", ("middle",), is_target=True, lean_name="my_target"),
    )


@pytest.fixture
def wide_blueprint() -> Blueprint:
    """Two independent helpers that the target uses together."""
    return make_blueprint(
        make_node("left"),
        make_node("right"),
        make_node("target", ("left", "right"), is_target=True, lean_name="my_target"),
    )


TARGET_FILE_SOURCE = """import Mathlib.Tactic

set_option maxHeartbeats 400000

theorem preceding_lemma : True := trivial

/-- The user's own docstring, which must survive assembly. -/
theorem my_target (n : Nat) : n + 0 = n := by
  sorry

theorem following_lemma : True := trivial
"""

TARGET_DECLARATION_TEXT = """/-- The user's own docstring, which must survive assembly. -/
theorem my_target (n : Nat) : n + 0 = n := by
  sorry"""


@pytest.fixture
def workspace(tmp_path):
    """A workspace over a small Lean file whose target sits between other declarations."""
    path = tmp_path / "Mod.lean"
    path.write_text(TARGET_FILE_SOURCE, encoding="utf-8")

    declaration = build_declaration(
        TARGET_FILE_SOURCE,
        TARGET_DECLARATION_TEXT,
        name="my_target",
        signature="(n : Nat) : n + 0 = n",
        type_pp="n + 0 = n",
    )

    return BlueprintWorkspace(
        base_folder=str(tmp_path),
        location=Location(name="my_target", module_path="Mod"),
        target_declaration=declaration,
        lean_config=LeanConfig(),
        semaphore=asyncio.Semaphore(1),
        trusted_declarations=[
            build_declaration(
                TARGET_FILE_SOURCE,
                "theorem preceding_lemma : True := trivial",
                name="preceding_lemma",
                signature=": True",
            ),
            build_declaration(
                TARGET_FILE_SOURCE,
                "theorem following_lemma : True := trivial",
                name="following_lemma",
                signature=": True",
            ),
        ],
    )
