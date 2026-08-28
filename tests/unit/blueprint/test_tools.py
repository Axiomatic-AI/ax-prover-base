"""The node prover's output contract: only a proof body can ever be compiled."""

import pytest

from ax_prover.blueprint.tools import (
    LEAN_COMPILE_TOOL_NAME,
    MATHLIB_SEARCH_TOOL_NAME,
    format_compile_output,
    make_node_compile_tool,
    make_skeleton_compile_tool,
    normalize_proof_body,
)

from .conftest import make_node


def test_keeps_a_plain_proof_body_untouched():
    body, notes = normalize_proof_body("by\n  simpa using h")

    assert body == "by\n  simpa using h"
    assert notes == []


def test_strips_a_markdown_fence():
    body, notes = normalize_proof_body("```lean\nby positivity\n```")

    assert body == "by positivity"
    assert notes == ["removed a markdown code fence"]


def test_discards_a_restated_theorem_declaration():
    body, notes = normalize_proof_body("theorem foo (n : Nat) : n = n := by rfl")

    assert body == "by rfl"
    assert any("restated theorem" in note for note in notes)


def test_discards_a_restated_declaration_with_its_own_docstring():
    body, _ = normalize_proof_body("/-- doc -/\nlemma foo : True := trivial")

    assert body == "trivial"


def test_strips_a_leading_assignment_token():
    body, _ = normalize_proof_body(":= by simp")

    assert body == "by simp"


@pytest.mark.parametrize(
    "command",
    ["import Mathlib.Tactic", "open Nat", "namespace Foo", "axiom cheat : True", "def helper := 1"],
)
def test_discards_top_level_commands(command):
    body, notes = normalize_proof_body(f"{command}\nby simp")

    assert body == "by simp"
    assert any(command in note for note in notes)


def test_keeps_a_term_level_open_in_prefix():
    body, notes = normalize_proof_body("open Nat in\nby simp")

    assert body == "open Nat in\nby simp"
    assert notes == []


def test_reports_an_empty_body():
    body, _ = normalize_proof_body("import Mathlib.Tactic")

    assert body == ""


def test_compile_output_formatting():
    assert "accepted" in format_compile_output(True, "")
    assert "boom" in format_compile_output(False, "boom")
    assert "no compiler output" in format_compile_output(False, "")


async def test_node_compile_tool_only_compiles_the_normalized_body(workspace, monkeypatch):
    compiled: list[str] = []

    async def fake_compile(source, label="scratch"):
        compiled.append(source)
        return type("Result", (), {"success": True, "output": "", "source": source})()

    monkeypatch.setattr(workspace, "compile_source", fake_compile)
    tool = make_node_compile_tool(workspace, make_node("helper", signature=": True"), ())

    message = await tool.ainvoke({"proof_body": "import Mathlib\ntheorem rogue : True := trivial"})

    assert tool.name == LEAN_COMPILE_TOOL_NAME
    assert "accepted" in message
    assert "theorem rogue" not in compiled[0]
    assert "import Mathlib\n" not in compiled[0].replace("import Mathlib.Tactic", "")


async def test_node_compile_tool_rejects_an_empty_body(workspace):
    tool = make_node_compile_tool(workspace, make_node("helper", signature=": True"), ())

    message = await tool.ainvoke({"proof_body": "import Mathlib.Tactic"})

    assert "empty after normalization" in message


async def test_skeleton_compile_tool_reports_failures_verbatim(workspace, monkeypatch):
    async def fake_compile(source, label="scratch"):
        return type(
            "Result", (), {"success": False, "output": "unknown identifier 'foo'", "source": source}
        )()

    monkeypatch.setattr(workspace, "compile_source", fake_compile)
    tool = make_skeleton_compile_tool(workspace)

    message = await tool.ainvoke({"helpers": "theorem helper : True := by sorry"})

    assert "unknown identifier 'foo'" in message


async def test_mathlib_search_is_absent_when_no_backend_is_configured():
    from ax_prover.blueprint.tools import make_mathlib_search_tool

    assert await make_mathlib_search_tool({}, None) is None
    assert await make_mathlib_search_tool({"a": None}, None) is None
    assert MATHLIB_SEARCH_TOOL_NAME == "mathlib_search"


def test_strip_outer_fence_removes_a_wrapping_markdown_fence():
    from ax_prover.blueprint.tools import strip_outer_fence

    body, notes = strip_outer_fence("```lean\ntheorem a : True := by sorry\n```")

    assert body == "theorem a : True := by sorry"
    assert notes


def test_strip_outer_fence_keeps_inner_ax_blueprint_blocks():
    """Helper docstrings legitimately contain fenced metadata; only the wrapper goes."""
    from ax_prover.blueprint.tools import strip_outer_fence

    helpers = """```lean
/--
```ax-blueprint
{"version": 1, "id": "h", "parents": []}
```
-/
theorem h : True := by sorry
```"""

    body, notes = strip_outer_fence(helpers)

    assert body.startswith("/--")
    assert "```ax-blueprint" in body
    assert '{"version": 1, "id": "h", "parents": []}' in body
    assert notes


def test_strip_outer_fence_leaves_unfenced_source_alone():
    from ax_prover.blueprint.tools import strip_outer_fence

    helpers = (
        '/--\n```ax-blueprint\n{"version": 1, "id": "h"}\n```\n-/\ntheorem h : True := by sorry'
    )

    body, notes = strip_outer_fence(helpers)

    assert body == helpers
    assert notes == []


async def test_skeleton_compile_tool_strips_a_fence_before_compiling(workspace, monkeypatch):
    """A fenced skeleton would fail to parse with `unexpected token '`'`."""
    compiled: list[str] = []

    async def fake_compile(source, label="scratch", **kwargs):
        compiled.append(source)
        return type("R", (), {"success": True, "output": "", "source": source})()

    monkeypatch.setattr(workspace, "compile_candidate", fake_compile)
    tool = make_skeleton_compile_tool(workspace)

    message = await tool.ainvoke({"helpers": "```lean\ntheorem h : True := by sorry\n```"})

    assert "```lean" not in compiled[0]
    assert "theorem h : True := by sorry" in compiled[0]
    assert "markdown code fence" in message
