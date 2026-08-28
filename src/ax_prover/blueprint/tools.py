"""The two tools a blueprint role may use: `lean_compile` and `mathlib_search`.

Blueprint mode deliberately exposes nothing else - no repository search, no web search, no
memory, no LLM reviewer - so node proving matches the paper's loop.
"""

import re
from collections.abc import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..runtime import Runtime
from ..tools import create_tool
from ..utils.logging import get_logger
from .lean_service import CompilePriority
from .models import BlueprintNode
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)

MATHLIB_SEARCH_TOOL_NAME = "mathlib_search"
LEAN_COMPILE_TOOL_NAME = "lean_compile"

_FENCE = re.compile(r"^```[a-zA-Z0-9]*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)
_DECLARATION_START = re.compile(r"^\s*(?:/--.*?-/\s*)?(?:theorem|lemma|example)\b", re.DOTALL)
_TOP_LEVEL_COMMAND = re.compile(
    r"^(import|open|namespace|end|variable|variables|set_option|attribute|instance|def|"
    r"structure|class|axiom|abbrev|macro|notation)\b"
)

_COMPILE_SUCCESS = "Compiled successfully. This proof body is accepted."
_NO_ERRORS = "(no compiler output)"


class ProofBodyInput(BaseModel):
    """Argument schema for `lean_compile` in node proving."""

    proof_body: str = Field(description="Proof body only: the text that follows `:=`")


class HelpersInput(BaseModel):
    """Argument schema for `lean_compile` in architect and refiner roles."""

    helpers: str = Field(description="Lean source of the docstringed helper lemmas")


class SearchInput(BaseModel):
    """Argument schema for `mathlib_search`."""

    query: str = Field(description="Declaration name, module path, or natural language query")


def strip_outer_fence(raw: str) -> tuple[str, list[str]]:
    """Remove a markdown code fence wrapping an entire Lean source block.

    Models routinely wrap `helpers` in ```lean ... ```, which makes the assembled skeleton
    fail to parse with `unexpected token '`'`. Only the outermost pair is removed: helper
    docstrings legitimately contain fenced ```ax-blueprint blocks, and the anchored,
    non-greedy pattern keeps those intact.
    """
    body = raw.strip()
    match = _FENCE.match(body)
    if not match:
        return body, []
    return match.group("body").strip(), ["removed a markdown code fence around `helpers`"]


def normalize_proof_body(raw: str) -> tuple[str, list[str]]:
    """Reduce a model response to a bare proof body.

    The harness owns the statement, so anything the model emits around the proof - a
    markdown fence, a restated theorem, top-level commands - is discarded rather than
    compiled.

    Returns:
        The proof body and a list of notes describing what was discarded.
    """
    notes: list[str] = []
    body = raw.strip()

    fence = _FENCE.match(body)
    if fence:
        body = fence.group("body").strip()
        notes.append("removed a markdown code fence")

    # Strip stray commands before looking for a restated declaration: a response that
    # leads with `import ...` would otherwise hide the theorem line behind it.
    kept_lines = []
    for line in body.splitlines():
        # `open Foo in` and similar term-level prefixes belong to a proof body, so only a
        # bare command line is discarded.
        if _TOP_LEVEL_COMMAND.match(line) and not line.rstrip().endswith(" in"):
            notes.append(f"discarded top-level command: {line.strip()!r}")
            continue
        kept_lines.append(line)
    body = "\n".join(kept_lines).strip()

    if _DECLARATION_START.match(body):
        _, separator, remainder = body.partition(":=")
        if separator:
            body = remainder.strip()
            notes.append("discarded a restated theorem declaration; only the proof body is used")

    if body.startswith(":="):
        body = body[2:].strip()

    return body, notes


def format_compile_output(success: bool, output: str) -> str:
    """Format a compile result for a model, keeping failures verbatim."""
    if success:
        return _COMPILE_SUCCESS
    return f"Compilation failed:\n{output or _NO_ERRORS}"


def make_node_compile_tool(
    workspace: BlueprintWorkspace,
    node: BlueprintNode,
    parents: tuple[BlueprintNode, ...],
    on_result: Callable[[str, bool, str], None] | None = None,
) -> StructuredTool:
    """Build the node prover's `lean_compile`, bound to one node's isolated module."""

    async def compile_body(proof_body: str) -> str:
        body, notes = normalize_proof_body(proof_body)
        if not body:
            return "Compilation failed: the proof body was empty after normalization."

        source = workspace.render_node_module(node, parents, body)
        result = await workspace.compile_candidate(
            source,
            node_id=node.id,
            check_axioms_of=node.lean_name,
            allowed_axioms=workspace.allowed_axioms(parents),
            label=f"node_{node.id}",
        )

        if on_result is not None:
            on_result(body, result.success, result.output)

        message = format_compile_output(result.success, result.output)
        if notes:
            message = "Note: " + "; ".join(notes) + "\n" + message
        return message

    return StructuredTool(
        name=LEAN_COMPILE_TOOL_NAME,
        description=(
            "Compile a candidate proof body under the exact statement you were given, in an "
            "isolated module. Returns compiler errors and remaining goals, or confirmation "
            "that the proof is accepted. Pass the proof body only."
        ),
        coroutine=compile_body,
        args_schema=ProofBodyInput,
    )


def make_skeleton_compile_tool(
    workspace: BlueprintWorkspace,
    on_result: Callable[[str, bool, str], None] | None = None,
) -> StructuredTool:
    """Build the architect's and refiner's `lean_compile` over a whole helper skeleton."""

    async def compile_helpers(helpers: str) -> str:
        helpers, notes = strip_outer_fence(helpers)
        source = workspace.render_skeleton(helpers)
        # Structural work unblocks the whole graph, so it outranks node candidates.
        result = await workspace.compile_candidate(
            source, priority=CompilePriority.STRUCTURAL, label="skeleton"
        )

        if on_result is not None:
            on_result(helpers, result.success, result.output)

        message = format_compile_output(result.success, result.output)
        if notes:
            message = "Note: " + "; ".join(notes) + "\n" + message
        return message

    return StructuredTool(
        name=LEAN_COMPILE_TOOL_NAME,
        description=(
            "Assemble your helper lemmas with the real file context and the real target, "
            "compile the module, and return the errors. `declaration uses 'sorry'` warnings "
            "are expected and are not errors."
        ),
        coroutine=compile_helpers,
        args_schema=HelpersInput,
    )


async def make_mathlib_search_tool(
    tool_configs: dict[str, dict | None], runtime: Runtime
) -> StructuredTool | None:
    """Expose the project's Lean search capability under the paper's tool name.

    Returns None when no search tool is configured or its warm-up failed, in which case
    node proving runs with `lean_compile` alone.
    """
    for tool_config in tool_configs.values():
        if tool_config is None:
            continue

        tool = await create_tool(tool_config, runtime)
        if tool is None:
            continue

        tool.name = MATHLIB_SEARCH_TOOL_NAME
        tool.args_schema = SearchInput
        return tool

    logger.info("No mathlib_search backend configured; node provers will use lean_compile only")
    return None


__all__ = [
    "LEAN_COMPILE_TOOL_NAME",
    "MATHLIB_SEARCH_TOOL_NAME",
    "format_compile_output",
    "make_mathlib_search_tool",
    "make_node_compile_tool",
    "make_skeleton_compile_tool",
    "normalize_proof_body",
]
