"""Canonicalization adapter: compiled Lean declarations to canonical blueprint nodes.

The plan names `lean-extract` as the extraction authority. `lean_interact`, already the
project's Lean bridge, exposes the identical canonical fields per declaration -
`full_name`, `kind`, elaborated `signature`/`type` pretty-printing, the docstring, the
source range, and `type`/`value` constant dependencies - so it backs this adapter. The
module boundary is the seam: swapping in a `lean-extract` backend means replacing
`extract_declarations` only.
"""

import re

from lean_interact.interface import DeclarationInfo, Pos

from ..models.declaration import Declaration
from ..utils.lean_interact import LeanInteractServer
from ..utils.lean_parsing import list_declarations_from_file
from .metadata import MetadataError, parse_metadata
from .models import BlueprintNode, BlueprintValidationError

_LEADING_DOC = re.compile(r"^\s*/--.*?-/[ \t]*\n?", re.DOTALL)

#: Suffixes Lean attaches to compiler-generated companions of a declaration. These are not
#: authored by the architect, so they carry no blueprint metadata and are skipped.
_LEAN_GENERATED_SUFFIXES = (".match_", ".eq_def", ".eq_", ".proof_", ".induct", ".sizeOf_spec")


def _is_lean_generated(full_name: str) -> bool:
    """True for a declaration Lean synthesized rather than one written in the source."""
    if any(component.startswith("_") for component in full_name.split(".")):
        return True
    return any(suffix in full_name for suffix in _LEAN_GENERATED_SUFFIXES)


async def extract_declarations(server: LeanInteractServer, file_path: str) -> list[Declaration]:
    """Compile a Lean file and return its declarations with canonical metadata."""
    return await list_declarations_from_file(server, file_path)


def to_offset(source: str, position: Pos) -> int:
    """Convert a 1-indexed line / 0-indexed column position into a character offset."""
    lines = source.splitlines(keepends=True)
    return sum(len(line) for line in lines[: position.line - 1]) + position.column


def statement_slice(source: str, info: DeclarationInfo) -> tuple[str, str]:
    """Return the declaration text up to `:=`, with and without its docstring.

    `lean_interact` reports `value.range.start` at the `:=` token, so the slice ends just
    before it. Appending `":= " + proof_body` therefore reproduces a full declaration.
    """
    start = to_offset(source, info.range.start)
    if info.value is None:
        raise BlueprintValidationError(
            [f"declaration {info.full_name!r} has no proof body; it must end in `:= by sorry`"]
        )
    end = to_offset(source, info.value.range.start)
    statement = source[start:end]

    doc_string = info.modifiers.doc_string
    if doc_string is not None:
        doc_end = to_offset(source, doc_string.range.finish) - start
        without_doc = statement[doc_end:].lstrip("\n")
    else:
        without_doc = _LEADING_DOC.sub("", statement)

    return statement, without_doc


def extract_nodes(
    declarations: list[Declaration],
    source: str,
    namespace: str,
    target_lean_name: str,
    trusted_names: frozenset[str] = frozenset(),
) -> list[BlueprintNode]:
    """Build canonical nodes from a compiled skeleton's declarations.

    A declaration becomes a node when it is the target or lives in the generated
    namespace. Declarations named in `trusted_names` are the user's own and are ignored.
    Anything else at top level is a declaration the architect smuggled past the namespace
    boundary, and is rejected.

    Raises:
        BlueprintValidationError: A generated declaration has missing or malformed
            blueprint metadata, or an unexpected declaration escaped the namespace.
    """
    problems: list[str] = []
    nodes: list[BlueprintNode] = []
    prefix = f"{namespace}."

    for declaration in declarations:
        info = declaration.info
        is_target = info.full_name == target_lean_name
        is_generated = info.full_name.startswith(prefix)

        if _is_lean_generated(info.full_name):
            continue

        if not is_target and not is_generated:
            if info.full_name not in trusted_names:
                problems.append(
                    f"declaration {info.full_name!r} is outside the generated namespace "
                    f"{namespace!r}; only helper lemmas inside it may be added"
                )
            continue

        docstring = info.modifiers.doc_string
        if docstring is None:
            problems.append(
                f"generated declaration {info.full_name!r} has no docstring; it needs an "
                "```ax-blueprint metadata block"
            )
            continue

        try:
            metadata, prose = parse_metadata(docstring.content)
        except MetadataError as e:
            problems.append(f"{info.full_name}: {e}")
            continue

        try:
            statement, statement_no_doc = statement_slice(source, info)
        except BlueprintValidationError as e:
            problems += e.problems
            continue

        nodes.append(
            BlueprintNode(
                id=metadata.id,
                parents=metadata.parents,
                lean_name=info.full_name,
                kind=info.kind,
                signature=info.signature.pp,
                type_pp=info.type.pp if info.type else "",
                statement_source=statement,
                statement_source_no_doc=statement_no_doc,
                doc_text=prose,
                type_deps=tuple(info.type.constants) if info.type else (),
                value_deps=tuple(info.value.constants) if info.value else (),
                start_line=info.range.start.line,
                is_target=is_target,
            )
        )

    if problems:
        raise BlueprintValidationError(problems)

    return nodes


def target_signature(declarations: list[Declaration], target_lean_name: str) -> str:
    """Pretty-printed elaborated signature of the original target declaration.

    Raises:
        BlueprintValidationError: The target declaration is absent.
    """
    for declaration in declarations:
        if declaration.info.full_name == target_lean_name:
            return declaration.info.signature.pp
    raise BlueprintValidationError([f"declaration {target_lean_name!r} not found"])
