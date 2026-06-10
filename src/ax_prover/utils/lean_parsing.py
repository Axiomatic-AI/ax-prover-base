"""Utilities for parsing Lean code structure and declarations."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from lean_interact import Command
from lean_interact.interface import DeclarationInfo, Sorry

from ..models.declaration import Declaration, DeclarationType
from ..models.files import Location
from .lean_interact import LeanInteractServer
from .logging import get_logger

logger = get_logger(__name__)

# Lean keywords for declarations
LEAN_KEYWORDS = [d.value for d in DeclarationType]


def count_pattern(
    content: str,
    pattern: str,
    context_lines: int = 1,
) -> tuple[int, list[tuple[int, str]]]:
    """Count pattern matches in Lean code with context.

    Args:
        content: The Lean file content
        context_lines: Number of lines to show before and after
        pattern: Regex pattern to search for (default: sorry/admit)

    Returns:
        Tuple of (count, locations) where locations is a list of (line_num, formatted_context)
    """
    sorry_locations = []
    lines = content.splitlines()

    for i, line in enumerate(lines):
        for match in re.finditer(pattern, line):
            line_num = i + 1
            col = match.start()

            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)

            context = []
            for j in range(start, end):
                context.append(f"  {lines[j]}")

                if j == i:
                    context.append("  " + " " * col + "^^^^^")

            sorry_locations.append((line_num, "\n".join(context)))

    return len(sorry_locations), sorry_locations


def strip_comments(src: str) -> str:
    """
    Remove Lean comments from src.
    Handles nested block comments '/- ... -/' and '--' line comments.
    Leaves string literals intact.
    """

    class ParsingState(Enum):
        Out = 1
        LineComment = 2
        BlockComment = 3
        StringLiteral = 4

    state = ParsingState.Out
    i = 0
    depth = 0
    out = []
    n = len(src)

    while i < n:
        c = src[i]
        c2 = src[i : i + 2]

        if state == ParsingState.Out:
            if c == '"':
                state = ParsingState.StringLiteral
            if c2 == "--":
                state = ParsingState.LineComment
                out.append("  ")
                i += 2
            elif c2 == "/-":
                state = ParsingState.BlockComment
                depth = 1
                out.append("  ")
                i += 2
            else:
                out.append(c)
                i += 1

        elif state == ParsingState.LineComment:
            if c == "\n":
                state = ParsingState.Out
                out.append("\n")
            else:
                out.append(" ")  # preserve byte count
            i += 1

        elif state == ParsingState.BlockComment:
            if c2 == "/-":
                depth += 1
                out.append("  ")
                i += 2
            elif c2 == "-/":
                depth -= 1
                out.append("  ")
                i += 2
                if depth == 0:
                    state = ParsingState.Out
            else:
                out.append(" " if c != "\n" else "\n")
                i += 1

        elif state == ParsingState.StringLiteral:
            if c == '"':
                state = ParsingState.Out
            out.append(c)
            i += 1

    return "".join(out)


def extract_function_from_content(content: str, function_name: str) -> str | None:
    """Extract a function/theorem/lemma definition from Lean code.

    Args:
        content: Lean code content as string
        function_name: Name of the function/theorem/lemma to extract

    Returns:
        The complete definition block including doc comments, or None
    """
    keywords_pattern = "|".join(LEAN_KEYWORDS)
    pattern = rf"^(\s*)({keywords_pattern})\s+{re.escape(function_name)}\b"

    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        return None

    start_pos = match.start()
    start_indent = len(match.group(1))

    # Look backwards for Lean4 doc comment (/-- ... -/)
    before_def = content[:start_pos]
    all_doc_comments = list(re.finditer(r"/--[\s\S]*?-/", before_def))

    # Check doc comments in reverse order to find the closest one
    for doc_match in reversed(all_doc_comments):
        between = content[doc_match.end() : start_pos]
        # If no definition keyword between comment and target, use it
        if not re.search(rf"\b(?:{keywords_pattern})\s+\w+", between):
            start_pos = doc_match.start()
            break

    # Find next definition, doc comment, structural keyword, or top-level comment
    # at same or lower indentation
    end_pattern = rf"^[ \t]{{0,{start_indent}}}(/--|--|{keywords_pattern}(?:\s+|\b))"

    remaining_content = content[match.end() :]
    end_match = re.search(end_pattern, remaining_content, re.MULTILINE)

    if end_match:
        end_pos = match.end() + end_match.start()
    else:
        end_pos = len(content)

    return content[start_pos:end_pos].strip()


def get_function_from_location(base_folder: str, location: Location) -> str | None:
    """Get a function/theorem/lemma definition using a Location object.

    Args:
        base_folder: Base folder path
        location: Location object with import path (dot notation), name, and is_external flag

    Returns:
        The complete definition block, or None if not found
    """
    full_path = location.absolute_path(base_folder)

    if not full_path or not full_path.exists():
        logger.warning(f"This path does not exist: {location.module_path}.")
        return None

    try:
        content = full_path.read_text(encoding="utf-8")
        return extract_function_from_content(content, location.name)
    except Exception as e:
        logger.error(f"Error in get_function_from_location: {e}")
        return None


async def get_unproven(server: LeanInteractServer, base_folder: str, file_path: str) -> list[str]:
    """Get all function/theorem/lemma names that contain 'sorry' in their body.

    Args:
        server: Lean interact server
        base_folder: Base folder path
        file_path: Path to file relative to base_folder

    Returns:
        List of function names that contain 'sorry' in their implementation
    """

    all_defs = await list_all_declarations_in_path_as_text(
        server, base_folder, file_path, show_statements=False
    )

    if not all_defs:
        return []

    module_path = file_path.replace("/", ".").removesuffix(".lean")
    unproven_functions = []

    for line in all_defs.strip().split("\n"):
        if not line:
            continue

        func_name = extract_theorem_name(line)
        if not func_name:
            continue

        location = Location(module_path=module_path, name=func_name, is_external=False)
        func_body = get_function_from_location(base_folder, location)
        if func_body and re.search(r"\bsorry\b", func_body):
            unproven_functions.append(func_name)

    return unproven_functions


def extract_theorem_name(theorem_statement: str) -> str | None:
    """Extract theorem name from a theorem statement.

    Args:
        theorem_statement: A Lean theorem/lemma/def/etc statement

    Returns:
        The theorem name, or None if not found

    Example:
        >>> extract_theorem_name("theorem foo : P := sorry")
        'foo'
        >>> extract_theorem_name("lemma bar (n : Nat) : n > 0 := by sorry")
        'bar'
        >>> extract_theorem_name("theorem Polynomial.not_isPrincipalIdealRing : ¬IsPrincipalIdealRing R[X] := sorry")
        'Polynomial.not_isPrincipalIdealRing'
    """
    theorem_statement = strip_comments(theorem_statement)

    keywords_pattern = "|".join(re.escape(kw) for kw in LEAN_KEYWORDS)
    match = re.search(rf"\b(?:{keywords_pattern})\s+([\w.]+)", theorem_statement)
    if match:
        return match.group(1)
    return None


async def _list_all_declarations_in_path(
    server: LeanInteractServer, base_folder: str = ".", path: str = ""
) -> list[tuple[Path, Declaration]]:
    """
    List all theorems, definitions, lemmas, axioms, and other Lean constructs; in a given path.

    Args:
        server: Lean interact server
        base_folder: Base folder to search in
        path: Path to subfolder or file to search in
    Returns:
        List of tuples (file_path, declaration)
    """

    if path:
        full_path = Path(base_folder) / path
    else:
        full_path = Path(base_folder)

    file_list = None
    if full_path.is_dir():
        file_list = list(
            filter(lambda p: p.is_file() and p.suffix == ".lean", full_path.rglob("*"))
        )
    else:
        assert full_path.suffix == ".lean"
        file_list = [full_path]

    declarations = []
    for file_path in file_list:
        for declaration in await list_declarations_from_file(server, file_path):
            declarations.append((file_path, declaration))

    return declarations


async def list_all_declarations_in_path_as_text(
    server: LeanInteractServer,
    base_folder: str = ".",
    path: str = "",
    show_statements: bool = False,
) -> str:
    """
    List all theorems, definitions, lemmas, axioms, and other Lean constructs as text; in a given path.

    Args:
        server: Lean interact server
        base_folder: Base folder to search in
        path: Path to subfolder or file to search in
        show_statements: If True, show full statements

    Returns:
        Text (string) containing all paths and declarations
    """
    declarations = await _list_all_declarations_in_path(server, base_folder, path)
    if show_statements:
        return "\n".join(f"{decl_path}:{str(decl)}" for decl_path, decl in declarations)
    else:
        return "\n".join(f"{decl_path}:{decl.kind} {decl.name}" for decl_path, decl in declarations)


def find_declaration_by_name(declarations: list[Declaration], name: str) -> Declaration | None:
    """Find a declaration by name.

    Args:
        declarations: List of declarations
        name: Name of the declaration to find

    Returns:
        The declaration, or None if not found
    """
    for declaration in declarations:
        if declaration.name == name:
            return declaration
    return None


def find_declaration_at_line(content: str, line_number: int) -> str | None:
    """Find the declaration name containing the given line number.

    Args:
        content: Lean code content as string
        line_number: 1-indexed line number to search for

    Returns:
        The name of the declaration containing the line, or None if not found
    """
    if line_number < 1:
        return None

    # strip_comments preserves newlines, so line numbers remain valid
    stripped = strip_comments(content)
    lines = stripped.split("\n")

    if line_number > len(lines):
        return None

    keywords_pattern = "|".join(LEAN_KEYWORDS)
    pattern = rf"^(\s*)({keywords_pattern})\s+([\w.]+)"

    declarations: list[tuple[str, int, int]] = []

    for i, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            name = match.group(3)
            # Split on punctuation that can follow the name
            name = re.split(r"[:({[\[]", name)[0]
            start_line = i + 1  # Convert to 1-indexed

            # Close previous declaration at same or lower indent
            if declarations:
                prev_name, prev_start, _ = declarations[-1]
                declarations[-1] = (prev_name, prev_start, i + 1)  # end is exclusive, 1-indexed

            declarations.append((name, start_line, len(lines) + 1))

    for name, start, end in declarations:
        if start <= line_number < end:
            return name

    return None


def format_goal_state_at_sorries(sorries: list[Sorry]) -> str:
    """
    Get the goal state at all sorry locations in a declaration.

    Args:
        sorries: List of Sorry objects

    Returns:
        Formatted string with goal states at each sorry location
    """
    if not sorries:
        return "No sorries found in code."

    goal_states = []
    for idx, sorry in enumerate(sorries, start=1):
        goal_states.append(
            f"Sorry #{idx} at line {sorry.start_pos.line}, column {sorry.start_pos.column}:\n"
            f"{sorry.goal}\n"
        )

    return "\n".join(goal_states)


async def list_declarations_from_code(
    server: LeanInteractServer, code: str
) -> list[DeclarationInfo]:
    """List all declarations from a code snippet."""
    response = await server.run(Command(cmd=code, declarations=True))
    return _get_declarations_with_sorries(response.declarations, response.sorries)


async def list_declarations_from_file(
    server: LeanInteractServer, file_path: Path
) -> list[DeclarationInfo]:
    """List all declarations from a file."""
    code = file_path.read_text()
    return await list_declarations_from_code(server, code)


def _get_declarations_with_sorries(
    declaration_infos: list[DeclarationInfo], sorries: list[Sorry]
) -> list[Declaration]:
    """Match the sorries with the declaration information from the lean interact response,
    and combine them into a single Declaration object."""
    declarations = []
    for declaration_info in declaration_infos:
        sorries_in_declaration = []
        for sorry in sorries:
            if (
                sorry.start_pos > declaration_info.range.start
                and sorry.start_pos < declaration_info.range.finish
            ):
                sorries_in_declaration.append(sorry)
        declarations.append(Declaration(info=declaration_info, sorries=sorries_in_declaration))

    return declarations
