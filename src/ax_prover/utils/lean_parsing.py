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

# Search/suggestion tactics that emit "Try this" and must not appear in a final proof.
# These names are tactic-only — none are API methods ending in "?", so real code like
# List.find?, xs.head?, Array.get?, m.lookup? is NOT matched. Extend as needed.
SEARCH_TACTICS = ("apply", "exact", "rw", "simp", "simp_all", "aesop", "observe")
# Longer names first so "simp_all?" isn't shadowed by "simp".
SEARCH_TACTIC_PATTERN = rf"\b({'|'.join(sorted(SEARCH_TACTICS, key=len, reverse=True))})\?"


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


def blank_string_literals(src: str) -> str:
    """Replace the CONTENTS of string literals with spaces, preserving the quotes.

    Same state machine as `strip_comments` but inverted: comments are left intact while
    the inside of `"..."` string literals is blanked (length and positions preserved,
    surrounding quotes kept). Used before pattern checks (e.g. search-tactic or `axiom`
    detection) so a tactic-like substring inside a string literal does not falsely match.
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
                out.append(c)
                i += 1
            elif c2 == "--":
                state = ParsingState.LineComment
                out.append(c2)
                i += 2
            elif c2 == "/-":
                state = ParsingState.BlockComment
                depth = 1
                out.append(c2)
                i += 2
            else:
                out.append(c)
                i += 1

        elif state == ParsingState.LineComment:
            if c == "\n":
                state = ParsingState.Out
            out.append(c)
            i += 1

        elif state == ParsingState.BlockComment:
            if c2 == "/-":
                depth += 1
                out.append(c2)
                i += 2
            elif c2 == "-/":
                depth -= 1
                out.append(c2)
                i += 2
                if depth == 0:
                    state = ParsingState.Out
            else:
                out.append(c)
                i += 1

        elif state == ParsingState.StringLiteral:
            if c == '"':
                state = ParsingState.Out
                out.append(c)
            else:
                out.append(" " if c != "\n" else "\n")
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
        location: Location object with import path (dot notation) and name

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


def find_declaration_at_line(
    declarations: list[Declaration], line_number: int
) -> Declaration | None:
    """Find the declaration that contains the given line number.

    Args:
        declarations: List of declarations
        line_number: 1-indexed line number to search for

    Returns:
        The declaration, or None if not found
    """
    matches = [
        declaration for declaration in declarations if declaration.contains_line(line_number)
    ]

    if not matches:
        return None

    if len(matches) > 1:
        logger.warning(f"Multiple declarations found at line {line_number}: {matches}")

    # In case of multiple matches, return the one with the smallest range that contains the line
    return min(matches, key=lambda d: d.info.range.finish.line - d.info.range.start.line)


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
