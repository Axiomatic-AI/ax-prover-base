"""Utilities for parsing Lean code structure and declarations."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from lean_interact import Command, FileCommand
from lean_interact.interface import DeclarationInfo, Sorry

from ..models.declaration import Declaration
from .lean_interact import LeanInteractServer
from .logging import get_logger

logger = get_logger(__name__)


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
    response = await server.run(FileCommand(path=str(file_path), declarations=True))
    return _get_declarations_with_sorries(response.declarations, response.sorries)


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


def read_declaration_source_code(declaration: Declaration, file_path: Path) -> str:
    """Read the source code of a declaration from a file."""
    with open(file_path) as file:
        lines = file.readlines()
    # Lines in code are 1-indexed, so it's important to enumerate from 1. Each line read from the
    # file already has its trailing newline.
    return "".join(
        line for line_number, line in enumerate(lines, 1) if declaration.contains_line(line_number)
    )
