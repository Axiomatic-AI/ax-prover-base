"""Utilities for parsing Lean code structure and declarations."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from lean_interact import Command, FileCommand
from lean_interact.interface import DeclarationInfo, Sorry, Tactic

from ..models.declaration import Declaration
from .lean_interact import LeanInteractServer
from .logging import get_logger

logger = get_logger(__name__)


def read_declaration_source_code(declaration: Declaration, file_path: Path) -> str:
    """Read the source code of a declaration from a file."""
    with open(file_path) as file:
        lines = file.readlines()
    # Lines in code are 1-indexed, so it's important to enumerate from 1. Each line read from the
    # file already has its trailing newline.
    return "".join(
        line for line_number, line in enumerate(lines, 1) if declaration.contains_line(line_number)
    )


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
    response = await server.run(Command(cmd=code, declarations=True, all_tactics=True))
    return _bundle_declarations(response.declarations, response.sorries, response.tactics)


async def list_declarations_from_file(
    server: LeanInteractServer, file_path: Path, all_tactics: bool = False
) -> list[DeclarationInfo]:
    """List all declarations from a file.

    Set `all_tactics=True` to also collect the tactics used in each declaration (needed for
    detecting search tactics). It makes the REPL response heavier, so leave it off when only
    declaration/sorry information is required.
    """
    response = await server.run(
        FileCommand(path=str(file_path), declarations=True, all_tactics=all_tactics)
    )
    return _bundle_declarations(response.declarations, response.sorries, response.tactics)


def _bundle_declarations(
    declaration_infos: list[DeclarationInfo], sorries: list[Sorry], tactics: list[Tactic]
) -> list[Declaration]:
    """Match the sorries with the declaration information from the lean interact response,
    and combine them into a single Declaration object."""
    declarations = []
    for declaration_info in declaration_infos:
        sorries_in_declaration = [
            sorry for sorry in sorries if _within_declaration_range(declaration_info, sorry)
        ]

        tactics_in_declaration = [
            tactic for tactic in tactics if _within_declaration_range(declaration_info, tactic)
        ]

        declarations.append(
            Declaration(
                info=declaration_info,
                sorries=sorries_in_declaration,
                tactics=tactics_in_declaration,
            )
        )

    return declarations


def _within_declaration_range(declaration_info: DeclarationInfo, object: Sorry | Tactic) -> bool:
    return (
        object.start_pos > declaration_info.range.start
        and object.start_pos < declaration_info.range.finish
    )
