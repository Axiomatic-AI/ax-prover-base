"""Utilities for creating proving targets."""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import TargetItem
from ..models.declaration import Declaration
from ..models.files import Location
from ..models.proving import ProverAgentState
from .lean_interact import LeanInteractServer
from .lean_parsing import (
    count_pattern,
    find_declaration_at_line,
    find_declaration_by_name,
    get_function_from_location,
    get_unproven,
    list_declarations_from_file,
)
from .logging import get_logger

if TYPE_CHECKING:
    from ..prover.agent import ProverAgent

logger = get_logger(__name__)


def get_item_from_location(folder: str, location_str: str) -> TargetItem | None:
    """Create a TargetItem from a location string."""
    logger.info(f"Proving theorem at: {location_str}")

    try:
        location = Location.parse(location_str)
    except ValueError as e:
        logger.error(str(e))
        return None

    theorem_content = get_function_from_location(folder, location)
    if not theorem_content:
        logger.error(f"Theorem not found: {location.formatted_context}")
        return None

    sorry_count, _ = count_pattern(theorem_content, pattern=r"\b(sorry|admit)\b")
    logger.debug(f"Found theorem with {sorry_count} sorrie(s)")

    item = TargetItem(
        location=location,
        is_proven=sorry_count == 0,
    )
    return item


async def get_items_from_lean_file(
    server: LeanInteractServer, folder: str, target: str
) -> list[TargetItem]:
    """Get all unproven functions from a Lean file."""
    file_path = target if target.endswith(".lean") else target.replace(".", "/") + ".lean"

    if not (Path(folder) / file_path).exists():
        logger.error(f"File not found: {file_path}")
        return []

    unproven_names = await get_unproven(server, folder, file_path)
    if not unproven_names:
        logger.info(f"No unproven functions found in {file_path}")
        return []

    logger.info(
        f"Found {len(unproven_names)} unproven function(s) in {file_path}: {', '.join(unproven_names)}"
    )

    module_path = file_path.replace("/", ".").removesuffix(".lean")
    items = []
    for func_name in unproven_names:
        item = get_item_from_location(folder, f"{module_path}:{func_name}")
        if item:
            items.append(item)

    return items


def get_item_from_line(folder: str, target: str, line: int) -> TargetItem | None:
    """Create a TargetItem from a file path and line number."""
    file_path = target if target.endswith(".lean") else target.replace(".", "/") + ".lean"
    full_path = Path(folder) / file_path

    if not full_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    content = full_path.read_text(encoding="utf-8")
    decl_name = find_declaration_at_line(content, line)

    if not decl_name:
        logger.error(f"No declaration found at line {line} in {file_path}")
        return None

    module_path = file_path.replace("/", ".").removesuffix(".lean")
    location_str = f"{module_path}:{decl_name}"

    return get_item_from_location(folder, location_str)


_LINE_SUFFIX_RE = re.compile(r"#L(\d+)$")


async def parse_prove_target(
    server: LeanInteractServer, folder: str, target: str
) -> list[TargetItem]:
    """Parse a prove target string and return items to prove.

    Supports formats:
    - Module.Path:theorem_name
    - path/to/file.lean:theorem_name
    - Module.Path (all unproven)
    - path/to/file.lean (all unproven)
    - path/to/file.lean#L42 (theorem at line 42)

    Returns:
        List of items to prove.

    Raises:
        ValueError: If target is a location string that doesn't exist or
                    if #L<line> is used with incompatible targets.
    """
    location_str, name, line = _parse_target_components(target)

    file_path = _resolve_file_path(folder, location_str)
    declarations = await list_declarations_from_file(server, file_path)

    if line is not None:
        declaration = find_declaration_at_line(declarations, line)
        if declaration is None:
            raise ValueError(f"No declaration found at line {line} in {file_path}")
        return [_make_target_item(location_str, declaration)]

    if name is not None:
        declaration = find_declaration_by_name(declarations, name)
        if declaration is None:
            raise ValueError(f"No declaration found with name {name} in {file_path}")
        return [_make_target_item(location_str, declaration)]

    # No line or name specified, so prove all unproven declarations in the file
    unproven = [declaration for declaration in declarations if declaration.sorries]
    if not unproven:
        logger.info(f"No unproven declarations found in {file_path}")
        return []

    logger.info(
        f"Found {len(unproven)} unproven declarations in {file_path}: "
        f"{', '.join(declaration.name for declaration in unproven)}"
    )

    return [_make_target_item(location_str, declaration) for declaration in unproven]


def _parse_target_components(target: str) -> tuple[str, str | None, int | None]:
    """Parse the target string into a tuple of (location_part, name, line).

    Raises:
        ValueError when '#L<line>' is used with a named target.
    """
    line_match = _LINE_SUFFIX_RE.search(target)

    line: int | None = None
    if line_match:
        line = int(line_match.group(1))
        target = target[: line_match.start()]

    name: str | None = None
    if ":" in target:
        target, name = target.rsplit(":", 1)

    if line is not None and name is not None:
        # Either specify by line or by name, not both. Can even not provide any of them at all!
        raise ValueError("Cannot use #L<line> with a target that already specifies a name")

    return target, name, line


def _resolve_file_path(folder: str, location: str) -> Path:
    """Resolve a file path from a location string."""
    relative_path = location if location.endswith(".lean") else location.replace(".", "/") + ".lean"
    full_path = Path(folder) / relative_path

    if not full_path.exists():
        raise ValueError(f"File not found: {full_path}")

    return full_path


def _make_target_item(location_str: str, declaration: Declaration) -> TargetItem:
    """Make a target item from a location string and a declaration."""
    location = Location.parse(f"{location_str}:{declaration.name}")
    return TargetItem(
        location=location, original_source=declaration.code, is_proven=not declaration.sorries
    )


async def prove_single_item(
    prover: "ProverAgent",
    item: TargetItem,
    thread_id: str | None = None,
) -> ProverAgentState:
    """Prove a single item and return the full state."""
    initial_state = ProverAgentState(item=item)
    run_name = f"prove:{item.name}"
    return await prover.chat(initial_state, run_name=run_name, thread_id=thread_id)
