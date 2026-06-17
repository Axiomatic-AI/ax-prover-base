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
    find_declaration_at_line,
    find_declaration_by_name,
    list_declarations_from_file,
    read_declaration_source_code,
)
from .logging import get_logger

if TYPE_CHECKING:
    from ..prover.agent import ProverAgent

logger = get_logger(__name__)


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
        return [_make_target_item(file_path, location_str, declaration)]

    if name is not None:
        declaration = find_declaration_by_name(declarations, name)
        if declaration is None:
            raise ValueError(f"No declaration found with name {name} in {file_path}")
        return [_make_target_item(file_path, location_str, declaration)]

    # No line or name specified, so prove all unproven declarations in the file
    unproven = [declaration for declaration in declarations if declaration.sorries]
    if not unproven:
        logger.info(f"No unproven declarations found in {file_path}")
        return []

    logger.info(
        f"Found {len(unproven)} unproven declarations in {file_path}: "
        f"{', '.join(declaration.name for declaration in unproven)}"
    )

    return [_make_target_item(file_path, location_str, declaration) for declaration in unproven]


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


def _make_target_item(file_path: Path, location_str: str, declaration: Declaration) -> TargetItem:
    """Make a target item from a location string and a declaration."""
    location = Location.parse(f"{location_str}:{declaration.name}")
    source_code = read_declaration_source_code(declaration, file_path)
    return TargetItem(
        location=location, original_source=source_code, is_proven=not declaration.sorries
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
