"""Utilities for creating proving targets."""

import re
from asyncio import Semaphore
from pathlib import Path

from ..config import Config
from ..models import TargetItem
from ..models.files import Location
from ..models.proving import ProverAgentState
from .lean_parsing import (
    count_sorries,
    find_declaration_at_line,
    get_function_from_location,
    get_unproven,
    normalize_location,
)
from .logging import get_logger

logger = get_logger(__name__)


def get_item_from_location(folder: str, location_str: str) -> TargetItem | None:
    """Create a TargetItem from a location string."""
    logger.info(f"Proving theorem at: {location_str}")

    try:
        location = Location.from_formatted_context(location_str)
    except ValueError as e:
        logger.error(str(e))
        return None

    theorem_content = get_function_from_location(folder, location)
    if not theorem_content:
        logger.error(f"Theorem not found: {location.formatted_context}")
        return None

    sorry_count, _ = count_sorries(theorem_content)
    logger.debug(f"Found theorem with {sorry_count} sorrie(s)")

    item = TargetItem(
        title=location.name,
        location=location,
        proven=sorry_count == 0,
    )
    return item


def get_items_from_lean_file(folder: str, target: str) -> list[TargetItem]:
    """Get all unproven functions from a Lean file."""
    file_path = target if target.endswith(".lean") else target.replace(".", "/") + ".lean"

    if not (Path(folder) / file_path).exists():
        logger.error(f"File not found: {file_path}")
        return []

    unproven_names = get_unproven(folder, file_path)
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


def parse_prove_target(folder: str, target: str) -> list[TargetItem]:
    """Parse a prove target string and return items to prove.

    Supports formats:
    - Module.Path:theorem_name
    - path/to/file.lean:theorem_name
    - Module.Path (all unproven)
    - path/to/file.lean (all unproven)
    - path/to/file.lean#L42 (theorem at line 42)

    Raises:
        ValueError: If target is a location string that doesn't exist or
                    if #L<line> is used with incompatible targets.
    """
    line_match = re.search(r"#L(\d+)$", target)
    if line_match:
        line = int(line_match.group(1))
        file_path = target[: line_match.start()]
        if ":" in file_path:
            raise ValueError("Cannot use #L<line> with a location that already specifies a theorem")
        item = get_item_from_line(folder, file_path, line)
        if not item:
            raise ValueError(f"No declaration found at line {line} in {file_path}")
        return [item]

    if ":" in target:
        item = get_item_from_location(folder, normalize_location(target))
        if not item:
            raise ValueError(f"Could not create item from location: {target}")
        return [item]
    return get_items_from_lean_file(folder, target)


async def prove_single_item(
    config: Config,
    folder: str,
    item: TargetItem,
    lean_semaphore: Semaphore | None = None,
    thread_id: str | None = None,
) -> ProverAgentState:
    """Prove a single item and return the full state."""
    prover = await config.create_prover(lean_semaphore=lean_semaphore, base_folder=folder)
    initial_state = ProverAgentState(item=item)
    run_name = f"prove:{item.title}"
    return await prover.chat(initial_state, run_name=run_name, thread_id=thread_id)
