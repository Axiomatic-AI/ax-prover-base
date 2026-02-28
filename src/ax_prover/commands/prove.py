"""Prover command implementation for ax-prover CLI."""

from datetime import datetime
from pathlib import Path

from ..config import Config
from ..models import ProverAgentState, ProverOutput, TargetItem
from ..tools.lean_search import lean_search_session_manager
from ..utils import get_logger, parse_prove_target, prove_single_item, write_json_output

logger = get_logger(__name__)


async def prove(
    folder: str,
    target: str,
    config: Config,
    overwrite: bool = False,
    output_file: str | None = None,
) -> int:
    """Run the prover agent on items from a plan, specific theorem, or all unproven in a file.

    Args:
        folder: Base folder path
        target: Location string, or file/module path (supports #L<line> suffix)
        config: Configuration object
        overwrite: Whether to re-prove already proven items
        output_file: File path to write JSON output
    """
    base_path = str(Path(folder).resolve())

    try:
        items_to_prove = parse_prove_target(base_path, target)
    except ValueError as e:
        logger.error(str(e))
        return 1

    if not items_to_prove:
        return 1

    return await _prove_all_items(base_path, items_to_prove, config, overwrite, output_file)


async def _prove_all_items(
    folder: str,
    items: list[TargetItem],
    config: Config,
    overwrite: bool,
    output_file: str | None = None,
) -> int:
    """Prove all items in the list."""
    async with lean_search_session_manager():
        failed = False
        outputs: dict[str, ProverOutput] = {}

        for item in items:
            if item.proven and not overwrite:
                logger.info(f"Already proven: {item.location.formatted_context}")
                continue

            key = item.location.formatted_context

            try:
                result_state = await _prove_item(config, folder, item)

                if not result_state.item.proven:
                    failed = True

                outputs[key] = ProverOutput.from_prover_state(result_state)

            except Exception as e:
                logger.exception(f"Error proving {key}")
                failed = True
                outputs[key] = ProverOutput.from_exception(e)
                if not output_file:
                    raise

        if output_file:
            write_json_output(outputs, output_file)

        return 1 if failed else 0


async def _prove_item(
    config: Config,
    folder: str,
    item: TargetItem,
) -> ProverAgentState:
    """Prove a single item."""
    logger.info(f"Proving: {item.location.formatted_context}")

    # Generate unique thread_id for this proving session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread_id = f"prove_{item.location.name}_{timestamp}"
    logger.debug(f"Using thread_id: {thread_id}")

    result = await prove_single_item(config, folder, item, thread_id=thread_id)

    if result.item.proven:
        logger.info(f"✓ Proven: {result.item.location.formatted_context}")
    else:
        logger.warning(f"✗ Not proven: {result.item.location.formatted_context}")

    return result
