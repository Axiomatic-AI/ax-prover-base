"""Prover command implementation for ax-prover CLI."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..blueprint import BlueprintOptions, BlueprintOrchestrator
from ..config import Config
from ..models import ProverAgentState, ProverOutput, TargetItem
from ..prover.agent import ProverAgent
from ..runtime import Runtime
from ..tools import create_tool_lifespans
from ..utils import get_logger, parse_prove_target, prove_single_item, write_json_output

logger = get_logger(__name__)


@dataclass
class BlueprintOverrides:
    """CLI switches that override `config.blueprint` for one invocation."""

    enabled: bool = False
    resume: bool = False
    restart: bool = False
    max_refinements: int | None = None
    max_node_agents: int | None = None
    max_lean_compiles: int | None = None
    checkpoint_dir: str | None = None
    require_comparator: bool = False

    def apply(self, config: Config) -> None:
        """Fold the overrides into the config's blueprint section."""
        blueprint = config.blueprint
        if self.enabled:
            blueprint.enabled = True
        if self.max_refinements is not None:
            blueprint.max_refinement_rounds = self.max_refinements
        if self.max_node_agents is not None:
            blueprint.max_node_agents = self.max_node_agents
        if self.max_lean_compiles is not None:
            blueprint.max_lean_compiles = self.max_lean_compiles
        if self.checkpoint_dir is not None:
            blueprint.checkpoint_dir = self.checkpoint_dir
        if self.require_comparator:
            blueprint.require_comparator = True


async def prove(
    folder: str,
    target: str,
    config: Config,
    overwrite: bool = False,
    output_file: str | None = None,
    blueprint: BlueprintOverrides | None = None,
) -> int:
    """Run the prover agent on items from a plan, specific theorem, or all unproven in a file.

    Args:
        folder: Base folder path
        target: Location string, or file/module path (supports #L<line> suffix)
        config: Configuration object
        overwrite: Whether to re-prove already proven items
        output_file: File path to write JSON output
        blueprint: Blueprint-mode CLI overrides; blueprint mode runs when enabled
    """
    base_path = str(Path(folder).resolve())

    blueprint = blueprint or BlueprintOverrides()
    blueprint.apply(config)

    return await _prove_all_items(base_path, target, config, overwrite, output_file, blueprint)


async def _prove_all_items(
    folder: str,
    target: str,
    config: Config,
    overwrite: bool,
    output_file: str | None = None,
    blueprint: BlueprintOverrides | None = None,
) -> int:
    """Prove all items in the list."""
    blueprint = blueprint or BlueprintOverrides()
    use_blueprint = config.blueprint.enabled

    tool_configs = config.blueprint.prover_tools if use_blueprint else config.prover.proposer_tools
    tool_lifespans = await create_tool_lifespans(tool_configs)

    async with Runtime.open(config.runtime, folder, tool_lifespans) as rt:
        try:
            items = await parse_prove_target(rt.lean_interact_server, folder, target)
        except ValueError as e:
            logger.error(str(e))
            return 1

        if not items:
            logger.warning(f"No items to prove in {target}")
            return 1

        orchestrator = (
            await BlueprintOrchestrator.create(config.blueprint, rt) if use_blueprint else None
        )

        failed = False
        outputs: dict[str, ProverOutput] = {}

        for item in items:
            if item.is_proven and not overwrite:
                logger.info(f"Already proven: {item.location.formatted_context}")
                continue

            key = item.location.formatted_context

            try:
                if orchestrator is not None:
                    output = await _prove_item_blueprint(orchestrator, item, blueprint)
                else:
                    result_state = await _prove_item(config, rt, item)
                    output = ProverOutput.from_prover_state(result_state)

                if not output.success:
                    failed = True

                outputs[key] = output

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
    runtime: Runtime,
    item: TargetItem,
) -> ProverAgentState:
    """Prove a single item."""
    logger.info(f"Proving: {item.location.formatted_context}")

    # Generate unique thread_id for this proving session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread_id = f"prove_{item.location.name}_{timestamp}"
    logger.debug(f"Using thread_id: {thread_id}")

    prover = await ProverAgent.create(config=config.prover, runtime=runtime)

    result = await prove_single_item(prover, item, thread_id=thread_id)

    if result.item.is_proven:
        logger.info(f"✓ Proven: {result.item.location.formatted_context}")
    else:
        logger.warning(f"✗ Not proven: {result.item.location.formatted_context}")

    return result


async def _prove_item_blueprint(
    orchestrator: BlueprintOrchestrator,
    item: TargetItem,
    overrides: BlueprintOverrides,
) -> ProverOutput:
    """Prove a single item through the blueprint pipeline."""
    logger.info(f"Proving (blueprint): {item.location.formatted_context}")

    result = await orchestrator.prove(
        item, BlueprintOptions(resume=overrides.resume, restart=overrides.restart)
    )

    if result.is_success:
        logger.info(f"✓ {result.status}: {result.target} (comparator: {result.comparator_status})")
    else:
        logger.warning(f"✗ {result.status}: {result.target}: {result.error}")

    return ProverOutput(
        success=result.is_success,
        error=result.error or None,
        summary=(
            f"{result.status}: {len(result.node_records)} node(s), "
            f"{result.refinement_rounds} refinement round(s), "
            f"{result.reused_proofs} reused proof(s)"
        ),
        details=result.model_dump(mode="json"),
    )
