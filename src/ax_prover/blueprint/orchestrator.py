"""End-to-end blueprint proving state machine.

    architect -> canonical graph -> frontier scheduling -> refinement -> assembly
    -> full build -> Comparator -> one atomic source edit

Every stage runs in a scratch workspace. The user's file is read at run start and written
exactly once, at the very end, only after every required check passes. Failed, cancelled,
and interrupted runs write nothing.
"""

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from ..config import BlueprintConfig
from ..models.proving import TargetItem
from ..runtime import Runtime
from ..utils.build import LeanBuildError
from ..utils.lean_interact import LeanInteractServer
from ..utils.lean_parsing import find_declaration_by_name
from ..utils.llm import LLMClient
from ..utils.logging import get_logger
from .assembly import (
    AssemblyError,
    artifact_path,
    check_generated_region,
    commit_source,
    render_assembly,
    render_helper_block,
)
from .comparator import run_comparator
from .generation import SkeletonCandidate, build_blueprint, generate_blueprint
from .lean_service import LeanCompileService
from .models import (
    TARGET_NODE_ID,
    Blueprint,
    BlueprintError,
    BlueprintRunResult,
    ComparatorStatus,
    RunStatus,
)
from .proof_store import ProofStore
from .refinement import refine_blueprint
from .scheduler import blocked_nodes, is_complete, run_schedule
from .tools import make_mathlib_search_tool
from .workspace import BlueprintWorkspace

logger = get_logger(__name__)


@dataclass
class BlueprintOptions:
    """Per-run switches the CLI exposes."""

    resume: bool = False
    restart: bool = False
    extra_context: str = ""


class BlueprintOrchestrator:
    """Runs one target through the blueprint pipeline."""

    def __init__(
        self,
        config: BlueprintConfig,
        runtime: Runtime,
        search_tool: BaseTool | None = None,
    ):
        self.config = config
        self.runtime = runtime
        self.search_tool = search_tool

        self.architect_role = config.role("architect")
        self.prover_role = config.role("prover")
        self.refiner_role = config.role("refiner")

        self._compile_stats: dict = {}
        self._extra_servers: list[LeanInteractServer] = []
        self.service = LeanCompileService(runtime.lean_interact_server)
        self.architect_client = LLMClient(self.architect_role.llm)
        self.prover_client = LLMClient(self.prover_role.llm)
        self.refiner_client = LLMClient(self.refiner_role.llm)

    @classmethod
    async def create(cls, config: BlueprintConfig, runtime: Runtime) -> "BlueprintOrchestrator":
        """Build an orchestrator, warming up the optional `mathlib_search` backend."""
        search_tool = await make_mathlib_search_tool(config.prover_tools, runtime)
        instance = cls(config, runtime, search_tool)
        instance._open_pool()
        return instance

    def _open_pool(self) -> None:
        """Create the run's Lean server pool, shared by every target this orchestrator proves.

        The pool is orchestrator-scoped, not per-target: a batch run proves many targets
        concurrently, and creating a pool per target would multiply servers by the batch
        concurrency and exhaust memory.
        """
        size = max(1, self.config.max_lean_compiles)
        self._extra_servers = [
            LeanInteractServer(self.runtime.base_folder, self.runtime.config.lean_interact)
            for _ in range(size - 1)
        ]
        # Slot 0 is the runtime's own server, so a single-server pool adds nothing.
        self.service = LeanCompileService([self.runtime.lean_interact_server, *self._extra_servers])
        if self._extra_servers:
            logger.info(f"Lean server pool: {size} servers ({len(self._extra_servers)} new)")

    def parse_server(self) -> LeanInteractServer:
        """The next pool server to parse a target file with, round-robin.

        Target parsing elaborates a whole Mathlib-importing file, and one server serializes
        internally, so routing every concurrent target at one server makes batch start-up
        serial. Round-robin also spreads the initial Mathlib load across the whole pool.

        Deliberately not the target's proving lease: that lease is keyed on `Module:name`,
        which is only known after parsing, so keying it on the dataset path would strand a
        warm prefix on a server the target never uses.
        """
        return self.service.servers[self.service.lease("")]

    async def aclose(self) -> None:
        """Shut down the compile queue and the servers this orchestrator created."""
        await self.service.aclose()
        for server in self._extra_servers:
            await server.aclose()
        self._extra_servers = []

    async def prove(
        self, item: TargetItem, options: BlueprintOptions | None = None
    ) -> BlueprintRunResult:
        """Prove one target end to end.

        Returns a result describing the outcome; exceptions are converted into an
        `infrastructure_error` result so batch runs never crash on one item.
        """
        options = options or BlueprintOptions()
        target = item.location.formatted_context

        try:
            return await self._prove(item, options)
        except BlueprintError as e:
            logger.error(f"Blueprint run failed for {target}: {e}")
            return BlueprintRunResult(status=RunStatus.FAILED, target=target, error=str(e))
        except LeanBuildError as e:
            logger.error(f"Lean infrastructure failure for {target}: {e}")
            return BlueprintRunResult(
                status=RunStatus.INFRASTRUCTURE_ERROR, target=target, error=str(e)
            )
        except Exception as e:  # noqa: BLE001 - one bad item must not abort a dataset run
            logger.exception(f"Unexpected error proving {target}")
            return BlueprintRunResult(
                status=RunStatus.INFRASTRUCTURE_ERROR, target=target, error=str(e)
            )

    async def _prove(self, item: TargetItem, options: BlueprintOptions) -> BlueprintRunResult:
        try:
            return await self._prove_with_service(item, options, self.service)
        finally:
            self.service.release(item.location.formatted_context)
            # Pool-wide, since the pool is shared across the batch. Per-node counts are
            # qualified by target so they stay attributable.
            self._compile_stats = self.service.stats.as_dict()

    async def _prove_with_service(
        self, item: TargetItem, options: BlueprintOptions, service: LeanCompileService
    ) -> BlueprintRunResult:
        target = item.location.formatted_context
        workspace = self._open_workspace(item, service)

        if len(service.servers) > 1:
            # Warm every server up front and in parallel: otherwise the first node to land
            # on a cold server pays a full Mathlib elaboration mid-round. A single server
            # is still warmed lazily, so a fully resumed run pays nothing.
            elapsed = await service.warm_all(workspace.stable_prefix)
            if elapsed:
                logger.info(f"Warmed {len(service.servers)} Lean servers in {elapsed:.1f}s")

        store = ProofStore.open(
            self.config.checkpoint_dir,
            target,
            source_hash=workspace.source_hash,
            resume=options.resume and not options.restart,
        )
        if options.restart:
            store.clear()

        candidate = await self._initial_blueprint(workspace, store, options)
        blueprint = candidate.blueprint
        environment = workspace.environment_fingerprint()
        reused = store.reconcile(blueprint, environment)

        for round_number in range(self.config.max_refinement_rounds + 1):
            report = await run_schedule(
                workspace,
                blueprint,
                store,
                self.prover_client,
                self.prover_role,
                self.search_tool,
                self.config.max_node_agents,
            )

            if report.infrastructure_error:
                return self._result(
                    RunStatus.INFRASTRUCTURE_ERROR,
                    target,
                    workspace,
                    blueprint,
                    store,
                    reused,
                    error=report.infrastructure_error,
                )

            if is_complete(blueprint, store):
                break

            if round_number >= self.config.max_refinement_rounds:
                blocked = ", ".join(blocked_nodes(blueprint, store))
                return self._result(
                    RunStatus.FAILED,
                    target,
                    workspace,
                    blueprint,
                    store,
                    reused,
                    error=f"refinement budget exhausted with unsolved nodes: {blocked}",
                )

            try:
                candidate = await refine_blueprint(
                    workspace,
                    self.runtime.lean_interact_server,
                    self.refiner_client,
                    self.refiner_role,
                    blueprint,
                    store.records,
                    round_number + 1,
                )
            except BlueprintError as e:
                blocked = ", ".join(blocked_nodes(blueprint, store))
                return self._result(
                    RunStatus.FAILED,
                    target,
                    workspace,
                    blueprint,
                    store,
                    reused,
                    error=f"refinement failed ({e}); unsolved nodes: {blocked}",
                )

            blueprint = candidate.blueprint
            # Parent signatures may have moved, so any queued compile is stale. The shared
            # prefix is unchanged, so the warm environment itself stays valid.
            for node_id in blocked_nodes(blueprint, store):
                service.cancel_node(node_id)
                service.resume_node(node_id)
            store.state.refinement_rounds = round_number + 1
            store.remember_skeleton(
                candidate.helpers, candidate.target_parents, candidate.target_proof_plan
            )
            reused = store.reconcile(blueprint, environment)

        return await self._finalize(target, workspace, blueprint, store, reused)

    def _open_workspace(
        self, item: TargetItem, service: LeanCompileService | None = None
    ) -> BlueprintWorkspace:
        """Build the run workspace from the target's already-parsed declarations."""
        declaration = find_declaration_by_name(item.original_declarations, item.location.name)
        if declaration is None:
            raise BlueprintError(
                f"declaration {item.location.name!r} not found in {item.location.path}"
            )

        return BlueprintWorkspace(
            base_folder=self.runtime.base_folder,
            location=item.location,
            target_declaration=declaration,
            lean_config=self.runtime.config.lean,
            semaphore=self.runtime.lean_semaphore,
            trusted_declarations=item.original_declarations,
            compile_service=service,
        )

    async def _initial_blueprint(
        self, workspace: BlueprintWorkspace, store: ProofStore, options: BlueprintOptions
    ) -> SkeletonCandidate:
        """Rebuild the checkpointed skeleton when resuming, otherwise call the architect."""
        if store.state.helpers:
            logger.info("Resuming from the checkpointed blueprint skeleton")
            blueprint = await build_blueprint(
                workspace,
                self.runtime.lean_interact_server,
                store.state.helpers,
                tuple(store.state.target_parents),
                store.state.target_proof_plan,
            )
            return SkeletonCandidate(
                blueprint=blueprint,
                helpers=store.state.helpers,
                target_parents=tuple(store.state.target_parents),
                target_proof_plan=store.state.target_proof_plan,
            )

        candidate = await generate_blueprint(
            workspace,
            self.runtime.lean_interact_server,
            self.architect_client,
            self.architect_role,
            options.extra_context,
        )
        store.remember_skeleton(
            candidate.helpers, candidate.target_parents, candidate.target_proof_plan
        )
        return candidate

    async def _finalize(
        self,
        target: str,
        workspace: BlueprintWorkspace,
        blueprint: Blueprint,
        store: ProofStore,
        reused: int,
    ) -> BlueprintRunResult:
        """Assemble, verify, judge, and perform the single atomic source edit."""
        proofs = store.solved_proofs()
        assembled = render_assembly(workspace, blueprint, proofs)

        problems = check_generated_region(workspace, assembled)
        if problems:
            return self._result(
                RunStatus.FAILED,
                target,
                workspace,
                blueprint,
                store,
                reused,
                error="; ".join(problems),
            )

        artifact_path(
            self.config.artifacts_dir, f"{workspace.location.name}_assembled.lean"
        ).write_text(assembled, encoding="utf-8")

        # The final gate deliberately uses a fresh `lake env lean`, independent of any
        # incremental REPL state.
        build = await workspace.compile_source(assembled, label="final")
        if not build.success:
            return self._result(
                RunStatus.FAILED,
                target,
                workspace,
                blueprint,
                store,
                reused,
                error=f"the assembled file does not compile:\n{build.output}",
            )

        helper_block = render_helper_block(workspace, blueprint, proofs)
        comparator = await run_comparator(
            workspace,
            blueprint,
            self.config.comparator,
            helper_block,
            proofs[TARGET_NODE_ID],
        )

        if comparator.status is ComparatorStatus.REJECTED:
            return self._result(
                RunStatus.FAILED,
                target,
                workspace,
                blueprint,
                store,
                reused,
                comparator_status=comparator.status,
                error=f"Comparator rejected the proof: {comparator.detail}\n{comparator.output}",
            )

        if comparator.status is ComparatorStatus.PENDING and self.config.require_comparator:
            return self._result(
                RunStatus.FAILED,
                target,
                workspace,
                blueprint,
                store,
                reused,
                comparator_status=comparator.status,
                error=f"--require-comparator was set but Comparator is unavailable: "
                f"{comparator.detail}",
            )

        try:
            commit_source(workspace, assembled)
        except AssemblyError as e:
            return self._result(
                RunStatus.FAILED,
                target,
                workspace,
                blueprint,
                store,
                reused,
                comparator_status=comparator.status,
                error=str(e),
            )

        status = (
            RunStatus.COMPARATOR_PENDING
            if comparator.status is ComparatorStatus.PENDING
            else RunStatus.SOLVED
        )
        return self._result(
            status,
            target,
            workspace,
            blueprint,
            store,
            reused,
            comparator_status=comparator.status,
            source_modified=True,
        )

    def _result(
        self,
        status: RunStatus,
        target: str,
        workspace: BlueprintWorkspace,
        blueprint: Blueprint,
        store: ProofStore,
        reused: int,
        comparator_status: ComparatorStatus = ComparatorStatus.SKIPPED,
        source_modified: bool = False,
        error: str = "",
    ) -> BlueprintRunResult:
        """Assemble the run result from current state."""
        if error:
            logger.warning(f"Blueprint run for {target}: {error}")

        return BlueprintRunResult(
            status=status,
            target=target,
            namespace=workspace.namespace_full,
            graph_size=len(blueprint.nodes),
            refinement_rounds=store.state.refinement_rounds,
            reused_proofs=reused,
            node_records=list(store.records.values()),
            comparator_status=comparator_status,
            source_modified=source_modified,
            error=error,
            compile_stats=dict(self._compile_stats),
        )
