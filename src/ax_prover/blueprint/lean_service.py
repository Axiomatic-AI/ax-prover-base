"""Serialized Lean compilation for blueprint mode, backed by one warm REPL.

Measured on a Mathlib project: a fresh `lake env lean` per candidate costs a ~39s median,
while resubmitting the full source to a warm `AutoLeanServer` costs ~0.02s, because
LeanInteract reuses the elaborated state at the longest matching command prefix. The prefix
(imports, trusted context, parent signatures) is byte-identical across a node's attempts, so
only the candidate declaration is elaborated.

Two limits are deliberately separate:

- `max_node_agents` - how many node agents reason, search, and wait on the model at once.
- `max_lean_compiles` - how many compilations run at once, default 1.

Several agents therefore make progress in parallel while their `lean_compile` calls queue
against a single warm server. Competing Mathlib environments are the thing to avoid: each
needs ~2GB resident, and a restart costs a full 80-130s re-elaboration.
"""

import asyncio
import itertools
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum

from lean_interact import Command
from lean_interact.interface import CommandResponse

from ..models.declaration import Declaration
from ..utils.lean_interact import LeanInteractServer
from ..utils.lean_parsing import _bundle_declarations
from ..utils.logging import get_logger

logger = get_logger(__name__)

#: `#print axioms` emits `'name' depends on axioms: [a, b]` as an info message.
_AXIOM_REPORT = re.compile(r"depends on axioms:\s*\[(?P<axioms>[^\]]*)\]")

#: Any proof reaching a `sorry`, by any route, ends up depending on this.
SORRY_AXIOM = "sorryAx"


class CompilePriority(IntEnum):
    """Queue ordering. Architect and refiner work unblocks the whole graph, so it wins."""

    STRUCTURAL = 0  # architect / refiner skeletons
    NODE = 1  # node prover candidates
    FINAL = 2  # final assembled verification


class CompileCancelled(Exception):
    """A queued compilation was cancelled because its node was solved or invalidated."""


@dataclass(frozen=True)
class CompileOutcome:
    """Everything one compilation yields, from a single REPL response."""

    success: bool
    output: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    declarations: tuple[Declaration, ...] = ()
    sorries: tuple[object, ...] = ()
    axioms: tuple[str, ...] = ()
    disallowed_axioms: tuple[str, ...] = ()
    elapsed_s: float = 0.0

    @property
    def depends_on_sorry(self) -> bool:
        """True when the checked declaration reaches a `sorry` by any route."""
        return SORRY_AXIOM in self.axioms


@dataclass
class CompileStats:
    """Observability for the queue: volume, latency, and warm-path effectiveness."""

    submitted: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    warm_calls: int = 0
    total_seconds: float = 0.0
    warmups: int = 0
    warmup_seconds: float = 0.0
    per_node: dict[str, int] = field(default_factory=dict)

    @property
    def mean_warm_seconds(self) -> float:
        """Mean seconds per compilation excluding warm-up."""
        return round(self.total_seconds / self.warm_calls, 4) if self.warm_calls else 0.0

    def as_dict(self) -> dict:
        """Flat summary for run artifacts."""
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "mean_warm_seconds": self.mean_warm_seconds,
            "total_seconds": round(self.total_seconds, 1),
            "warmups": self.warmups,
            "warmup_seconds": round(self.warmup_seconds, 1),
            "per_node": dict(self.per_node),
        }


@dataclass(order=True)
class _Request:
    """One queued compilation, ordered by priority then submission order."""

    priority: int
    sequence: int
    source: str = field(compare=False)
    node_id: str = field(compare=False, default="")
    check_axioms_of: str | None = field(compare=False, default=None)
    allowed_axioms: frozenset[str] = field(compare=False, default=frozenset())
    future: asyncio.Future = field(compare=False, default=None)


def parse_axioms(response: CommandResponse) -> tuple[str, ...]:
    """Extract the axiom list `#print axioms` reported, if the command included one."""
    for message in getattr(response, "messages", None) or []:
        match = _AXIOM_REPORT.search(message.data)
        if match:
            listed = [a.strip() for a in match.group("axioms").split(",") if a.strip()]
            return tuple(listed)
    return ()


def split_messages(response: CommandResponse) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a response's messages into errors and warnings."""
    messages = getattr(response, "messages", None) or []
    errors = tuple(m.data for m in messages if m.severity == "error")
    warnings = tuple(m.data for m in messages if m.severity == "warning")
    return errors, warnings


class LeanCompileService:
    """A fair, serialized compile queue over one warm LeanInteract server.

    The stable prefix is warmed once and marked for replay, so a restart restores the
    common environment rather than a pile of obsolete graph branches.
    """

    def __init__(
        self,
        server: LeanInteractServer,
        max_lean_compiles: int = 1,
        timeout: float | None = None,
    ):
        self.server = server
        self.max_lean_compiles = max(1, max_lean_compiles)
        self.timeout = timeout
        self.stats = CompileStats()

        self._queue: asyncio.PriorityQueue[_Request] = asyncio.PriorityQueue()
        self._sequence = itertools.count()
        self._workers: list[asyncio.Task] = []
        self._cancelled_nodes: set[str] = set()
        self._warmed_prefix: str | None = None
        self._started = False

    async def start(self) -> None:
        """Start the worker tasks that drain the queue."""
        if self._started:
            return
        self._started = True
        self._workers = [
            asyncio.create_task(self._worker(i)) for i in range(self.max_lean_compiles)
        ]

    async def aclose(self) -> None:
        """Stop the workers, failing anything still queued."""
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._workers = []
        self._started = False

    async def warm(self, stable_prefix: str) -> float:
        """Elaborate the stable prefix once and mark it for replay after a restart.

        Only this common environment enters the session cache; candidate commands never do.
        Re-warming a prefix that is already current is a no-op.
        """
        if self._warmed_prefix == stable_prefix:
            return 0.0

        start = time.monotonic()
        await self.server.run(Command(cmd=stable_prefix), add_to_session_cache=True)
        elapsed = time.monotonic() - start

        self._warmed_prefix = stable_prefix
        self.stats.warmups += 1
        self.stats.warmup_seconds += elapsed
        logger.info(f"Warmed the Lean environment in {elapsed:.1f}s")
        return elapsed

    def invalidate(self, reason: str) -> None:
        """Drop the warm prefix when the trusted environment changes.

        Imports, toolchain, lake options, the original context, and parent signatures all
        invalidate reuse. A changed proof body does not.
        """
        logger.info(f"Invalidating the warm Lean environment: {reason}")
        self._warmed_prefix = None

    def cancel_node(self, node_id: str) -> None:
        """Cancel a node's queued compilations, once it is solved or invalidated."""
        self._cancelled_nodes.add(node_id)

    def resume_node(self, node_id: str) -> None:
        """Allow a previously cancelled node to queue work again."""
        self._cancelled_nodes.discard(node_id)

    async def compile(
        self,
        source: str,
        node_id: str = "",
        priority: CompilePriority = CompilePriority.NODE,
        check_axioms_of: str | None = None,
        allowed_axioms: frozenset[str] = frozenset(),
    ) -> CompileOutcome:
        """Queue a compilation and await its outcome.

        Args:
            source: The complete module source. Always send the full text: LeanInteract
                finds the reuse point itself, and splitting imports from the proof defeats
                it.
            node_id: Owning node, for per-node accounting and cancellation.
            priority: Structural work (architect, refiner) outranks node candidates.
            check_axioms_of: Fully qualified declaration to run `#print axioms` on.
            allowed_axioms: Axioms the declaration may legitimately depend on - the
                permitted list plus the placeholder axioms standing in for proven parents.
                Anything else, `sorryAx` included, fails the compile even when the compiler
                reported no error.

        Raises:
            CompileCancelled: The node was cancelled before this request ran.
        """
        await self.start()

        if node_id and node_id in self._cancelled_nodes:
            raise CompileCancelled(f"node {node_id!r} was cancelled")

        request = _Request(
            priority=int(priority),
            sequence=next(self._sequence),
            source=source,
            node_id=node_id,
            check_axioms_of=check_axioms_of,
            allowed_axioms=frozenset(allowed_axioms),
            future=asyncio.get_running_loop().create_future(),
        )

        self.stats.submitted += 1
        if node_id:
            self.stats.per_node[node_id] = self.stats.per_node.get(node_id, 0) + 1

        await self._queue.put(request)
        return await request.future

    async def _worker(self, index: int) -> None:
        """Drain the queue, one compilation at a time per worker."""
        while True:
            request = await self._queue.get()
            try:
                if request.node_id and request.node_id in self._cancelled_nodes:
                    self.stats.cancelled += 1
                    request.future.set_exception(
                        CompileCancelled(f"node {request.node_id!r} was cancelled")
                    )
                    continue

                outcome = await self._run(request)
                self.stats.completed += 1
                if not request.future.done():
                    request.future.set_result(outcome)

            except asyncio.CancelledError:
                if not request.future.done():
                    request.future.set_exception(CompileCancelled("the service was closed"))
                raise
            except Exception as e:  # noqa: BLE001 - delivered to the awaiting caller
                self.stats.failed += 1
                logger.warning(f"Compile worker {index} error: {e}")
                if not request.future.done():
                    request.future.set_exception(e)
            finally:
                self._queue.task_done()

    async def _run(self, request: _Request) -> CompileOutcome:
        """Compile one request, harvesting everything from a single response."""
        source = request.source
        if request.check_axioms_of:
            source = f"{source.rstrip()}\n\n#print axioms {request.check_axioms_of}\n"

        start = time.monotonic()
        response = await self.server.run(
            Command(cmd=source, declarations=True, all_tactics=True), timeout=self.timeout
        )
        elapsed = time.monotonic() - start

        self.stats.warm_calls += 1
        self.stats.total_seconds += elapsed

        if not isinstance(response, CommandResponse):
            detail = getattr(response, "message", None) or str(response)
            return CompileOutcome(
                success=False,
                output=f"Lean server error: {detail}",
                errors=(str(detail),),
                elapsed_s=elapsed,
            )

        errors, warnings = split_messages(response)
        axioms = parse_axioms(response)
        declarations = tuple(
            _bundle_declarations(
                response.declarations or [], response.sorries or [], response.tactics or []
            )
        )

        # An unexpected axiom means the declaration did not really prove the statement,
        # even when the compiler reported no error.
        disallowed = ()
        if request.check_axioms_of:
            disallowed = tuple(a for a in axioms if a not in request.allowed_axioms)
        success = not errors and not disallowed

        output = format_diagnostics(errors, warnings, disallowed, request.check_axioms_of)
        return CompileOutcome(
            success=success,
            output=output,
            errors=errors,
            warnings=warnings,
            declarations=declarations,
            sorries=tuple(response.sorries or []),
            axioms=axioms,
            disallowed_axioms=disallowed,
            elapsed_s=elapsed,
        )


def format_diagnostics(
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    disallowed_axioms: tuple[str, ...],
    checked: str | None,
) -> str:
    """Render model-facing diagnostics, keeping compiler text verbatim."""
    if disallowed_axioms:
        listed = ", ".join(f"`{a}`" for a in disallowed_axioms)
        if SORRY_AXIOM in disallowed_axioms:
            return (
                f"Rejected: {checked or 'the declaration'} depends on {listed}, so it does "
                "not actually prove the statement. Every step needs a real proof."
            )
        return f"Rejected: {checked or 'the declaration'} depends on disallowed axioms: {listed}."
    if errors:
        return "\n".join(errors)
    if warnings:
        return "Compiled successfully.\n\nWarnings:\n" + "\n".join(warnings)
    return "Compiled successfully."
