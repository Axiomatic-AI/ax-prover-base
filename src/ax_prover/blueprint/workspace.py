"""Isolated temporary Lean workspaces.

Every skeleton compile, node attempt, and final verification happens in a scratch file
created beside the user's target file, so imports and `lake env lean` resolve exactly as
they do for the real module. The user's source is never written during a run; the single
atomic edit happens in `assembly.commit_source`.
"""

import asyncio
import hashlib
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_PERMITTED_AXIOMS, LeanConfig
from ..models.declaration import Declaration
from ..models.files import Location
from ..utils.build import check_lean_file
from ..utils.lean_interact import LeanInteractServer
from ..utils.logging import get_logger
from .extraction import extract_declarations, statement_slice
from .lean_service import CompilePriority, LeanCompileService
from .metadata import render_docstring
from .models import TARGET_NODE_ID, BlueprintNode, NodeMetadata

logger = get_logger(__name__)

_ANONYMOUS_NAMESPACE = "[anonymous]"
_DECLARATION_KEYWORD = re.compile(r"^\s*(theorem|lemma)\b")
_TRAILING_DOC = re.compile(r"/--.*?-/[ \t]*\n?\s*$", re.DOTALL)
_NON_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class CompileResult:
    """Outcome of compiling one scratch module.

    `declarations`, `sorries`, and `axioms` are populated on the warm REPL path, where a
    single response carries them, so no second parsing pass is needed after success.
    """

    success: bool
    output: str
    source: str
    declarations: tuple = ()
    sorries: tuple = ()
    axioms: tuple = ()
    elapsed_s: float = 0.0


def generated_namespace(location: Location) -> str:
    """Deterministic, collision-isolated namespace name for a target's helpers."""
    digest = hashlib.sha256(location.formatted_context.encode("utf-8")).hexdigest()[:8]
    safe_name = _NON_IDENTIFIER.sub("_", location.name)
    return f"AxProverGenerated_{safe_name}_{digest}"


class BlueprintWorkspace:
    """Owns the target's file context and every scratch compilation for one run."""

    def __init__(
        self,
        base_folder: str,
        location: Location,
        target_declaration: Declaration,
        lean_config: LeanConfig,
        semaphore: asyncio.Semaphore,
        trusted_declarations: list[Declaration] | None = None,
        compile_service: "LeanCompileService | None" = None,
    ):
        self.base_folder = base_folder
        # When present, candidate compiles go through one warm REPL instead of a fresh
        # `lake env lean` per attempt: ~0.02s versus a ~39s median on a Mathlib project.
        self.compile_service = compile_service
        self.location = location
        self.lean_config = lean_config
        self.semaphore = semaphore

        self.file_path = location.absolute_path(base_folder)
        self.original_source = self.file_path.read_text(encoding="utf-8")
        self.source_hash = hashlib.sha256(self.original_source.encode("utf-8")).hexdigest()

        info = target_declaration.info
        self.target_lean_name = info.full_name
        self.target_signature = info.signature.pp
        self.target_statement_with_doc, self.target_statement = statement_slice(
            self.original_source, info
        )

        lines = self.original_source.splitlines(keepends=True)
        # `range` covers the docstring too, so slicing by line cleanly splits the file
        # around the whole target declaration.
        self.prefix = _TRAILING_DOC.sub("", "".join(lines[: info.range.start.line - 1]))
        self.suffix = "".join(lines[info.range.finish.line :])

        # Names the user's file already declares. Everything else at top level in a
        # skeleton is something the architect added outside the generated namespace.
        self.trusted_names = frozenset(
            declaration.info.full_name for declaration in (trusted_declarations or [])
        ) - {info.full_name}

        curr_namespace = info.scope.curr_namespace
        self.enclosing_namespace = "" if curr_namespace == _ANONYMOUS_NAMESPACE else curr_namespace
        self.namespace = generated_namespace(location)
        self.namespace_full = (
            f"{self.enclosing_namespace}.{self.namespace}"
            if self.enclosing_namespace
            else self.namespace
        )

    def render_parent_placeholder(self, parent: BlueprintNode) -> str:
        """Render a proven parent as a named axiom standing in for its proof.

        An `axiom` rather than `:= sorry` keeps the placeholder out of `sorryAx`, so the
        post-compile axiom check can tell "used a proven parent" apart from "left a hole".
        The placeholder names are on the allow-list; `sorryAx` never is.
        """
        statement = parent.statement_source_no_doc.rstrip()
        return _DECLARATION_KEYWORD.sub(" axiom", statement, count=1).lstrip()

    def allowed_axioms(self, parents: tuple[BlueprintNode, ...]) -> frozenset[str]:
        """Axioms a node's proof may legitimately depend on."""
        return (
            frozenset(DEFAULT_PERMITTED_AXIOMS)
            | {parent.short_name for parent in parents}
            | {parent.lean_name for parent in parents}
        )

    def render_helper_block(self, helpers: str) -> str:
        """Wrap architect-authored helper source in the deterministic namespace."""
        body = helpers.strip()
        if not body:
            return ""
        return f"namespace {self.namespace}\n\n{body}\n\nend {self.namespace}\n"

    def render_target(self, parents: tuple[str, ...], proof_plan: str, proof_body: str) -> str:
        """Render the immutable target with harness-owned blueprint metadata.

        The statement text comes from the user's file, never from a model response, so the
        target cannot drift.
        """
        metadata = NodeMetadata(version=1, id=TARGET_NODE_ID, parents=tuple(parents))
        docstring = render_docstring(metadata, proof_plan)
        return f"{docstring}\n{self.target_statement.rstrip()} := {proof_body.strip()}\n"

    def render_skeleton(
        self,
        helpers: str,
        target_parents: tuple[str, ...] = (),
        target_proof_plan: str = "",
    ) -> str:
        """Render a full placeholder skeleton: helpers plus the sorried target."""
        blocks = [self.prefix.rstrip(), self.render_helper_block(helpers)]
        blocks.append(self.render_target(target_parents, target_proof_plan, "by sorry"))
        return "\n\n".join(block for block in blocks if block.strip()) + "\n"

    def render_node_module(
        self,
        node: BlueprintNode,
        parents: tuple[BlueprintNode, ...],
        proof_body: str,
    ) -> str:
        """Render the isolated scratch module for one node attempt.

        Contains the trusted file prefix, placeholder statements for the node's direct
        parents only, and the node itself. Unrelated generated siblings and parent proof
        bodies are deliberately absent.
        """
        placeholders = "\n\n".join(self.render_parent_placeholder(p) for p in parents)

        if node.is_target:
            blocks = [self.prefix.rstrip(), self.render_helper_block(placeholders)]
            blocks.append(self.render_target(node.parents, "", proof_body))
        else:
            inner = f"{placeholders}\n\n{node.render(proof_body, include_doc=False)}".strip()
            blocks = [self.prefix.rstrip(), self.render_helper_block(inner)]

        return "\n\n".join(block for block in blocks if block.strip()) + "\n"

    def render_final_file(self, helper_block: str, target: str) -> str:
        """Render the complete candidate file: prefix, helpers, target, original suffix."""
        blocks = [self.prefix.rstrip(), helper_block.strip(), target.strip(), self.suffix.strip()]
        return "\n\n".join(block for block in blocks if block) + "\n"

    @asynccontextmanager
    async def scratch_file(self, source: str, label: str) -> AsyncIterator[Path]:
        """Write `source` to a uniquely named module beside the target file, then remove it.

        The scratch file must live in the project tree so `lake env lean` and the Lean
        interact server resolve the same imports the real module does.
        """
        safe_label = _NON_IDENTIFIER.sub("_", label)
        scratch = self.file_path.parent / f"tmp_bp_{safe_label}_{uuid.uuid4().hex[:8]}.lean"
        try:
            scratch.write_text(source, encoding="utf-8")
            yield scratch
        finally:
            scratch.unlink(missing_ok=True)

    def _clean_output(self, output: str, scratch: Path) -> str:
        """Replace the random scratch path with a stable placeholder for model feedback."""
        relative = str(scratch.relative_to(self.base_folder))
        return output.replace(relative, "<scratch>.lean").replace(scratch.name, "<scratch>.lean")

    @property
    def stable_prefix(self) -> str:
        """The common environment worth warming once: imports, options, trusted context.

        Node-specific parent signatures are deliberately excluded. They differ per node, and
        the incremental cache extends the common branch for each of them on demand, so only
        this shared prefix belongs in the crash-replay cache.
        """
        return self.prefix.rstrip() + "\n"

    async def compile_candidate(
        self,
        source: str,
        node_id: str = "",
        priority: "CompilePriority | None" = None,
        check_axioms_of: str | None = None,
        allowed_axioms: frozenset[str] = frozenset(),
        label: str = "candidate",
    ) -> CompileResult:
        """Compile a candidate through the warm REPL, falling back to a subprocess.

        `check_axioms_of` runs `#print axioms` on that declaration in the same command, so a
        proof reaching a `sorry` by any route is rejected even when the compiler reports no
        error.
        """
        if self.compile_service is None:
            return await self.compile_source(source, label=label)

        # Warm only the server this request will land on; the pool leases stickily.
        await self.compile_service.warm(
            self.stable_prefix, index=self.compile_service.lease(node_id)
        )
        outcome = await self.compile_service.compile(
            source,
            node_id=node_id,
            priority=priority or CompilePriority.NODE,
            check_axioms_of=check_axioms_of,
            allowed_axioms=allowed_axioms,
        )
        return CompileResult(
            success=outcome.success,
            output=outcome.output,
            source=source,
            declarations=outcome.declarations,
            sorries=outcome.sorries,
            axioms=outcome.axioms,
            elapsed_s=outcome.elapsed_s,
        )

    async def compile_source(self, source: str, label: str = "scratch") -> CompileResult:
        """Compile `source` with a fresh `lake env lean`, the authoritative batch route.

        Used for final verification of the assembled file, and as the fallback when no warm
        service is available.
        """
        async with self.scratch_file(source, label) as scratch:
            success, output = await check_lean_file(
                self.base_folder,
                str(scratch.relative_to(self.base_folder)),
                self.lean_config,
                self.semaphore,
                show_warnings=False,
            )
            output = self._clean_output(output, scratch)

        return CompileResult(success=success, output=output.strip(), source=source)

    async def compile_and_extract(
        self, source: str, server: LeanInteractServer, label: str = "skeleton"
    ) -> tuple[CompileResult, list[Declaration]]:
        """Compile `source` and, when it compiles, extract its canonical declarations.

        Compilation and extraction run against the same scratch file, so the declarations
        always describe exactly the module that was checked.
        """
        async with self.scratch_file(source, label) as scratch:
            relative = str(scratch.relative_to(self.base_folder))
            success, output = await check_lean_file(
                self.base_folder,
                relative,
                self.lean_config,
                self.semaphore,
                show_warnings=False,
            )
            output = self._clean_output(output, scratch)

            declarations: list[Declaration] = []
            if success:
                declarations = await extract_declarations(server, scratch)

        return CompileResult(success=success, output=output.strip(), source=source), declarations

    def source_unchanged(self) -> bool:
        """True when the target file still matches its content hash from run start."""
        if not self.file_path.exists():
            return False
        current = hashlib.sha256(self.file_path.read_bytes()).hexdigest()
        return current == self.source_hash

    def environment_fingerprint(self) -> str:
        """Hash of the trusted context a proof was checked against.

        Covers the file prefix (imports, options, variables, preceding declarations), the
        Lean toolchain, the Lake manifest, and the generated namespace, so a change to any
        of them invalidates stored proofs.
        """
        base = Path(self.base_folder)
        parts = [self.prefix, self.namespace_full]
        for name in ("lean-toolchain", "lake-manifest.json"):
            candidate = base / name
            parts.append(candidate.read_text(encoding="utf-8") if candidate.exists() else "")
        return hashlib.sha256(" ".join(parts).encode("utf-8")).hexdigest()
