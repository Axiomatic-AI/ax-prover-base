"""Checkpointed proof storage with interface fingerprinting and invalidation.

Solved proof bodies live outside the Lean source so refinement can rewire the graph
without discarding expensive work. A proof is reusable only while the interface it was
checked against is unchanged; the docstring is deliberately excluded from the fingerprint
so a prose-only refinement keeps every proof.
"""

import hashlib
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ..utils.logging import get_logger
from .graph import normalize_signature, transitive_descendants
from .models import Blueprint, BlueprintNode, NodeDiagnosis, NodeRecord, NodeStatus

logger = get_logger(__name__)

STORE_VERSION = 1

_NON_FILENAME = re.compile(r"[^A-Za-z0-9_.-]")


def node_fingerprint(
    blueprint: Blueprint, node: BlueprintNode, environment_fingerprint: str
) -> str:
    """Fingerprint the interface a node's proof was checked against."""
    by_id = blueprint.by_id
    payload = {
        "lean_name": node.lean_name,
        "signature": normalize_signature(node.signature),
        "parents": [
            {
                "id": parent_id,
                "signature": normalize_signature(by_id[parent_id].signature)
                if parent_id in by_id
                else "",
            }
            for parent_id in sorted(node.parents)
        ],
        "environment": environment_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ProofStoreState(BaseModel):
    """On-disk checkpoint contents."""

    version: int = STORE_VERSION
    target: str = ""
    source_hash: str = ""
    refinement_rounds: int = 0
    helpers: str = Field(default="", description="Architect-authored helper skeleton source")
    target_parents: list[str] = Field(default_factory=list)
    target_proof_plan: str = ""
    records: dict[str, NodeRecord] = Field(default_factory=dict)


def checkpoint_path(checkpoint_dir: str | Path, target: str) -> Path:
    """Path of the checkpoint file for a target, keyed by its formatted location."""
    return Path(checkpoint_dir) / f"{_NON_FILENAME.sub('_', target)}.json"


class ProofStore:
    """Atomic, resumable checkpoint of per-node proving state."""

    def __init__(self, path: Path, target: str, source_hash: str = ""):
        self.path = path
        self.state = ProofStoreState(target=target, source_hash=source_hash)

    @classmethod
    def open(
        cls,
        checkpoint_dir: str | Path,
        target: str,
        source_hash: str = "",
        resume: bool = False,
    ) -> "ProofStore":
        """Open the store for a target, loading prior state only when resuming."""
        store = cls(checkpoint_path(checkpoint_dir, target), target, source_hash)
        if resume:
            store.load()
        return store

    def load(self) -> None:
        """Load persisted state, ignoring an unreadable or stale-format checkpoint."""
        if not self.path.exists():
            return

        try:
            loaded = ProofStoreState.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring unreadable checkpoint {self.path}: {e}")
            return

        if loaded.version != STORE_VERSION:
            logger.warning(
                f"Ignoring checkpoint {self.path} with version {loaded.version} "
                f"(expected {STORE_VERSION})"
            )
            return

        self.state = loaded

    def save(self) -> None:
        """Persist state atomically so an interrupted run never leaves a partial file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    @property
    def records(self) -> dict[str, NodeRecord]:
        """Mutable per-node records keyed by blueprint id."""
        return self.state.records

    @property
    def statuses(self) -> dict[str, NodeStatus]:
        """Current status of every known node."""
        return {node_id: record.status for node_id, record in self.records.items()}

    def record(self, node_id: str) -> NodeRecord:
        """Record for a node, created as pending if absent."""
        return self.records.setdefault(node_id, NodeRecord(node_id=node_id))

    def solved_proofs(self) -> dict[str, str]:
        """Stored proof bodies of every solved node."""
        return {
            node_id: record.proof_body
            for node_id, record in self.records.items()
            if record.status is NodeStatus.SOLVED and record.proof_body is not None
        }

    def reconcile(self, blueprint: Blueprint, environment_fingerprint: str) -> int:
        """Align stored records with a (possibly refined) blueprint.

        Keeps a solved proof only when the node's fingerprint is unchanged, then
        invalidates the transitive descendants of every node that did change.

        Returns:
            The number of solved proofs preserved.
        """
        fingerprints = {
            node.id: node_fingerprint(blueprint, node, environment_fingerprint)
            for node in blueprint.nodes
        }

        reconciled: dict[str, NodeRecord] = {}
        invalidated: set[str] = set()

        for node_id, fingerprint in fingerprints.items():
            previous = self.records.get(node_id)
            unchanged = previous is not None and previous.fingerprint == fingerprint

            if unchanged and previous.status is NodeStatus.SOLVED and previous.proof_body:
                reconciled[node_id] = previous.model_copy(update={"reused": True})
            elif unchanged and previous.status is NodeStatus.FAILED:
                # The refiner saw this node's diagnosis and left it identical, so retrying
                # it would burn budget on the same statement. Keep the diagnosis instead.
                reconciled[node_id] = previous.model_copy()
            else:
                reconciled[node_id] = NodeRecord(node_id=node_id, fingerprint=fingerprint)
                if previous is not None and previous.status is NodeStatus.SOLVED:
                    invalidated.add(node_id)

        for node_id in invalidated:
            for descendant in transitive_descendants(blueprint, node_id):
                if reconciled[descendant].status is NodeStatus.SOLVED:
                    logger.info(f"Invalidating {descendant!r}: ancestor {node_id!r} changed")
                    reconciled[descendant] = NodeRecord(
                        node_id=descendant, fingerprint=fingerprints[descendant]
                    )

        self.state.records = reconciled
        reused = sum(1 for record in reconciled.values() if record.reused)
        self.save()
        return reused

    def mark_solved(self, node_id: str, proof_body: str, attempts: int) -> None:
        """Record a node as solved and persist immediately."""
        record = self.record(node_id)
        record.status = NodeStatus.SOLVED
        record.proof_body = proof_body
        record.attempts += attempts
        record.diagnosis = None
        self.save()

    def mark_failed(self, node_id: str, diagnosis: NodeDiagnosis | None, attempts: int) -> None:
        """Record a failed node with its diagnosis and persist immediately."""
        record = self.record(node_id)
        record.status = NodeStatus.FAILED
        record.attempts += attempts
        record.diagnosis = diagnosis
        self.save()

    def remember_skeleton(
        self, helpers: str, target_parents: tuple[str, ...], target_proof_plan: str
    ) -> None:
        """Persist the current skeleton so `--resume` rebuilds it without the architect."""
        self.state.helpers = helpers
        self.state.target_parents = list(target_parents)
        self.state.target_proof_plan = target_proof_plan
        self.save()

    def clear(self) -> None:
        """Discard all state and remove the checkpoint file."""
        self.state = ProofStoreState(target=self.state.target, source_hash=self.state.source_hash)
        self.path.unlink(missing_ok=True)
