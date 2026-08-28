"""Blueprint-driven proving: decompose a target into a helper DAG, prove it node by node.

An additional proving mode, selected with `ax-prover prove ... --blueprint`. The existing
direct prover is untouched and remains the default.
"""

from .models import (
    TARGET_NODE_ID,
    Blueprint,
    BlueprintError,
    BlueprintNode,
    BlueprintRunResult,
    BlueprintValidationError,
    ComparatorStatus,
    NodeDiagnosis,
    NodeOutcome,
    NodeRecord,
    NodeStatus,
    RunStatus,
)
from .orchestrator import BlueprintOptions, BlueprintOrchestrator

__all__ = [
    "TARGET_NODE_ID",
    "Blueprint",
    "BlueprintError",
    "BlueprintNode",
    "BlueprintOptions",
    "BlueprintOrchestrator",
    "BlueprintRunResult",
    "BlueprintValidationError",
    "ComparatorStatus",
    "NodeDiagnosis",
    "NodeOutcome",
    "NodeRecord",
    "NodeStatus",
    "RunStatus",
]
