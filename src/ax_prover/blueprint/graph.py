"""Graph validation, topological ordering, and ready-frontier calculation."""

from collections import deque
from collections.abc import Iterable, Mapping

from .models import (
    TARGET_NODE_ID,
    Blueprint,
    BlueprintNode,
    BlueprintValidationError,
    NodeStatus,
)

#: Only helper lemmas are permitted in milestone one. `lean_interact` reports `lemma`
#: declarations as `theorem`, but both spellings are accepted defensively.
ALLOWED_HELPER_KINDS = frozenset({"theorem", "lemma"})


def normalize_signature(signature: str) -> str:
    """Collapse whitespace so pretty-printer line breaks do not read as a type change."""
    return " ".join(signature.split())


def validate_blueprint(
    nodes: Iterable[BlueprintNode],
    namespace: str,
    target_lean_name: str,
    target_signature: str,
    skeleton: str = "",
) -> Blueprint:
    """Validate extracted nodes and return the canonical blueprint.

    Raises:
        BlueprintValidationError: With every problem found, so one refinement round can
            fix all of them.
    """
    nodes = tuple(nodes)
    problems: list[str] = []

    problems += _check_uniqueness(nodes)
    problems += _check_kinds_and_namespace(nodes, namespace)
    problems += _check_target(nodes, target_lean_name, target_signature)
    problems += _check_edges(nodes)

    if problems:
        raise BlueprintValidationError(problems)

    return Blueprint(namespace=namespace, nodes=nodes, skeleton=skeleton)


def _check_uniqueness(nodes: tuple[BlueprintNode, ...]) -> list[str]:
    problems = []
    for attribute, label in (("id", "blueprint id"), ("lean_name", "Lean declaration name")):
        seen: set[str] = set()
        for node in nodes:
            value = getattr(node, attribute)
            if value in seen:
                problems.append(f"duplicate {label}: {value!r}")
            seen.add(value)
    return problems


def _check_kinds_and_namespace(nodes: tuple[BlueprintNode, ...], namespace: str) -> list[str]:
    problems = []
    prefix = f"{namespace}."
    for node in nodes:
        if node.is_target:
            continue
        if node.id == TARGET_NODE_ID:
            problems.append(f"helper may not use the reserved blueprint id {TARGET_NODE_ID!r}")
        if node.kind not in ALLOWED_HELPER_KINDS:
            problems.append(
                f"node {node.id!r} is a {node.kind}; only helper lemmas are permitted "
                "(no definitions, structures, classes, instances, or axioms)"
            )
        if not node.lean_name.startswith(prefix):
            problems.append(
                f"helper {node.lean_name!r} is outside the generated namespace {namespace!r}"
            )
    return problems


def _check_target(
    nodes: tuple[BlueprintNode, ...], target_lean_name: str, target_signature: str
) -> list[str]:
    targets = [node for node in nodes if node.is_target]

    if not targets:
        return [f"missing target node (expected declaration {target_lean_name!r})"]
    if len(targets) > 1:
        return [f"found {len(targets)} target nodes, expected exactly 1"]

    target = targets[0]
    problems = []
    if target.id != TARGET_NODE_ID:
        problems.append(f"target node must use blueprint id {TARGET_NODE_ID!r}, got {target.id!r}")
    if target.lean_name != target_lean_name:
        problems.append(
            f"target declaration renamed: expected {target_lean_name!r}, got {target.lean_name!r}"
        )
    if normalize_signature(target.signature) != normalize_signature(target_signature):
        problems.append(
            "the original target's type may not change; expected "
            f"{normalize_signature(target_signature)!r}, got "
            f"{normalize_signature(target.signature)!r}"
        )
    return problems


def _check_edges(nodes: tuple[BlueprintNode, ...]) -> list[str]:
    problems = []
    known = {node.id for node in nodes}

    for node in nodes:
        # Only declared parents can be wrong: statement parents are resolved from
        # elaborated types, so they always name a real node and never the node itself.
        for parent in node.declared_parents:
            if parent == node.id:
                problems.append(f"node {node.id!r} declares itself as a parent")
            elif parent not in known:
                problems.append(f"node {node.id!r} declares unknown parent {parent!r}")

    if not problems:
        cycle = find_cycle(nodes)
        if cycle:
            problems.append(f"parent edges form a cycle: {' -> '.join(cycle)}")

    return problems


def find_cycle(nodes: Iterable[BlueprintNode]) -> list[str]:
    """Return one cycle in the parent graph as a list of ids, or an empty list."""
    parents = {node.id: tuple(node.parents) for node in nodes}
    visiting: set[str] = set()
    done: set[str] = set()
    stack: list[str] = []

    def walk(node_id: str) -> list[str]:
        if node_id in done:
            return []
        if node_id in visiting:
            start = stack.index(node_id)
            return stack[start:] + [node_id]

        visiting.add(node_id)
        stack.append(node_id)
        for parent in parents.get(node_id, ()):
            cycle = walk(parent)
            if cycle:
                return cycle
        stack.pop()
        visiting.discard(node_id)
        done.add(node_id)
        return []

    for node_id in sorted(parents):
        cycle = walk(node_id)
        if cycle:
            return cycle
    return []


def topological_order(blueprint: Blueprint) -> tuple[BlueprintNode, ...]:
    """Order every node parents-first, deterministically by id within each layer.

    Raises:
        BlueprintValidationError: The graph contains a cycle.
    """
    by_id = blueprint.by_id
    pending = {node.id: set(node.parents) for node in blueprint.nodes}
    children: dict[str, list[str]] = {node_id: [] for node_id in pending}
    for node_id, parents in pending.items():
        for parent in parents:
            children[parent].append(node_id)

    ready = deque(sorted(node_id for node_id, parents in pending.items() if not parents))
    ordered: list[BlueprintNode] = []

    while ready:
        node_id = ready.popleft()
        ordered.append(by_id[node_id])
        newly_ready = []
        for child in children[node_id]:
            pending[child].discard(node_id)
            if not pending[child]:
                newly_ready.append(child)
        for child in sorted(newly_ready):
            ready.append(child)
        # Keep the queue sorted so the order depends only on the graph, not insertion order.
        ready = deque(sorted(ready))

    if len(ordered) != len(blueprint.nodes):
        raise BlueprintValidationError(["parent edges form a cycle"])

    return tuple(ordered)


def ready_frontier(blueprint: Blueprint, statuses: Mapping[str, NodeStatus]) -> tuple[str, ...]:
    """Ids of pending nodes whose every declared direct parent is already solved."""
    return tuple(
        node.id
        for node in blueprint.nodes
        if statuses.get(node.id, NodeStatus.PENDING) is NodeStatus.PENDING
        and all(statuses.get(parent) is NodeStatus.SOLVED for parent in node.parents)
    )


def transitive_ancestors(blueprint: Blueprint, node_id: str) -> set[str]:
    """Ids of every node `node_id` depends on, directly or transitively."""
    by_id = blueprint.by_id
    ancestors: set[str] = set()
    queue = deque(by_id[node_id].parents) if node_id in by_id else deque()

    while queue:
        current = queue.popleft()
        if current in ancestors or current not in by_id:
            continue
        ancestors.add(current)
        queue.extend(by_id[current].parents)

    return ancestors


def required_nodes(blueprint: Blueprint) -> set[str]:
    """Ids the target actually needs: itself plus its transitive ancestors.

    Helpers the refiner left behind but nothing depends on are not proven and not
    assembled, so a dead helper cannot block a run.
    """
    return {TARGET_NODE_ID} | transitive_ancestors(blueprint, TARGET_NODE_ID)


def transitive_descendants(blueprint: Blueprint, node_id: str) -> set[str]:
    """Ids of every node that depends on `node_id`, directly or transitively."""
    children: dict[str, list[str]] = {node.id: [] for node in blueprint.nodes}
    for node in blueprint.nodes:
        for parent in node.parents:
            if parent in children:
                children[parent].append(node.id)

    descendants: set[str] = set()
    queue = deque(children.get(node_id, []))
    while queue:
        current = queue.popleft()
        if current in descendants:
            continue
        descendants.add(current)
        queue.extend(children.get(current, []))

    return descendants
