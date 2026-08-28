"""The `ax-blueprint` docstring protocol: parsing, stripping, and rendering.

A generated declaration declares its graph identity in a fenced JSON block inside its
Lean docstring. Only the fenced block is a stable protocol; the surrounding Markdown is
model-facing planning context.
"""

import json
import re

from .models import BLUEPRINT_METADATA_VERSION, NodeMetadata

FENCE_LANGUAGE = "ax-blueprint"

_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches a fenced block whose info string is exactly `ax-blueprint`. Backtick fences of
# three or more are accepted so a docstring can nest a Lean code fence inside its prose.
_FENCE_PATTERN = re.compile(
    rf"^(?P<fence>`{{3,}})[ \t]*{re.escape(FENCE_LANGUAGE)}[ \t]*\n(?P<body>.*?)^(?P=fence)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

_DOC_DELIMITERS = re.compile(r"^\s*/--(?P<content>.*?)-/\s*$", re.DOTALL)


class MetadataError(ValueError):
    """A docstring's blueprint metadata is missing or malformed."""


def unwrap_docstring(docstring: str) -> str:
    """Strip the `/-- -/` delimiters from a Lean docstring, if present."""
    match = _DOC_DELIMITERS.match(docstring)
    return match.group("content") if match else docstring


def parse_metadata(docstring: str) -> tuple[NodeMetadata, str]:
    """Parse the blueprint metadata out of a Lean docstring.

    Returns:
        The parsed metadata and the remaining human-readable docstring text.

    Raises:
        MetadataError: The fence is absent, duplicated, not valid JSON, or the payload
            violates the protocol.
    """
    content = unwrap_docstring(docstring)
    matches = list(_FENCE_PATTERN.finditer(content))

    if not matches:
        raise MetadataError(
            f"missing ```{FENCE_LANGUAGE} metadata block in docstring; "
            "every generated declaration needs one"
        )
    if len(matches) > 1:
        raise MetadataError(f"found {len(matches)} ```{FENCE_LANGUAGE} blocks, expected exactly 1")

    match = matches[0]
    try:
        payload = json.loads(match.group("body"))
    except json.JSONDecodeError as e:
        raise MetadataError(f"```{FENCE_LANGUAGE} block is not valid JSON: {e}") from e

    metadata = _validate_payload(payload)
    remaining = (content[: match.start()] + content[match.end() :]).strip()
    return metadata, remaining


def _validate_payload(payload: object) -> NodeMetadata:
    """Validate a decoded metadata payload against the protocol."""
    if not isinstance(payload, dict):
        raise MetadataError(f"metadata must be a JSON object, got {type(payload).__name__}")

    unknown = sorted(set(payload) - {"version", "id", "parents"})
    if unknown:
        raise MetadataError(f"unsupported metadata keys: {', '.join(unknown)}")

    version = payload.get("version")
    if version != BLUEPRINT_METADATA_VERSION:
        raise MetadataError(
            f"unsupported metadata version {version!r}, expected {BLUEPRINT_METADATA_VERSION}"
        )

    node_id = payload.get("id")
    if not isinstance(node_id, str) or not _ID_PATTERN.match(node_id):
        raise MetadataError(f"invalid id {node_id!r}, expected an identifier-like string")

    parents = payload.get("parents", [])
    if not isinstance(parents, list) or not all(isinstance(parent, str) for parent in parents):
        raise MetadataError(f"invalid parents {parents!r}, expected a list of node ids")

    duplicates = sorted({parent for parent in parents if parents.count(parent) > 1})
    if duplicates:
        raise MetadataError(f"duplicate parents in node {node_id!r}: {', '.join(duplicates)}")

    for parent in parents:
        if not _ID_PATTERN.match(parent):
            raise MetadataError(f"invalid parent id {parent!r} in node {node_id!r}")

    return NodeMetadata(version=version, id=node_id, parents=tuple(parents))


def strip_metadata_fence(docstring: str) -> str:
    """Return the human-readable docstring text with the metadata fence removed."""
    content = unwrap_docstring(docstring)
    return _FENCE_PATTERN.sub("", content).strip()


def render_docstring(metadata: NodeMetadata, prose: str = "") -> str:
    """Render a complete Lean docstring carrying `metadata` plus optional prose."""
    payload = json.dumps(
        {"version": metadata.version, "id": metadata.id, "parents": list(metadata.parents)}
    )
    block = f"```{FENCE_LANGUAGE}\n{payload}\n```"
    body = f"{block}\n\n{prose.strip()}" if prose.strip() else block
    return f"/--\n{body}\n-/"
