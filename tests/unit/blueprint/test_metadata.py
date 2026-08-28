"""Docstring metadata protocol parsing."""

import pytest

from ax_prover.blueprint.metadata import (
    MetadataError,
    parse_metadata,
    render_docstring,
    strip_metadata_fence,
)
from ax_prover.blueprint.models import NodeMetadata

VALID = """/--
```ax-blueprint
{"version": 1, "id": "positive_denominator", "parents": ["nonzero"]}
```

## Statement

The denominator is positive.
-/"""


def test_parses_metadata_and_returns_remaining_prose():
    metadata, prose = parse_metadata(VALID)

    assert metadata == NodeMetadata(version=1, id="positive_denominator", parents=("nonzero",))
    assert prose.startswith("## Statement")
    assert "ax-blueprint" not in prose


def test_parses_metadata_without_docstring_delimiters():
    metadata, prose = parse_metadata('```ax-blueprint\n{"version": 1, "id": "a"}\n```')

    assert metadata.id == "a"
    assert metadata.parents == ()
    assert prose == ""


def test_accepts_a_nested_lean_fence_in_the_prose():
    docstring = """/--
````ax-blueprint
{"version": 1, "id": "a", "parents": []}
````

```lean
example : True := trivial
```
-/"""

    metadata, prose = parse_metadata(docstring)

    assert metadata.id == "a"
    assert "example : True" in prose


@pytest.mark.parametrize(
    ("docstring", "message"),
    [
        ("/-- just prose -/", "missing"),
        (
            '```ax-blueprint\n{"version": 1, "id": "a"}\n```\n```ax-blueprint\n{}\n```',
            "expected exactly 1",
        ),
        ("```ax-blueprint\nnot json\n```", "not valid JSON"),
        ('```ax-blueprint\n{"version": 2, "id": "a"}\n```', "unsupported metadata version"),
        ('```ax-blueprint\n{"id": "a"}\n```', "unsupported metadata version"),
        ('```ax-blueprint\n{"version": 1, "id": "9bad"}\n```', "invalid id"),
        ('```ax-blueprint\n{"version": 1, "id": "a", "parents": "b"}\n```', "invalid parents"),
        (
            '```ax-blueprint\n{"version": 1, "id": "a", "parents": ["b", "b"]}\n```',
            "duplicate parents",
        ),
        (
            '```ax-blueprint\n{"version": 1, "id": "a", "extra": 1}\n```',
            "unsupported metadata keys",
        ),
        ("```ax-blueprint\n[1, 2]\n```", "must be a JSON object"),
    ],
)
def test_rejects_malformed_metadata(docstring, message):
    with pytest.raises(MetadataError, match=message):
        parse_metadata(docstring)


def test_strip_metadata_fence_keeps_only_prose():
    assert strip_metadata_fence(VALID).startswith("## Statement")


def test_render_docstring_round_trips():
    metadata = NodeMetadata(version=1, id="helper", parents=("a", "b"))

    rendered = render_docstring(metadata, "## Proof\n\nBy positivity.")
    parsed, prose = parse_metadata(rendered)

    assert parsed == metadata
    assert prose == "## Proof\n\nBy positivity."


def test_render_docstring_without_prose_is_still_parseable():
    metadata = NodeMetadata(version=1, id="target", parents=())

    parsed, prose = parse_metadata(render_docstring(metadata))

    assert parsed == metadata
    assert prose == ""
