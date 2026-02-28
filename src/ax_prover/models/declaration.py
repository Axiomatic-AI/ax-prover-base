"""Models for Lean4 declarations."""

import logging
from enum import StrEnum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Not very precise, since open, abbrev etc... are technically not declarations. But it's good enough for now.
class DeclarationType(StrEnum):
    Definition = "def"
    Theorem = "theorem"
    Lemma = "lemma"
    Instance = "instance"
    Structure = "structure"
    Class = "class"
    Inductive = "inductive"
    Axiom = "axiom"
    Abbrev = "abbrev"
    Notation = "notation"
    NoncomputableDef = "noncomputable def"
    NoncomputableAbbrev = "noncomputable abbrev"
    Macro = "macro"
    Syntax = "syntax"
    Elab = "elab"
    DeclareSyntaxCat = "declare_syntax_cat"
    Open = "open"
    End = "end"
    Section = "section"
    Namespace = "namespace"
    Import = "import"


class Declaration(BaseModel):
    declaration_type: DeclarationType = Field(
        description="What type of declaration is it (def, theorem, ...)"
    )
    name: str = Field(description="Name of the declaration")
    content: str = Field(
        description="Raw text of the full content right after the declaration type and the name"
    )

    def __str__(self):
        return f"{self.declaration_type.value} {self.name} {self.content.rstrip()}"
