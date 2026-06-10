"""Models for Lean4 declarations."""

import logging
from enum import StrEnum

from lean_interact.interface import DeclarationInfo, Sorry
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
    info: DeclarationInfo
    sorries: list[Sorry] = Field(default_factory=list)

    @property
    def name(self) -> str:
        """Name of the declaration."""
        return self.info.name

    @property
    def kind(self) -> str:
        "Kind of the declaration."
        return self.info.kind

    def __str__(self):
        return self.info.pp
