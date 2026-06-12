"""Models for Lean4 declarations."""

import logging

from lean_interact.interface import DeclarationInfo, Sorry
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Declaration(BaseModel):
    info: DeclarationInfo
    sorries: list[Sorry] = Field(default_factory=list)

    @property
    def code(self) -> str:
        "Source code of the declaration."
        return self.info.pp

    @property
    def kind(self) -> str:
        "Kind of the declaration."
        return self.info.kind

    @property
    def name(self) -> str:
        """Name of the declaration."""
        return self.info.name

    def contains_line(self, line_number: int) -> bool:
        """Check if the declaration contains the given line number."""
        return self.info.range.start.line <= line_number <= self.info.range.finish.line

    def __str__(self):
        return self.info.pp
