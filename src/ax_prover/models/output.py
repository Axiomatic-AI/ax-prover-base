"""Structured output models for prover results."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from .proving import ProverAgentState

logger = logging.getLogger(__name__)


class ProverOutput(BaseModel):
    """Minimal structured output from the prover agent."""

    success: bool = Field(description="Whether the proof succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    summary: str = Field(default="", description="LLM-generated summary of the prover run")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Mode-specific metrics, e.g. a blueprint run's graph and node records",
    )

    @classmethod
    def from_prover_state(cls, state: ProverAgentState) -> "ProverOutput":
        """Create output from ProverAgentState."""
        return cls(success=state.item.is_proven, error=None, summary=state.summary)

    @classmethod
    def from_exception(cls, exc: Exception) -> "ProverOutput":
        """Create output from an exception."""
        return cls(success=False, error=f"{type(exc).__name__}: {exc}")
