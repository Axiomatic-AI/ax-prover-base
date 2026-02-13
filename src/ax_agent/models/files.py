"""Models for file operations."""

import logging

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class Location(BaseModel):
    """Location information for a formalized item."""

    name: str = Field(description="Name of the definition/theorem/lemma in Lean code")
    module_path: str = Field(
        description="Import path in dot notation "
        "(e.g., Mathlib.Topology.Basic or MyProject.Algebra.Ring)"
    )
    is_external: bool = Field(
        description="Whether this references an external library (e.g., Mathlib) or project code",
    )  # default field kills the LLMs structured output

    @field_validator("module_path")
    @classmethod
    def validate_module_path(cls, v: str) -> str:
        """Validate and auto-fix module_path to use dot notation."""
        if "/" in v:
            # Auto-fix: convert filesystem path to module path
            fixed = v.replace("/", ".").removesuffix(".lean")
            logger.warning(f"module_path should use dot notation. Auto-fixing '{v}' -> '{fixed}'")
            return fixed
        return v

    @property
    def path(self) -> str:
        """Convert module path to file system path with .lean extension."""
        return self.module_path.replace(".", "/") + ".lean"

    @property
    def formatted_context(self) -> str:
        """Get formatted string representation of location."""
        location_str = f"{self.module_path}:{self.name}"
        if self.is_external:
            location_str += " (external)"
        return location_str

    @classmethod
    def from_formatted_context(cls, formatted_context: str) -> "Location":
        """Parse a 'ModulePath:name' string into a Location object."""
        if ":" not in formatted_context:
            raise ValueError(
                f"Invalid location format: '{formatted_context}'. "
                "Expected format: 'ModulePath:name' (e.g., 'QuantumLib.Operators:my_theorem')"
            )

        module_path, name = formatted_context.rsplit(":", 1)

        return cls(name=name, module_path=module_path, is_external=False)
