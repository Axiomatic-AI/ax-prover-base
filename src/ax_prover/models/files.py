"""Models for file operations."""

import logging
from pathlib import Path

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

    def absolute_path(self, base_folder: str) -> Path | None:
        """Resolve an absolute path to the location given a base folder.

        Returns None for external locations that cannot be resolved under
        .lake/packages (see _resolve_lake_package_path).
        """
        base_path = Path(base_folder)

        if self.is_external:
            return _resolve_lake_package_path(base_path, self.module_path)
        else:
            return base_path / self.path

    @classmethod
    def parse(cls, target: str, is_external: bool = False) -> "Location":
        """Parse a target string into a Location object.
        Accepts dotted and slash notation:

            Module.Path:name -> Location(name="name", module_path="Module.Path")
            path/to/file.lean:name -> Location(name="name", module_path="path.to.file")

        Raises:
            ValueError: If the target string does not contain a colon ':'.
        """
        if ":" not in target:
            raise ValueError(
                f"Invalid target string: '{target}'. Expected format: 'modulepath:name'"
            )

        module_path, name = target.rsplit(":", 1)
        module_path = module_path.replace("/", ".").removesuffix(".lean")

        return cls(name=name, module_path=module_path, is_external=is_external)


def _resolve_lake_package_path(base: Path, module_path: str) -> Path | None:
    """Resolve a dotted module path under .lake/packages/ to its .lean file.

    The first component of `module_path` is matched case-insensitively against
    Lake package directory names; the remaining components are joined as a
    filesystem path with a .lean suffix appended.

    Example::

        _resolve_lake_package_path(
            Path("/proj"), "Mathlib.Algebra.Group.Defs"
        )
        # -> Path("/proj/.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean")

    Returns None if the .lake/packages directory or the package itself cannot
    be located. (Existence of the .lean file is the caller's concern.)
    """
    packages_dir = base / ".lake" / "packages"
    if not packages_dir.is_dir():
        return None

    parts = module_path.split(".")
    if not parts:
        return None

    package_dirs = {
        entry.name.lower(): entry.name for entry in packages_dir.iterdir() if entry.is_dir()
    }
    pkg_dir = package_dirs.get(parts[0].lower())
    if pkg_dir is None:
        return None

    return packages_dir / pkg_dir / Path(*parts).with_suffix(".lean")
