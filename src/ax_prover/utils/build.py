"""Build, compilation, and code application tools for Lean projects."""

import asyncio
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import LeanConfig
from ..models.files import Location
from ..utils import get_logger
from ..utils.files import edit_function, edit_imports, edit_opens, read_file
from ..utils.lean_parsing import extract_function_from_content

if TYPE_CHECKING:
    from ax_prover.models.messages import ProposalMessage

logger = get_logger(__name__)


class LeanBuildError(Exception):
    """Base exception for Lean build failures."""

    pass


class LeanBuildTimeout(LeanBuildError):
    """Build exceeded timeout limit."""

    pass


class LeanToolNotFound(LeanBuildError):
    """Lean/Lake tools not available."""

    pass


def _uses_mathlib(repo_path: Path) -> bool:
    """Check if the Lean project depends on mathlib4 by inspecting lake-manifest.json."""
    manifest_path = repo_path / "lake-manifest.json"
    if not manifest_path.exists():
        return False
    try:
        content = manifest_path.read_text()
        return (
            "https://github.com/leanprover-community/mathlib4" in content
            or '"name": "mathlib"' in content
        )
    except OSError as e:
        logger.warning(f"Could not read lake-manifest.json: {e}")
        return False


def build_lean_repo(base_folder: str, config: LeanConfig) -> tuple[bool, str]:
    """Build a Lean repository by running cache get and build.

    Args:
        base_folder: Base folder path containing the Lean project
        config: Lean configuration

    Returns:
        Tuple of (success: bool, output: str) where output contains build logs
    """
    base_path = Path(base_folder).resolve()

    lakefile = base_path / "lakefile.lean"
    lakefile_toml = base_path / "lakefile.toml"
    if not lakefile.exists() and not lakefile_toml.exists():
        msg = f"No lakefile found in {base_folder}"
        logger.error(msg)
        return False, msg

    try:
        output_lines = []

        # Only run cache get if the project uses mathlib4
        use_cache = _uses_mathlib(base_path)

        if use_cache:
            logger.info("Running: lake exe cache get")
            result = subprocess.run(
                ["lake", "exe", "cache", "get"],
                cwd=base_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=config.cache_get_timeout,
            )
            output_lines.append("=== lake exe cache get ===")
            output_lines.append(result.stdout)
            if result.stderr:
                output_lines.append(result.stderr)

            cache_success = result.returncode == 0
            if not cache_success:
                logger.warning(f"cache get failed with code {result.returncode}")
        else:
            cache_success = True
            logger.info("Project does not use mathlib4, skipping cache get")

        logger.info("Running: lake build")
        result = subprocess.run(
            ["lake", "build"],
            cwd=base_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=config.build_timeout,
        )
        output_lines.append("\n=== lake build ===")
        output_lines.append(result.stdout)
        if result.stderr:
            output_lines.append(result.stderr)

        build_success = result.returncode == 0
        if not build_success:
            logger.error(f"lake build failed with code {result.returncode}")

        output = "\n".join(output_lines)
        success = cache_success and build_success

        if success:
            logger.info("✓ Lean repository built successfully")
        else:
            logger.warning("✗ Lean repository build completed with errors")

        return success, output

    except subprocess.TimeoutExpired as e:
        msg = f"Build timeout: {e}"
        logger.error(msg)
        return False, msg
    except FileNotFoundError:
        msg = "lake command not found. Make sure Lean 4 is installed."
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"Build error: {e}"
        logger.error(msg)
        return False, msg


def _trim_warnings(output: str) -> str:
    """Remove warning lines from Lean compiler output."""
    filtered_lines = []
    for line in output.splitlines():
        if any(
            warning in line.lower()
            for warning in [
                "warning:",
                "declaration uses 'sorry'",
                "uses sorry",
                "unused variable",
                "unused parameter",
                "trace:",
                "note:",
            ]
        ):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _format_lean_errors(error_output: str, file_path: str, file_content: str) -> str:
    """Format Lean compiler errors with code context (only for errors, not warnings)."""
    lines = file_content.splitlines()
    pattern1 = re.compile(rf"{re.escape(file_path)}:(\d+):(\d+):\s*(.*)")
    pattern2 = re.compile(rf"error:\s*{re.escape(file_path)}:(\d+):(\d+):\s*(.*)")
    formatted = []

    for error_line in error_output.splitlines():
        match = pattern1.match(error_line) or pattern2.match(error_line)
        if match:
            line_num = int(match.group(1))
            col_num = int(match.group(2))
            msg = match.group(3)

            if "error:" in error_line.lower():
                if 0 < line_num <= len(lines):
                    code = lines[line_num - 1]
                    marker = " " * (col_num - 1) + "^^^"

                    formatted.extend(
                        [
                            f"\n╭─ Error at line {line_num}:{col_num}",
                            f"│  {code}",
                            f"│  {marker}",
                            f"╰─ {msg}",
                        ]
                    )
                    continue

        formatted.append(error_line)

    return "\n".join(formatted)


async def _run_lean_subprocess(
    command: list[str],
    cwd: str,
    timeout: float,
) -> tuple[int, str, str]:
    """Run a Lean/Lake subprocess with timeout and graceful cleanup.

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as e:
        raise LeanBuildTimeout(f"Build timeout (exceeded {timeout} seconds)") from e
    finally:
        # Two-stage shutdown: try graceful (SIGINT) before force (SIGKILL)
        if process.returncode is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                logger.debug(f"Sent SIGINT to process group for {' '.join(command)}")

                for _ in range(30):  # Wait up to 3 seconds (30 * 0.1s)
                    await asyncio.sleep(0.1)
                    if process.returncode is not None:
                        break

                if process.returncode is None:
                    logger.debug(
                        f"Process did not respond to SIGINT, sending SIGKILL for {' '.join(command)}"
                    )
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)

            except (ProcessLookupError, AttributeError):
                pass

            try:
                await process.wait()
            except Exception:
                pass

    return (
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def check_lean_file(
    base_folder: str,
    file_path: str,
    config: LeanConfig,
    semaphore: asyncio.Semaphore,
    show_warnings: bool = True,
    build: bool = False,
) -> tuple[bool, str]:
    """Check if a Lean file compiles successfully.

    Uses asyncio subprocesses for non-blocking I/O operations.

    Args:
        base_folder: Base folder path (project root with lakefile.lean)
        file_path: Path to Lean file relative to base_folder
        config: Lean configuration
        semaphore: Semaphore to limit concurrent Lean process executions
        show_warnings: If False, suppress warning messages like 'declaration uses sorry'
        build: If True, use 'lake build file_path' instead of 'lake env lean file_path'.
               'lake build' is more precise for checking proofs as it builds the entire
               module and dependencies, while 'lake env lean' is generally faster and
               sufficient for checking statements and definitions.
               Falls back to 'lake env lean' if the module is not a known Lake target.

    Returns:
        Tuple of (success, message)
    """
    full_path = Path(base_folder) / file_path

    if not full_path.exists():
        return False, f"File not found: {file_path}"

    try:
        if build:
            module_path = file_path.replace("/", ".").removesuffix(".lean")
            command = ["lake", "build", module_path]
        else:
            command = ["lake", "env", "lean", file_path]

        # Acquire semaphore before starting subprocess to limit concurrent builds
        async with semaphore:
            returncode, stdout_str, stderr_str = await _run_lean_subprocess(
                command, base_folder, config.check_file_timeout
            )

            # Fallback: if lake build fails with "unknown target", retry with lake env lean
            if build and returncode != 0 and "unknown target" in (stdout_str + stderr_str).lower():
                logger.warning(
                    f"'lake build {module_path}' failed: unknown target. "
                    f"Falling back to 'lake env lean {file_path}'"
                )
                fallback_command = ["lake", "env", "lean", file_path]
                returncode, stdout_str, stderr_str = await _run_lean_subprocess(
                    fallback_command, base_folder, config.check_file_timeout
                )

        if returncode == 0:
            return True, "Build successful"

        output = stdout_str + stderr_str

        if not show_warnings:
            output = _trim_warnings(output)

        file_content = read_file(base_folder, file_path)
        formatted_output = _format_lean_errors(output, file_path, file_content)
        return False, formatted_output.strip()

    except FileNotFoundError as e:
        raise LeanToolNotFound("Lean/Lake not found. Please ensure Lean 4 is installed.") from e
    except LeanBuildTimeout:
        raise
    except Exception as e:
        logger.error(f"Error checking Lean file: {e}")
        return False, f"Check failed: {str(e)}"


class TemporaryProposal:
    """Context manager for testing proposals in temporary files.

    Creates a temporary copy of a file, applies a proposal (imports, opens, code),
    and provides methods to test and permanently apply the changes.

    Example:
        with TemporaryProposal(base_folder, original_location, proposal) as applier:
            if not applier.success:
                print(f"Failed to apply: {applier.error}")
                return

            # Test the temporary file
            build_success, message = await check_lean_file(
                base_folder, applier.location.path, config, semaphore
            )

            if build_success:
                # Apply permanently to original file
                applier.apply_permanently()
    """

    def __init__(
        self,
        base_folder: str,
        original_location: Location | None,
        proposal: "ProposalMessage",
    ):
        """Initialize the temporary proposal applier.

        Args:
            base_folder: Base folder path
            original_location: Location object for the original file (None means no location set)
            proposal: ProposalMessage with imports, opens, and code to apply
        """
        self.base_folder = base_folder
        self.original_location = original_location
        self.proposal = proposal
        self.location: Location | None = None  # Temp location, set in __enter__
        self.error: str = ""
        self.success: bool = False
        self._temp_file = None

    def __enter__(self) -> "TemporaryProposal":
        """Create temp file and apply proposal. Returns self for method access."""
        try:
            if not self.original_location:
                self.error = "No location set"
                return self

            if self.original_location.is_external and self.proposal.has_changes:
                self.error = (
                    f"Cannot modify external library location: "
                    f"{self.original_location.formatted_context}"
                )
                return self

            original_path = Path(self.base_folder) / self.original_location.path

            self._temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=original_path.suffix,
                prefix=f"tmp_{original_path.stem}_",
                dir=original_path.parent,
                delete=False,
                encoding="utf-8",
            )

            if original_path.exists():
                with open(original_path, encoding="utf-8") as f:
                    self._temp_file.write(f.read())
                self._temp_file.flush()

            temp_path_abs = Path(self._temp_file.name)
            temp_path_rel = str(temp_path_abs.relative_to(self.base_folder))

            temp_module_path = temp_path_rel.replace("/", ".").removesuffix(".lean")
            self.location = self.original_location.model_copy(
                update={"module_path": temp_module_path}
            )

            if self.proposal.imports:
                success = edit_imports(self.base_folder, self.location.path, self.proposal.imports)
                if not success:
                    self.error = "Failed to apply imports to temp file"
                    return self

            if self.proposal.opens:
                success = edit_opens(self.base_folder, self.location.path, self.proposal.opens)
                if not success:
                    self.error = "Failed to apply opens to temp file"
                    return self

            if self.proposal.code:
                # Only allow edits within the function definition
                filtered_code = extract_function_from_content(
                    self.proposal.code, self.location.name
                )
                success = edit_function(self.base_folder, self.location, filtered_code)
                if not success:
                    self.error = "Failed to apply code to temp file"
                    return self

            self.success = True

        except Exception as e:
            self.error = f"Error creating temporary proposal: {str(e)}"
            logger.error(self.error)

        return self

    def apply_permanently(self) -> bool:
        """Apply the proposal to the original file by copying the temp file."""
        if not self.success:
            logger.error("Cannot apply permanently: temporary application failed")
            return False

        if not self.location:
            logger.error("Cannot apply permanently: no temp location")
            return False

        try:
            temp_path = Path(self.base_folder) / self.location.path
            original_path = Path(self.base_folder) / self.original_location.path

            original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_path, original_path)

            return True

        except Exception as e:
            logger.error(f"Error applying permanently: {e}")
            return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and delete the temporary file."""
        if self._temp_file and self.location:
            try:
                self._temp_file.close()
                temp_path = Path(self.base_folder) / self.location.path
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
