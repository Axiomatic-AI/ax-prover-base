"""LeanInteract tools for goal state extraction."""

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from lean_interact import AutoLeanServer, Command, LeanREPLConfig, LocalProject

from ..config import LeanInteractConfig
from ..utils import get_logger

logger = get_logger(__name__)


# Module-level LeanInteract server for goal state extraction
# Shared across all experiments, created lazily on first use
# Thread-safe via internal locking (processes requests sequentially)
_lean_interact_server: AutoLeanServer | None = None


async def _get_lean_interact_server(base_folder: str, config: LeanInteractConfig) -> AutoLeanServer:
    """Get or create the shared LeanInteract server.

    The server is created lazily on first use and reused across all goal state
    extraction calls. Thread-safe via internal locking, processes requests sequentially.
    Uses lean_interact's default configuration values for memory management and restarts.

    Args:
        base_folder: Base folder of the Lean project
        config: LeanInteract configuration (minimal, mostly for verbose flag)

    Returns:
        Shared AutoLeanServer instance configured for the project
    """
    global _lean_interact_server
    if _lean_interact_server is None:
        project = LocalProject(
            directory=base_folder,
            auto_build=False,  # Project already built by builder node
        )

        repl_config = LeanREPLConfig(
            project=project,
            verbose=config.verbose,
        )

        _lean_interact_server = AutoLeanServer(repl_config)
        logger.debug(f"Created shared LeanInteract server for {base_folder}")

    return _lean_interact_server


@contextlib.asynccontextmanager
async def lean_interact_session_manager() -> AsyncIterator[None]:
    """Async context manager for LeanInteract server lifecycle.

    Ensures the shared LeanInteract server is properly cleaned up on exit.
    Server is lazily created on first goal state extraction request.

    Usage:
        async with lean_interact_session_manager():
            # All goal state extractions here will reuse the server
            # Server automatically cleaned up on exit
            await get_goal_state_at_sorries(...)
    """
    try:
        yield
    finally:
        global _lean_interact_server
        if _lean_interact_server is not None:
            _lean_interact_server = None
            logger.debug("Closed shared LeanInteract server")


async def get_goal_state_at_sorries(
    base_folder: str, file_path: str, config: LeanInteractConfig
) -> str:
    """Extract goal states at all sorry locations using LeanInteract (async).

    Uses a shared AutoLeanServer instance for efficient resource usage across
    multiple concurrent experiment runs. The server is thread-safe and processes
    requests sequentially.

    Args:
        base_folder: Base folder of the Lean project
        file_path: Relative path to the Lean file (relative to base_folder)
        config: LeanInteract configuration for server settings

    Returns:
        Formatted string with goal states at each sorry location
    """
    lean_code = (Path(base_folder) / file_path).read_text()

    server = await _get_lean_interact_server(base_folder, config)

    response = await server.async_run(Command(cmd=lean_code))

    if not response.sorries:
        return "No sorries found in code."

    goal_states = []
    for idx, sorry in enumerate(response.sorries, start=1):
        goal_states.append(
            f"Sorry #{idx} at line {sorry.start_pos.line}, column {sorry.start_pos.column}:\n"
            f"{sorry.goal}\n"
        )

    return "\n".join(goal_states)
