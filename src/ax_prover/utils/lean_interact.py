"""LeanInteract tools for goal state extraction."""

import asyncio
from pathlib import Path

from lean_interact import (
    AutoLeanServer,
    Command,
    LeanREPLConfig,
    LocalProject,
)
from lean_interact.interface import CommandResponse, LeanError

from ..config import LeanInteractConfig
from ..utils import get_logger

logger = get_logger(__name__)


class LeanInteractServer:
    "LeanInteract server that is lazily-instantiated and thread safe."

    def __init__(self, base_folder: str, config: LeanInteractConfig) -> None:
        self._base_folder = base_folder
        self._config = config
        self._server: AutoLeanServer | None = None
        self._lock = asyncio.Lock()  # lock to ensure only one server is created

    async def run(self, command: Command) -> CommandResponse | LeanError:
        "Run the command and return the response or error."
        server = await self._get_server()
        return await server.async_run(command)

    async def aclose(self) -> None:
        "Close the server."
        if self._server is not None:
            self._server.kill()
            self._server = None

    async def _get_server(self) -> AutoLeanServer:
        "Get the server or create it if it doesn't exist."
        if self._server is not None:
            return self._server

        async with self._lock:  # Prevent multiple servers from being created concurrently
            if self._server is None:
                # The project is already built by the builder node
                project = LocalProject(directory=self._base_folder, auto_build=False)
                repl_config = LeanREPLConfig(
                    project=project,
                    verbose=self._config.verbose,
                )
                self._server = AutoLeanServer(repl_config)
                logger.debug(f"Created LeanInteract server for {self._base_folder}")

        return self._server

    async def __aenter__(self) -> "LeanInteractServer":
        "Enter the context manager."
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        "Exit the context manager."
        await self.aclose()


async def get_goal_state_at_sorries(
    server: LeanInteractServer, base_folder: str, file_path: str
) -> str:
    """Extract goal states at all sorry locations using LeanInteract (async).

    Uses a shared AutoLeanServer instance for efficient resource usage across
    multiple concurrent experiment runs. The server is thread-safe and processes
    requests sequentially.

    Args:
        server: LeanInteractServer instance
        base_folder: Base folder of the Lean project
        file_path: Relative path to the Lean file (relative to base_folder)

    Returns:
        Formatted string with goal states at each sorry location
    """
    lean_code = (Path(base_folder) / file_path).read_text()

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
