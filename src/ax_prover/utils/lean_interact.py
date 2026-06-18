"""LeanInteract tools for goal state extraction."""

import asyncio

from lean_interact import (
    AutoLeanServer,
    Command,
    LeanREPLConfig,
    LocalProject,
)
from lean_interact.interface import CommandResponse, LeanError

from ..config import LeanInteractConfig
from .logging import get_logger

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
                self._server = AutoLeanServer(repl_config, max_total_memory=1.0)
                logger.debug(f"Created LeanInteract server for {self._base_folder}")

        return self._server

    async def __aenter__(self) -> "LeanInteractServer":
        "Enter the context manager."
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        "Exit the context manager."
        await self.aclose()
