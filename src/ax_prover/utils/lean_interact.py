"""LeanInteract tools for goal state extraction."""

import asyncio
from typing import NamedTuple

from lean_interact import (
    AutoLeanServer,
    Command,
    LeanREPLConfig,
    LocalProject,
)
from lean_interact.interface import CommandResponse, LeanError
from lean_interact.utils import (
    DEFAULT_REPL_GIT_URL,
    DEFAULT_REPL_VERSION,
    get_project_lean_version,
)

from ..config import LeanInteractConfig
from .logging import get_logger

logger = get_logger(__name__)


class ReplSource(NamedTuple):
    """A Lean REPL git fork lean_interact can build the server from.

    `rev` is the base REPL revision; lean_interact resolves the concrete build by
    checking out the tag `{rev}_lean-toolchain-{lean_version}`.
    """

    git: str
    rev: str


# REPL forks tried in order until one advertises the project's Lean version. The list is
# hard-coded (no config) so goal-state extraction works out of the box across toolchains.
# Add newer/internal forks here as they become available.
REPL_SOURCES: tuple[ReplSource, ...] = (
    # lean_interact's default fork: backports the REPL across v4.8.0-rc1 .. v4.31.0-rc1.
    ReplSource(git=DEFAULT_REPL_GIT_URL, rev=DEFAULT_REPL_VERSION),
    # Our fork extending the backports to the stable v4.31.0 release.
    ReplSource(git="https://github.com/austinletson-ax/repl", rev="v1.3.18"),
    # Official upstream, tracking the newest Lean version not yet backported above.
    ReplSource(git="https://github.com/leanprover-community/repl", rev="master"),
)


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
                repl_config = self._build_repl_config()
                self._server = AutoLeanServer(
                    repl_config, max_total_memory=self._config.max_total_memory
                )
                logger.debug(f"Created LeanInteract server for {self._base_folder}")

        return self._server

    def _build_repl_config(self) -> LeanREPLConfig:
        """Build a REPL config using the first fork that supports the project's Lean version.

        Any single REPL fork only tracks a bounded range of Lean versions (e.g. up to a
        release candidate, not the stable release). Rather than relying on the
        version-mismatch error lean_interact raises, we check each fork's advertised
        versions up front and pick one that covers the project, walking the hard-coded
        `REPL_SOURCES` list in order.
        """
        # The project is already built by the builder node.
        project = LocalProject(directory=self._base_folder, auto_build=False)
        project_version = get_project_lean_version(self._base_folder)

        if project_version is None:
            # Toolchain unreadable/unparseable: let lean_interact resolve the default fork.
            logger.warning(
                "Could not determine Lean version for %s; using default REPL fork.",
                self._base_folder,
            )
            return LeanREPLConfig(project=project, verbose=self._config.verbose)

        supported_by_source: list[tuple[ReplSource, list[str]]] = []
        for source in REPL_SOURCES:
            supported = self._available_versions(source)
            supported_by_source.append((source, supported))
            if project_version in supported:
                logger.info(
                    "Using REPL fork %s@%s for Lean %s",
                    source.git,
                    source.rev,
                    project_version,
                )
                return LeanREPLConfig(
                    project=project,
                    repl_git=source.git,
                    repl_rev=source.rev,
                    verbose=self._config.verbose,
                )

        ranges = "; ".join(
            f"{s.git}@{s.rev}: {versions[0]}..{versions[-1]}"
            if versions
            else f"{s.git}@{s.rev}: none"
            for s, versions in supported_by_source
        )
        raise RuntimeError(
            f"No known Lean REPL fork supports the project's Lean version {project_version}. "
            f"Tried [{ranges}]. Add a more up-to-date fork to `REPL_SOURCES` in "
            f"ax_prover.utils.lean_interact."
        )

    def _available_versions(self, source: ReplSource) -> list[str]:
        """Lean versions a REPL fork can build, per lean_interact's public API.

        Uses `build_repl=False` so this only clones/reads git tags (cheap) instead of
        compiling the REPL. The clone warms lean_interact's cache for the real build.
        """
        probe = LeanREPLConfig(
            repl_git=source.git,
            repl_rev=source.rev,
            build_repl=False,
            verbose=self._config.verbose,
        )
        return probe.get_available_lean_versions()

    async def __aenter__(self) -> "LeanInteractServer":
        "Enter the context manager."
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        "Exit the context manager."
        await self.aclose()
