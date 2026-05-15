import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any

from .config import RuntimeConfig
from .tools.registry import create_tool_lifespan
from .utils.lean_interact import LeanInteractServer


class Runtime:
    "Runtime environment for the prover agent."

    config: RuntimeConfig
    base_folder: str
    lean_interact_server: LeanInteractServer
    lean_semaphore: asyncio.Semaphore
    _tool_lifespans: dict[str, AbstractAsyncContextManager[Any]]

    def __init__(self, config: RuntimeConfig, base_folder: str) -> None:
        # Do not use __init__, the intended use is through Runtime.open(...)
        # Trying to access the servers or semaphores will raise an error if it's not within an open context.
        self.config = config
        self.base_folder = base_folder

    @classmethod
    @asynccontextmanager
    async def open(
        cls, config: RuntimeConfig, base_folder: str, tool_configs: dict[str, Any] | None = None
    ) -> AsyncIterator["Runtime"]:
        rt = cls(config, base_folder)

        async with AsyncExitStack() as stack:
            rt.lean_interact_server = await stack.enter_async_context(
                LeanInteractServer(base_folder, config.lean_interact)
            )
            rt.lean_semaphore = asyncio.Semaphore(config.lean.max_concurrent_builds)

            for tool_type, tool_config in (tool_configs or {}).items():
                lifespan = await create_tool_lifespan(tool_config)
                if lifespan is not None:
                    rt._tool_lifespans[tool_type] = await stack.enter_async_context(lifespan)

            yield rt

    def get_tool_lifespan(self, tool_type: str) -> AbstractAsyncContextManager[Any] | None:
        """Get the lifespan for a tool type."""
        return self._tool_lifespans.get(tool_type)
