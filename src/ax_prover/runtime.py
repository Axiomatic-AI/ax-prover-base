import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from .config import RuntimeConfig
from .utils.lean_interact import LeanInteractServer


class Runtime:
    "Runtime environment for the prover agent."

    config: RuntimeConfig
    base_folder: str
    lean_interact_server: LeanInteractServer
    lean_semaphore: asyncio.Semaphore

    def __init__(self, config: RuntimeConfig, base_folder: str) -> None:
        # Do not use __init__, the intended use is through Runtime.open(...)
        # Trying to access the servers or semaphores will raise an error if it's not within an open context.
        self.config = config
        self.base_folder = base_folder

    @classmethod
    @asynccontextmanager
    async def open(cls, config: RuntimeConfig, base_folder: str) -> AsyncIterator["Runtime"]:
        rt = cls(config, base_folder)

        async with AsyncExitStack() as stack:
            rt.lean_interact_server = await stack.enter_async_context(
                LeanInteractServer(base_folder, config.lean_interact)
            )
            rt.lean_semaphore = asyncio.Semaphore(config.lean.max_concurrent_builds)

            yield rt
