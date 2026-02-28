"""Configuration structures compatible with OmegaConf.

This module provides dataclass-based configuration that can be:
- Created programmatically (default)
- Loaded from YAML files via OmegaConf
- Overridden via command-line arguments

Example:
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load("config.yaml")
    >>> prover = ProverAgent(config=cfg.prover)
"""

from asyncio import Semaphore
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from .prover import ProverAgent

__all__ = [
    "LLMConfig",
    "LogLevel",
    "MemoryConfig",
    "ProverConfig",
    "SummarizeOutputConfig",
]


class LogLevel(StrEnum):
    """Logging level for ax-prover."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LLMConfig:
    """
    LLM configuration for creating chat models.

    The model string should follow LangChain's format: "provider:model_name"
    Examples: "anthropic:claude-haiku-4-5-20251001", "openai:gpt-4o"
    """

    model: str
    provider_config: dict[str, Any] = field(default_factory=dict)
    retry_config: dict[str, Any] = field(
        default_factory=lambda: {
            "stop_after_attempt": 10000,  # 10k attempts at 3s is about 8h 20min.
            "wait_exponential_jitter": True,
            "exponential_jitter_params": {
                "initial": 0.5,
                "max": 3,
                "exp_base": 2.0,
                "jitter": 1.0,
            },
        }
    )


@dataclass
class MemoryConfig:
    """Configuration for memory processor in ProverAgent."""

    class_name: str
    init_args: dict = field(default_factory=dict)


@dataclass
class SummarizeOutputConfig:
    """Configuration for the summarize_output node."""

    enabled: bool = True
    llm: LLMConfig | None = None  # None = use prover_llm


@dataclass
class ProverConfig:
    """Configuration for ProverAgent."""

    prover_llm: LLMConfig | None = None  # None is a placeholder to allow merging configs in main
    proposer_tools: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 0
    memory_config: MemoryConfig = field(
        default_factory=lambda: MemoryConfig(class_name="ExperienceProcessor")
    )
    summarize_output: SummarizeOutputConfig = field(default_factory=SummarizeOutputConfig)
    user_comments: str | None = None


@dataclass
class LeanConfig:
    """Configuration for Lean build and compilation tools."""

    cache_get_timeout: int = 600
    build_timeout: int = 1200
    check_file_timeout: int = 180
    max_concurrent_builds: int = 4


@dataclass
class LeanInteractConfig:
    """Configuration for LeanInteract server (goal state extraction).

    Used for extracting goal states at sorry locations in Lean code.
    Uses lean_interact's default configuration values.
    """

    verbose: bool = False


@dataclass
class RuntimeConfig:
    """Configuration for runtime infrastructure."""

    log_level: LogLevel = LogLevel.INFO
    max_tool_calling_iterations: int = 1
    lean: LeanConfig = field(default_factory=LeanConfig)
    lean_interact: LeanInteractConfig = field(default_factory=LeanInteractConfig)


@dataclass
class Config:
    """Root configuration object compatible with OmegaConf."""

    prover: ProverConfig = field(default_factory=ProverConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    async def create_prover(
        self,
        lean_semaphore: Semaphore | None = None,
        base_folder: str = ".",
    ) -> "ProverAgent":
        """Create a prover instance from the config.

        Args:
            lean_semaphore: Optional semaphore for limiting concurrent Lean operations (default: None).
            base_folder: Base folder for the Lean project (default: ".")

        Returns:
            ProverAgent: A fully initialized prover instance ready to use

        Raises:
            ValueError: If prover.prover_llm is not set (e.g. when no YAML config was provided).
        """
        from .prover import ProverAgent

        prover_config = (
            OmegaConf.to_object(self.prover) if isinstance(self.prover, DictConfig) else self.prover
        )
        runtime_config = (
            OmegaConf.to_object(self.runtime)
            if isinstance(self.runtime, DictConfig)
            else self.runtime
        )

        if prover_config.prover_llm is None:
            raise ValueError(
                "prover.prover_llm must be set in config (e.g. pass a YAML file with prover.prover_llm)"
            )

        return await ProverAgent.create(
            config=prover_config,
            runtime_config=runtime_config,
            lean_semaphore=lean_semaphore,
            base_folder=base_folder,
        )
