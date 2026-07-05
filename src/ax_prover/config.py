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

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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


DEFAULT_LLM_RETRY_CONFIG = {
    "stop_after_attempt": 10000,  # 10k attempts at 3s is about 8h 20min.
    "wait_exponential_jitter": True,
    "exponential_jitter_params": {
        "initial": 0.5,
        "max": 3,
        "exp_base": 2.0,
        "jitter": 1.0,
    },
}


@dataclass
class LLMConfig:
    """
    LLM configuration for creating chat models.

    The model string should follow LangChain's format: "provider:model_name"
    Examples: "anthropic:claude-haiku-4-5-20251001", "openai:gpt-4o"
    """

    model: str
    provider_config: dict[str, Any] = field(default_factory=dict)
    retry_config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LLM_RETRY_CONFIG))
    # Prices in USD per 1M tokens, documenting the model's cost near its config.
    # Cost is tracked natively by LangSmith via its model price map (keyed on the
    # model name); keep these in sync with that entry. Leave null if unknown.
    input_token_price: float | None = None
    output_token_price: float | None = None


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
    proposer_tools: dict[str, dict[str, Any] | None] = field(default_factory=dict)
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

    max_total_memory: float = 1.0
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
