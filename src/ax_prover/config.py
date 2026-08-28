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

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

__all__ = [
    "BlueprintConfig",
    "BlueprintRoleConfig",
    "ComparatorConfig",
    "LLMConfig",
    "LogLevel",
    "MemoryConfig",
    "OpenRouterConfig",
    "ProverConfig",
    "SummarizeOutputConfig",
]

#: Axioms a final Comparator check may rely on, per plan section 11.
DEFAULT_PERMITTED_AXIOMS = ["propext", "Quot.sound", "Classical.choice"]


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
class OpenRouterConfig:
    """OpenRouter endpoint settings shared by every blueprint role."""

    base_url: str = "https://openrouter.ai/api/v1"
    provider_sort: str | None = "exacto"


@dataclass
class BlueprintRoleConfig:
    """Budgets and model for one blueprint role (architect, node prover, or refiner).

    `max_total_tokens` is a per-role ceiling on accumulated LLM token usage; the role
    stops proposing once it is exceeded. `max_attempts` bounds harness-verified proposals
    and `max_tool_iterations` bounds tool calls within a single proposal.
    """

    llm: LLMConfig | None = None  # None = fall back to blueprint.llm
    max_total_tokens: int = 2_000_000
    max_attempts: int = 8
    max_tool_iterations: int = 8


@dataclass
class ComparatorConfig:
    """Final Comparator acceptance gate (plan section 11)."""

    binary: str = "comparator"
    module_prefix: str = "AxProverComparator"
    permitted_axioms: list[str] = field(default_factory=lambda: list(DEFAULT_PERMITTED_AXIOMS))
    timeout: int = 1800


@dataclass
class BlueprintConfig:
    """Configuration for the blueprint-driven proving mode."""

    enabled: bool = False
    llm: LLMConfig | None = None  # Shared default for roles that do not set their own
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    # `max_total_tokens` counts cumulative *billed* tokens, so a turn's re-sent transcript
    # is counted on every call. The paper's 65536 starved node provers after 1-2 of their 4
    # attempts while costing under a cent, so these are set generously; token spend is not
    # the binding constraint on a cheap model, wall-clock is.
    architect: BlueprintRoleConfig = field(
        default_factory=lambda: BlueprintRoleConfig(max_total_tokens=2_000_000, max_attempts=8)
    )
    prover: BlueprintRoleConfig = field(
        default_factory=lambda: BlueprintRoleConfig(max_total_tokens=1_000_000, max_attempts=4)
    )
    refiner: BlueprintRoleConfig = field(
        default_factory=lambda: BlueprintRoleConfig(max_total_tokens=2_000_000, max_attempts=4)
    )
    prover_tools: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    max_refinement_rounds: int = 8
    # Node agents reason, search, and wait on the model concurrently; their Lean
    # compilations are serialized separately. Competing Mathlib environments each need
    # ~2GB resident and a restart costs a full re-elaboration, so the Lean limit is 1.
    max_node_agents: int = 4
    max_lean_compiles: int = 1
    checkpoint_dir: str = ".axiomatic/blueprint"
    artifacts_dir: str = ".axiomatic/blueprint/artifacts"
    require_comparator: bool = False
    comparator: ComparatorConfig = field(default_factory=ComparatorConfig)

    def role(self, name: str) -> BlueprintRoleConfig:
        """Role config with the shared blueprint LLM filled in when unset.

        Raises:
            ValueError: Neither the role nor the blueprint config specifies an LLM.
        """
        role: BlueprintRoleConfig = getattr(self, name)
        if role.llm is not None:
            return role
        if self.llm is None:
            raise ValueError(
                f"blueprint.{name}.llm is unset and blueprint.llm has no fallback. "
                "Set one in your YAML config."
            )
        return replace(role, llm=self.llm)


@dataclass
class Config:
    """Root configuration object compatible with OmegaConf."""

    prover: ProverConfig = field(default_factory=ProverConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    blueprint: BlueprintConfig = field(default_factory=BlueprintConfig)
