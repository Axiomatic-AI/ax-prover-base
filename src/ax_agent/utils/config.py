"""Configuration utilities for loading and merging OmegaConf configurations."""

from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..config import Config

# Keys that are valid in the Config dataclass (used for filtering temporary keys)
_CONFIG_KEYS = frozenset(field.name for field in fields(Config))


def _load_yaml_with_imports(file_path: str) -> list[DictConfig]:
    """Recursively load a YAML file and all its imports.

    Returns a list of DictConfigs in merge order (imports first, then the file itself).
    This ensures proper precedence: base configs are loaded first, then overrides.
    """
    loaded = OmegaConf.load(file_path)
    import_files = loaded.pop("import", [])

    result = []
    # Process imports recursively (they have lower precedence)
    for import_file in import_files:
        result.extend(_load_yaml_with_imports(import_file))

    result.append(loaded)
    return result


def save_config(config: Config, file_path: str | Path) -> None:
    """Save a Config dataclass to a YAML file.

    Args:
        config: Config dataclass instance to save
        file_path: Path to save the YAML file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    dict_config = OmegaConf.structured(config)
    with open(file_path, "w") as f:
        OmegaConf.save(dict_config, f)


def merge_configs(
    configs: Iterable[Config | str | DictConfig | list[str] | dict],
) -> Config:
    """Merge multiple configurations in order of precedence.

    Configurations are merged left-to-right, with later configs overriding earlier ones.

    Supports OmegaConf variable interpolation (e.g., ${llm_configs.claude_opus_4_5})
    for referencing values defined in imported config files. Temporary keys used for
    interpolation (like `llm_configs`) are automatically removed after resolution.

    Args:
        configs: Iterable of configurations. Supported formats:
            - Config dataclass instance
            - str: path to YAML file (with optional "import" section for inheritance)
            - DictConfig: OmegaConf config object
            - list[str]: dot-notation overrides like ["prover.max_iterations=20"]
            - dict: Python dictionary

    Returns:
        Merged Config dataclass instance with type safety

    Example:
        ```yaml
        # configs/llms.yaml
        llm_configs:
          claude_opus_4_5:
            model: "anthropic:claude-opus-4-5"
            provider_config:
              betas: ["structured-outputs-2025-11-13"]

        # experiment.yaml
        import:
          - configs/llms.yaml

        prover:
          prover_llm: ${llm_configs.claude_opus_4_5}
        ```
    """
    # Separate structured configs (dataclasses) from unstructured configs
    # We need to merge unstructured first, resolve interpolations, then validate
    structured_configs = []
    unstructured_configs = []

    for config in configs:
        if isinstance(config, DictConfig):
            unstructured_configs.append(config)
            continue

        if isinstance(config, str):
            unstructured_configs.extend(_load_yaml_with_imports(config))
        elif isinstance(config, dict):
            unstructured_configs.append(OmegaConf.create(config))
        elif isinstance(config, list):
            unstructured_configs.append(OmegaConf.from_dotlist(config))
        else:
            structured_configs.append(OmegaConf.structured(config))

    # Merge structured configs first (provides defaults)
    if structured_configs:
        base = OmegaConf.merge(*structured_configs)
    else:
        base = OmegaConf.structured(Config())

    # Merge unstructured configs (may contain temporary keys like llm_configs)
    if unstructured_configs:
        unstructured_merged = OmegaConf.merge(*unstructured_configs)

        OmegaConf.resolve(unstructured_merged)

        # Remove temporary keys that aren't part of Config schema
        keys_to_remove = [k for k in unstructured_merged.keys() if k not in _CONFIG_KEYS]
        for key in keys_to_remove:
            del unstructured_merged[key]

        merged = OmegaConf.merge(base, unstructured_merged)
    else:
        merged = base

    return OmegaConf.to_object(merged)
