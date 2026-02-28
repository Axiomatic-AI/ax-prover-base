"""Configuration utilities for loading and merging OmegaConf configurations."""

import importlib.resources as pkg_resources
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from platformdirs import user_config_dir

from ..config import Config

# Keys that are valid in the Config dataclass (used for filtering temporary keys)
_CONFIG_KEYS = frozenset(field.name for field in fields(Config))

# Package root: utils/config.py -> ax_prover/ -> src/ -> repo root (works for editable installs)
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# User-global secrets location (written by `ax-prover configure`)
USER_SECRETS_PATH = Path(user_config_dir("ax-prover")) / ".env.secrets"


def _get_bundled_config_dir() -> Path:
    """Get path to bundled config files shipped inside the package."""
    return Path(str(pkg_resources.files("ax_prover.configs")))


def resolve_config_path(config_path: str | Path, folder: str | Path | None = None) -> Path:
    """Resolve a config file path with priority: CWD > --folder > bundled configs > package root.

    Args:
        config_path: Config file path (absolute or relative)
        folder: Optional project folder path (from --folder CLI arg)

    Raises:
        FileNotFoundError: If the config file cannot be found in any search path
    """
    path = Path(config_path)

    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found: {path}")

    # Priority 1: relative to CWD
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    # Priority 2: relative to --folder
    if folder is not None:
        folder_path = Path(folder) / path
        if folder_path.exists():
            return folder_path.resolve()

    # Priority 3: bundled configs inside the package (works for pip installs)
    bundled_path = _get_bundled_config_dir() / path
    if bundled_path.exists():
        return bundled_path.resolve()

    # Priority 4: relative to package root (works for editable installs)
    root_path = _PACKAGE_ROOT / path
    if root_path.exists():
        return root_path.resolve()
    # Also check configs/ subdirectory at package root
    root_configs_path = _PACKAGE_ROOT / "configs" / path
    if root_configs_path.exists():
        return root_configs_path.resolve()

    searched = [f"  - {Path.cwd()} (cwd)"]
    if folder is not None:
        searched.append(f"  - {Path(folder)} (--folder)")
    searched.append(f"  - {_get_bundled_config_dir()} (bundled)")
    searched.append(f"  - {_PACKAGE_ROOT} (package root)")
    raise FileNotFoundError(
        f"Config file not found: {config_path}\nSearched in:\n" + "\n".join(searched)
    )


def _load_yaml_with_imports(file_path: str | Path) -> list[DictConfig]:
    """Recursively load a YAML file and all its imports.

    Import paths are resolved relative to the importing file's directory.
    Returns a list of DictConfigs in merge order (imports first, then the file itself).
    """
    file_path = Path(file_path).resolve()
    loaded = OmegaConf.load(file_path)
    import_files = loaded.pop("import", [])

    result = []
    # Process imports recursively (they have lower precedence)
    for import_file in import_files:
        # Resolve relative to the importing file's directory
        resolved = file_path.parent / import_file
        result.extend(_load_yaml_with_imports(resolved))

    result.append(loaded)
    return result


def load_env_secrets(folder: str | Path | None = None) -> None:
    """Load secrets with cascading inheritance: CWD > folder > user global > package root.

    Each layer inherits variables it doesn't define from layers below.
    Shell environment variables are never overridden.
    """
    from dotenv import load_dotenv

    # Load highest-priority first with override=False.
    # First file to set a variable wins; shell env is never touched.
    load_dotenv(Path.cwd() / ".env.secrets", override=False)
    if folder is not None:
        load_dotenv(Path(folder) / ".env.secrets", override=False)
    load_dotenv(USER_SECRETS_PATH, override=False)
    load_dotenv(_PACKAGE_ROOT / ".env.secrets", override=False)


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
    folder: str | Path | None = None,
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
        folder: Optional project folder path for config file resolution (from --folder)

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
          - ../llms.yaml

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
            resolved = resolve_config_path(config, folder)
            unstructured_configs.extend(_load_yaml_with_imports(resolved))
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
