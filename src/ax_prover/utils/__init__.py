"""Utilities for ax-prover."""

from .config import load_env_secrets, merge_configs, resolve_config_path, save_config
from .files import write_json_output
from .git import get_git_hash, get_repo_metadata, is_git_dirty

# Export Lean parsing utilities
from .lean_parsing import (
    list_declarations_from_code,
    list_declarations_from_file,
)

# Export logging utilities
from .logging import (
    attach_builder_files,
    attach_lean_files,
    attach_prover_logs_if_enabled,
    get_logger,
    reconfigure_log_level,
)

# Export proving utilities
from .proving import (
    parse_prove_target,
    prove_single_item,
)

__all__ = [
    # Config
    "load_env_secrets",
    "merge_configs",
    "resolve_config_path",
    "save_config",
    # Files
    "write_json_output",
    # Logging
    "get_logger",
    "reconfigure_log_level",
    "attach_lean_files",
    "attach_builder_files",
    "attach_prover_logs_if_enabled",
    # Git
    "get_git_hash",
    "get_repo_metadata",
    "is_git_dirty",
    # Lean
    "list_declarations_from_code",
    "list_declarations_from_file",
    # Proving
    "parse_prove_target",
    "prove_single_item",
]
