"""Utilities for ax-prover."""

from .config import load_env_secrets, merge_configs, resolve_config_path, save_config
from .files import write_json_output
from .git import get_git_hash, get_repo_metadata, is_git_dirty

# Export Lean parsing utilities
from .lean_parsing import (
    LEAN_KEYWORDS,
    count_sorries,
    extract_function_from_content,
    extract_theorem_name,
    get_function_from_location,
    get_unproven,
    list_all_declarations_in_lean_code,
    list_all_declarations_in_path_as_text,
    normalize_location,
    strip_comments,
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
    get_item_from_line,
    get_item_from_location,
    get_items_from_lean_file,
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
    "extract_function_from_content",
    "extract_theorem_name",
    "get_function_from_location",
    "get_unproven",
    "list_all_declarations_in_lean_code",
    "list_all_declarations_in_path_as_text",
    "normalize_location",
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
    "LEAN_KEYWORDS",
    "count_sorries",
    "strip_comments",
    # Proving
    "get_item_from_line",
    "get_item_from_location",
    "get_items_from_lean_file",
    "parse_prove_target",
    "prove_single_item",
]
