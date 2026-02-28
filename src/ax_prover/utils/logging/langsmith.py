"""LangSmith integration for logging. Captures logs and sends them to LangSmith traces."""

from __future__ import annotations

import collections
import difflib
import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from langsmith import traceable
from langsmith.schemas import Attachment


class LangSmithLogAggregator(logging.Handler):
    """Simple log handler that aggregates logs in memory for later attachment to traces."""

    def __init__(self, maxlen: int = 10000):
        super().__init__()
        self._log_records: collections.deque = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord):
        """Collect a log record in memory."""
        try:
            log_message = self.format(record)
        except Exception:
            log_message = record.getMessage()

        extra_info = f"[{record.levelname}] {record.name}"
        if record.exc_info:
            extra_info += f"\n{''.join(traceback.format_exception(*record.exc_info))}"

        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": log_message,
            "extra_info": extra_info,
        }

        self._log_records.append(log_entry)

    def get_aggregated_logs(self) -> list[dict]:
        """Return all collected log entries."""
        return list(self._log_records)

    def clear_logs(self):
        """Clear all collected log entries."""
        self._log_records.clear()


def _is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing should be enabled."""
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    tracing = (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        or os.getenv("LANGSMITH_TRACING_V2", "").lower() == "true"
    )

    return api_key and tracing


@lru_cache(maxsize=1)
def get_langsmith_aggregator() -> LangSmithLogAggregator | None:
    """Get a LangSmith log aggregator. Uses LRU cache to avoid creating multiple instances."""
    if not _is_langsmith_enabled():
        return None

    aggregator = LangSmithLogAggregator()
    # No cleanup needed - aggregator has no background threads
    return aggregator


@traceable(name="attach_lean_files")
def attach_lean_files(
    original_file_path: str,
    modified_file_path: str,
    original_file: Attachment,
    modified_file: Attachment,
    diff_file: Attachment,
) -> None:
    """Attach original and modified Lean files plus their diff to the LangSmith trace.

    This function follows the pattern described in the LangSmith documentation:
    https://docs.langchain.com/langsmith/upload-files-with-traces

    IMPORTANT: Even though this function appears to do nothing, it actually attaches
    the files to the trace. The @traceable decorator automatically captures the
    Attachment objects passed as arguments and uploads them to LangSmith, making
    them visible and downloadable in the trace UI.

    Args:
        original_file_path: Path to the original file (for metadata/labeling)
        modified_file_path: Path to the modified file (for metadata/labeling)
        original_file: Attachment containing original file data (automatically uploaded by @traceable)
        modified_file: Attachment containing modified file data (automatically uploaded by @traceable)
        diff_file: Attachment containing unified diff between files (automatically uploaded by @traceable)
    """
    pass


def attach_builder_files(
    base_folder: str,
    original_file_relative_path: str,
    modified_file_relative_path: str,
) -> None:
    """Helper function to attach original, modified, and diff files from builder node.

    This is a convenience function used by both ProverAgent and FormalizationAgent
    to attach files to LangSmith traces during the builder node execution.

    Args:
        base_folder: Base folder path for the project
        original_file_relative_path: Relative path to the original file
        modified_file_relative_path: Relative path to the modified (temp) file
    """
    original_file_path = Path(base_folder) / original_file_relative_path
    modified_file_path = Path(base_folder) / modified_file_relative_path

    original_text = original_file_path.read_text()
    modified_text = modified_file_path.read_text()

    diff = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        modified_text.splitlines(keepends=True),
        fromfile=original_file_relative_path,
        tofile=modified_file_relative_path,
    )
    diff_text = "".join(diff)

    original_attachment = Attachment(
        mime_type="text/plain",
        data=original_file_path.read_bytes(),
    )
    modified_attachment = Attachment(
        mime_type="text/plain",
        data=modified_file_path.read_bytes(),
    )
    diff_attachment = Attachment(
        mime_type="text/plain",
        data=diff_text.encode("utf-8"),
    )

    attach_lean_files(
        original_file_path=original_file_relative_path,
        modified_file_path=modified_file_relative_path,
        original_file=original_attachment,
        modified_file=modified_attachment,
        diff_file=diff_attachment,
    )


@traceable(name="attach_aggregated_logs")
def _attach_aggregated_logs(log_data: Attachment, metadata: dict) -> None:
    """Attach aggregated logs to the LangSmith trace.

    This function follows the same pattern as attach_lean_files. The @traceable
    decorator automatically captures the Attachment object and uploads it to
    LangSmith, making it visible in the trace UI.

    Args:
        log_data: Attachment containing aggregated log data (JSON format)
        metadata: Metadata dict with stats about the logs (total_logs, time_range)
    """
    pass


def attach_prover_logs_if_enabled() -> None:
    """Attach aggregated logs to the current trace if LangSmith is enabled.

    This function collects all logs from the log aggregator, converts them to JSON,
    and attaches them to the trace.
    """
    if not _is_langsmith_enabled():
        return

    try:
        aggregator = get_langsmith_aggregator()
        if not aggregator:
            return

        logs = aggregator.get_aggregated_logs()
        if not logs:
            return

        json_data = json.dumps(logs, indent=2).encode("utf-8")

        log_attachment = Attachment(
            mime_type="application/json",
            data=json_data,
        )

        timestamps = [log.get("timestamp") for log in logs if log.get("timestamp")]
        time_range = {}
        if timestamps:
            time_range = {
                "first": min(timestamps),
                "last": max(timestamps),
            }

        metadata = {
            "total_logs": len(logs),
            "time_range": time_range,
        }

        _attach_aggregated_logs(log_attachment, metadata)

        aggregator.clear_logs()

    except Exception as e:
        # Don't fail the agent if log attachment fails
        logging.getLogger(__name__).warning(f"Failed to attach aggregated logs to LangSmith: {e}")
