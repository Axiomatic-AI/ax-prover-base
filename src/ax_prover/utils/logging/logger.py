"""Logging utilities for ax-prover."""

import logging
import os
import sys

from pylatexenc.latex2text import LatexNodes2Text

from .langsmith import get_langsmith_aggregator


def _latex_to_unicode(text: str) -> str:
    """Convert LaTeX markup to Unicode characters for terminal display."""
    if not text:
        return text
    converter = LatexNodes2Text(math_mode="text", keep_comments=False)
    try:
        return converter.latex_to_text(text)
    except Exception:
        return text


class _LaTeXFormatter(logging.Formatter):
    """Custom formatter that converts LaTeX to Unicode in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return _latex_to_unicode(formatted)


def _setup_logger(name: str | None = None, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with rich formatting that shows file, line, and function.

    Args:
        name: Logger name (defaults to root logger)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)

    formatter = _LaTeXFormatter(
        fmt="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s",  # noqa: E501
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Create LangSmith log aggregator (always if LangSmith is enabled)
    langsmith_aggregator = get_langsmith_aggregator()
    if langsmith_aggregator:
        logger.addHandler(langsmith_aggregator)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get or create a logger with the standard configuration.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect

        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else "ax_prover"

    level = _configured_level or os.getenv("LOG_LEVEL", "INFO")

    return _setup_logger(name, level=level)


_configured_level: str | None = None


def reconfigure_log_level(level: str) -> None:
    """Reconfigure all ax_prover loggers to a new level.

    Call after config is loaded to override the default level
    set during module-level logger creation. Also stores the level
    so that loggers created later (via get_logger) use it automatically.
    """
    global _configured_level
    _configured_level = level.upper()
    numeric_level = getattr(logging, _configured_level, logging.INFO)
    for name, logger_ref in logging.Logger.manager.loggerDict.items():
        if name.startswith("ax_prover") and isinstance(logger_ref, logging.Logger):
            logger_ref.setLevel(numeric_level)
            for handler in logger_ref.handlers:
                handler.setLevel(numeric_level)
