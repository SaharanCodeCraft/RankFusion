"""
Logging utilities for the Cranfield IR project.

Provides a consistent, configured logger so that every module can
``from src.utils.logging_utils import get_logger`` and get uniform output.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a project logger that writes to stderr with a standard format.

    Args:
        name:  Logical name for the logger, typically ``__name__``.
        level: Minimum severity level to emit (default ``INFO``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
