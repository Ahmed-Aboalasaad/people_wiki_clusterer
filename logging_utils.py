"""
Logging utilities for the NLP clustering framework.

Call ``setup_logging()`` once at the start of every entry-point script.
All modules should obtain their loggers via ``logging.getLogger(__name__)``.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logger with a console handler and an optional file handler.

    Args:
        level:    Minimum log level (default: INFO).
        log_file: If given, also write logs to this path.
        fmt:      Log record format string.
        datefmt:  Date/time format string.
    """
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when the function is called more than once
    if root.handlers:
        root.handlers.clear()

    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper around ``logging.getLogger``."""
    return logging.getLogger(name)
