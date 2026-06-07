"""Structured logging configuration for the whole application.

A single :func:`configure_logging` call wires up a consistent, level-aware
formatter. Modules obtain a namespaced logger via :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the process.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``). Invalid
            values fall back to ``INFO``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Avoid duplicate handlers if a framework already attached one.
    root.handlers.clear()
    root.addHandler(handler)

    # Quieten noisy third-party libraries.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
