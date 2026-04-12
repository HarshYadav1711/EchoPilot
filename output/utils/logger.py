"""Centralized logging configuration for EchoPilot."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO, name: str = "echopilot") -> logging.Logger:
    """Configure root logger once; idempotent for repeated calls."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the EchoPilot namespace."""
    base = "echopilot"
    full = f"{base}.{name}" if name else base
    return logging.getLogger(full)
