"""Constrained file read/write within the sandbox directory."""

from __future__ import annotations

from core.config import get_settings
from core.models import IntentResult, ToolResult
from utils.logger import get_logger

logger = get_logger("tools.file_ops")


def read_safe(intent: IntentResult) -> ToolResult:
    """Read a file under the data sandbox. Implementation to follow."""
    raise NotImplementedError("file_ops.read_safe deferred to implementation stage.")


def write_safe(intent: IntentResult) -> ToolResult:
    """
    Write a file under the data sandbox.

    Future: require explicit human confirmation before mutating paths.
    """
    raise NotImplementedError("file_ops.write_safe deferred to implementation stage.")


def sandbox_root() -> str:
    """Expose resolved sandbox path for callers and tests."""
    return str(get_settings().data_dir.resolve())
