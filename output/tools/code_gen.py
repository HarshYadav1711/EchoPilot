"""Generate small code snippets locally (sandboxed)."""

from __future__ import annotations

from core.models import IntentResult, ToolResult
from utils.logger import get_logger

logger = get_logger("tools.code_gen")


def generate(intent: IntentResult) -> ToolResult:
    """Produce code from intent slots; implementation to follow."""
    raise NotImplementedError("code_gen.generate deferred to implementation stage.")
