"""General chat / Q&A against local model."""

from __future__ import annotations

from core.models import IntentResult, ToolResult
from utils.logger import get_logger

logger = get_logger("tools.chat")


def reply(intent: IntentResult) -> ToolResult:
    """Respond to user message. Implementation to follow."""
    raise NotImplementedError("chat.reply deferred to implementation stage.")
