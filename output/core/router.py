"""Map classified intents to safe tool calls."""

from __future__ import annotations

from core.models import IntentName, IntentResult, ToolResult
from tools import chat, code_gen, file_ops, summarizer
from utils.logger import get_logger

logger = get_logger("router")


class IntentRouter:
    """Dispatch intents to tools; extend for compound commands and confirmation gates."""

    def route(self, intent: IntentResult) -> tuple[str, ToolResult]:
        """
        Return (action_label, tool_result).

        Future: multi-step plans, confirmation before destructive file writes, fallbacks.
        """
        name = intent.name
        if name == IntentName.CHAT:
            return "chat.reply", chat.reply(intent)
        if name == IntentName.SUMMARIZE:
            return "summarizer.run", summarizer.run(intent)
        if name == IntentName.FILE_READ:
            return "file_ops.read_safe", file_ops.read_safe(intent)
        if name == IntentName.FILE_WRITE:
            return "file_ops.write_safe", file_ops.write_safe(intent)
        if name == IntentName.CODE_GENERATE:
            return "code_gen.generate", code_gen.generate(intent)
        return "noop", ToolResult(ok=False, message="Unknown or unsupported intent.")
