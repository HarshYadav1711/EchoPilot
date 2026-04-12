"""Summarize text via local model (Ollama)."""

from __future__ import annotations

from core.models import IntentAnalysis, ToolResult
from utils.logger import get_logger

logger = get_logger("tools.summarizer")


def run(intent: IntentAnalysis) -> ToolResult:
    """Summarize content from intent slots. Implementation to follow."""
    raise NotImplementedError("summarizer.run deferred to implementation stage.")
