"""Intent classification via local Ollama LLM. Implementation to follow."""

from __future__ import annotations

from core.config import Settings, get_settings
from core.models import IntentName, IntentResult
from utils.logger import get_logger

logger = get_logger("intent")


class IntentClassifier:
    """Parse user text into a structured intent using Ollama JSON-style prompts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def classify(self, user_text: str) -> IntentResult:
        """Return structured intent; placeholder until Ollama client is wired."""
        raise NotImplementedError("Intent classification deferred to implementation stage.")

    @staticmethod
    def _default_unknown() -> IntentResult:
        return IntentResult(name=IntentName.UNKNOWN, confidence=0.0, slots={}, raw_llm=None)
