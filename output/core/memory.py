"""Session memory and action history (in-memory first; persistence hooks later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.models import PipelineResult


@dataclass
class SessionMemory:
    """
    Holds recent turns for compound commands and audit.

    Future: persist to disk under the sandbox, human-confirmation queues, TTL.
    """

    max_entries: int = 50
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, result: PipelineResult) -> None:
        """Record a completed pipeline result."""
        self.entries.append(
            {
                "primary_intent": result.intent_analysis.primary_intent.value,
                "requires_confirmation": result.action_plan.requires_confirmation,
                "steps": [s.tool_route for s in result.action_plan.steps],
            }
        )
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def clear(self) -> None:
        self.entries.clear()
