"""Per-run execution state passed through tools (dry-run, confirmation, step handoff)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExecutionContext:
    """Human-in-the-loop flags and data passed between plan steps."""

    user_utterance: str
    transcription_text: str
    dry_run: bool
    confirm_writes: bool
    allow_overwrite: bool
    accumulated: Dict[str, Any] = field(default_factory=dict)
