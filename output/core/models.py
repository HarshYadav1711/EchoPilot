"""Shared datatypes for transcription, intent, and tool outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class IntentName(str, Enum):
    """Canonical intent labels produced by the local classifier (extend as needed)."""

    UNKNOWN = "unknown"
    CHAT = "chat"
    SUMMARIZE = "summarize"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    CODE_GENERATE = "code_generate"


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    duration_s: Optional[float] = None


@dataclass
class IntentResult:
    name: IntentName
    confidence: float
    slots: Dict[str, Any] = field(default_factory=dict)
    raw_llm: Optional[str] = None


@dataclass
class ToolResult:
    ok: bool
    message: str
    payload: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Aggregate result for UI: transcription → intent → tool → summary."""

    transcription: TranscriptionResult
    intent: IntentResult
    action_label: str
    tool: ToolResult
