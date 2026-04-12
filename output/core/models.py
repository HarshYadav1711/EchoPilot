"""Shared datatypes for transcription, intent, and tool outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentName(str, Enum):
    """Canonical intent labels produced by the local classifier (extend as needed)."""

    UNKNOWN = "unknown"
    CHAT = "chat"
    SUMMARIZE = "summarize"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    CODE_GENERATE = "code_generate"


class TranscriptionSource(str, Enum):
    """Where the audio came from (UI layer)."""

    MICROPHONE = "microphone"
    UPLOAD = "upload"


@dataclass
class TranscriptionResult:
    """
    STT output. When ``ok`` is False, ``error`` explains the failure; ``text`` is empty.
    ``warnings`` collects non-fatal issues (e.g. skipped ffmpeg normalization).
    """

    ok: bool
    text: str
    language: Optional[str] = None
    duration_s: Optional[float] = None
    source_type: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        source_type: Optional[str] = None,
        warnings: Optional[List[str]] = None,
    ) -> "TranscriptionResult":
        return cls(
            ok=False,
            text="",
            source_type=source_type,
            warnings=list(warnings or []),
            error=error,
        )

    @classmethod
    def success(
        cls,
        text: str,
        *,
        language: Optional[str],
        duration_s: Optional[float],
        source_type: Optional[str],
        warnings: Optional[List[str]] = None,
    ) -> "TranscriptionResult":
        return cls(
            ok=True,
            text=text,
            language=language,
            duration_s=duration_s,
            source_type=source_type,
            warnings=list(warnings or []),
            error=None,
        )


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
