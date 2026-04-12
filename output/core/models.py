"""Shared datatypes for transcription, intent, and tool outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PrimaryIntent(str, Enum):
    """Allowed intent labels (must match LLM JSON contract)."""

    CREATE_FILE = "create_file"
    WRITE_CODE = "write_code"
    SUMMARIZE = "summarize"
    GENERAL_CHAT = "general_chat"


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
class IntentAnalysis:
    """
    Validated structured output from the intent layer (Ollama JSON).

    ``parse_warnings`` records recovery steps (e.g. invalid JSON repaired, confidence adjusted).
    """

    primary_intent: PrimaryIntent
    sub_intents: List[PrimaryIntent]
    arguments: Dict[str, Any]
    confidence: float
    requires_confirmation: bool
    explanation_for_ui: str
    parse_warnings: List[str] = field(default_factory=list)
    raw_llm_text: Optional[str] = None
    # Simple two-clause "X and Y" compounds: raw segments for UI; per-step arg overrides for tools.
    compound_parts: List[str] = field(default_factory=list)
    per_step_arguments: Optional[List[Dict[str, Any]]] = None
    # When True, the classifier used a safe fallback (see ``user_notice`` for a short UI message).
    intent_degraded: bool = False
    user_notice: Optional[str] = None

    @property
    def primary_intent_value(self) -> str:
        return self.primary_intent.value

    def effective_arguments_for_step(self, order_1based: int) -> Dict[str, Any]:
        """Merge shared ``arguments`` with overrides for this plan step (1-based order)."""
        base = dict(self.arguments)
        ps = self.per_step_arguments
        if ps and 1 <= order_1based <= len(ps):
            base = {**base, **ps[order_1based - 1]}
        return base

    def is_compound_two_part(self) -> bool:
        return len(self.compound_parts) == 2


@dataclass
class ActionPlanStep:
    """One ordered step in a safe local execution plan."""

    order: int
    intent: PrimaryIntent
    tool_route: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPlan:
    """Ordered plan derived from ``IntentAnalysis`` (display and future executor)."""

    steps: List[ActionPlanStep]
    requires_confirmation: bool
    explanation_for_ui: str


@dataclass
class ToolResult:
    ok: bool
    message: str
    payload: Optional[Dict[str, Any]] = None
    dry_run_preview: Optional[str] = None
    affected_paths: List[str] = field(default_factory=list)


@dataclass
class RouterExecutionResult:
    """
    Outcome of running an action plan through local tools.

    ``step_logs`` holds structured entries for auditing (intent, status, notes).
    """

    action_taken: str
    files_created_or_modified: List[str]
    execution_status: str
    final_output: str
    warnings: List[str]
    step_logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Aggregate result for UI: transcription → intent → action plan → optional execution."""

    transcription: TranscriptionResult
    intent_analysis: IntentAnalysis
    action_plan: ActionPlan
    execution: Optional["RouterExecutionResult"] = None
