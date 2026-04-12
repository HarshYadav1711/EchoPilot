"""Execution confidence guard (intent layer)."""

from __future__ import annotations

from core.intent import apply_requires_user_confirmation
from core.models import IntentAnalysis, PrimaryIntent


def _analysis(conf: float) -> IntentAnalysis:
    return IntentAnalysis(
        primary_intent=PrimaryIntent.GENERAL_CHAT,
        sub_intents=[PrimaryIntent.GENERAL_CHAT],
        arguments={},
        confidence=conf,
        requires_confirmation=False,
        explanation_for_ui="test",
        why_this_action="test",
        parse_warnings=[],
    )


def test_apply_requires_user_confirmation_below_threshold() -> None:
    assert apply_requires_user_confirmation(_analysis(0.59), min_confidence=0.6) is True


def test_apply_requires_user_confirmation_at_or_above_threshold() -> None:
    assert apply_requires_user_confirmation(_analysis(0.6), min_confidence=0.6) is False
    assert apply_requires_user_confirmation(_analysis(0.9), min_confidence=0.6) is False
