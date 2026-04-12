"""Executor aggregation, empty plans, and error handling (tools mocked)."""

from __future__ import annotations

from unittest.mock import patch

from core.executor import execute_action_plan
from core.models import ActionPlan, ActionPlanStep, IntentAnalysis, PrimaryIntent, ToolResult
from core.router import compile_action_plan


def test_execute_empty_plan() -> None:
    plan = ActionPlan(steps=[], requires_confirmation=False, explanation_for_ui="")
    analysis = IntentAnalysis(
        primary_intent=PrimaryIntent.GENERAL_CHAT,
        sub_intents=[PrimaryIntent.GENERAL_CHAT],
        arguments={},
        confidence=0.5,
        requires_confirmation=True,
        explanation_for_ui="x",
        why_this_action="Test.",
        parse_warnings=[],
    )
    r = execute_action_plan(
        plan,
        analysis,
        user_utterance="hi",
        transcription_text="hi",
        dry_run=True,
        confirm_writes=False,
        allow_overwrite=False,
    )
    assert r.execution_status == "failure"
    assert "Empty" in r.warnings[0]


def test_execute_partial_failure_via_mock() -> None:
    analysis = IntentAnalysis(
        primary_intent=PrimaryIntent.SUMMARIZE,
        sub_intents=[PrimaryIntent.SUMMARIZE, PrimaryIntent.GENERAL_CHAT],
        arguments={},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="x",
        why_this_action="Test.",
        parse_warnings=[],
    )
    plan = compile_action_plan(analysis)

    def _side_effect(intent: PrimaryIntent, *_args, **_kwargs) -> ToolResult:
        if intent == PrimaryIntent.SUMMARIZE:
            return ToolResult(ok=True, message="summary text")
        return ToolResult(ok=False, message="chat failed")

    with patch("core.executor.dispatch_intent_step", side_effect=_side_effect):
        r = execute_action_plan(
            plan,
            analysis,
            user_utterance="u",
            transcription_text="u",
            dry_run=False,
            confirm_writes=False,
            allow_overwrite=False,
        )
    assert r.execution_status == "partial_failure"
    assert any("chat failed" in w for w in r.warnings)


def test_execute_blocked_confirmation_flag() -> None:
    analysis = IntentAnalysis(
        primary_intent=PrimaryIntent.CREATE_FILE,
        sub_intents=[PrimaryIntent.CREATE_FILE],
        arguments={"path": "x.txt", "content": "c"},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="x",
        why_this_action="Test.",
        parse_warnings=[],
    )
    plan = compile_action_plan(analysis)

    def _blocked(*_a, **_k) -> ToolResult:
        return ToolResult(
            ok=False,
            message="need confirm",
            payload={"blocked": "confirmation"},
        )

    with patch("core.executor.dispatch_intent_step", side_effect=_blocked):
        r = execute_action_plan(
            plan,
            analysis,
            user_utterance="u",
            transcription_text="u",
            dry_run=False,
            confirm_writes=False,
            allow_overwrite=False,
        )
    assert r.execution_status == "blocked"
    assert any("blocked" in w.lower() for w in r.warnings)


def test_dispatch_exception_in_execute_loop() -> None:
    analysis = IntentAnalysis(
        primary_intent=PrimaryIntent.GENERAL_CHAT,
        sub_intents=[PrimaryIntent.GENERAL_CHAT],
        arguments={},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="x",
        why_this_action="Test.",
        parse_warnings=[],
    )
    plan = compile_action_plan(analysis)

    with patch("core.executor.dispatch_intent_step", side_effect=RuntimeError("x")):
        r = execute_action_plan(
            plan,
            analysis,
            user_utterance="u",
            transcription_text="u",
            dry_run=True,
            confirm_writes=False,
            allow_overwrite=False,
        )
    assert r.execution_status == "failure"
    assert not r.step_logs[0]["ok"]
