"""Router resilience when execution layer raises."""

from __future__ import annotations

from unittest.mock import patch

from core.models import ActionPlan, ActionPlanStep, IntentAnalysis, PrimaryIntent
from core.router import IntentRouter


def test_execute_plan_returns_structured_failure_when_executor_raises() -> None:
    plan = ActionPlan(
        steps=[
            ActionPlanStep(
                order=1,
                intent=PrimaryIntent.GENERAL_CHAT,
                tool_route="tools.chat.reply",
                description="x",
                params={},
            )
        ],
        requires_confirmation=False,
        explanation_for_ui="x",
    )
    analysis = IntentAnalysis(
        primary_intent=PrimaryIntent.GENERAL_CHAT,
        sub_intents=[PrimaryIntent.GENERAL_CHAT],
        arguments={},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="x",
        parse_warnings=[],
    )
    router = IntentRouter()
    with patch("core.router.execute_action_plan", side_effect=RuntimeError("internal")):
        r = router.execute_plan(
            plan,
            analysis,
            user_utterance="hi",
            transcription_text="hi",
            dry_run=True,
            confirm_writes=False,
            allow_overwrite=False,
            action_timeline=None,
        )
    assert r.execution_status == "failure"
    assert "try again" in r.final_output.lower()
    assert r.step_logs == []
