"""Simple two-clause `` and `` compounds: split, merge, and per-step execution args."""

from __future__ import annotations

from unittest.mock import patch

from core.executor import execute_action_plan
from core.intent import _merge_compound_two, try_split_compound_two
from core.models import IntentAnalysis, PrimaryIntent, ToolResult
from core.router import compile_action_plan


def test_try_split_compound_two_basic() -> None:
    assert try_split_compound_two("summarize this text and save it to summary.txt") == (
        "summarize this text",
        "save it to summary.txt",
    )


def test_try_split_compound_two_case_insensitive_and() -> None:
    assert try_split_compound_two("Do A AND B") == ("Do A", "B")


def test_try_split_compound_two_none_when_no_and() -> None:
    assert try_split_compound_two("single phrase") is None


def test_try_split_compound_two_none_empty_side() -> None:
    assert try_split_compound_two("only left and ") is None
    assert try_split_compound_two(" and only right") is None


def test_merge_compound_two_sub_intents_and_per_step_args() -> None:
    left = IntentAnalysis(
        primary_intent=PrimaryIntent.SUMMARIZE,
        sub_intents=[PrimaryIntent.SUMMARIZE],
        arguments={},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="Summarize clause",
        parse_warnings=[],
    )
    right = IntentAnalysis(
        primary_intent=PrimaryIntent.CREATE_FILE,
        sub_intents=[PrimaryIntent.CREATE_FILE],
        arguments={"path": "summary.txt"},
        confidence=0.85,
        requires_confirmation=False,
        explanation_for_ui="Save file",
        parse_warnings=[],
    )
    m = _merge_compound_two("summarize this text", "save it to summary.txt", left, right)
    assert m.sub_intents == [PrimaryIntent.SUMMARIZE, PrimaryIntent.CREATE_FILE]
    assert m.compound_parts == ["summarize this text", "save it to summary.txt"]
    assert m.per_step_arguments is not None
    assert m.per_step_arguments[0]["text"] == "summarize this text"
    assert m.per_step_arguments[1]["path"] == "summary.txt"
    assert m.confidence == 0.85


def test_effective_arguments_for_step() -> None:
    a = IntentAnalysis(
        primary_intent=PrimaryIntent.SUMMARIZE,
        sub_intents=[PrimaryIntent.SUMMARIZE, PrimaryIntent.GENERAL_CHAT],
        arguments={"base": 1},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="x",
        parse_warnings=[],
        per_step_arguments=[{"text": "L"}, {"query": "R"}],
    )
    assert a.effective_arguments_for_step(1)["text"] == "L"
    assert a.effective_arguments_for_step(2)["query"] == "R"


def test_executor_passes_merged_arguments_per_step() -> None:
    calls: list[dict] = []

    def capture(intent: PrimaryIntent, analysis: IntentAnalysis, ctx) -> ToolResult:
        calls.append(dict(analysis.arguments))
        return ToolResult(ok=True, message="ok")

    analysis = IntentAnalysis(
        primary_intent=PrimaryIntent.SUMMARIZE,
        sub_intents=[PrimaryIntent.SUMMARIZE, PrimaryIntent.GENERAL_CHAT],
        arguments={"base": 1},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="x",
        parse_warnings=[],
        per_step_arguments=[{"text": "first"}, {"query": "second"}],
    )
    plan = compile_action_plan(analysis)
    with patch("core.executor.dispatch_intent_step", side_effect=capture):
        r = execute_action_plan(
            plan,
            analysis,
            user_utterance="u",
            transcription_text="u",
            dry_run=True,
            confirm_writes=False,
            allow_overwrite=False,
        )
    assert r.execution_status == "dry_run"
    assert calls[0]["text"] == "first"
    assert calls[1]["query"] == "second"
