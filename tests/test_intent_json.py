"""Unit tests for strict JSON intent parsing and action plans (no Ollama)."""

from __future__ import annotations

from core.models import IntentAnalysis, PrimaryIntent
from core.router import compile_action_plan
from utils.json_intent import parse_json_loose, validate_intent_payload


def test_parse_json_strict() -> None:
    payload = '{"primary_intent":"summarize","sub_intents":["summarize"],"arguments":{},"confidence":0.9,"requires_confirmation":false,"explanation_for_ui":"ok"}'
    data, tag = parse_json_loose(payload)
    assert data is not None
    assert tag is None


def test_parse_json_fenced() -> None:
    text = '```json\n{"primary_intent":"general_chat","sub_intents":["general_chat"],"arguments":{},"confidence":0.5,"requires_confirmation":true,"explanation_for_ui":"x"}\n```'
    data, tag = parse_json_loose(text)
    assert data is not None


def test_parse_json_embedded() -> None:
    text = 'Here you go: {"primary_intent":"write_code","sub_intents":["write_code"],"arguments":{"language":"python"},"confidence":0.8,"requires_confirmation":false,"explanation_for_ui":"code"} trailing'
    data, tag = parse_json_loose(text)
    assert data is not None
    assert tag == "recovered_from_embedded_json"


def test_validate_low_confidence_forces_confirmation() -> None:
    raw = {
        "primary_intent": "summarize",
        "sub_intents": ["summarize"],
        "arguments": {},
        "confidence": 0.2,
        "requires_confirmation": False,
        "explanation_for_ui": "test",
    }
    primary, subs, args, conf, req, expl, why, w = validate_intent_payload(raw, confidence_threshold=0.55)
    assert primary == PrimaryIntent.SUMMARIZE
    assert req is True
    assert "low_confidence_forces_confirmation" in w
    assert conf == 0.2
    assert why == "test"


def test_validate_compound_sub_intents() -> None:
    raw = {
        "primary_intent": "summarize",
        "sub_intents": ["summarize", "create_file"],
        "arguments": {"path": "summary.txt"},
        "confidence": 0.9,
        "requires_confirmation": False,
        "explanation_for_ui": "Summarize then save.",
    }
    primary, subs, _, _, _, _, _, _ = validate_intent_payload(raw, confidence_threshold=0.55)
    assert primary == PrimaryIntent.SUMMARIZE
    assert [s.value for s in subs] == ["summarize", "create_file"]


def test_compile_action_plan_order() -> None:
    analysis = IntentAnalysis(
        primary_intent=PrimaryIntent.SUMMARIZE,
        sub_intents=[PrimaryIntent.SUMMARIZE, PrimaryIntent.CREATE_FILE],
        arguments={"path": "summary.txt"},
        confidence=0.9,
        requires_confirmation=False,
        explanation_for_ui="Two steps",
        why_this_action="User asked to summarize then save to a file.",
        parse_warnings=[],
    )
    plan = compile_action_plan(analysis)
    assert len(plan.steps) == 2
    assert plan.steps[0].intent == PrimaryIntent.SUMMARIZE
    assert plan.steps[1].intent == PrimaryIntent.CREATE_FILE
    assert "summarizer.run" in plan.steps[0].tool_route
    assert "file_ops.write_safe" in plan.steps[1].tool_route


def test_invalid_primary_coerces_to_general_chat() -> None:
    raw = {
        "primary_intent": "not_a_real_intent",
        "sub_intents": [],
        "arguments": {},
        "confidence": 1.0,
        "requires_confirmation": False,
        "explanation_for_ui": "x",
    }
    primary, subs, _, _, _, _, _, w = validate_intent_payload(raw, confidence_threshold=0.55)
    assert primary == PrimaryIntent.GENERAL_CHAT
    assert "invalid_or_missing_primary_intent" in w
    assert subs == [PrimaryIntent.GENERAL_CHAT]


def test_validate_why_this_action_explicit() -> None:
    raw = {
        "primary_intent": "write_code",
        "sub_intents": ["write_code"],
        "arguments": {"language": "python"},
        "confidence": 0.9,
        "requires_confirmation": False,
        "explanation_for_ui": "Code task",
        "why_this_action": "User requested creation of a Python file with specific functionality.",
    }
    _, _, _, _, _, _, why, _ = validate_intent_payload(raw, confidence_threshold=0.55)
    assert "Python" in why
