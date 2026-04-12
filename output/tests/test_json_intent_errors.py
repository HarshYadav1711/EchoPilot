"""Strict JSON parsing edge cases and validation errors."""

from __future__ import annotations

from utils.json_intent import parse_json_loose, validate_intent_payload


def test_parse_json_rejects_garbage() -> None:
    data, tag = parse_json_loose("not json at all {{{")
    assert data is None
    assert tag == "invalid_json"


def test_parse_json_rejects_array_root() -> None:
    data, _ = parse_json_loose("[1,2,3]")
    assert data is None


def test_validate_requires_confirmation_string_coercion() -> None:
    raw = {
        "primary_intent": "summarize",
        "sub_intents": ["summarize"],
        "arguments": {},
        "confidence": 0.9,
        "requires_confirmation": "true",
        "explanation_for_ui": "ok",
    }
    _, _, _, _, req, _, _, _ = validate_intent_payload(raw, confidence_threshold=0.55)
    assert req is True


def test_validate_arguments_coerce_non_primitive() -> None:
    raw = {
        "primary_intent": "general_chat",
        "sub_intents": ["general_chat"],
        "arguments": {"nested": {"x": 1}},
        "confidence": 0.8,
        "requires_confirmation": False,
        "explanation_for_ui": "ok",
    }
    _, _, args, _, _, _, _, _ = validate_intent_payload(raw, confidence_threshold=0.55)
    assert isinstance(args["nested"], str)
    assert "x" in args["nested"]
