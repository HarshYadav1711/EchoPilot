"""Edge cases for sub_intents coercion."""

from __future__ import annotations

from core.models import PrimaryIntent
from utils.json_intent import validate_intent_payload


def test_sub_intents_all_invalid_strings_use_primary() -> None:
    raw = {
        "primary_intent": "summarize",
        "sub_intents": ["not_an_intent", "also_bad"],
        "arguments": {},
        "confidence": 0.9,
        "requires_confirmation": False,
        "explanation_for_ui": "x",
    }
    primary, subs, _, _, _, _, w = validate_intent_payload(raw, confidence_threshold=0.55)
    assert primary == PrimaryIntent.SUMMARIZE
    assert subs == [PrimaryIntent.SUMMARIZE]
    assert "sub_intents_invalid_all_dropped" in w


def test_arguments_dict_capped_at_64_keys() -> None:
    raw = {
        "primary_intent": "general_chat",
        "sub_intents": ["general_chat"],
        "arguments": {f"k{i}": "v" for i in range(100)},
        "confidence": 0.8,
        "requires_confirmation": False,
        "explanation_for_ui": "x",
    }
    _, _, args, _, _, _, _ = validate_intent_payload(raw, confidence_threshold=0.55)
    assert len(args) == 64
