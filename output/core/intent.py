"""Intent classification via local Ollama — strict JSON only, validated schemas."""

from __future__ import annotations

from core.config import Settings, get_settings
from core.models import IntentAnalysis, PrimaryIntent
from utils.json_intent import parse_json_loose, validate_intent_payload
from utils.logger import get_logger

logger = get_logger("intent")


def apply_requires_user_confirmation(analysis: IntentAnalysis, *, min_confidence: float) -> bool:
    """
    When True, the UI must not run Apply immediately — user confirms first (low-confidence guard).

    Independent of the model's ``requires_confirmation`` flag and of validation thresholds
    used inside :func:`validate_intent_payload`.
    """
    return analysis.confidence < min_confidence


_SYSTEM_PROMPT = """You are a strict JSON intent classifier for a local desktop assistant.
Reply with one JSON object only. No markdown fences, no commentary.

Allowed intent strings (exactly these tokens):
create_file, write_code, summarize, general_chat

Required JSON shape:
{
  "primary_intent": "<string>",
  "sub_intents": ["<strings in execution order>"],
  "arguments": { },
  "confidence": <number from 0.0 to 1.0>,
  "requires_confirmation": <boolean>,
  "explanation_for_ui": "<one short English sentence>"
}

Rules:
- sub_intents is the ordered plan (first step first). For a single action use one element equal to primary_intent.
- Compound examples:
  - Summarize text then save: sub_intents ["summarize","create_file"], arguments.path filename, arguments.text the source.
  - New Python file with retry logic: sub_intents ["create_file","write_code"] or ["write_code"] if only coding; set arguments.path and arguments.language "python" when known.
- Use summarize for condensation; create_file for writing a new file; write_code for code generation; general_chat for questions, greetings, or unclear requests.
- If unsure, use general_chat and low confidence (<0.5).
- arguments values must be JSON primitives or short strings only.
"""


def _fallback_analysis(reason: str, raw_llm_text: str | None = None) -> IntentAnalysis:
    return IntentAnalysis(
        primary_intent=PrimaryIntent.GENERAL_CHAT,
        sub_intents=[PrimaryIntent.GENERAL_CHAT],
        arguments={},
        confidence=0.0,
        requires_confirmation=True,
        explanation_for_ui="Classification unavailable; using safe general-chat fallback.",
        parse_warnings=[reason],
        raw_llm_text=raw_llm_text,
    )


class IntentClassifier:
    """Call Ollama with JSON mode, then validate — never trust raw text as intent."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def classify(self, user_text: str) -> IntentAnalysis:
        text = (user_text or "").strip()
        if not text:
            return _fallback_analysis("empty_user_text")

        raw: str | None = None
        try:
            from ollama import Client

            client = Client(host=self._settings.ollama_host)
            logger.info("Intent request model=%s", self._settings.ollama_model)
            resp = client.chat(
                model=self._settings.ollama_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"User text:\n{text}\n"},
                ],
                format="json",
                options={"temperature": self._settings.ollama_intent_temperature},
            )
            raw = resp.get("message", {}).get("content")
            if not raw or not str(raw).strip():
                logger.warning("Ollama returned empty content")
                return _fallback_analysis("empty_ollama_response")
        except Exception as exc:
            logger.warning(
                "Ollama intent call failed: %s: %s",
                exc.__class__.__name__,
                exc,
            )
            return _fallback_analysis(f"ollama_error:{exc.__class__.__name__}")

        data, recovery = parse_json_loose(str(raw))
        parse_warnings: list[str] = []
        if recovery == "invalid_json" or data is None:
            logger.warning("Intent JSON parse failed after recovery attempts")
            return _fallback_analysis("invalid_json", raw_llm_text=str(raw)[:4000])
        if recovery == "recovered_from_embedded_json":
            parse_warnings.append("recovered_from_embedded_json")

        primary, subs, args, conf, req, expl, val_warnings = validate_intent_payload(
            data,
            confidence_threshold=self._settings.intent_confidence_threshold,
        )
        parse_warnings.extend(val_warnings)
        parse_warnings.extend(_resolve_primary_sub_mismatch(primary, subs))

        return IntentAnalysis(
            primary_intent=primary,
            sub_intents=subs,
            arguments=args,
            confidence=conf,
            requires_confirmation=req,
            explanation_for_ui=expl,
            parse_warnings=parse_warnings,
            raw_llm_text=str(raw)[:4000],
        )


def _resolve_primary_sub_mismatch(primary: PrimaryIntent, subs: list[PrimaryIntent]) -> list[str]:
    """Flag inconsistent primary vs ordered steps (informational)."""
    out: list[str] = []
    if len(subs) > 1 and primary not in subs:
        out.append("primary_intent_not_in_sub_intents")
    return out
