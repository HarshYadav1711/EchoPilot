"""Intent classification via local Ollama — strict JSON only, validated schemas."""

from __future__ import annotations

from core.config import Settings, get_settings
from core.models import IntentAnalysis, PrimaryIntent
from utils.json_intent import parse_json_loose, validate_intent_payload
from utils.logger import get_logger

logger = get_logger("intent")


_COMPOUND_SPLIT = " and "


def try_split_compound_two(user_text: str) -> tuple[str, str] | None:
    """
    Detect a simple two-clause command joined by `` and `` (spaces required).

    Splits on the **first** occurrence only so the right clause may still contain ``and``.
    Returns None if there is no split or either side is empty.
    """
    t = (user_text or "").strip()
    if not t:
        return None
    lower = t.lower()
    idx = lower.find(_COMPOUND_SPLIT)
    if idx == -1:
        return None
    left = t[:idx].strip()
    right = t[idx + len(_COMPOUND_SPLIT) :].strip()
    if not left or not right:
        return None
    return (left, right)


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
  "explanation_for_ui": "<one short English sentence — label for the choice>",
  "why_this_action": "<1–2 short sentences in plain English: why these intents fit what the user asked. No system prompts or JSON talk.>"
}

Rules:
- why_this_action must be readable by a non-technical reviewer (e.g. why write_code vs chat).
- sub_intents is the ordered plan (first step first). For a single action use one element equal to primary_intent.
- Compound examples:
  - Summarize text then save: sub_intents ["summarize","create_file"], arguments.path filename, arguments.text the source.
  - New Python file with retry logic: sub_intents ["create_file","write_code"] or ["write_code"] if only coding; set arguments.path and arguments.language "python" when known.
- Use summarize for condensation; create_file for writing a new file; write_code for code generation; general_chat for questions, greetings, or unclear requests.
- If unsure, use general_chat and low confidence (<0.5).
- arguments values must be JSON primitives or short strings only.
"""


_FALLBACK_EXPLANATION = "Could not parse intent reliably; continuing with a safe chat response."
_DEFAULT_USER_NOTICE = "Intent parsing failed, falling back to chat."


def _fallback_analysis(
    reason: str,
    raw_llm_text: str | None = None,
    *,
    user_notice: str | None = None,
) -> IntentAnalysis:
    return IntentAnalysis(
        primary_intent=PrimaryIntent.GENERAL_CHAT,
        sub_intents=[PrimaryIntent.GENERAL_CHAT],
        arguments={},
        confidence=0.0,
        requires_confirmation=True,
        explanation_for_ui=_FALLBACK_EXPLANATION,
        why_this_action="Intent could not be interpreted reliably, so the assistant will respond with general chat.",
        parse_warnings=[reason],
        raw_llm_text=raw_llm_text,
        compound_parts=[],
        per_step_arguments=None,
        intent_degraded=True,
        user_notice=user_notice or _DEFAULT_USER_NOTICE,
    )


def _per_segment_arguments(intent: PrimaryIntent, segment_text: str, model_args: dict) -> dict:
    """Prefer model args; pin clause text for summarize / chat so tools do not see the full compound."""
    args = dict(model_args)
    seg = segment_text.strip()
    if intent == PrimaryIntent.SUMMARIZE:
        args["text"] = seg
    elif intent == PrimaryIntent.GENERAL_CHAT:
        args.setdefault("query", seg)
    return args


def _merge_compound_two(
    left_text: str,
    right_text: str,
    left: IntentAnalysis,
    right: IntentAnalysis,
) -> IntentAnalysis:
    la = left.sub_intents[0] if left.sub_intents else left.primary_intent
    rb = right.sub_intents[0] if right.sub_intents else right.primary_intent
    sub_intents = [la, rb]
    per_step_arguments = [
        _per_segment_arguments(la, left_text, left.arguments),
        _per_segment_arguments(rb, right_text, right.arguments),
    ]
    merged_flat = {**left.arguments, **right.arguments}
    confidence = min(left.confidence, right.confidence)
    requires_confirmation = left.requires_confirmation or right.requires_confirmation
    parse_warnings: list[str] = []
    parse_warnings.extend(f"compound_left:{w}" for w in left.parse_warnings)
    parse_warnings.extend(f"compound_right:{w}" for w in right.parse_warnings)
    explanation = (
        f"Two-part request (in order): (1) {left.explanation_for_ui} "
        f"(2) {right.explanation_for_ui}"
    )
    raw: str | None = None
    if left.raw_llm_text or right.raw_llm_text:
        raw = (
            f"--- clause 1 ---\n{left.raw_llm_text or ''}\n"
            f"--- clause 2 ---\n{right.raw_llm_text or ''}"
        )
    why = (
        f"First part: {left.why_this_action or left.explanation_for_ui} "
        f"Second part: {right.why_this_action or right.explanation_for_ui}"
    )
    if len(why) > 450:
        why = why[:447] + "…"

    return IntentAnalysis(
        primary_intent=la,
        sub_intents=sub_intents,
        arguments=merged_flat,
        confidence=confidence,
        requires_confirmation=requires_confirmation,
        explanation_for_ui=explanation[:2000],
        why_this_action=why,
        parse_warnings=parse_warnings,
        raw_llm_text=raw,
        compound_parts=[left_text.strip(), right_text.strip()],
        per_step_arguments=per_step_arguments,
        intent_degraded=False,
        user_notice=None,
    )


class IntentClassifier:
    """Call Ollama with JSON mode, then validate — never trust raw text as intent."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def classify(self, user_text: str) -> IntentAnalysis:
        """Always returns a usable analysis; logs failures and never raises to the UI."""
        try:
            return self._classify_inner(user_text)
        except Exception as exc:
            logger.exception("Intent classification failed unexpectedly")
            return _fallback_analysis(
                f"unexpected:{exc.__class__.__name__}",
                user_notice="Intent parsing failed, falling back to chat.",
            )

    def _classify_inner(self, user_text: str) -> IntentAnalysis:
        text = (user_text or "").strip()
        if not text:
            return _fallback_analysis("empty_user_text")

        pair = try_split_compound_two(text)
        if pair is not None:
            left_t, right_t = pair
            try:
                left_a = self._analyze_once(left_t)
                right_a = self._analyze_once(right_t)
                return _merge_compound_two(left_t, right_t, left_a, right_a)
            except Exception as exc:
                logger.exception("Compound intent handling failed; falling back to single clause")
                return self._analyze_once(text)

        return self._analyze_once(text)

    def _analyze_once(self, text: str) -> IntentAnalysis:
        """Single LLM intent classification for one user clause (used alone or as one side of a compound)."""
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
                return _fallback_analysis(
                    "empty_ollama_response",
                    user_notice="Intent parsing failed, falling back to chat.",
                )
        except Exception as exc:
            logger.exception("Ollama intent call failed")
            return _fallback_analysis(
                f"ollama_error:{exc.__class__.__name__}",
                user_notice="Could not reach the intent model. Check Ollama is running, then try again.",
            )

        data, recovery = parse_json_loose(str(raw))
        parse_warnings: list[str] = []
        if recovery == "invalid_json" or data is None:
            logger.warning("Intent JSON parse failed after recovery attempts")
            return _fallback_analysis(
                "invalid_json",
                raw_llm_text=str(raw)[:4000],
                user_notice="Intent parsing failed, falling back to chat.",
            )
        if recovery == "recovered_from_embedded_json":
            parse_warnings.append("recovered_from_embedded_json")

        primary, subs, args, conf, req, expl, why, val_warnings = validate_intent_payload(
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
            why_this_action=why,
            parse_warnings=parse_warnings,
            raw_llm_text=str(raw)[:4000],
            compound_parts=[],
            per_step_arguments=None,
            intent_degraded=False,
            user_notice=None,
        )


def _resolve_primary_sub_mismatch(primary: PrimaryIntent, subs: list[PrimaryIntent]) -> list[str]:
    """Flag inconsistent primary vs ordered steps (informational)."""
    out: list[str] = []
    if len(subs) > 1 and primary not in subs:
        out.append("primary_intent_not_in_sub_intents")
    return out
