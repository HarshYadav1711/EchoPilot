"""Extract and strictly validate intent JSON from model output (no free-form intent decisions)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from core.models import PrimaryIntent


def strip_code_fences(text: str) -> str:
    """Remove optional ```json ... ``` wrappers."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_first_json_object(text: str) -> str | None:
    """
    Find the first balanced `{ ... }` substring that looks like JSON.

    Used only after strict full-string parse fails.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
                quote = ""
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_loose(text: str) -> Tuple[Dict[str, Any] | None, str | None]:
    """
    Try strict parse, then fenced strip, then first JSON object.

    Returns ``(data, tag)``. On success: ``tag`` is ``None`` or ``"recovered_from_embedded_json"``.
    On failure: ``data`` is ``None`` and ``tag`` is ``"invalid_json"``.
    """
    raw = text.strip()
    for candidate in (raw, strip_code_fences(raw)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data, None
        except json.JSONDecodeError:
            continue

    sub = extract_first_json_object(raw)
    if sub:
        try:
            data = json.loads(sub)
            if isinstance(data, dict):
                return data, "recovered_from_embedded_json"
        except json.JSONDecodeError:
            pass
    return None, "invalid_json"


def _coerce_primary(s: Any) -> PrimaryIntent | None:
    if s is None:
        return None
    if isinstance(s, PrimaryIntent):
        return s
    if not isinstance(s, str):
        return None
    key = s.strip().lower().replace("-", "_")
    for p in PrimaryIntent:
        if p.value == key:
            return p
    return None


def _coerce_sub_intents(val: Any) -> List[PrimaryIntent]:
    out: List[PrimaryIntent] = []
    if val is None:
        return out
    if not isinstance(val, list):
        return out
    for item in val:
        p = _coerce_primary(item)
        if p is not None:
            out.append(p)
    return out


def _coerce_arguments(val: Any) -> Dict[str, Any]:
    if val is None:
        return {}
    if not isinstance(val, dict):
        return {}
    safe: Dict[str, Any] = {}
    for i, (k, v) in enumerate(val.items()):
        if i >= 64:
            break
        if not isinstance(k, str) or not k.strip():
            continue
        key = k.strip()[:128]
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[key] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            safe[key] = v
        else:
            safe[key] = str(v)[:8000]
    return safe


def _clamp_confidence(val: Any) -> float:
    try:
        c = float(val)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, c))


_MAX_WHY_CHARS = 450


def _clamp_why_this_action(text: str) -> str:
    """Keep explainability text concise (about 1–2 short lines); no raw prompts."""
    s = " ".join(text.split())
    if len(s) > _MAX_WHY_CHARS:
        s = s[: _MAX_WHY_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return s


def validate_intent_payload(
    data: Dict[str, Any],
    *,
    confidence_threshold: float,
) -> Tuple[PrimaryIntent, List[PrimaryIntent], Dict[str, Any], float, bool, str, str, List[str]]:
    """
    Map a parsed JSON dict into validated fields.

    Returns (primary, sub_intents, arguments, confidence, requires_confirmation, explanation,
    why_this_action, warnings).
    """
    warnings: List[str] = []

    primary = _coerce_primary(data.get("primary_intent"))
    if primary is None:
        primary = PrimaryIntent.GENERAL_CHAT
        warnings.append("invalid_or_missing_primary_intent")

    raw_sub = data.get("sub_intents")
    subs = _coerce_sub_intents(raw_sub)
    if isinstance(raw_sub, list) and len(raw_sub) > 0 and len(subs) == 0:
        warnings.append("sub_intents_invalid_all_dropped")
        subs = [primary]
    elif not subs:
        subs = [primary]
        warnings.append("sub_intents_defaulted_to_primary")

    # Single-step plans must execute primary_intent; models sometimes emit mismatched pairs
    # (e.g. primary general_chat + sub_intents ["summarize"]). Coerce to avoid wrong tools.
    if len(subs) == 1 and subs[0] != primary:
        warnings.append("sub_intents_coerced_to_match_primary")
        subs = [primary]

    args = _coerce_arguments(data.get("arguments"))
    conf = _clamp_confidence(data.get("confidence", 0.5))

    req = data.get("requires_confirmation")
    if isinstance(req, bool):
        requires_confirmation = req
    elif isinstance(req, str):
        requires_confirmation = req.strip().lower() in ("1", "true", "yes", "on")
        if req.strip().lower() not in ("0", "1", "true", "false", "yes", "no", "on", "off"):
            warnings.append("requires_confirmation_coerced_from_string")
    else:
        requires_confirmation = False
        warnings.append("requires_confirmation_defaulted")

    expl = data.get("explanation_for_ui")
    if isinstance(expl, str) and expl.strip():
        explanation = expl.strip()[:2000]
    else:
        explanation = "No explanation provided by model."
        warnings.append("explanation_defaulted")

    raw_why = data.get("why_this_action")
    if isinstance(raw_why, str) and raw_why.strip():
        why_this_action = _clamp_why_this_action(raw_why.strip())
    elif explanation != "No explanation provided by model.":
        why_this_action = _clamp_why_this_action(explanation)
    else:
        why_this_action = "These actions were chosen from the words you provided."
        warnings.append("why_this_action_defaulted")

    if conf < confidence_threshold:
        requires_confirmation = True
        warnings.append("low_confidence_forces_confirmation")

    return primary, subs, args, conf, requires_confirmation, explanation, why_this_action, warnings
