"""In-session action timeline: append-only list capped in place (no DB; UI passes session_state list)."""

from __future__ import annotations

import datetime as dt
from typing import Any, List

# Maximum executed step rows kept for the session (oldest dropped first).
ACTION_TIMELINE_MAX = 10


def append_executed_actions(timeline: List[dict[str, Any]], step_logs: list[dict[str, Any]]) -> None:
    """
    Append one entry per executor step log from a non–dry-run Apply.

    Mutates ``timeline`` in place and trims to :data:`ACTION_TIMELINE_MAX` rows.
    Each entry: ``timestamp``, ``intent``, ``summary``, ``result_status`` (success|failure).
    """
    for log in step_logs:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        intent = str(log.get("intent") or "unknown")
        ok = bool(log.get("ok"))
        raw = (log.get("message") or "").strip()
        summary = raw.replace("\n", " ")[:240]
        if not summary:
            summary = "—"
        timeline.append(
            {
                "timestamp": ts,
                "intent": intent,
                "summary": summary,
                "result_status": "success" if ok else "failure",
            }
        )
    while len(timeline) > ACTION_TIMELINE_MAX:
        timeline.pop(0)
