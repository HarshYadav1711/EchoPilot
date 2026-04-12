"""Minimal UI helpers for EchoPilot Streamlit (badges, timeline, status mapping)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

import streamlit as st

TimelineStatus = Literal["success", "warning", "failure", "info", "neutral"]

MAX_TIMELINE = 12


def inject_ep_style() -> None:
    """Subtle typography and badge styles (no animations)."""
    st.markdown(
        """
<style>
  .ep-wrap { max-width: 920px; margin: 0 auto; }
  .ep-muted { color: #6b7280; font-size: 0.9rem; }
  .ep-section-title {
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.35rem;
    font-weight: 600;
  }
  .ep-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    background: #fafafa;
    margin-bottom: 1rem;
  }
  .ep-timeline-item {
    border-left: 2px solid #e5e7eb;
    padding-left: 0.85rem;
    margin-bottom: 0.75rem;
    margin-left: 0.2rem;
  }
  .ep-badge {
    display: inline-block;
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .ep-badge-success { background: #d1fae5; color: #065f46; }
  .ep-badge-warning { background: #fef3c7; color: #92400e; }
  .ep-badge-failure { background: #fee2e2; color: #991b1b; }
  .ep-badge-info { background: #e0e7ff; color: #3730a3; }
  .ep-badge-neutral { background: #f3f4f6; color: #374151; }
</style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(kind: TimelineStatus, text: str) -> str:
    m = {
        "success": "ep-badge-success",
        "warning": "ep-badge-warning",
        "failure": "ep-badge-failure",
        "info": "ep-badge-info",
        "neutral": "ep-badge-neutral",
    }
    cls = m.get(kind, "ep-badge-neutral")
    return f'<span class="ep-badge {cls}">{text}</span>'


def render_badge(kind: TimelineStatus, text: str) -> None:
    st.markdown(badge_html(kind, text), unsafe_allow_html=True)


def execution_status_badge(status: str) -> tuple[TimelineStatus, str]:
    """Map router execution_status to badge kind and label."""
    s = (status or "").lower()
    if s == "success":
        return "success", "Success"
    if s == "dry_run":
        return "info", "Preview only"
    if s == "blocked":
        return "warning", "Blocked"
    if s in ("partial_failure", "failure"):
        return "warning" if s == "partial_failure" else "failure", status.replace("_", " ").title()
    return "neutral", status or "Unknown"


def append_timeline(
    label: str,
    *,
    detail: str = "",
    status: TimelineStatus = "neutral",
    phase: str = "",
) -> None:
    if "ep_timeline" not in st.session_state:
        st.session_state.ep_timeline = []
    entry = {
        "ts": dt.datetime.now().strftime("%H:%M:%S"),
        "label": label,
        "detail": (detail or "")[:500],
        "status": status,
        "phase": phase,
    }
    st.session_state.ep_timeline.append(entry)
    st.session_state.ep_timeline = st.session_state.ep_timeline[-MAX_TIMELINE:]


def clear_timeline() -> None:
    st.session_state.ep_timeline = []


def reset_ep_session() -> None:
    """Clear pipeline and timeline; keep widget keys minimal."""
    for key in (
        "echo_pipeline",
        "ep_timeline",
        "ep_last_preview",
        "ep_raw_intent_open",
        "last_stt",
    ):
        st.session_state.pop(key, None)


def transcription_badge(ok: bool) -> TimelineStatus:
    return "success" if ok else "failure"


def intent_confidence_status(confidence: float, threshold: float) -> TimelineStatus:
    if confidence >= threshold:
        return "success"
    if confidence >= threshold * 0.7:
        return "warning"
    return "failure"
