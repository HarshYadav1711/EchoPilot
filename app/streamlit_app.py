"""
EchoPilot Streamlit entrypoint — full pipeline: input → reasoning → execution → timeline.

Run from the repository root:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.ui_helpers import (  # noqa: E402
    MAX_TIMELINE,
    append_timeline,
    badge_html,
    execution_status_badge,
    inject_ep_style,
    intent_confidence_status,
    render_badge,
    reset_ep_session,
    transcription_badge,
    write_apply_allowed,
)
from core.config import get_settings  # noqa: E402
from core.intent import IntentClassifier, apply_requires_user_confirmation  # noqa: E402
from core.models import PipelineResult, TranscriptionSource  # noqa: E402
from core.router import IntentRouter  # noqa: E402
from core.stt import SpeechTranscriber  # noqa: E402
from tools.file_ops import sandbox_root  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.streamlit_audio import read_audio_bytes  # noqa: E402

_LOG = get_logger("streamlit.app")
_SETTINGS = get_settings()
_TRANSCRIBER = SpeechTranscriber(_SETTINGS)
_CLASSIFIER = IntentClassifier(_SETTINGS)
_ROUTER = IntentRouter()
_CONF_THRESHOLD = _SETTINGS.intent_confidence_threshold
_EXEC_CONF_THRESHOLD = _SETTINGS.execution_confidence_threshold


def _empty_card(message: str) -> None:
    st.markdown(f'<p class="ep-muted">{message}</p>', unsafe_allow_html=True)


def _section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<p class="ep-section-title">{title}</p>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def main() -> None:
    st.set_page_config(page_title="EchoPilot", layout="wide", initial_sidebar_state="collapsed")
    inject_ep_style()
    st.session_state.setdefault("ep_action_timeline", [])

    st.markdown('<div class="ep-wrap">', unsafe_allow_html=True)

    h1, h2 = st.columns([4, 1])
    with h1:
        st.title("EchoPilot")
        st.caption("Local pipeline: audio → transcript → intent → plan → safe execution")
    with h2:
        if st.button("Reset session", use_container_width=True, help="Clear transcript, plan, previews, and timeline"):
            reset_ep_session()
            for k in (
                "ep_intent_ack",
                "ep_confirm_writes",
                "ep_allow_overwrite",
                "ep_await_exec_low_conf",
            ):
                st.session_state.pop(k, None)
            st.rerun()

    # —— 1. Input ——
    _section_header("1 · Input", "Microphone or file — audio never leaves your machine for STT.")
    with st.container():
        st.markdown('<div class="ep-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            mic_audio = st.audio_input("Microphone", label_visibility="visible")
        with c2:
            uploaded = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3", "m4a", "webm", "ogg", "flac", "aac"],
                help="Processed only inside the project sandbox.",
            )
        run = st.button("Run transcription & intent", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        extra_warnings: list[str] = []
        if mic_audio is not None and uploaded is not None:
            extra_warnings.append("Both sources provided — using the microphone recording.")

        tx = None
        if mic_audio is not None:
            try:
                data = read_audio_bytes(mic_audio)
            except (TypeError, ValueError, OSError) as exc:
                _LOG.warning("Microphone read failed: %s", exc)
                st.error("Could not read microphone audio. Try recording again or upload a file instead.")
                append_timeline("Input error", "microphone read failed", status="failure", phase="input")
            else:
                tx = _TRANSCRIBER.transcribe_from_bytes(
                    data,
                    filename_for_suffix="recording.wav",
                    source=TranscriptionSource.MICROPHONE,
                )
        elif uploaded is not None:
            try:
                data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
            except (OSError, ValueError) as exc:
                _LOG.warning("Upload read failed: %s", exc)
                st.error("Could not read the uploaded file. Try another file or a different format.")
                append_timeline("Upload read failed", "file read failed", status="failure", phase="input")
            else:
                if not data:
                    st.error("Uploaded file is empty.")
                    append_timeline("Empty upload", "", status="failure", phase="input")
                else:
                    tx = _TRANSCRIBER.transcribe_from_bytes(
                        data,
                        filename_for_suffix=getattr(uploaded, "name", None),
                        source=TranscriptionSource.UPLOAD,
                    )
        else:
            st.warning("Add a recording or a file to continue.")
            append_timeline("No audio provided", "", status="warning", phase="input")

        if tx is not None:
            if extra_warnings:
                tx.warnings = [*extra_warnings, *tx.warnings]

            st.session_state["last_stt"] = tx
            st.session_state.pop("echo_pipeline", None)
            st.session_state.pop("ep_last_preview", None)
            st.session_state.pop("ep_await_exec_low_conf", None)

            if not tx.ok or not (tx.text or "").strip():
                append_timeline(
                    "Transcription failed",
                    tx.error or "No usable text",
                    status="failure",
                    phase="input",
                )
                st.rerun()

            try:
                analysis = _CLASSIFIER.classify(tx.text)
                plan = _ROUTER.build_action_plan(analysis)
                pipeline = PipelineResult(transcription=tx, intent_analysis=analysis, action_plan=plan)
            except Exception:
                _LOG.exception("Pipeline build after transcription failed")
                st.session_state["echo_pipeline"] = None
                append_timeline(
                    "Intent plan failed",
                    "unexpected error",
                    status="failure",
                    phase="reasoning",
                )
                st.error("Could not build an intent plan from the transcript. Try again or rephrase.")
                st.rerun()

            st.session_state["echo_pipeline"] = pipeline

            append_timeline(
                "Transcription ready",
                (tx.text[:120] + "…") if len(tx.text) > 120 else tx.text,
                status=transcription_badge(True),
                phase="input",
            )
            append_timeline(
                "Intent classified",
                f"{analysis.primary_intent.value} · conf {analysis.confidence:.0%}",
                status=intent_confidence_status(analysis.confidence, _CONF_THRESHOLD),
                phase="reasoning",
            )
            st.rerun()

    # —— 2. Reasoning (transcript + intent + confidence + plan) ——
    st.divider()
    _section_header("2 · Reasoning", "What was said, what the model understood, and the ordered plan.")
    st.markdown('<div class="ep-card">', unsafe_allow_html=True)

    last_stt = st.session_state.get("last_stt")
    pipeline = st.session_state.get("echo_pipeline")

    if last_stt is None:
        _empty_card("No transcription yet. Use **Run transcription & intent** in the section above.")
        r1, r2 = st.columns([3, 1])
        with r2:
            render_badge("neutral", "Empty")
    else:
        r1, r2 = st.columns([3, 1])
        with r1:
            st.markdown("**Transcription**")
            if last_stt.ok:
                st.text_area(
                    "Transcript",
                    value=last_stt.text,
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                    key="tx_display",
                )
                meta = []
                if last_stt.language:
                    meta.append(f"Lang: {last_stt.language}")
                if last_stt.duration_s is not None:
                    meta.append(f"Duration: {last_stt.duration_s:.1f}s")
                if last_stt.source_type:
                    meta.append(f"Source: {last_stt.source_type}")
                if meta:
                    st.caption(" · ".join(meta))
                if last_stt.warnings:
                    for w in last_stt.warnings:
                        st.warning(w)
            else:
                st.error(last_stt.error or "Transcription failed.")
        with r2:
            render_badge(transcription_badge(last_stt.ok), "STT")

        if pipeline is not None:
            an = pipeline.intent_analysis
            pl = pipeline.action_plan
            st.markdown("---")
            if an.intent_degraded and an.user_notice:
                st.info(an.user_notice)
            ic1, ic2 = st.columns([3, 1])
            with ic1:
                st.markdown("**Detected intent**")
                st.caption(f"Primary: `{an.primary_intent.value}`")
                st.markdown("**Why this action?**")
                st.write(an.why_this_action or an.explanation_for_ui)
            with ic2:
                bc = intent_confidence_status(an.confidence, _CONF_THRESHOLD)
                render_badge(bc, f"Confidence {an.confidence:.0%}")

            st.markdown("**Confidence**")
            st.progress(min(1.0, max(0.0, an.confidence)))
            st.caption(f"{an.confidence:.0%} · model threshold {_CONF_THRESHOLD:.0%}")

            if an.is_compound_two_part():
                st.markdown("**Compound command**")
                st.caption("Detected two clauses joined by “and”; each is classified separately, then run in order.")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**1 · First clause**")
                    st.write(an.compound_parts[0])
                with c2:
                    st.markdown("**2 · Second clause**")
                    st.write(an.compound_parts[1])

            st.markdown("**Planned actions (execution order)**")
            st.caption("Steps run top to bottom; later steps can use outputs from earlier ones (e.g. summary → file).")
            if not pl.steps:
                _empty_card("No steps in this plan.")
            else:
                for step in pl.steps:
                    st.markdown(
                        f"{step.order}. **{step.intent.value}** — _{step.description}_  \n"
                        f"<span class='ep-muted'>{step.tool_route}</span>",
                        unsafe_allow_html=True,
                    )

            if an.parse_warnings and not an.intent_degraded:
                st.warning("Parser / model notes: " + "; ".join(an.parse_warnings))

            with st.expander("Technical details (intent JSON)", expanded=False):
                detail: dict = {
                    "primary_intent": an.primary_intent.value,
                    "sub_intents": [s.value for s in an.sub_intents],
                    "arguments": an.arguments,
                    "requires_confirmation": an.requires_confirmation,
                    "explanation_for_ui": an.explanation_for_ui,
                    "why_this_action": an.why_this_action,
                    "parse_warnings": an.parse_warnings,
                    "intent_degraded": an.intent_degraded,
                }
                if an.user_notice:
                    detail["user_notice"] = an.user_notice
                if an.compound_parts:
                    detail["compound_parts"] = an.compound_parts
                if an.per_step_arguments:
                    detail["per_step_arguments"] = an.per_step_arguments
                st.json(detail)
            if an.raw_llm_text:
                with st.expander("Raw model output", expanded=False):
                    st.code(an.raw_llm_text, language="json")
        else:
            st.info("Intent appears after a successful transcription with non-empty text.")

    st.markdown("</div>", unsafe_allow_html=True)

    # —— 3. Execution ——
    st.divider()
    _section_header("3 · Execution", "Preview is always safe; Apply writes only after you confirm.")
    st.markdown('<div class="ep-card">', unsafe_allow_html=True)

    if pipeline is None:
        _empty_card("Nothing to execute yet. Complete step **1 · Input** with a successful transcription first.")
        st.caption(f"Sandbox directory: `{sandbox_root()}`")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        _footer_timeline()
        return

    write_steps = any(
        s.intent.value in ("create_file", "write_code") for s in pipeline.action_plan.steps
    )
    an = pipeline.intent_analysis
    needs_review_checkbox = (
        write_steps
        or an.requires_confirmation
        or bool(an.parse_warnings)
    )
    low_conf_apply = apply_requires_user_confirmation(an, min_confidence=_EXEC_CONF_THRESHOLD)

    st.caption(
        f"Sandbox root: `{sandbox_root()}` — flat filenames only; extensions allowlisted. "
        "Flow: **Preview** (safe) → confirm if prompted → **Apply**."
    )

    if write_steps or needs_review_checkbox:
        st.markdown("**Review**")
        if needs_review_checkbox:
            st.warning(
                "Parser flags or model confirmation flags need a quick review before you apply changes to disk."
            )
        if write_steps:
            st.info("File tools require preview or explicit write confirmation before Apply.")

        intent_ack = st.checkbox(
            "I have reviewed the transcript, intent, and plan",
            value=st.session_state.get("ep_intent_ack", False),
            key="ep_intent_ack",
        )
    else:
        intent_ack = True

    if low_conf_apply:
        st.info(
            f"Intent confidence is **{an.confidence:.0%}**, below the execution threshold "
            f"({_EXEC_CONF_THRESHOLD:.0%}). **Apply** will ask you to confirm before any action runs."
        )

    col_p, col_e = st.columns(2)
    with col_p:
        preview = st.button("Preview (dry-run)", use_container_width=True, help="Runs tools without writing files.")
    with col_e:
        apply_run = st.button("Apply", use_container_width=True, type="primary", help="Writes only if the plan includes files and you confirmed.")

    st.checkbox(
        "I confirm writes inside the sandbox",
        value=False,
        help="Required when applying plans that create or overwrite files.",
        key="ep_confirm_writes",
    )
    allow_overwrite = st.checkbox(
        "Allow overwriting an existing file in the sandbox",
        value=False,
        key="ep_allow_overwrite",
    )

    def _run_apply() -> None:
        if not (intent_ack if needs_review_checkbox else True):
            st.error("Check the review checkbox above before applying.")
            return
        if not write_apply_allowed(write_steps, bool(st.session_state.get("ep_confirm_writes", False))):
            st.error("Non–dry-run file writes require “I confirm writes”.")
            return
        cw = bool(st.session_state.get("ep_confirm_writes", False))
        ao = bool(st.session_state.get("ep_allow_overwrite", False))
        execution = _ROUTER.execute_plan(
            pipeline.action_plan,
            pipeline.intent_analysis,
            user_utterance=pipeline.transcription.text,
            transcription_text=pipeline.transcription.text,
            dry_run=False,
            confirm_writes=cw,
            allow_overwrite=ao,
            action_timeline=st.session_state["ep_action_timeline"],
        )
        updated = PipelineResult(
            transcription=pipeline.transcription,
            intent_analysis=pipeline.intent_analysis,
            action_plan=pipeline.action_plan,
            execution=execution,
        )
        st.session_state["echo_pipeline"] = updated
        ek, el = execution_status_badge(execution.execution_status)
        append_timeline("Execution finished", el, status=ek, phase="execution")
        st.rerun()

    awaiting_low_conf = bool(st.session_state.get("ep_await_exec_low_conf"))
    if awaiting_low_conf and not low_conf_apply:
        st.session_state.pop("ep_await_exec_low_conf", None)
        awaiting_low_conf = False

    if awaiting_low_conf and low_conf_apply:
        st.warning("Low confidence detected. Please confirm before executing.")
        st.caption(
            f"Confidence {an.confidence:.0%} is below {_EXEC_CONF_THRESHOLD:.0%}. "
            "Cancel returns you to the plan without running tools."
        )
        c_yes, c_no = st.columns(2)
        with c_yes:
            if st.button("Confirm", use_container_width=True, type="primary", key="ep_exec_low_conf_confirm"):
                st.session_state["ep_await_exec_low_conf"] = False
                _run_apply()
        with c_no:
            if st.button("Cancel", use_container_width=True, key="ep_exec_low_conf_cancel"):
                st.session_state["ep_await_exec_low_conf"] = False
                st.rerun()

    if preview:
        ex = _ROUTER.execute_plan(
            pipeline.action_plan,
            pipeline.intent_analysis,
            user_utterance=pipeline.transcription.text,
            transcription_text=pipeline.transcription.text,
            dry_run=True,
            confirm_writes=False,
            allow_overwrite=bool(st.session_state.get("ep_allow_overwrite", False)),
            action_timeline=None,
        )
        st.session_state["ep_last_preview"] = ex
        kind, label = execution_status_badge(ex.execution_status)
        append_timeline("Preview (dry-run)", label, status=kind, phase="execution")
        st.rerun()

    preview_ex = st.session_state.get("ep_last_preview")
    if preview_ex is not None:
        st.markdown("**Write preview**")
        pr_kind, pr_label = execution_status_badge(preview_ex.execution_status)
        st.markdown(badge_html(pr_kind, pr_label), unsafe_allow_html=True)
        for log in preview_ex.step_logs:
            prev = log.get("dry_run_preview_excerpt") or ""
            if prev:
                with st.expander(f"Step {log.get('order')} · {log.get('intent')} — preview"):
                    st.code(prev, language="text")
        if preview_ex.final_output and preview_ex.final_output != "No successful output from tools.":
            st.markdown("**Output (preview run)**")
            st.text_area(
                "preview_out",
                value=preview_ex.final_output[:8000],
                height=min(400, 120 + preview_ex.final_output.count("\n") * 20),
                disabled=True,
                label_visibility="collapsed",
            )

    if apply_run:
        if low_conf_apply:
            st.session_state["ep_await_exec_low_conf"] = True
            st.rerun()
        else:
            _run_apply()

    current = st.session_state.get("echo_pipeline")
    if current and current.execution:
        ex = current.execution
        st.markdown("---")
        st.markdown("**Final execution result**")
        st.caption("Reflects the last **Apply plan** run (not dry-run preview).")
        fk, fl = execution_status_badge(ex.execution_status)
        st.markdown(badge_html(fk, fl), unsafe_allow_html=True)
        if ex.files_created_or_modified:
            st.success("Files touched: " + ", ".join(ex.files_created_or_modified))
        elif ex.execution_status not in ("dry_run",):
            st.caption("No files written in this run.")
        st.text_area(
            "final_out",
            value=ex.final_output[:12000],
            height=min(420, 160 + ex.final_output.count("\n") * 18),
            disabled=True,
            label_visibility="collapsed",
        )
        if ex.warnings:
            for w in ex.warnings:
                st.warning(w)
        with st.expander("Structured logs", expanded=False):
            st.json(ex.step_logs)

    st.markdown("</div>", unsafe_allow_html=True)

    _render_action_timeline()

    _footer_timeline()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_action_timeline() -> None:
    """Structured log of each tool step from Apply runs (session_state, max 10 rows)."""
    st.divider()
    _section_header("Action Timeline", "Applied tool steps only — preview runs are not logged here.")
    st.markdown('<div class="ep-card">', unsafe_allow_html=True)
    entries = st.session_state.get("ep_action_timeline") or []
    if not entries:
        _empty_card("Nothing logged yet. **Apply** a plan to record tool steps.")
    else:
        for row in reversed(entries):
            ts = row.get("timestamp", "")
            intent = row.get("intent", "")
            summ = row.get("summary", "")
            st_t = row.get("result_status", "failure")
            kind: str = "success" if st_t == "success" else "failure"
            label = "OK" if st_t == "success" else "FAIL"
            st.markdown(
                f"**{ts}** · `{intent}` · {badge_html(kind, label)}",
                unsafe_allow_html=True,
            )
            st.caption(summ)
    st.markdown("</div>", unsafe_allow_html=True)


def _footer_timeline() -> None:
    st.divider()
    _section_header("Session timeline", "Recent steps in this browser session.")
    st.markdown('<div class="ep-card">', unsafe_allow_html=True)
    tl = st.session_state.get("ep_timeline") or []
    if not tl:
        _empty_card("No events yet — transcription and execution steps will appear here.")
    else:
        for ev in reversed(tl[-MAX_TIMELINE:]):
            ev_status = ev.get("status", "neutral")
            lab = ev.get("label", "")
            ts = ev.get("ts", "")
            detail = ev.get("detail", "")
            phase = ev.get("phase", "")
            ph = f"[{phase}] " if phase else ""
            label_badge = str(ev_status).replace("_", " ").title()
            st.markdown(
                f'<div class="ep-timeline-item">{badge_html(ev_status, label_badge)} '
                f"<strong>{ph}{lab}</strong> · <span class='ep-muted'>{ts}</span><br/>"
                f"<span class='ep-muted'>{detail}</span></div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
