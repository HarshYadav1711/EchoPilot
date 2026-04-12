"""
EchoPilot Streamlit entrypoint.

Run from the project root (`output/`) so package imports resolve:
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (parent of `app/`) is importable without editable install
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.intent import IntentClassifier  # noqa: E402
from core.memory import SessionMemory  # noqa: E402
from core.models import TranscriptionSource  # noqa: E402
from core.router import IntentRouter  # noqa: E402
from core.stt import SpeechTranscriber  # noqa: E402
from utils.streamlit_audio import read_audio_bytes  # noqa: E402

_SETTINGS = get_settings()
_TRANSCRIBER = SpeechTranscriber(_SETTINGS)
_CLASSIFIER = IntentClassifier(_SETTINGS)
_ROUTER = IntentRouter()
_MEMORY = SessionMemory()


def main() -> None:
    st.set_page_config(page_title="EchoPilot", layout="wide")
    st.title("EchoPilot")
    st.caption("Local voice AI — audio capture and transcription (STT).")

    st.subheader("Audio input")
    mic_audio = st.audio_input("Record from microphone", label_visibility="visible")
    uploaded = st.file_uploader(
        "Or upload audio",
        type=["wav", "mp3", "m4a", "webm", "ogg", "flac", "aac"],
        help="Common speech formats; files are processed under the app sandbox only.",
    )

    run = st.button("Transcribe", type="primary")

    if run:
        extra_warnings: list[str] = []
        if mic_audio is not None and uploaded is not None:
            extra_warnings.append("Both microphone and file provided; using microphone recording.")

        if mic_audio is not None:
            try:
                data = read_audio_bytes(mic_audio)
            except (TypeError, ValueError, OSError) as exc:
                st.error(f"Could not read microphone audio: {exc}")
                return
            result = _TRANSCRIBER.transcribe_from_bytes(
                data,
                filename_for_suffix="recording.wav",
                source=TranscriptionSource.MICROPHONE,
            )
        elif uploaded is not None:
            try:
                data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
            except (OSError, ValueError) as exc:
                st.error(f"Could not read upload: {exc}")
                return
            if not data:
                st.error("Uploaded file is empty.")
                return
            result = _TRANSCRIBER.transcribe_from_bytes(
                data,
                filename_for_suffix=getattr(uploaded, "name", None),
                source=TranscriptionSource.UPLOAD,
            )
        else:
            st.warning("Provide a microphone recording or upload a file.")
            return

        if extra_warnings:
            result.warnings = [*extra_warnings, *result.warnings]

        st.subheader("Transcription")
        st.json(
            {
                "ok": result.ok,
                "text": result.text,
                "language": result.language,
                "duration_s": result.duration_s,
                "source_type": result.source_type,
                "warnings": result.warnings,
                "error": result.error,
            }
        )


if __name__ == "__main__":
    main()
