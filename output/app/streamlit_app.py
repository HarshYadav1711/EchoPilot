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
from core.router import IntentRouter  # noqa: E402
from core.stt import SpeechTranscriber  # noqa: E402

# Wire singletons for import-time validation; UI flow to be implemented next
_SETTINGS = get_settings()
_TRANSCRIBER = SpeechTranscriber(_SETTINGS)
_CLASSIFIER = IntentClassifier(_SETTINGS)
_ROUTER = IntentRouter()
_MEMORY = SessionMemory()


def main() -> None:
    st.set_page_config(page_title="EchoPilot", layout="wide")
    st.title("EchoPilot")
    st.caption("Local voice AI — scaffold; STT, intent, and tools not yet implemented.")

    st.subheader("Setup check")
    st.code(f"project_root: {_SETTINGS.project_root}\ndata_dir: {_SETTINGS.data_dir}", language="text")

    st.subheader("Pipeline (placeholder)")
    st.write(
        "Sections for **transcription**, **intent**, **action**, and **result** will render here "
        "once STT, Ollama, and tools are wired."
    )


if __name__ == "__main__":
    main()
