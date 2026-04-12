# EchoPilot — troubleshooting (short)

| Symptom | What to check |
|--------|----------------|
| Intent step fails or shows fallback `general_chat` | Ollama running (`ollama serve`), model pulled (`ollama pull` your `OLLAMA_MODEL`). |
| Empty or bad transcription | Mic permission in browser; try file upload; quieter room; smaller `WHISPER_MODEL` if CPU-bound. |
| Writes blocked | Use dry-run first; enable "I confirm writes"; filename must be a single segment with an allowed extension under the sandbox. |
| `ffmpeg` warnings | Optional: install ffmpeg for consistent WAV normalization; STT still works without it. |
| Streamlit `audio_input` missing | Upgrade Streamlit (`pip install -U streamlit` per `requirements.txt`). |
| Import errors running tests | `cd` to this folder; `pip install -r requirements.txt`; run `python -m pytest tests/`. |

Full setup, architecture, and safety notes: see `README.md`.
