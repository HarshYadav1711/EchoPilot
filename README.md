# EchoPilot

Local-first voice assistant: **Streamlit** UI, **faster-whisper** speech-to-text, **Ollama** for intent JSON and tool-backed text generation. All file output is confined to a single project subdirectory; there are no paid cloud APIs in the default path.

## Why local and free

- **STT**: `faster-whisper` runs on your CPU/GPU with open weights; no per-minute billing.
- **Intent and LLM tools**: Ollama serves models you already pulled; traffic stays on `127.0.0.1` unless you change `OLLAMA_HOST`.
- **No accounts**: The stock configuration does not send audio or transcripts to a third-party inference API.

Trade-off: you install models and accept local CPU/GPU limits instead of outsourcing scale.

## Setup

**Requirements**

- Python 3.10+ (3.11+ recommended).
- [Ollama](https://ollama.com/) installed and running; pull a model matching `OLLAMA_MODEL` (e.g. `ollama pull llama3.2`).
- Optional: **ffmpeg** on `PATH` for deterministic audio normalization before STT (the app still runs if ffmpeg is missing, with a warning).

**Install**

```bash
cd <repository_root>
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
copy .env.example .env          # optional; edit if needed
```

**Run the app**

```bash
streamlit run app/streamlit_app.py
```

**Run tests**

```bash
python -m pytest tests/ -q
```

## Architecture (compact)

```
                    ┌─────────────────┐
  mic / upload ────►│  streamlit_app  │
                    │  (input + UI)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌─────────────┐
       │ core/stt │   │core/     │   │core/executor│
       │ faster-  │   │intent    │   │ + router    │
       │ whisper  │   │+ Ollama  │   └──────┬──────┘
       └────┬─────┘   │ JSON    │          │
            │         └────┬────┘          │
            │              │              ▼
            │         ┌────┴────┐    ┌──────────┐
            │         │ models  │    │ tools/   │
            │         └─────────┘    │ chat,    │
            │                        │summarize,│
            │                        │code_gen, │
            │                        │file_ops  │
            │                        └────┬─────┘
            │                             │
            └─────────────────────────────┼──► utils/ (safety, json_intent,
                                              audio, logging)
                                          │
                                          ▼
                               writable root: ./output/  (see Safety)
```

- **`core/`**: configuration, datatypes, STT wrapper, Ollama JSON intent, plan compilation, execution loop.
- **`tools/`**: narrow, side-effect-bounded operations (summarize, chat, code generation, sandboxed writes).
- **`utils/`**: path policy, strict JSON coercion for intents, audio staging, logging.
- **`app/`**: Streamlit layout, session timeline, confirmation UX.

## Microphone vs file upload

- **Microphone**: `st.audio_input` returns browser-recorded audio (typically WAV). The bytes are read in-process and passed to the STT layer; nothing is uploaded to a remote host by EchoPilot itself.
- **File upload**: `st.file_uploader` reads bytes into memory. Common speech extensions are accepted; unusual containers may rely on faster-whisper’s decoder or optional ffmpeg normalization.

If both are provided, the UI prefers the microphone clip and notes that in warnings.

## Where writes go (sandbox)

- Config resolves **`data_dir`** to **`<project_root>/output`** (runtime-generated files and sandbox writes only; source code lives beside this folder at the repo root).
- **All user-driven file creation** goes through `utils/file_sanitizer.py`: single filename segment, allowlisted extension, no `..`, no absolute paths, no subdirectories in the user string.
- **`utils/safety.ensure_within_root`** is used when resolving paths so resolved paths cannot sit outside `data_dir`.
- **Dry-run** executes tools without calling `Path.write_bytes` for real; **confirmation checkboxes** gate real writes.

## Model choices (defaults)

| Piece | Default env | Notes |
|-------|-------------|--------|
| STT | `WHISPER_MODEL=tiny` | Light on laptops; use `base` or larger if accuracy matters. |
| Intent + tools | `OLLAMA_MODEL=llama3.2` | Must exist in Ollama (`ollama pull`). |
| Intent JSON | Low temperature in client options | Reduces creative drift off-schema. |

## Safety boundaries

- Intent output is **validated JSON** (`utils/json_intent.py`); invalid payloads fall back to `general_chat` with low confidence and forced confirmation semantics in validation.
- **Executor** catches per-step failures; one bad step does not crash the process.
- **Streamlit** separates preview (dry-run) from apply; disk writes require explicit confirmation when the plan includes file tools.

## Fallback behavior on limited hardware

- **Slow STT**: Use `tiny` / `int8` on CPU; shorten recordings; prefer file upload of a pre-clipped WAV.
- **Slow or missing Ollama**: Intent classification returns a safe fallback; tool steps that call Ollama will error gracefully with a `ToolResult` message rather than crashing the app.
- **No ffmpeg**: Normalization is skipped; STT may still decode many formats via faster-whisper’s stack.

## Demo: example phrases

Say clearly, or paste text into a workflow that hits transcription + intent:

| Goal | Example utterance |
|------|-------------------|
| Chat | “What is the capital of France?” |
| Summarize | “Summarize the main idea in two sentences.” |
| Save summary | “Summarize this text and save it to `summary.txt`.” (then confirm writes after preview) |
| Code file | “Create a Python file `retry_demo.py` with a simple retry helper.” |

Structured demo walkthrough and pacing: `demo/DEMO_SCRIPT.txt`.  
Static text for summarize tests (no mic): `demo/sample_text_for_summarize.txt`.

## Troubleshooting

See **`TROUBLESHOOTING.md`** for a compact symptom → fix table. Common issues: Ollama not running, model not pulled, browser blocking the microphone, or missing write confirmation when applying file steps.

## Tests

The suite covers:

- **Path safety** (`utils/safety`, `file_sanitizer`)
- **JSON intent parsing and validation** (`utils/json_intent`)
- **Router plan order** (`compile_action_plan`)
- **Executor** aggregation, empty plans, blocked writes, mocked partial failures
- **Audio helpers** (WAV metrics, temp dirs; ffmpeg-dependent test skipped if absent)
- **UI helpers** (execution status → badge mapping, write-confirm policy)

```bash
python -m pytest tests/ -q
```

## License / attribution

Stack components (Streamlit, faster-whisper, Ollama, CTranslate2, etc.) are governed by their respective licenses. This project does not bundle model weights; you download them via Ollama or faster-whisper’s default cache behavior.
