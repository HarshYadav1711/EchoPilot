# EchoPilot — Local Voice-Controlled AI Agent

EchoPilot is a **local-first, production-minded AI agent** that converts voice input into structured intent and safely executes actions on the local system.

Unlike typical demos, EchoPilot focuses on **reliability, safety, and clear system design**, making every step of the pipeline transparent and testable.

---

## What It Does

* Accepts audio via **microphone or file upload**
* Transcribes speech locally using `faster-whisper`
* Detects structured intent using a local LLM (Ollama)
* Executes actions safely on the local machine
* Displays the full pipeline in a clean UI

---

## Supported Intents

* Create files
* Write code into files
* Summarize text
* General chat

Also supports:

* Compound commands (e.g., “summarize and save”)
* Confidence-based execution safeguards
* Action timeline (stateful memory)

---

## Architecture

```
Audio Input
   ↓
Speech-to-Text (faster-whisper)
   ↓
Intent Classification (Ollama - JSON structured)
   ↓
Execution Planner (Router)
   ↓
Tool Layer (Safe, sandboxed)
   ↓
Streamlit UI (Transparent pipeline)
```

---

## Safety & Design Principles

* All file operations are restricted to:

  ```
  /output
  ```
* No writes outside the sandbox (prevents accidental system changes)
* Confirmation required for low-confidence or destructive actions
* Structured JSON intent → deterministic execution
* Graceful fallback for errors and unclear inputs

---

## Stateful Behavior

EchoPilot includes a lightweight **action timeline**, allowing the system to behave like a **stateful agent** rather than a stateless script.

This design is inspired by modern memory-layer systems used in real-world AI applications.

---

## Model Choices

| Component | Choice          | Reason                         |
| --------- | --------------- | ------------------------------ |
| STT       | faster-whisper  | Fast, local, lightweight       |
| LLM       | Ollama (Llama3) | Fully local, no API dependency |
| UI        | Streamlit       | Simple, effective, transparent |

---

## Setup

**Ollama must be running** before you use intent / chat tools. Install from [ollama.com](https://ollama.com/), start the Ollama app (Windows: it stays in the system tray), or run `ollama serve` in another terminal. Pull a model that matches `OLLAMA_MODEL` (default `llama3.2`), e.g. `ollama pull llama3.2`.

```bash
git clone https://github.com/HarshYadav1711/EchoPilot
cd EchoPilot
python -m venv .venv
# Windows: .\.venv\Scripts\activate   |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Demo

[Add your 2–3 min video link here]

---

## Technical Article

[Add your Dev.to / Medium link here]

---

## Example Commands

* “Create a Python file with retry logic”
* “Summarize this paragraph and save it”
* “Write code for binary search”

---

## Key Design Decisions

* Enforced **structured JSON outputs** for reliable intent handling
* Built **safe execution boundaries** for local actions
* Focused on **modularity and clarity over complexity**
* Avoided overengineering to keep the system maintainable

---

## Future Work

* Persistent memory (cross-session)
* Multi-step agent planning
* Model benchmarking

---

## Final Note

EchoPilot was intentionally designed to reflect how **real AI systems are built**:

* structured
* safe
* modular
* and production-aware

---
