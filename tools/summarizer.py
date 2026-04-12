"""Summarize text via local Ollama — never writes to disk (saving is a separate create_file step)."""

from __future__ import annotations

from core.config import get_settings
from core.execution_context import ExecutionContext
from core.models import IntentAnalysis, ToolResult
from utils.logger import get_logger

logger = get_logger("tools.summarizer")


def run(intent: IntentAnalysis, ctx: ExecutionContext) -> ToolResult:
    """Produce a concise summary; does not create or modify files."""
    args = intent.arguments
    text = (
        args.get("text")
        or args.get("body")
        or ctx.transcription_text
        or ctx.user_utterance
    )
    if isinstance(text, (int, float)):
        text = str(text)
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ToolResult(ok=False, message="No text available to summarize.")

    try:
        from ollama import Client

        client = Client(host=get_settings().ollama_host)
        prompt = (
            "Summarize the following in 2–5 short, clear sentences. "
            "Do not add a preamble or title.\n\n"
            f"{text[:20000]}"
        )
        logger.info("summarize chars=%s", len(text))
        resp = client.chat(
            model=get_settings().ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        summary = (resp.get("message") or {}).get("content") or ""
    except Exception as exc:
        logger.exception("summarize failed")
        return ToolResult(ok=False, message=f"Summarization failed: {exc!s}")

    summary = summary.strip()
    if not summary:
        return ToolResult(ok=False, message="Model returned an empty summary.")

    ctx.accumulated["summary_text"] = summary
    return ToolResult(
        ok=True,
        message=summary,
        payload={"summary": summary},
    )
