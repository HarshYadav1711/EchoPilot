"""General chat via local Ollama — no file or tool side effects."""

from __future__ import annotations

from core.config import get_settings
from core.execution_context import ExecutionContext
from core.models import IntentAnalysis, ToolResult
from utils.logger import get_logger

logger = get_logger("tools.chat")


def reply(intent: IntentAnalysis, ctx: ExecutionContext) -> ToolResult:
    """Short conversational reply; does not read or write files."""
    msg = (ctx.user_utterance or "").strip()
    if not msg:
        msg = (ctx.transcription_text or "").strip()
    if not msg:
        return ToolResult(ok=False, message="Nothing to respond to.")

    try:
        from ollama import Client

        client = Client(host=get_settings().ollama_host)
        prompt = (
            "You are a brief, helpful assistant in a local demo app. "
            "Answer in a few sentences. Do not claim to browse the web or run shell commands.\n\n"
            f"User: {msg}"
        )[:12000]
        logger.info("chat request chars=%s", len(msg))
        resp = client.chat(
            model=get_settings().ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4},
        )
        answer = (resp.get("message") or {}).get("content") or ""
    except Exception as exc:
        logger.exception("chat failed")
        return ToolResult(ok=False, message=f"Chat failed: {exc!s}")

    answer = answer.strip()
    if not answer:
        return ToolResult(ok=False, message="Model returned an empty reply.")

    ctx.accumulated["last_chat_reply"] = answer
    return ToolResult(ok=True, message=answer, payload={"reply": answer})
