"""Generate code via local Ollama and save only under the sandbox."""

from __future__ import annotations

from core.config import get_settings
from core.execution_context import ExecutionContext
from core.models import IntentAnalysis, ToolResult
from tools.file_ops import write_text_under_sandbox
from utils.file_sanitizer import resolve_flat_sandbox_file
from utils.logger import get_logger

logger = get_logger("tools.code_gen")


def generate(intent: IntentAnalysis, ctx: ExecutionContext) -> ToolResult:
    """Generate source from a short prompt, then write through the same sandbox guardrails as create_file."""
    args = intent.arguments
    root = get_settings().data_dir.resolve()
    try:
        raw_name = args.get("path") or args.get("filename") or args.get("file")
        if not raw_name:
            return ToolResult(ok=False, message="Missing path or filename for code output.")
        target = resolve_flat_sandbox_file(str(raw_name), root)
    except ValueError as exc:
        logger.info("code_gen path rejected: %s", exc)
        return ToolResult(ok=False, message=str(exc))

    topic = (
        args.get("topic")
        or args.get("description")
        or args.get("content_hint")
        or ctx.user_utterance
    )
    lang = str(args.get("language") or "python")

    try:
        from ollama import Client

        client = Client(host=get_settings().ollama_host)
        prompt = (
            f"You write {lang} source code only. No markdown, no explanation. "
            f"Implement: {topic}"
        )[:12000]
        logger.info("code_gen request path=%s", target.name)
        resp = client.chat(
            model=get_settings().ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.15},
        )
        code = (resp.get("message") or {}).get("content") or ""
    except Exception as exc:
        logger.exception("code_gen LLM failed")
        return ToolResult(ok=False, message=f"Code generation failed: {exc!s}")

    code = code.strip()
    if not code:
        return ToolResult(ok=False, message="Model returned no code.")

    ctx.accumulated["last_generated_code"] = code
    return write_text_under_sandbox(target, code, ctx)
