"""Create files only under the configured sandbox (flat names, validated)."""

from __future__ import annotations

from pathlib import Path

from core.config import get_settings
from core.execution_context import ExecutionContext
from core.models import IntentAnalysis, ToolResult
from utils.file_sanitizer import resolve_flat_sandbox_file
from utils.logger import get_logger
from utils.safety import ensure_within_root

logger = get_logger("tools.file_ops")

_MAX_BYTES = 512_000


def _intent_arguments(intent: IntentAnalysis) -> dict:
    return intent.arguments


def write_text_under_sandbox(
    path: Path,
    content: str,
    ctx: ExecutionContext,
) -> ToolResult:
    """Write UTF-8 text; respect dry-run, overwrite policy, and size limits."""
    data = content.encode("utf-8")
    if len(data) > _MAX_BYTES:
        return ToolResult(
            ok=False,
            message=f"Content too large (max {_MAX_BYTES} bytes).",
            payload={"path": str(path)},
        )

    exists = path.exists()
    if exists and not ctx.allow_overwrite:
        return ToolResult(
            ok=False,
            message=f"File already exists: {path.name}. Enable “allow overwrite” or choose another name.",
            payload={"path": str(path.name), "blocked": "overwrite"},
        )

    if ctx.dry_run:
        preview = content if len(content) <= 8000 else content[:8000] + "\n… [truncated]"
        logger.info(
            "Dry-run create_file path=%s bytes=%s",
            path.name,
            len(data),
        )
        return ToolResult(
            ok=True,
            message=f"[Dry-run] Would write {len(data)} bytes to {path.name}.",
            payload={
                "path": str(path.name),
                "dry_run": True,
                "would_overwrite": exists,
            },
            dry_run_preview=preview,
        )

    if not ctx.confirm_writes:
        return ToolResult(
            ok=False,
            message="Writes require confirmation. Check “I confirm writes” and run again.",
            payload={"blocked": "confirmation"},
        )

    root = get_settings().data_dir.resolve()
    try:
        ensure_within_root(path, root)
    except ValueError as exc:
        logger.warning("Write path rejected post-resolve: %s", exc)
        return ToolResult(ok=False, message=str(exc), payload={"blocked": "path"})

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Write failed: %s", exc)
        return ToolResult(
            ok=False,
            message=f"Could not write file: {exc}",
            payload={"path": str(path.name)},
        )

    logger.info("Wrote file path=%s bytes=%s", path.name, len(data))
    return ToolResult(
        ok=True,
        message=f"Saved {path.name} ({len(data)} bytes).",
        payload={"path": str(path.name), "bytes": len(data)},
        affected_paths=[str(path.name)],
    )


def create_file(intent: IntentAnalysis, ctx: ExecutionContext) -> ToolResult:
    """Create a new file under the sandbox using validated filename and content."""
    args = _intent_arguments(intent)
    root = get_settings().data_dir.resolve()
    try:
        raw_name = args.get("path") or args.get("filename") or args.get("file")
        if not raw_name:
            return ToolResult(ok=False, message="Missing path or filename in arguments.")
        target = resolve_flat_sandbox_file(str(raw_name), root)
    except ValueError as exc:
        logger.info("Path rejected: %s", exc)
        return ToolResult(ok=False, message=str(exc))

    content = (
        args.get("content")
        or args.get("body")
        or ctx.accumulated.get("summary_text")
        or ""
    )
    if isinstance(content, (int, float)):
        content = str(content)
    if not isinstance(content, str):
        content = str(content)
    if not content.strip():
        return ToolResult(
            ok=False,
            message="No content to write. Provide content in arguments or run summarize first.",
            payload={"path": str(target.name)},
        )

    return write_text_under_sandbox(target, content, ctx)


def write_safe(intent: IntentAnalysis, ctx: ExecutionContext) -> ToolResult:
    """Alias for create_file (router compatibility)."""
    return create_file(intent, ctx)


def sandbox_root() -> str:
    return str(get_settings().data_dir.resolve())
