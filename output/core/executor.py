"""Execute validated action plans with dry-run, confirmation, and structured logging."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from core.execution_context import ExecutionContext
from core.models import (
    ActionPlan,
    IntentAnalysis,
    PrimaryIntent,
    RouterExecutionResult,
    ToolResult,
)
from tools import chat, code_gen, file_ops, summarizer
from utils.logger import get_logger

logger = get_logger("executor")


def execute_action_plan(
    plan: ActionPlan,
    analysis: IntentAnalysis,
    *,
    user_utterance: str,
    transcription_text: str,
    dry_run: bool,
    confirm_writes: bool,
    allow_overwrite: bool,
) -> RouterExecutionResult:
    """
    Run each plan step in order. Writes require ``confirm_writes`` when ``dry_run`` is False.

    Summarize never touches disk; create_file / write_code use sandbox rules in tools.
    """
    ctx = ExecutionContext(
        user_utterance=user_utterance,
        transcription_text=transcription_text,
        dry_run=dry_run,
        confirm_writes=confirm_writes,
        allow_overwrite=allow_overwrite,
        accumulated={},
    )

    warnings: list[str] = []
    files: list[str] = []
    outputs: list[str] = []
    step_logs: list[dict[str, Any]] = []
    any_ok = False
    any_fail = False
    blocked = False

    action_taken = ", ".join(s.intent.value for s in plan.steps) if plan.steps else "none"

    if not plan.steps:
        return RouterExecutionResult(
            action_taken="none",
            files_created_or_modified=[],
            execution_status="failure",
            final_output="No steps to execute.",
            warnings=["Empty action plan."],
            step_logs=[],
        )

    for step in plan.steps:
        r: ToolResult
        try:
            eff = replace(analysis, arguments=analysis.effective_arguments_for_step(step.order))
            r = dispatch_intent_step(step.intent, eff, ctx)
        except Exception as exc:
            logger.exception("Unhandled tool error step=%s", step.order)
            r = ToolResult(ok=False, message=f"Internal error: {exc!s}")
            any_fail = True

        log_entry = {
            "order": step.order,
            "intent": step.intent.value,
            "tool_route": step.tool_route,
            "ok": r.ok,
            "message": (r.message or "")[:2000],
            "dry_run": dry_run,
            "affected_paths": list(r.affected_paths),
            "blocked": (r.payload or {}).get("blocked"),
            "dry_run_preview_excerpt": (r.dry_run_preview or "")[:600],
        }
        step_logs.append(log_entry)
        logger.info("exec_step %s", json.dumps(log_entry, default=str)[:4000])

        if r.payload and r.payload.get("blocked") == "confirmation":
            blocked = True
            warnings.append(
                "Write blocked: enable “I confirm writes” to apply changes to disk."
            )
        if r.payload and r.payload.get("blocked") == "overwrite":
            warnings.append(
                "Overwrite blocked: enable “allow overwrite” or pick a new filename."
            )

        if r.ok:
            any_ok = True
            if r.message:
                outputs.append(r.message)
            files.extend(r.affected_paths)
        else:
            any_fail = True
            warnings.append(f"Step {step.order} ({step.intent.value}): {r.message}")

    execution_status = _resolve_status(
        dry_run=dry_run,
        blocked=blocked,
        any_ok=any_ok,
        any_fail=any_fail,
    )

    final_output = "\n\n".join(outputs).strip()
    if not final_output:
        final_output = "No successful output from tools."
    if len(final_output) > 12000:
        final_output = final_output[:12000] + "\n… [truncated]"

    # Dedupe files list preserving order
    seen: set[str] = set()
    unique_files: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return RouterExecutionResult(
        action_taken=action_taken,
        files_created_or_modified=unique_files,
        execution_status=execution_status,
        final_output=final_output,
        warnings=warnings,
        step_logs=step_logs,
    )


def dispatch_intent_step(intent: PrimaryIntent, analysis: IntentAnalysis, ctx: ExecutionContext) -> ToolResult:
    if intent == PrimaryIntent.CREATE_FILE:
        return file_ops.create_file(analysis, ctx)
    if intent == PrimaryIntent.WRITE_CODE:
        return code_gen.generate(analysis, ctx)
    if intent == PrimaryIntent.SUMMARIZE:
        return summarizer.run(analysis, ctx)
    if intent == PrimaryIntent.GENERAL_CHAT:
        return chat.reply(analysis, ctx)
    return ToolResult(ok=False, message=f"Unsupported intent: {intent}")


def _resolve_status(
    *,
    dry_run: bool,
    blocked: bool,
    any_ok: bool,
    any_fail: bool,
) -> str:
    if blocked:
        return "blocked"
    if dry_run:
        if not any_fail:
            return "dry_run"
        return "partial_failure" if any_ok else "failure"
    if not any_fail and any_ok:
        return "success"
    if any_ok and any_fail:
        return "partial_failure"
    return "failure"
