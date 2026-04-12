"""Map validated intent analysis to an ordered, local-only action plan."""

from __future__ import annotations

from core.models import ActionPlan, ActionPlanStep, IntentAnalysis, PrimaryIntent, ToolResult
from tools import chat, code_gen, file_ops, summarizer
from utils.logger import get_logger

logger = get_logger("router")

# Maps intent → tool module function name for display and dispatch.
_INTENT_TOOL_ROUTE: dict[PrimaryIntent, str] = {
    PrimaryIntent.CREATE_FILE: "tools.file_ops.write_safe",
    PrimaryIntent.WRITE_CODE: "tools.code_gen.generate",
    PrimaryIntent.SUMMARIZE: "tools.summarizer.run",
    PrimaryIntent.GENERAL_CHAT: "tools.chat.reply",
}


def compile_action_plan(analysis: IntentAnalysis) -> ActionPlan:
    """
    Build an ordered execution plan from validated sub_intents.

    Each step carries a copy of shared ``arguments`` for the tools layer (future execution).
    """
    steps: list[ActionPlanStep] = []
    shared = dict(analysis.arguments)
    for i, intent in enumerate(analysis.sub_intents, start=1):
        route = _INTENT_TOOL_ROUTE.get(intent, "tools.chat.reply")
        desc = _step_description(intent, shared)
        steps.append(
            ActionPlanStep(
                order=i,
                intent=intent,
                tool_route=route,
                description=desc,
                params=shared,
            )
        )
    return ActionPlan(
        steps=steps,
        requires_confirmation=analysis.requires_confirmation,
        explanation_for_ui=analysis.explanation_for_ui,
    )


def _step_description(intent: PrimaryIntent, args: dict) -> str:
    path = args.get("path") or args.get("filename") or args.get("file")
    lang = args.get("language")
    if intent == PrimaryIntent.CREATE_FILE:
        return f"Create or write file{f' {path}' if path else ''}"
    if intent == PrimaryIntent.WRITE_CODE:
        extra = f" ({lang})" if lang else ""
        return f"Generate code{extra}"
    if intent == PrimaryIntent.SUMMARIZE:
        return "Summarize provided text"
    return "General conversation"


class IntentRouter:
    """Bridge validated intents to tools; prefer ``build_action_plan`` for structured UI."""

    def build_action_plan(self, analysis: IntentAnalysis) -> ActionPlan:
        return compile_action_plan(analysis)

    def route_first_step(self, analysis: IntentAnalysis) -> tuple[str, ToolResult]:
        """
        Run only the first planned step (legacy helper until full executor exists).

        Not used when the UI only displays the plan.
        """
        plan = compile_action_plan(analysis)
        if not plan.steps:
            return "noop", ToolResult(ok=False, message="Empty action plan.")
        first = plan.steps[0].intent
        if first == PrimaryIntent.GENERAL_CHAT:
            return "chat.reply", chat.reply(analysis)
        if first == PrimaryIntent.SUMMARIZE:
            return "summarizer.run", summarizer.run(analysis)
        if first == PrimaryIntent.CREATE_FILE:
            return "file_ops.write_safe", file_ops.write_safe(analysis)
        if first == PrimaryIntent.WRITE_CODE:
            return "code_gen.generate", code_gen.generate(analysis)
        return "noop", ToolResult(ok=False, message="Unsupported intent.")
