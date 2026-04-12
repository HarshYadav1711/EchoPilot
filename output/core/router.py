"""Map validated intent analysis to an ordered, local-only action plan and execution."""

from __future__ import annotations

from core.execution_context import ExecutionContext
from core.executor import dispatch_intent_step, execute_action_plan
from core.memory import append_executed_actions
from core.models import ActionPlan, ActionPlanStep, IntentAnalysis, PrimaryIntent, RouterExecutionResult, ToolResult

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
    """Bridge validated intents to tools and structured execution results."""

    def build_action_plan(self, analysis: IntentAnalysis) -> ActionPlan:
        return compile_action_plan(analysis)

    def execute_plan(
        self,
        plan: ActionPlan,
        analysis: IntentAnalysis,
        *,
        user_utterance: str,
        transcription_text: str,
        dry_run: bool,
        confirm_writes: bool,
        allow_overwrite: bool,
        action_timeline: list | None = None,
    ) -> RouterExecutionResult:
        result = execute_action_plan(
            plan,
            analysis,
            user_utterance=user_utterance,
            transcription_text=transcription_text,
            dry_run=dry_run,
            confirm_writes=confirm_writes,
            allow_overwrite=allow_overwrite,
        )
        if action_timeline is not None and not dry_run and result.step_logs:
            append_executed_actions(action_timeline, result.step_logs)
        return result

    def route_first_step(self, analysis: IntentAnalysis, ctx: ExecutionContext | None = None) -> tuple[str, ToolResult]:
        """Run only the first planned step (defaults to dry-run safe context)."""
        plan = compile_action_plan(analysis)
        if not plan.steps:
            return "noop", ToolResult(ok=False, message="Empty action plan.")
        if ctx is None:
            ctx = ExecutionContext(
                user_utterance="",
                transcription_text="",
                dry_run=True,
                confirm_writes=False,
                allow_overwrite=False,
            )
        first = plan.steps[0].intent
        label = plan.steps[0].tool_route
        return label, dispatch_intent_step(first, analysis, ctx)
