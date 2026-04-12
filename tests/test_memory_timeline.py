"""In-session action timeline (core.memory): append and cap."""

from __future__ import annotations

from core.memory import ACTION_TIMELINE_MAX, append_executed_actions


def test_append_executed_actions_empty_step_logs() -> None:
    timeline: list[dict] = []
    append_executed_actions(timeline, [])
    assert timeline == []


def test_append_executed_actions_maps_fields() -> None:
    timeline: list[dict] = []
    append_executed_actions(
        timeline,
        [
            {
                "intent": "general_chat",
                "ok": True,
                "message": "Hello\nworld",
            },
            {
                "intent": "summarize",
                "ok": False,
                "message": "bad",
            },
        ],
    )
    assert len(timeline) == 2
    assert timeline[0]["intent"] == "general_chat"
    assert timeline[0]["result_status"] == "success"
    assert "Hello" in timeline[0]["summary"]
    assert timeline[1]["result_status"] == "failure"
    assert "timestamp" in timeline[0]


def test_append_executed_actions_caps_at_max() -> None:
    timeline: list[dict] = []
    for i in range(ACTION_TIMELINE_MAX + 5):
        append_executed_actions(
            timeline,
            [{"intent": "x", "ok": True, "message": f"step {i}"}],
        )
    assert len(timeline) == ACTION_TIMELINE_MAX
    # Oldest dropped: first retained summary should be step 5
    assert "step 5" in timeline[0]["summary"]
    assert "step 14" in timeline[-1]["summary"]
