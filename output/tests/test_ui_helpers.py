"""UI policy helpers (pure functions, no Streamlit runtime required)."""

from __future__ import annotations

from app.ui_helpers import execution_status_badge, write_apply_allowed


def test_write_apply_allowed_no_write_steps() -> None:
    assert write_apply_allowed(False, False) is True
    assert write_apply_allowed(False, True) is True


def test_write_apply_allowed_requires_confirm() -> None:
    assert write_apply_allowed(True, False) is False
    assert write_apply_allowed(True, True) is True


def test_execution_status_badge_mapping() -> None:
    assert execution_status_badge("success")[0] == "success"
    assert execution_status_badge("blocked")[0] == "warning"
    assert execution_status_badge("dry_run")[0] == "info"
