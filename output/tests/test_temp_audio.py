"""Temp session directory stays inside sandbox."""

from __future__ import annotations

from pathlib import Path


def test_session_temp_dir_cleanup() -> None:
    from core.config import Settings
    from utils.temp_audio import session_temp_dir

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    created: Path | None = None
    with session_temp_dir(settings) as d:
        created = d
        assert d.is_dir()
        assert str(d.resolve()).startswith(str(settings.data_dir.resolve()))
        (d / "x.bin").write_bytes(b"test")
    assert created is not None
    assert not created.exists()
