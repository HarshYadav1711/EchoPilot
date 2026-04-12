"""Isolated temporary directories under the sandbox for audio staging (safe cleanup)."""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.config import Settings
from utils.safety import ensure_within_root


@contextmanager
def session_temp_dir(settings: Settings, prefix: str = "stt_") -> Iterator[Path]:
    """
    Create a unique subdirectory under ``settings.data_dir / "tmp"`` and remove it afterwards.

    All intermediate audio files for one transcription job should stay inside this directory.
    """
    base = settings.data_dir / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    job = base / f"{prefix}{uuid.uuid4().hex}"
    job.mkdir(parents=False)
    resolved_job = ensure_within_root(job, settings.data_dir)
    try:
        yield resolved_job
    finally:
        try:
            shutil.rmtree(resolved_job, ignore_errors=True)
        except OSError:
            pass
