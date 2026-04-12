"""Normalize speech audio to 16 kHz mono PCM WAV via ffmpeg (deterministic pipeline)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AudioNormalizationError(Exception):
    """Raised when ffmpeg cannot produce a normalized WAV."""


def ffmpeg_available() -> bool:
    """True if ``ffmpeg`` is on PATH."""
    return shutil.which("ffmpeg") is not None


def normalize_to_wav_16k_mono(src: Path, dst: Path) -> None:
    """
    Convert arbitrary audio to 16-bit PCM WAV, mono, 16 kHz.

    Raises:
        AudioNormalizationError: on missing ffmpeg or nonzero exit.
    """
    if not ffmpeg_available():
        raise AudioNormalizationError("ffmpeg not found on PATH")

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or "ffmpeg failed"
        raise AudioNormalizationError(err[:2000])
