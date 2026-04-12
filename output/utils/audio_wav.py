"""Lightweight WAV inspection (stdlib ``wave``) for duration and silence heuristics."""

from __future__ import annotations

import struct
import wave
from pathlib import Path


def wav_duration_seconds(path: Path) -> float | None:
    """Return duration in seconds for a readable WAV file, or None if unsupported/corrupt."""
    try:
        with wave.open(str(path), "rb") as w:
            nframes = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return None
            return nframes / float(rate)
    except (wave.Error, OSError, EOFError):
        return None


def wav_peak_float(path: Path) -> float | None:
    """
    Approximate peak absolute sample value in [0, 1] for 16-bit PCM mono/stereo WAV.

    Returns None if format is not trivially readable.
    """
    try:
        with wave.open(str(path), "rb") as w:
            nch = w.getnchannels()
            sampwidth = w.getsampwidth()
            nframes = w.getnframes()
            if sampwidth != 2 or nframes <= 0:
                return None
            data = w.readframes(nframes)
    except (wave.Error, OSError, EOFError):
        return None

    if not data:
        return 0.0

    n = len(data) // 2
    peak = 0
    # Unpack as signed int16 little-endian
    for i in range(0, len(data), 2 * nch):
        sample = struct.unpack_from("<h", data, i)[0]
        a = abs(sample)
        if a > peak:
            peak = a
    return peak / 32768.0


def is_effectively_silent_wav(path: Path, peak_threshold: float) -> bool:
    """True if peak energy is at or below threshold (constant silence / digital zeros)."""
    peak = wav_peak_float(path)
    if peak is None:
        return False
    return peak <= peak_threshold
