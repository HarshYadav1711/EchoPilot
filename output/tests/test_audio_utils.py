"""Unit tests for audio helpers (no heavy model fixtures)."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import pytest

from utils.audio_naming import safe_suffix_from_filename
from utils.audio_normalize import ffmpeg_available, normalize_to_wav_16k_mono
from utils.audio_wav import is_effectively_silent_wav, wav_duration_seconds, wav_peak_float
from utils.streamlit_audio import read_audio_bytes


def _write_silent_wav(path: Path, duration_s: float = 0.5, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * nframes)


def _write_tone_wav(path: Path, duration_s: float = 0.5, sample_rate: int = 16000) -> None:
    """Non-silent sine-ish signal (16-bit mono)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(duration_s * sample_rate)
    frames = bytearray()
    for i in range(nframes):
        v = int(8000 * (1 if (i // 100) % 2 == 0 else -1))
        frames.extend(struct.pack("<h", max(-32768, min(32767, v))))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))


def test_safe_suffix_from_filename() -> None:
    assert safe_suffix_from_filename("foo.MP3") == ".mp3"
    assert safe_suffix_from_filename("x") == ".bin"
    assert safe_suffix_from_filename("../../../etc/passwd") == ".bin"


def test_wav_duration_and_silence(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    _write_silent_wav(p, duration_s=0.4, sample_rate=16000)
    assert wav_duration_seconds(p) == pytest.approx(0.4, rel=1e-3)
    assert wav_peak_float(p) == pytest.approx(0.0, abs=1e-9)
    assert is_effectively_silent_wav(p, peak_threshold=1e-6) is True


def test_wav_not_silent(tmp_path: Path) -> None:
    p = tmp_path / "b.wav"
    _write_tone_wav(p)
    assert is_effectively_silent_wav(p, peak_threshold=1e-6) is False


def test_read_audio_bytes_from_bytesio() -> None:
    buf = io.BytesIO(b"abc123")
    assert read_audio_bytes(buf) == b"abc123"
    assert read_audio_bytes(b"xyz") == b"xyz"


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_ffmpeg_normalize_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    _write_tone_wav(src, duration_s=0.3)
    normalize_to_wav_16k_mono(src, dst)
    assert dst.is_file()
    assert wav_duration_seconds(dst) is not None
