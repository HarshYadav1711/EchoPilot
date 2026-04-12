"""Helpers for Streamlit microphone / upload objects (file-like or raw bytes)."""

from __future__ import annotations

from typing import Any


def read_audio_bytes(value: Any) -> bytes:
    """
    Read all bytes from ``st.audio_input`` (UploadedFile-like) or raw buffer.

    Resets stream position when possible so the same object can be reused by Streamlit.
    """
    if value is None:
        raise ValueError("audio value is None")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    pos = None
    if hasattr(value, "tell"):
        try:
            pos = value.tell()
        except OSError:
            pos = None
    if hasattr(value, "seek"):
        try:
            value.seek(0)
        except OSError:
            pass
    if hasattr(value, "read"):
        data = value.read()
    else:
        data = bytes(value)
    if pos is not None and hasattr(value, "seek"):
        try:
            value.seek(pos)
        except OSError:
            pass
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"Unsupported audio payload type: {type(value)!r}")
