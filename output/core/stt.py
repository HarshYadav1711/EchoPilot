"""Speech-to-text via faster-whisper (local). Implementation to follow."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from core.config import Settings, get_settings
from core.models import TranscriptionResult
from utils.logger import get_logger

logger = get_logger("stt")

PathLike = Union[str, Path]


class SpeechTranscriber:
    """Thin wrapper around faster-whisper; lazy-load model in `transcribe`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None  # WhisperModel, loaded on first use

    def transcribe(self, audio_path: PathLike) -> TranscriptionResult:
        """Transcribe audio file to text. Raises on missing file or model errors."""
        raise NotImplementedError("STT wiring deferred to implementation stage.")
