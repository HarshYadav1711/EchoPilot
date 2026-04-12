"""Speech-to-text via faster-whisper (local)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from core.config import Settings, get_settings
from core.models import TranscriptionResult, TranscriptionSource
from utils.audio_naming import safe_suffix_from_filename
from utils.audio_normalize import AudioNormalizationError, ffmpeg_available, normalize_to_wav_16k_mono
from utils.audio_wav import is_effectively_silent_wav, wav_duration_seconds
from utils.logger import get_logger
from utils.temp_audio import session_temp_dir

logger = get_logger("stt")

PathLike = Union[str, Path]


def _user_facing_stt_error(internal: str) -> str:
    """Map internal failure hints to short, non-technical UI copy (details stay in logs)."""
    s = (internal or "").strip().lower()
    if "empty" in s and "buffer" in s:
        return "No audio data was received."
    if "not installed" in s or "faster-whisper" in s:
        return "Speech recognition isn’t available. Install faster-whisper or check your environment."
    if "too short" in s or "empty" in s and "audio" in s:
        return "The recording is too short to transcribe clearly."
    if "silent" in s:
        return "Could not understand the audio — it may be silent or too quiet."
    if "unintelligible" in s or "no speech" in s:
        return "Could not understand the audio. Try a clearer recording or a different file."
    if "save" in s and "audio" in s:
        return "Could not process this audio. Try again or use another file."
    if "read" in s and "file" in s:
        return "Could not read the audio file."
    if "not found" in s:
        return "Audio file was not found."
    return "Could not understand the audio. Try another recording or file."


class SpeechTranscriber:
    """faster-whisper wrapper with sandboxed temp files and graceful error results."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None  # type: ignore[assignment]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            logger.exception("faster-whisper import failed")
            raise RuntimeError("faster-whisper is not installed") from exc

        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            self._settings.whisper_model,
            self._settings.whisper_device,
            self._settings.whisper_compute_type,
        )
        self._model = WhisperModel(
            self._settings.whisper_model,
            device=self._settings.whisper_device,
            compute_type=self._settings.whisper_compute_type,
        )
        logger.info("Whisper model ready.")

    def transcribe_from_bytes(
        self,
        data: bytes,
        *,
        filename_for_suffix: str | None,
        source: TranscriptionSource,
    ) -> TranscriptionResult:
        """
        Write bytes into a sandbox temp session, normalize when possible, transcribe.

        Does not raise on bad audio; returns ``TranscriptionResult.ok == False`` instead.
        """
        warnings: list[str] = []
        if not data:
            return TranscriptionResult.failure(
                _user_facing_stt_error("Empty audio buffer."),
                source_type=source.value,
                warnings=warnings,
            )

        suffix = safe_suffix_from_filename(filename_for_suffix, default=".wav")
        try:
            with session_temp_dir(self._settings) as job_dir:
                raw_path = job_dir / f"input{suffix}"
                raw_path.write_bytes(data)
                return self._transcribe_file_in_session(
                    raw_path,
                    source=source,
                    warnings=warnings,
                )
        except OSError as exc:
            logger.warning("Failed to stage audio: %s", exc)
            return TranscriptionResult.failure(
                _user_facing_stt_error("Could not save audio for transcription."),
                source_type=source.value,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001 — top-level guard; return structured error
            logger.exception("Unexpected STT pipeline error")
            return TranscriptionResult.failure(
                _user_facing_stt_error(_short_error(exc)),
                source_type=source.value,
                warnings=warnings,
            )

    def transcribe_file(
        self,
        audio_path: PathLike,
        *,
        source: TranscriptionSource,
    ) -> TranscriptionResult:
        """Transcribe an existing file path (must be readable)."""
        warnings: list[str] = []
        path = Path(audio_path)
        try:
            if not path.is_file():
                return TranscriptionResult.failure(
                    _user_facing_stt_error("Audio file not found."),
                    source_type=source.value,
                    warnings=warnings,
                )
            data = path.read_bytes()
        except OSError as exc:
            logger.warning("Failed to read audio file: %s", exc)
            return TranscriptionResult.failure(
                _user_facing_stt_error("Could not read audio file."),
                source_type=source.value,
                warnings=warnings,
            )

        return self.transcribe_from_bytes(
            data,
            filename_for_suffix=path.name,
            source=source,
        )

    def _transcribe_file_in_session(
        self,
        raw_path: Path,
        *,
        source: TranscriptionSource,
        warnings: list[str],
    ) -> TranscriptionResult:
        """Assume ``raw_path`` lives inside an isolated job directory."""
        normalized_path = raw_path.parent / "normalized.wav"
        transcribe_path = raw_path

        if ffmpeg_available():
            try:
                normalize_to_wav_16k_mono(raw_path, normalized_path)
                transcribe_path = normalized_path
            except AudioNormalizationError as exc:
                msg = str(exc)
                warnings.append(
                    f"ffmpeg normalization skipped ({msg[:200]}); using direct decode."
                )
                logger.warning("ffmpeg normalization failed: %s", msg[:500])
                transcribe_path = raw_path
        else:
            warnings.append(
                "ffmpeg not found on PATH; using faster-whisper decoder without WAV normalization."
            )

        if transcribe_path.suffix.lower() == ".wav":
            dur = wav_duration_seconds(transcribe_path)
            if dur is not None and dur < self._settings.whisper_min_duration_s:
                logger.info("Rejecting audio: duration=%s below minimum", dur)
                return TranscriptionResult.failure(
                    _user_facing_stt_error("Audio too short or empty."),
                    source_type=source.value,
                    warnings=warnings,
                )
            if dur is not None and dur > 0 and is_effectively_silent_wav(
                transcribe_path,
                self._settings.whisper_silence_peak,
            ):
                logger.info("Rejecting audio: silent WAV peak below threshold")
                return TranscriptionResult.failure(
                    _user_facing_stt_error("Audio appears silent."),
                    source_type=source.value,
                    warnings=warnings,
                )

        try:
            self._ensure_model()
        except (RuntimeError, ImportError, OSError) as exc:
            logger.exception("Whisper model unavailable")
            return TranscriptionResult.failure(
                _user_facing_stt_error(str(exc)),
                source_type=source.value,
                warnings=warnings,
            )
        assert self._model is not None

        logger.info(
            "Transcription start path=%s source=%s",
            transcribe_path.name,
            source.value,
        )

        try:
            segments, info = self._model.transcribe(
                str(transcribe_path),
                beam_size=self._settings.whisper_beam_size,
                temperature=self._settings.whisper_temperature,
                vad_filter=self._settings.whisper_vad_filter,
                condition_on_previous_text=False,
            )
            parts: list[str] = []
            for seg in segments:
                parts.append(seg.text)
            text = "".join(parts).strip()
            language = getattr(info, "language", None)
            duration_s = getattr(info, "duration", None)
            if duration_s is None and transcribe_path.suffix.lower() == ".wav":
                duration_s = wav_duration_seconds(transcribe_path)

        except Exception as exc:
            logger.exception("Transcription failed")
            return TranscriptionResult.failure(
                _user_facing_stt_error(_short_error(exc)),
                source_type=source.value,
                warnings=warnings,
            )

        if duration_s is not None and duration_s < self._settings.whisper_min_duration_s:
            logger.info("Transcription rejected: duration=%s after decode", duration_s)
            return TranscriptionResult.failure(
                _user_facing_stt_error("Audio too short or empty."),
                source_type=source.value,
                warnings=warnings,
            )

        if not text:
            logger.info("No text from Whisper (duration_s=%s)", duration_s)
            return TranscriptionResult.failure(
                _user_facing_stt_error(
                    "No speech could be detected or audio was unintelligible.",
                ),
                source_type=source.value,
                warnings=warnings,
            )

        logger.info(
            "Transcription success chars=%s language=%s duration_s=%s",
            len(text),
            language,
            duration_s,
        )
        return TranscriptionResult.success(
            text,
            language=language,
            duration_s=duration_s,
            source_type=source.value,
            warnings=warnings,
        )


def _short_error(exc: Exception, limit: int = 240) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    if len(msg) > limit:
        return msg[: limit - 3] + "..."
    return msg
