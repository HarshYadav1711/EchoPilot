"""Load and validate application settings (local-only; no cloud)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Directory containing this package's parent (the `output` project root)."""
    return Path(__file__).resolve().parent.parent


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration; extend for Whisper/Ollama without adding cloud deps."""

    project_root: Path = field(default_factory=_project_root)
    # Writable sandbox for generated files and uploads (never outside project_root/output/output)
    data_dir: Path = field(init=False)
    # Ollama HTTP API (default local)
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2"))
    # faster-whisper — default `tiny` is suitable for typical laptops; override via WHISPER_MODEL
    whisper_model: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL", "tiny"))
    whisper_device: str = field(default_factory=lambda: os.environ.get("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(default_factory=lambda: os.environ.get("WHISPER_COMPUTE_TYPE", "int8"))
    whisper_beam_size: int = field(default_factory=lambda: _env_int("WHISPER_BEAM_SIZE", 1))
    whisper_temperature: float = field(default_factory=lambda: _env_float("WHISPER_TEMPERATURE", 0.0))
    whisper_vad_filter: bool = field(default_factory=lambda: _env_bool("WHISPER_VAD_FILTER", True))
    whisper_min_duration_s: float = field(default_factory=lambda: _env_float("WHISPER_MIN_DURATION_S", 0.25))
    whisper_silence_peak: float = field(default_factory=lambda: _env_float("WHISPER_SILENCE_PEAK", 1.0 / 32768.0))
    # Intent layer (Ollama JSON)
    intent_confidence_threshold: float = field(
        default_factory=lambda: _env_float("INTENT_CONFIDENCE_THRESHOLD", 0.55)
    )
    ollama_intent_temperature: float = field(
        default_factory=lambda: _env_float("OLLAMA_INTENT_TEMPERATURE", 0.1)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", self.project_root / "output")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        return cls()


def get_settings() -> Settings:
    """Singleton-style accessor for tests and app wiring."""
    return Settings.from_env()
