"""Load and validate application settings (local-only; no cloud)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Directory containing this package's parent (the `output` project root)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime configuration; extend for Whisper/Ollama without adding cloud deps."""

    project_root: Path = field(default_factory=_project_root)
    # Writable sandbox for generated files and uploads (never outside project_root/output/output)
    data_dir: Path = field(init=False)
    # Ollama HTTP API (default local)
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2"))
    # faster-whisper model name or path
    whisper_model: str = field(default_factory=lambda: os.environ.get("WHISPER_MODEL", "base"))
    whisper_device: str = field(default_factory=lambda: os.environ.get("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(default_factory=lambda: os.environ.get("WHISPER_COMPUTE_TYPE", "int8"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", self.project_root / "output")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        return cls()


def get_settings() -> Settings:
    """Singleton-style accessor for tests and app wiring."""
    return Settings.from_env()
