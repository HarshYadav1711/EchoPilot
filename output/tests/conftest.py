"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def add_project_root() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
