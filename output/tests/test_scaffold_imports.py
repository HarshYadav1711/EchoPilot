"""Verify scaffold modules import when project root is on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def add_project_root() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def test_import_core() -> None:
    import core.config  # noqa: F401
    import core.intent  # noqa: F401
    import core.memory  # noqa: F401
    import core.models  # noqa: F401
    import core.router  # noqa: F401
    import core.stt  # noqa: F401


def test_import_tools_utils() -> None:
    import tools.chat  # noqa: F401
    import tools.code_gen  # noqa: F401
    import tools.file_ops  # noqa: F401
    import tools.summarizer  # noqa: F401
    import utils.logger  # noqa: F401
    import utils.safety  # noqa: F401


def test_safety_rejects_escape() -> None:
    from utils.safety import ensure_within_root

    root = ROOT / "output"
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        ensure_within_root(ROOT / ".." / "etc" / "passwd", root)
