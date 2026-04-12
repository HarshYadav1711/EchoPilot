"""Path safety: safe_join and traversal rejection."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.safety import ensure_within_root, safe_join


def test_safe_join_rejects_parent_segment(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        safe_join(root, "..", "x.txt")


def test_safe_join_rejects_absolute_second_segment(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    with pytest.raises(ValueError):
        safe_join(root, "/etc", "passwd")


def test_ensure_within_root_relative_resolution(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"
    root.mkdir()
    p = ensure_within_root("f.txt", root)
    assert p.resolve() == (root / "f.txt").resolve()
