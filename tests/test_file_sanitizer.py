"""Sandbox filename policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.file_sanitizer import sanitize_basename
from utils.safety import ensure_within_root


def test_sanitize_basename_ok(tmp_path: Path) -> None:
    assert sanitize_basename("notes.txt") == "notes.txt"
    assert sanitize_basename("  my-file_v2.py  ") == "my-file_v2.py"


def test_sanitize_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        sanitize_basename("../x.txt")
    with pytest.raises(ValueError):
        sanitize_basename("a/b.txt")


def test_sanitize_rejects_bad_ext() -> None:
    with pytest.raises(ValueError):
        sanitize_basename("x.exe")
    with pytest.raises(ValueError):
        sanitize_basename("x.unknownext")


def test_resolve_flat_rejects_nested(tmp_path: Path) -> None:
    from utils.file_sanitizer import resolve_flat_sandbox_file

    root = tmp_path / "out"
    root.mkdir()
    with pytest.raises(ValueError):
        resolve_flat_sandbox_file("sub/x.txt", root)


def test_ensure_within_root_accepts_sandbox_file(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"
    root.mkdir()
    p = root / "a.txt"
    ensure_within_root(p, root)
