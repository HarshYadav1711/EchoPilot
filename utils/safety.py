"""
Path safety: all user-facing paths must resolve inside the configured sandbox directory.

Designed to block traversal (.., absolute paths outside root, symlinks escaping root on supported platforms).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _realpath_strict(path: Path) -> Path:
    """Resolve path; on POSIX, follow symlinks for canonical comparison."""
    try:
        return path.resolve(strict=False)
    except OSError:
        return path


def ensure_within_root(candidate: PathLike, root: PathLike) -> Path:
    """
    Resolve `candidate` relative to `root` if not absolute, then verify it stays under `root`.

    Raises:
        ValueError: if the path escapes the root directory.
    """
    root_path = _realpath_strict(Path(root))
    p = Path(candidate)
    if not p.is_absolute():
        p = root_path / p
    resolved = _realpath_strict(p)

    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Path escapes sandbox: {resolved} not under {root_path}") from exc

    return resolved


def safe_join(root: PathLike, *parts: str) -> Path:
    """
    Join path segments under root and reject any segment that walks upward.

    Raises:
        ValueError: on empty parts, '..' segments, or absolute second+ segments.
    """
    root_path = _realpath_strict(Path(root))
    current = root_path
    for part in parts:
        if not part or part == ".":
            continue
        if part == ".." or Path(part).is_absolute():
            raise ValueError(f"Unsafe path segment: {part!r}")
        current = current / part
    return ensure_within_root(current, root_path)
