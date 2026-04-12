"""Aggressive filename and path validation for sandbox writes (flat layout under data_dir)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from utils.safety import ensure_within_root

# Extensions allowed for user-created content (narrow by design).
ALLOWED_WRITE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".rst",
        ".log",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
    }
)

BLOCKED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".exe",
        ".bat",
        ".cmd",
        ".com",
        ".dll",
        ".msi",
        ".scr",
        ".ps1",
        ".vbs",
        ".jar",
        ".app",
        ".deb",
        ".rpm",
        ".sh",
    }
)

_MAX_BASENAME_LEN = 180
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9._\-]+$")


def _strip_control_chars(s: str) -> str:
    return "".join(c for c in s if unicodedata.category(c) != "Cc")


def sanitize_basename(raw: str) -> str:
    """
    Reduce user input to a single safe filename segment (no directories).

    Raises:
        ValueError: if nothing usable remains.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("Filename is empty or invalid.")
    s = raw.strip().replace("\\", "/")
    if ".." in s:
        raise ValueError("Path traversal is not allowed.")
    if s.startswith("/"):
        raise ValueError("Absolute paths are not allowed.")
    if len(s) >= 2 and s[1] == ":":
        raise ValueError("Absolute paths are not allowed.")
    parts = [p for p in s.split("/") if p and p != "."]
    if len(parts) > 1:
        raise ValueError("Only a single filename is allowed; no directories.")
    base = parts[0] if parts else ""
    base = _strip_control_chars(base)
    base = base.replace("\x00", "")
    if not base or base in (".", ".."):
        raise ValueError("Filename is empty after sanitization.")
    if ".." in base or "/" in base or "\\" in base:
        raise ValueError("Path separators are not allowed in filenames.")
    if len(base) > _MAX_BASENAME_LEN:
        base = base[:_MAX_BASENAME_LEN]
    if not _SAFE_NAME_RE.match(base):
        raise ValueError("Filename contains disallowed characters; use letters, digits, ._- only.")
    ext = Path(base).suffix.lower()
    if not ext:
        raise ValueError("A file extension is required (e.g. .txt, .py).")
    if ext in BLOCKED_EXTENSIONS:
        raise ValueError(f"Extension {ext!r} is not allowed for safety reasons.")
    if ext not in ALLOWED_WRITE_EXTENSIONS:
        raise ValueError(f"Extension {ext!r} is not in the allowed list for writes.")
    return base


def resolve_flat_sandbox_file(user_hint: str, sandbox_root: Path) -> Path:
    """
    Map user-provided path hint to a single file path directly under ``sandbox_root``.

    Rejects absolute paths, traversal, and nested directories.
    """
    if not user_hint or not str(user_hint).strip():
        raise ValueError("No file path provided.")
    hint = str(user_hint).strip()
    p = Path(hint)
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    parts = p.parts
    if ".." in parts or any(".." in x for x in parts):
        raise ValueError("Path traversal is not allowed.")
    if len(parts) != 1:
        raise ValueError("Only a single filename in the sandbox is allowed (no subfolders).")
    name = sanitize_basename(parts[0])
    out = (sandbox_root / name).resolve()
    ensure_within_root(out, sandbox_root)
    return out
