"""Safe file suffixes for staged audio (path traversal / weird extensions)."""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_EXT = re.compile(r"^[.][a-zA-Z0-9]{1,12}$")


def safe_suffix_from_filename(name: str | None, default: str = ".bin") -> str:
    """Return a safe lowercase extension like ``.wav`` or ``default``."""
    if not name:
        return default if _SAFE_EXT.match(default) else ".bin"
    ext = Path(name).suffix.lower()
    if ext and _SAFE_EXT.match(ext):
        return ext
    return default if _SAFE_EXT.match(default) else ".bin"
