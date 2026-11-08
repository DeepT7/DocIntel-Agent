"""Scope helpers.

A *scope* is the folder the user is currently asking about. It is stored on every
point as the ``root`` payload field, so retrieval can be restricted to one folder
even though many folders share a single Qdrant collection.

"""

import platform
import re
from pathlib import Path


def resolve_user_path(raw_path: str | Path) -> Path:
    """Return the absolute folder path the user typed."""
    path_str = str(raw_path).strip().strip('"').strip("'")

    # WSL reports a Microsoft kernel release string.
    is_wsl = "microsoft" in platform.uname().release.lower()

    if is_wsl:
        # Windows drive paths: "F:\LLM\DocIntel" or "C:/Users/..."
        match = re.match(r"^([a-zA-Z]):[\\/](.*)", path_str)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")

    return Path(path_str).expanduser().resolve()


def validate_folder(raw: str | Path) -> Path:
    """Resolve the folder and check that it is a directory we can read."""
    path = resolve_user_path(raw)
    if not path.exists():
        raise ValueError(
            f"Folder not found: {str(raw).strip()} (resolved to {path.as_posix()})"
        )
    if not path.is_dir():
        raise ValueError(f"Not a folder: {path.as_posix()}")
    return path


def normalize_root(root: str | Path) -> str:
    """Canonical string used as the ``root`` payload value and as a query scope.

    Resolves to an absolute path and renders it with forward slashes so the same
    folder always produces the same key on a given machine.
    """
    return resolve_user_path(root).as_posix()
