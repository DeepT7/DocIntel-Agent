"""Scope helpers.

A *scope* is the folder the user is currently asking about. It is stored on every
point as the ``root`` payload field, so retrieval can be restricted to one folder
even though many folders share a single Qdrant collection.

"""

import os
from pathlib import Path


def _clean(raw: str | Path) -> str:
    """Strip whitespace/quoting and expand ``~`` and environment variables."""
    text = str(raw).strip()
    if text.startswith("& "):
        text = text[2:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return os.path.expandvars(os.path.expanduser(text))


def resolve_folder(raw: str | Path, base: str | Path | None = None) -> Path:
    """Return the absolute path of the folder the user typed."""
    text = _clean(raw)
    if not text:
        raise ValueError("Folder path is empty")
    path = Path(text)
    if not path.is_absolute():
        path = Path(base or os.getcwd()) / path
    return path


def validate_folder(raw: str | Path) -> Path:
    """Resolve the folder and check that it is a directory we can read."""
    path = resolve_folder(raw).resolve()
    if not path.exists():
        raise ValueError(
            f"Folder not found: {_clean(raw)} (resolved to {path.as_posix()})"
        )
    if not path.is_dir():
        raise ValueError(f"Not a folder: {path.as_posix()}")
    return path


def normalize_root(root: str | Path) -> str:
    """Canonical string used as the ``root`` payload value and as a query scope.

    Resolves to an absolute path and renders it with forward slashes so the same
    folder always produces the same key on a given machine.
    """
    return resolve_folder(root).resolve().as_posix()
