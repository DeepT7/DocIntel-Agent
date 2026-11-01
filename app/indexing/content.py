"""Mechanism 4 — on-demand content loading.

Content is read from disk only when a caller explicitly asks for one file
(e.g. the raw-content API or a preview). The path is confined to the scope so a
``../`` request can never escape the folder.
"""

from __future__ import annotations

import base64
from pathlib import Path

from app.core.config import CONTENT_MAX_BYTES

TEXT_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".py", ".json", ".jsonl", ".html", ".htm",
    ".css", ".js", ".ts", ".tsx", ".jsx", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".csv", ".tsv", ".xml", ".log", ".rst", ".sql", ".sh",
}


class ContentError(ValueError):
    """The requested file is outside the scope, missing, or too large."""


def _safe_path(scope: str | Path, rel_path: str) -> tuple[Path, Path]:
    root = Path(scope).resolve()
    relative = rel_path.strip().replace("\\", "/").lstrip("/")
    if not relative:
        raise ContentError("path is required")
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ContentError(f"path escapes the folder: {rel_path}")
    if not target.is_file():
        raise ContentError(f"file not found: {rel_path}")
    return root, target


def _looks_like_text(raw: bytes) -> bool:
    if b"\x00" in raw:
        return False
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def read_content(
    scope: str | Path,
    rel_path: str,
    *,
    max_bytes: int = CONTENT_MAX_BYTES,
) -> dict:
    """Read one file from disk (raw bytes, decoded text when possible)."""
    _root, target = _safe_path(scope, rel_path)
    stat = target.stat()
    with target.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]

    payload = {
        "path": target.relative_to(Path(scope).resolve()).as_posix(),
        "name": target.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "truncated": truncated,
        "max_bytes": max_bytes,
    }
    if target.suffix.lower() in TEXT_EXTENSIONS or _looks_like_text(raw):
        payload["encoding"] = "utf-8"
        payload["text"] = raw.decode("utf-8", errors="replace")
    else:
        payload["encoding"] = "base64"
        payload["content"] = base64.b64encode(raw).decode("ascii")
    return payload
