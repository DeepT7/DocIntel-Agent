"""Mechanism 1 — lazy directory traversal.

Only directory entries and file metadata (path, name, size, mtime) are read.
File contents are never opened here, so scanning a folder with thousands of
large PDFs costs a handful of ``stat`` calls and no RAM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.core.config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True, slots=True)
class FileMeta:
    """Cheap metadata for one supported file (no content)."""

    rel_path: str
    abs_path: str
    name: str
    ext: str
    size: int
    mtime_ns: int


def _is_supported(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS


def iter_files(root: str | Path) -> Iterator[FileMeta]:
    """Yield one ``FileMeta`` per supported file, depth-first and lazily.

    Hidden files/directories (``.git``, ``.obsidian``, ...) are skipped, as are
    symlinked files and directories (they can create cycles).
    """
    root_path = Path(root)
    stack: list[Path] = [root_path]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                if not _is_supported(entry.name):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            yield FileMeta(
                rel_path=Path(entry.path).relative_to(root_path).as_posix(),
                abs_path=Path(entry.path).as_posix(),
                name=entry.name,
                ext=os.path.splitext(entry.name)[1].lower(),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )


def snapshot(root: str | Path) -> dict[str, FileMeta]:
    """Metadata snapshot of a folder, keyed by relative POSIX path."""
    return {meta.rel_path: meta for meta in iter_files(root)}


def file_meta(root: str | Path, rel_path: str) -> FileMeta | None:
    """Read metadata for a single known path (no directory walk)."""
    root_path = Path(root)
    path = root_path / rel_path
    try:
        if not path.is_file():
            return None
        stat = path.stat()
    except OSError:
        return None
    name = path.name
    return FileMeta(
        rel_path=rel_path,
        abs_path=path.as_posix(),
        name=name,
        ext=os.path.splitext(name)[1].lower(),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def scan_tree(root: str | Path) -> dict:
    """Nested folder/file tree built from metadata only.

    Shape::

        {"name": "docs", "path": "", "type": "folder", "children": [
            {"name": "guide.md", "path": "guide.md", "type": "file",
             "size": 1203, "mtime_ns": 1700000000000000000},
            {"name": "api", "path": "api", "type": "folder", "children": [...]},
        ]}
    """
    root_path = Path(root)
    tree: dict = {"name": root_path.name, "path": "", "type": "folder", "children": []}
    folders: dict[str, dict] = {"": tree}
    for meta in sorted(iter_files(root_path), key=lambda item: item.rel_path):
        parent_rel = meta.rel_path.rsplit("/", 1)[0] if "/" in meta.rel_path else ""
        parent = folders.get(parent_rel)
        if parent is None:
            # Defensive: create any missing folders in the relative chain.
            parts = parent_rel.split("/") if parent_rel else []
            current_rel = ""
            parent = tree
            for part in parts:
                current_rel = f"{current_rel}/{part}".lstrip("/")
                node = folders.setdefault(
                    current_rel,
                    {"name": part, "path": current_rel, "type": "folder", "children": []},
                )
                if node not in parent["children"]:
                    parent["children"].append(node)
                parent = node
        parent["children"].append(
            {
                "name": meta.name,
                "path": meta.rel_path,
                "type": "file",
                "size": meta.size,
                "mtime_ns": meta.mtime_ns,
            }
        )
    return tree
