"""Capability: workspace metadata (files in the current folder).

Answers questions about the workspace itself — "list the files", "how many
documents", "what is in this folder" — which are not answerable from document
content. Kept separate from retrieval on purpose.
"""

from pathlib import Path

from app.core.config import SUPPORTED_EXTENSIONS, WORKSPACE_MAX_FILES
from app.core.scope import normalize_root


def list_files(scope: str, limit: int = WORKSPACE_MAX_FILES) -> list[str]:
    """Relative paths of supported files under ``scope``, sorted."""
    if not scope:
        return []
    root = Path(normalize_root(scope))
    if not root.exists():
        return []

    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path.relative_to(root).as_posix())
            if len(files) >= limit:
                break
    return files
