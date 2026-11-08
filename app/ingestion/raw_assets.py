"""Raw asset storage for tables and images (the "stub + pointer" design).

Images and tables are no longer embedded as text into the vector index. Instead
their full, original content is stored on disk in the local asset store, and only
a compact, context-rich *stub* chunk (``table_stub`` / ``image_stub``) is embedded
for retrieval. A later Agent tool can read the raw content back by ``asset_id``.

Layout (all under ``ASSETS_DIR``, i.e. ``data/index/assets``):

    images/<file_id>/...            file names produced by ``app.ingestion.assets``
    tables/<file_id>.json           full raw tables (headers + rows) keyed by id

Identifiers are deterministic so re-indexing the same file yields the same ids:

    table_id  = f"{file_id}_{section_id}_t{index}"
    image_id  = f"{file_id}_{section_id}_p{index}"

This module only deals with the *raw* JSON manifests and id generation. The chunk
stubs and vector payloads are produced by ``app.ingestion.chunking``.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import ASSETS_DIR
from app.ingestion.identity import file_id_for

TABLES_DIRNAME = "tables"
IMAGES_DIRNAME = "images"


def tables_dir(out_dir: Path | None = None) -> Path:
    return Path(out_dir or ASSETS_DIR) / TABLES_DIRNAME


def images_dir(out_dir: Path | None = None) -> Path:
    return Path(out_dir or ASSETS_DIR) / IMAGES_DIRNAME


def table_manifest_path(file_id: str, out_dir: Path | None = None) -> Path:
    return tables_dir(out_dir) / f"{file_id}.json"


def image_manifest_path(file_id: str, out_dir: Path | None = None) -> Path:
    return images_dir(out_dir) / file_id / "manifest.json"


def table_id_for(file_id: str, section_id: str, index: int) -> str:
    return f"{file_id}_{section_id}_t{index}"


def image_id_for(file_id: str, section_id: str, index: int) -> str:
    return f"{file_id}_{section_id}_p{index}"


def save_tables(
    file_id: str,
    tables: list[dict],
    *,
    out_dir: Path | None = None,
) -> None:
    """Write the raw table manifest for one file (atomics replace)."""
    path = table_manifest_path(file_id, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"file_id": file_id, "tables": tables}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_tables(file_id: str, *, out_dir: Path | None = None) -> list[dict]:
    """Read back a file's raw tables (empty list when absent)."""
    path = table_manifest_path(file_id, out_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data.get("tables") or []


def read_raw_table(table_id: str, *, out_dir: Path | None = None) -> dict | None:
    """Look up one raw table by its deterministic id.

    The id encodes the file, so we can resolve the manifest without a scan.
    The leading 32 characters are the md5 ``file_id`` (see ``identity``).
    """
    file_id = table_id[:32] if len(table_id) >= 32 else ""
    if not file_id:
        return None
    for table in load_tables(file_id, out_dir=out_dir):
        if table.get("table_id") == table_id:
            return table
    return None


def read_raw_image(
    image_id: str = "",
    image_path: str = "",
    *,
    out_dir: Path | None = None,
) -> dict | None:
    """Resolve one stored image to its metadata plus an absolute file path.

    Accepts either the deterministic ``image_id`` (``<file_id>_<section>_p<n>``,
    which encodes the file id in its first 32 characters) or an ``image_path``
    like ``images/<file_id>/<name>.png``. Returns the manifest record with an
    added ``abs_path`` when the file exists on disk, else ``None``.
    """
    file_id = str(image_id)[:32] if len(str(image_id)) >= 32 else ""
    name = ""
    if not file_id and image_path:
        parts = Path(image_path).parts
        if len(parts) >= 3 and parts[0] == IMAGES_DIRNAME:
            file_id = parts[1]
            name = parts[-1]
    if not file_id:
        return None

    path = image_manifest_path(file_id, out_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    assets = data.get("assets") or []
    for asset in assets:
        if name and asset.get("file_name") == name:
            record = asset
        elif asset.get("asset_id") == image_id:
            record = asset
        else:
            continue
        record = dict(record)
        abs_path = Path(out_dir or ASSETS_DIR) / (record.get("image_path") or "")
        record["abs_path"] = str(abs_path) if abs_path.exists() else ""
        return record
    return None


def remove_raw_assets(rel_path: str, *, out_dir: Path | None = None) -> None:
    """Drop both the table manifest and the image directory for one document."""
    file_id = file_id_for(rel_path)
    table_manifest_path(file_id, out_dir).unlink(missing_ok=True)
    image_dir = images_dir(out_dir) / file_id
    if image_dir.exists():
        import shutil

        shutil.rmtree(image_dir, ignore_errors=True)