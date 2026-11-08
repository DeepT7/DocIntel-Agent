"""Query-time access to the raw image store.

Ingestion keeps an ``image_stub`` (id + path + caption) in Qdrant and writes the
actual PNG to ``data/index/assets/images/<file_id>/``. Because no VLM is called at
ingest time, this module lets the answer step reach the stored image and its
metadata by id, closing the same loop that ``table_tools`` closes for tables.

The image content itself (pixels) is not inlined into the prompt; instead the stub
is enriched with the file's absolute path and recorded metadata (size, page,
caption) so a caller can attach or open it, and the model can reason about what
figure exists and where it lives.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.scope import normalize_root
from app.ingestion.raw_assets import read_raw_image

# ``<md5 file_id>_<section id>_p<n>`` — the deterministic image id from chunking.
_IMAGE_ID_RE = re.compile(r"[0-9a-f]{32}_[A-Za-z0-9.\-]+_p\d+")

# Describes the one image tool the answer step may call.
VIEW_IMAGE_TOOL = {
    "name": "view_image",
    "description": (
        "Return one stored figure's file path and metadata (size, page, "
        "caption) by image id or asset path. Use it to point the user at a "
        "specific figure or to attach it to the answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_id": {"type": "string"},
            "image_path": {"type": "string"},
        },
    },
}


def view_image(
    image_id: str = "",
    image_path: str = "",
    *,
    out_dir: Path | None = None,
) -> dict | None:
    """The ``view_image`` tool: validate the id, then read the stored image."""
    value = (image_id or "").strip()
    if value and not _IMAGE_ID_RE.fullmatch(value):
        return None
    if not value and not (image_path or "").strip():
        return None
    try:
        return read_raw_image(value, image_path=(image_path or "").strip(), out_dir=out_dir)
    except Exception:
        return None


def _image_block(record: dict, caption: str) -> str:
    width = record.get("width")
    height = record.get("height")
    size = f" ({width}x{height})" if width and height else ""
    lines = [f"[Image available{size}: {record.get('abs_path') or record.get('image_path')}]"]
    if caption:
        lines.append(f"Caption: {caption}")
    if record.get("page"):
        lines.append(f"Page: {record['page']}")
    return "\n".join(lines)


def hydrate_image_stubs(
    chunks: list[dict],
    *,
    out_dir: Path | None = None,
) -> list[dict]:
    """Annotate every retrieved ``image_stub`` with its stored file + metadata.

    Only image chunks are touched; the stub text is extended (never replaced) so
    the caption and id stay visible even when the file cannot be resolved.
    """
    hydrated: list[dict] = []
    for chunk in chunks:
        if chunk.get("chunk_type") != "image_stub":
            hydrated.append(chunk)
            continue
        image_id = str(chunk.get("asset_id") or "").strip()
        image_path = str(chunk.get("asset_path") or "").strip()
        record = None
        try:
            record = read_raw_image(image_id, image_path=image_path, out_dir=out_dir)
        except Exception:
            record = None
        if not record:
            hydrated.append(chunk)
            continue
        caption = str(chunk.get("caption") or record.get("caption") or "").strip()
        copy = dict(chunk)
        base = str(chunk.get("text") or "").strip()
        block = _image_block(record, caption)
        copy["text"] = f"{base}\n\n{block}" if base else block
        copy["image_resolved"] = True
        copy["image_abs_path"] = record.get("abs_path") or ""
        copy["image_width"] = record.get("width")
        copy["image_height"] = record.get("height")
        hydrated.append(copy)
    return hydrated


def available_images(scope: str | None, paths: list[str] | None = None) -> list[dict]:
    """Image metadata from the Document Cards (RAM-cached, no Qdrant scroll)."""
    if not scope:
        return []
    from app.capabilities.registry import list_cards

    wanted = {path for path in (paths or []) if path} or None
    images: list[dict] = []
    seen: set[str] = set()
    for card in list_cards(normalize_root(scope)):
        path = card.get("path")
        if wanted and path not in wanted:
            continue
        for item in card.get("images") or []:
            image_id = item.get("image_id")
            if not image_id or image_id in seen:
                continue
            seen.add(image_id)
            images.append(
                {
                    "image_id": image_id,
                    "image_path": item.get("image_path") or "",
                    "caption": item.get("caption") or "",
                    "section_id": item.get("section_id") or "",
                    "path": path,
                    "file_name": card.get("file_name") or Path(path or "").name,
                }
            )
    return images
