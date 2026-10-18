"""Folder-based knowledge base: ingest local documents into Qdrant incrementally.

Design
------
- A **manifest** records a content hash per file, so the next sync knows which
  files were added, modified or deleted.
- Only changed files are re-chunked and re-embedded. Embeddings are cached by
  exact chunk text, so unchanged text is reused instead of re-sent to the API.
- Every chunk gets a **deterministic point id** (uuid5 of its chunk_id) and the
  file's chunks share a ``file_id``. On modify/delete we drop all points for that
  ``file_id`` first and then re-add the new ones, so nothing goes stale.

Result: adding/editing one note re-embeds only the chunks that actually changed.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import (
    CHUNK_MAX_CHARS,
    CHUNK_TARGET_CHARS,
    DENSE_VECTOR_NAME,
    DOCS_DIR,
    EMBEDDING_DIM,
    INDEX_DIR,
    QDRANT_COLLECTION,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
    SUPPORTED_EXTENSIONS,
)
from app.core.scope import normalize_root
from app.core.sections import clean_heading, split_section_number
from app.ingestion.extract import extract_blocks, table_caption, table_markdown
from app.retrieval.embeddings import embed_texts, load_cached_embeddings, record_model
from app.retrieval.store import get_client

MANIFEST_PATH = INDEX_DIR / "manifest.json"
CACHE_PATH = INDEX_DIR / "embeddings_cache.jsonl"
BATCH_SIZE = 50
# Bump when the extraction/chunking logic changes, so sync re-indexes everything
# automatically instead of relying on a manual --reset.
INDEX_VERSION = "12"
# Auto-prune the embedding cache only when it has grown well past the live index,
# so frequent edits do not trigger a cleanup on every sync.
CACHE_PRUNE_RATIO = 1.5
CACHE_PRUNE_MIN_EXTRA = 200


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_id_for(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()


def point_id_for(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


def iter_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)
    return sorted(files)


def _split_text(text: str, target_chars: int, max_chars: int) -> list[str]:
    """Greedily pack paragraphs up to target_chars, hard-splitting long ones."""
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        for start in range(0, len(paragraph), max_chars):
            chunks.append(paragraph[start : start + max_chars])

    if current:
        chunks.append(current)
    return chunks


# An author-written caption, e.g. "Table 3: Results", "Table 2.1: ...", "Table A.1:",
# "Figure 3.1:", "Bảng 3:". pymupdf4llm puts text taken from a picture (chart axis
# labels, a boxed table, ...) between "<!-- Start of picture text -->" markers;
# the caption right after tells us whether that region is a table (keep) or a
# figure (drop): figure text is meaningless without the image.
# A trailing ":" "." or "-" is required so a reference like "Table 3 shows" does
# not match. The number may be dotted (2.1, A.1, 3.2a) and is kept whole.
TABLE_RE = re.compile(r"^\s*\**\s*(table|bảng)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[a-z]?)\s*[:.\-]", re.IGNORECASE)
FIGURE_RE = re.compile(r"^\s*\**\s*(figure|hình)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[a-z]?)\s*[:.\-]", re.IGNORECASE)


def _adjacent_caption(blocks: list[dict], index: int) -> tuple[int, str, str, str] | None:
    """Return (caption_index, text, kind, number) for a caption next to index.

    The caption below the region is checked before the one above: a figure is
    usually followed by its own caption and may itself follow an unrelated table
    caption. ``kind`` is "table" or "figure".
    """
    for neighbour in (index + 1, index - 1):
        if neighbour < 0 or neighbour >= len(blocks):
            continue
        other = blocks[neighbour]
        if other["type"] != "text":
            continue
        text = other["text"].strip()
        table_match = TABLE_RE.match(text)
        if table_match:
            return neighbour, text, "table", table_match.group(2)
        figure_match = FIGURE_RE.match(text)
        if figure_match:
            return neighbour, text, "figure", figure_match.group(2)
    return None


def _attach_table_captions(blocks: list[dict]) -> list[dict]:
    """Resolve picture-text regions and attach author captions to tables.

    - A table with an adjacent author caption gets that caption (primary signal).
    - A picture-text region is kept only when its caption says "Table N" (its
      text is the table content); figure/chart text is dropped.
    - The caption block is removed from the body so it is not duplicated.
    """
    consumed: set[int] = set()
    promotions: dict[int, tuple[str, str]] = {}
    for index, block in enumerate(blocks):
        if block["type"] not in ("table", "picture"):
            continue
        found = _adjacent_caption(blocks, index)
        if not found:
            continue
        caption_index, caption, kind, number = found
        if kind != "table" or caption_index in consumed:
            continue
        consumed.add(caption_index)
        promotions[index] = (caption, number)

    output: list[dict] = []
    for index, block in enumerate(blocks):
        if index in consumed:
            continue
        if block["type"] == "picture":
            # Keep only regions promoted to a table by a "Table N" caption.
            if index in promotions:
                caption, number = promotions[index]
                output.append(
                    {
                        "type": "table",
                        "rows": [],
                        "content": block["text"],
                        "caption": caption,
                        "table_number": number,
                    }
                )
            continue
        if index in promotions:
            caption, number = promotions[index]
            block = {**block, "caption": caption, "table_number": number}
        output.append(block)
    return output


def _chunk_record(
    file_id: str,
    root: str,
    rel_path: str,
    file_name: str,
    title: str,
    section: dict,
    part: str,
    seq: int,
    extra: dict | None = None,
) -> dict:
    """One embedded chunk, prefixed with a contextual header.

    ``section`` is the materialized path produced by ``build_chunks_for_file``:
    ``id``, ``ancestors``, ``titles``, ``depths``, ``numbers``, ``title``,
    ``raw``, ``level`` and ``chunk_type``. The contextual header uses the clean
    titles (up to level 3), so the embedded text no longer contains Markdown
    stars or a synthetic ``(Table)`` suffix.
    """
    section_titles = list(section.get("titles") or [])
    section_depths = list(section.get("depths") or [])
    section_numbers = list(section.get("numbers") or [])
    section_title = section.get("title") or "Overview"
    raw_heading = section.get("raw") or section_title
    chunk_type = section.get("chunk_type", "text")
    header_titles = section_titles[:3] or [section_title]
    breadcrumb = " > ".join(header_titles)
    header = f"[Document: {file_name} | Section: {breadcrumb}]"
    text = f"{header}\n\n{part}".strip()
    record = {
        "chunk_id": f"{file_id}:{seq}",
        "file_id": file_id,
        "root": root,
        "path": rel_path,
        "url": rel_path,
        "title": title,
        "file_name": file_name,
        # Display/evaluation metadata. Retrieval filters on section_id(s).
        "section_heading": raw_heading,
        "section_level": section.get("level", 1),
        "section_depth": section.get("depth", len(section_depths) or 1),
        "breadcrumbs": section_titles or [section_title],
        "breadcrumb_levels": section_depths or [1],
        # Canonical materialized path (the retrieval contract).
        "section_id": section.get("id") or "s0",
        "section_ancestors": list(section.get("ancestors") or ["s0"]),
        "section_titles": section_titles or [section_title],
        "section_depths": section_depths or [1],
        "section_numbers": section_numbers or [""],
        "section_title_clean": section_title,
        "section_number": section_numbers[-1] if section_numbers else "",
        "raw_heading": raw_heading,
        "chunk_type": chunk_type,
        "context_header": header,
        "doc_type": "local",
        "char_count": len(text),
        "text": text,
    }
    if extra:
        record.update(extra)
    return record


def build_chunks_for_file(root: Path, path: Path) -> list[dict]:
    rel_path = path.relative_to(root).as_posix()
    file_id = file_id_for(rel_path)
    root_str = normalize_root(root)
    title = path.stem
    file_name = path.name

    chunks: list[dict] = []
    seq = 0
    # Structural heading tree: a new heading pops deeper/equal raw levels. Each
    # node gets a materialized path id (s1, s1.1, ...) that is independent of
    # the heading text, so duplicate/un-numbered headings remain distinct.
    stack: list[dict] = []
    top_count = 0
    overview = {
        "id": "s0",
        "title": "Overview",
        "number": "",
        "raw_level": 0,
        "depth": 1,
    }
    body: list[str] = []

    def section_info(chunk_type: str = "text", raw: str | None = None) -> dict:
        nodes = stack or [overview]
        current = nodes[-1]
        titles = [node["title"] for node in nodes]
        return {
            "id": current["id"],
            "ancestors": [node["id"] for node in nodes],
            "titles": titles,
            "depths": [node["depth"] for node in nodes],
            "numbers": [node["number"] for node in nodes],
            "title": current["title"],
            "raw": raw or current["title"],
            "level": current["raw_level"],
            "depth": current["depth"],
            "chunk_type": chunk_type,
        }

    def emit_body() -> None:
        nonlocal seq, body
        text = "\n\n".join(body).strip()
        body = []
        if not text:
            return
        section = section_info("text")
        for part in _split_text(text, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS):
            seq += 1
            chunks.append(
                _chunk_record(
                    file_id, root_str, rel_path, file_name, title,
                    section, part, seq,
                )
            )

    for block in _attach_table_captions(extract_blocks(path)):
        if block["type"] == "heading":
            emit_body()
            raw_heading = block["text"]
            clean = clean_heading(raw_heading) or raw_heading
            number, _ = split_section_number(clean)
            level = int(block.get("level") or 1)
            while stack and stack[-1]["raw_level"] >= level:
                stack.pop()
            if stack:
                parent = stack[-1]
                parent["child_count"] = parent.get("child_count", 0) + 1
                depth = parent["depth"] + 1
                section_id = f"{parent['id']}.{parent['child_count']}"
            else:
                top_count += 1
                depth = 1
                section_id = f"s{top_count}"
            stack.append(
                {
                    "id": section_id,
                    "title": clean,
                    "number": number,
                    "raw_level": level,
                    "depth": depth,
                    "child_count": 0,
                }
            )
        elif block["type"] in ("text", "picture"):
            body.append(block["text"])
        elif block["type"] == "table":
            # A table is one atomic chunk (never split). It shares its section's
            # id, so asking for that section returns the prose AND the table. The
            # synthetic "(Table)" suffix stays display-only (section_heading).
            emit_body()
            seq += 1
            current_title = stack[-1]["title"] if stack else overview["title"]
            section = section_info("table", raw=f"{current_title} (Table)")
            real_caption = block.get("caption")
            caption = real_caption or table_caption(block["rows"])
            if block["rows"]:
                body_text = table_markdown(block["rows"])
            else:
                body_text = block.get("content", "")  # single-cell text-box table
            content_parts: list[str] = []
            if not real_caption:
                content_parts.append(current_title)
            content_parts.extend([caption, body_text])
            part = "\n\n".join(p for p in content_parts if p).strip()
            extra = {"caption": caption}
            if block.get("table_number"):
                extra["table_number"] = block["table_number"]
            chunks.append(
                _chunk_record(
                    file_id, root_str, rel_path, file_name, title,
                    section, part, seq, extra,
                )
            )

    emit_body()
    return chunks


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"files": {}, "updated_at": None}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
def _ensure_collection(client: QdrantClient, reset: bool = False) -> bool:
    """Ensure the collection exists with the current schema.

    Returns True when the collection was (re)created, so the caller can force a
    full re-index (the stored points are gone).
    """
    collections = [c.name for c in client.get_collections().collections]
    exists = QDRANT_COLLECTION in collections
    recreated = False

    if exists and reset:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        exists = False

    if exists:
        params = client.get_collection(QDRANT_COLLECTION).config.params
        dense = params.vectors if isinstance(params.vectors, dict) else {}
        sparse = params.sparse_vectors if isinstance(params.sparse_vectors, dict) else {}
        dense_ok = DENSE_VECTOR_NAME in dense and dense[DENSE_VECTOR_NAME].size == EMBEDDING_DIM
        sparse_ok = SPARSE_VECTOR_NAME in sparse
        if not (dense_ok and sparse_ok):
            # A dense-dim change (new embedding model) or a missing sparse vector
            # (upgrade to native hybrid) invalidates the stored points, so rebuild.
            print("collection schema changed (dense dim / sparse vector); recreating collection")
            client.delete_collection(collection_name=QDRANT_COLLECTION)
            exists = False

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=EMBEDDING_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        recreated = True
    return recreated


def _delete_file_points(client: QdrantClient, file_id: str) -> None:
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]
        ),
    )


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #
def sync_folder(root: Path = DOCS_DIR, reset: bool = False, auto_prune: bool = True) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    client = get_client()
    recreated = _ensure_collection(client, reset=reset)

    manifest = load_manifest()
    previous = manifest.get("files", {}) if not reset else {}
    # Force a full re-index when the collection was rebuilt or the chunker changed.
    if recreated or manifest.get("version") != INDEX_VERSION:
        previous = {}

    path_by_rel = {p.relative_to(root).as_posix(): p for p in iter_files(root)}
    current = {rel: file_hash(p) for rel, p in path_by_rel.items()}

    changed = [rel for rel, h in current.items() if previous.get(rel, {}).get("hash") != h]
    deleted = [rel for rel in previous if rel not in current]

    # Drop stale vectors for modified and removed files.
    for rel in (*changed, *deleted):
        _delete_file_points(client, file_id_for(rel))

    # Re-chunk only the changed files.
    chunks: list[dict] = []
    for rel in changed:
        file_chunks = build_chunks_for_file(root, path_by_rel[rel])
        if not file_chunks:
            print(f"warning: no extractable text in {rel} (scanned/image PDF?)")
        chunks.extend(file_chunks)

    # Embed incrementally: reuse cached vectors by exact text.
    model = record_model()
    cache = load_cached_embeddings(CACHE_PATH, model)
    pending = [c for c in chunks if c["text"] not in cache]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            vectors = embed_texts([c["text"] for c in batch])
            for chunk, vector in zip(batch, vectors):
                cache[chunk["text"]] = vector
                handle.write(
                    json.dumps({"model": model, "text": chunk["text"], "embedding": vector}, ensure_ascii=False)
                    + "\n"
                )

    # Upsert points carrying both the dense and the sparse vector. Qdrant's
    # client encodes the BM25 sparse vector from the text (via fastembed); the
    # collection applies IDF when scoring.
    points = [
        models.PointStruct(
            id=point_id_for(chunk["chunk_id"]),
            vector={
                DENSE_VECTOR_NAME: cache[chunk["text"]],
                SPARSE_VECTOR_NAME: models.Document(text=chunk["text"], model=SPARSE_MODEL),
            },
            payload={
                "chunk_id": chunk["chunk_id"],
                "file_id": chunk["file_id"],
                "root": chunk["root"],
                "path": chunk["path"],
                "url": chunk["url"],
                "title": chunk["title"],
                "file_name": chunk.get("file_name"),
                "section_heading": chunk["section_heading"],
                "section_level": chunk.get("section_level", 1),
                "section_depth": chunk.get("section_depth", 1),
                "breadcrumbs": chunk.get("breadcrumbs", []),
                "breadcrumb_levels": chunk.get("breadcrumb_levels", []),
                "section_id": chunk["section_id"],
                "section_ancestors": chunk.get("section_ancestors", []),
                "section_titles": chunk.get("section_titles", []),
                "section_depths": chunk.get("section_depths", []),
                "section_numbers": chunk.get("section_numbers", []),
                "section_title_clean": chunk.get("section_title_clean"),
                "section_number": chunk.get("section_number", ""),
                "raw_heading": chunk.get("raw_heading"),
                "chunk_type": chunk.get("chunk_type", "text"),
                "context_header": chunk.get("context_header"),
                "doc_type": chunk["doc_type"],
                "char_count": chunk["char_count"],
                "text": chunk["text"],
                "caption": chunk.get("caption"),
                "table_number": chunk.get("table_number"),
            },
        )
        for chunk in chunks
    ]
    for start in range(0, len(points), 100):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[start : start + 100])

    # Single source of truth: refresh each Document Card's outline tree from the
    # chunks just written (changed files) or from Qdrant once (missing/stale
    # cards). Chat then reads the tree from the RAM-cached registry instead of
    # scrolling Qdrant on every turn.
    chunks_by_path: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_path.setdefault(chunk["path"], []).append(chunk)
    try:
        from app.capabilities.registry import sync_outlines

        stats_cards = sync_outlines(root, current, chunks_by_path, deleted)
    except Exception as error:  # cards must never break indexing
        print(f"warning: could not update document outlines: {error}")
        stats_cards = {"updated": 0, "deleted": 0, "error": str(error)}

    manifest["files"] = {rel: {"hash": current[rel]} for rel in current}
    manifest["version"] = INDEX_VERSION
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)

    stats = {
        "files": len(current),
        "changed": len(changed),
        "deleted": len(deleted),
        "chunks_reindexed": len(chunks),
        "embeddings_computed": len(pending),
        "outlines": stats_cards,
    }

    # Threshold-based cleanup: only when the cache has grown well past the index.
    if auto_prune:
        live_count = client.count(collection_name=QDRANT_COLLECTION).count
        if len(cache) > live_count * CACHE_PRUNE_RATIO + CACHE_PRUNE_MIN_EXTRA:
            stats["cache_pruned"] = prune_cache()["removed"]
    return stats


def prune_cache() -> dict:
    """Drop embedding-cache entries whose text is no longer present in the index.

    The cache is keyed by text (not by file) and is append-only, so after files
    are deleted or their text changes, old entries linger. This removes the ones
    the current index no longer references.
    """
    client = get_client()
    live_texts: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            text = (point.payload or {}).get("text")
            if text:
                live_texts.add(text)
        if offset is None:
            break

    if not CACHE_PATH.exists():
        return {"cache_before": 0, "cache_after": 0, "removed": 0}

    kept_lines: list[str] = []
    before = 0
    with CACHE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            before += 1
            if json.loads(line).get("text") in live_texts:
                kept_lines.append(line)
    CACHE_PATH.write_text("".join(kept_lines), encoding="utf-8")
    return {
        "cache_before": before,
        "cache_after": len(kept_lines),
        "removed": before - len(kept_lines),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync a local document folder into Qdrant.")
    parser.add_argument("--root", default=str(DOCS_DIR), help="Folder to ingest")
    parser.add_argument("--reset", action="store_true", help="Rebuild the collection from scratch")
    parser.add_argument("--prune-cache", action="store_true", help="Drop cache entries not in the index")
    parser.add_argument("--no-prune", action="store_true", help="Disable the automatic cache cleanup")
    args = parser.parse_args()

    if args.prune_cache:
        print("prune:", prune_cache())
    else:
        print("sync:", sync_folder(Path(args.root), reset=args.reset, auto_prune=not args.no_prune))