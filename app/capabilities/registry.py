"""Capability: the Document Registry — a reusable "card" per file (Cấp độ 2).

Chunks are small and often never name their own document ("Open port 8502"), so a
question like "what does the Docker setup guide say about the port?" cannot be
matched from chunk text alone. The registry keeps one card per file — real title,
summary, keywords/entities and a compact indented outline — built once after
indexing and cached by content hash.

The outline is stored twice:

* ``outline_tree``: the canonical nested section tree, read by the resolver.
  Each node has a materialized-path id (``s4``, ``s4.1``; structural fallbacks
  are ``x1``/``s4.x1``), its clean title, its depth and its children. Chat reads
  this from the RAM-cached card, so it does **not** scroll Qdrant on every turn.
* ``sections``: the same tree rendered as compact text for the UI and the card
  prompt.

Both are refreshed during ``/api/sync`` from the chunks just written (or from
Qdrant once when a card is missing). An LLM card (title/summary/keywords) is
enriched separately by ``build_card``; ``summary_stale`` marks a summary whose
file content changed since it was written.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.capabilities.insights import _file_chunks
from app.core.config import (
    CACHE_HASH_LEN,
    CARD_MAX_KEYWORDS,
    CARD_SINGLE_PASS_CHARS,
    INDEX_DIR,
    MAX_TOKENS,
    OUTLINE_VERSION,
)
from app.core.prompts import build_document_card_prompt
from app.core.scope import normalize_root
from app.core.sections import (
    build_section_tree,
    render_outline,
)
from app.ingestion.folder import file_hash
from app.reasoning.answer import _generate

REGISTRY_DIR = INDEX_DIR / "registry"
# Only this much text feeds the card prompt (cards summarize, they don't index).
SINGLE_PASS_CHARS = CARD_SINGLE_PASS_CHARS

# RAM cache keyed by the registry file's mtime. Chat reads a card on every turn,
# so re-reading and re-parsing the JSON each time is wasted work.
_CACHE: dict[str, tuple[int, dict]] = {}


def _registry_path(root: str) -> Path:
    key = hashlib.md5(root.encode("utf-8")).hexdigest()[:CACHE_HASH_LEN]
    return REGISTRY_DIR / f"{key}.json"


def _load(root: str) -> dict:
    path = _registry_path(root)
    if not path.exists():
        return {}
    try:
        mtime = path.stat().st_mtime_ns
        cached = _CACHE.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        _CACHE[str(path)] = (mtime, data)
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def _save(root: str, data: dict) -> None:
    path = _registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace: a crash mid-write must not leave a truncated card file.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    try:
        _CACHE[str(path)] = (path.stat().st_mtime_ns, data)
    except OSError:
        _CACHE.pop(str(path), None)


def _parse_card(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    keywords = data.get("keywords") or []
    if isinstance(keywords, str):
        keywords = re.split(r"[,\n]", keywords)
    return {
        "title": str(data.get("title") or "").strip(),
        "summary": str(data.get("summary") or "").strip(),
        "keywords": [str(k).strip() for k in keywords if str(k).strip()][:CARD_MAX_KEYWORDS],
    }


def _structural_fields(
    chunks: list[dict],
    rel_path: str,
    current_hash: str,
    existing: dict | None = None,
) -> dict:
    """Deterministic fields that always follow the chunks (no LLM call)."""
    tree = build_section_tree(chunks)
    record = dict(existing or {})
    record.update(
        {
            "path": rel_path,
            "file_name": Path(rel_path).name,
            "outline_tree": tree,
            "sections": render_outline(tree),
            "outline_version": OUTLINE_VERSION,
            "outline_hash": current_hash,
            "outline_updated_at": datetime.now(timezone.utc).isoformat(),
            "chunks": len(chunks) or record.get("chunks", 0),
        }
    )
    record.setdefault("title", Path(rel_path).stem)
    record.setdefault("summary", "")
    record.setdefault("keywords", [])
    # ``file_hash`` belongs to the LLM card; the outline has its own hash. A
    # content edit therefore refreshes the tree immediately while the summary is
    # marked stale until ``build_card`` runs.
    record["summary_stale"] = record.get("file_hash") != current_hash
    return record


def build_card(scope: str, rel_path: str, force: bool = False) -> dict:
    """Return a cached or freshly built card for one document."""
    root = normalize_root(scope)
    abs_path = Path(root) / rel_path
    if not abs_path.exists():
        return {"error": f"file not found: {rel_path}"}

    current_hash = file_hash(abs_path)
    registry = _load(root)
    cached = registry.get(rel_path)
    if cached and cached.get("file_hash") == current_hash and not force:
        # The LLM fields are fresh. Make sure the deterministic outline exists
        # and matches the current corpus schema.
        if (
            not cached.get("outline_tree")
            or cached.get("outline_hash") != current_hash
            or cached.get("outline_version") != OUTLINE_VERSION
        ):
            chunks = _file_chunks(root, rel_path)
            if chunks:
                cached = _structural_fields(chunks, rel_path, current_hash, cached)
                registry[rel_path] = cached
                _save(root, registry)
        return {**cached, "cached": True}

    chunks = _file_chunks(root, rel_path)
    if not chunks:
        return {"error": "no indexed chunks for this file", "path": rel_path}

    full_text = "\n\n".join(chunk.get("text", "") for chunk in chunks)
    parsed = _parse_card(
        _generate(
            build_document_card_prompt(abs_path.name, full_text[:SINGLE_PASS_CHARS]),
            max_tokens=MAX_TOKENS["document_card"],
        )
    )
    record = dict(cached or {})
    record.update(
        {
            "path": rel_path,
            "file_name": abs_path.name,
            "title": parsed.get("title") or record.get("title") or abs_path.stem,
            "summary": parsed.get("summary", ""),
            "keywords": parsed.get("keywords", []),
            "file_hash": current_hash,
            "summary_stale": False,
        }
    )
    # ``file_hash`` is current before this call, so summary_stale stays False.
    record.update(_structural_fields(chunks, rel_path, current_hash, record))
    registry[rel_path] = record
    _save(root, registry)
    return {**record, "cached": False}


def ensure_outline(scope: str, rel_path: str) -> dict:
    """Return a card with a current outline tree, rebuilding it once if needed.

    The fast path never hashes the source file and never touches Qdrant: it is a
    RAM read. The slow path is a one-time migration for cards created before the
    canonical tree existed.
    """
    root = normalize_root(scope)
    registry = _load(root)
    card = registry.get(rel_path) or {}
    if (
        card.get("outline_tree")
        and card.get("outline_version") == OUTLINE_VERSION
    ):
        return card

    abs_path = Path(root) / rel_path
    if not abs_path.exists():
        return card
    current_hash = file_hash(abs_path)
    chunks = _file_chunks(root, rel_path)
    if not chunks:
        return card
    card = _structural_fields(chunks, rel_path, current_hash, card)
    registry[rel_path] = card
    _save(root, registry)
    return card


def sync_outlines(
    root: str | Path,
    current_hashes: dict[str, str],
    chunks_by_path: dict[str, list[dict]] | None = None,
    deleted: list[str] | None = None,
) -> dict:
    """Refresh every card's canonical tree after an index sync.

    Changed files reuse the chunks already built by the indexer. Unchanged files
    with a valid outline are skipped. A missing/stale outline is read from Qdrant
    exactly once, so a normal sync does not scroll the collection per file.
    """
    root = normalize_root(root)
    registry = _load(root)
    updated = 0
    for rel_path, current_hash in current_hashes.items():
        card = registry.get(rel_path) or {}
        if (
            card.get("outline_tree")
            and card.get("outline_hash") == current_hash
            and card.get("outline_version") == OUTLINE_VERSION
        ):
            continue
        chunks = (chunks_by_path or {}).get(rel_path)
        if chunks is None:
            chunks = _file_chunks(root, rel_path)
        if not chunks:
            continue
        registry[rel_path] = _structural_fields(chunks, rel_path, current_hash, card)
        updated += 1

    removed = 0
    for rel_path in deleted or []:
        if rel_path in registry:
            registry.pop(rel_path, None)
            removed += 1

    if updated or removed:
        _save(root, registry)
    return {"updated": updated, "deleted": removed}


def get_card(scope: str, rel_path: str) -> dict | None:
    """Return the stored card only (no LLM call, no Qdrant scroll)."""
    return _load(normalize_root(scope)).get(rel_path)


def list_cards(scope: str) -> list[dict]:
    return [card for _, card in sorted(_load(normalize_root(scope)).items())]


def build_missing(scope: str, force: bool = False) -> dict:
    """Build cards for every source that lacks a fresh one."""
    from app.capabilities.sources import list_sources

    built, skipped, errors = [], [], []
    for item in list_sources(scope):
        result = build_card(scope, item["path"], force=force)
        if result.get("error"):
            errors.append({"path": item["path"], "error": result["error"]})
        elif result.get("cached"):
            skipped.append(item["path"])
        else:
            built.append(item["path"])
    return {"built": built, "skipped": skipped, "errors": errors}
