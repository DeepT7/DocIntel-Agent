"""Retrieval over the local Qdrant collection.

The collection stores two vectors per point:

- ``dense``: the semantic embedding (OpenRouter/Gemini model).
- ``bm25``: a sparse BM25 vector (FastEmbed), scored with Qdrant's IDF modifier.

``vector_search`` fuses the two with Qdrant's native Reciprocal Rank Fusion via
the Query API (``prefetch`` + ``Fusion.RRF``). ``dense_search`` exposes the
semantic signal alone, which is useful for comparison.
"""

import json
import re

from qdrant_client.http import models

from app.core.config import (
    DEFAULT_TOP_K,
    DENSE_VECTOR_NAME,
    FETCH_MIN_EXTRA,
    FETCH_MULTIPLIER,
    HYBRID_PREFETCH_LIMIT,
    QDRANT_COLLECTION,
    REFERENCE_DENSITY_THRESHOLD,
    REFERENCE_DENSITY_WINDOW,
    REFERENCE_MIN_TEXT,
    REFERENCE_PENALTY,
    SECTION_CHUNKS_LIMIT,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
)
from app.core.scope import normalize_root
from app.retrieval.embeddings import embed_texts
from app.retrieval.store import get_client

# Resolved lazily at query time: importing this module (e.g. for `answer._generate`
# or the intent router) must not open the embedded DB and take its single-process
# lock. `get_client()` returns the shared, cached instance.
COLLECTION_NAME = QDRANT_COLLECTION


def _path_condition(paths: "str | list[str] | None"):
    """Build a payload condition matching one or many document paths."""
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]
    paths = [p for p in dict.fromkeys(paths) if p]
    if not paths:
        return None
    if len(paths) == 1:
        return models.FieldCondition(key="path", match=models.MatchValue(value=paths[0]))
    return models.FieldCondition(key="path", match=models.MatchAny(any=paths))


def _scope_filter(scope: str | None, paths: "str | list[str] | None" = None):
    """Restrict a query to one folder root and/or a set of documents.

    ``scope`` filters on the folder root; ``paths`` narrows to one or many
    documents (the relative path stored in the payload). Returns ``None`` when
    neither is set, meaning "search everything".
    """
    must = []
    if scope:
        must.append(
            models.FieldCondition(
                key="root", match=models.MatchValue(value=normalize_root(scope))
            )
        )
    path_cond = _path_condition(paths)
    if path_cond is not None:
        must.append(path_cond)
    return models.Filter(must=must) if must else None


# Bibliography-like sections rank high for short queries but are rarely the
# answer. We do NOT drop them (a user may ask about the references themselves);
# we only down-weight them. Detection combines a heading hint with a
# language-independent "citation density" signal so it generalizes beyond
# English/AI papers.
_REFERENCE_HEADING_RE = re.compile(
    r"\b(references|bibliography|works\s+cited|acknowledg|tài\s+liệu\s+tham\s+khảo)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"(\(?\b(?:19|20)\d{2}\b\)?|\bet\s+al\b|https?://|\bdoi\b|\bpp\.\s*\d|\bvol\.\s*\d)",
    re.IGNORECASE,
)


def _is_reference_like(payload: dict) -> bool:
    heading = payload.get("section_title_clean") or payload.get("section_heading") or ""
    if _REFERENCE_HEADING_RE.search(heading):
        return True
    text = payload.get("text") or ""
    if len(text) < REFERENCE_MIN_TEXT:
        return False
    # ~2+ citation markers per N characters => bibliography-like.
    density = len(_CITATION_RE.findall(text)) / (len(text) / REFERENCE_DENSITY_WINDOW)
    return density >= REFERENCE_DENSITY_THRESHOLD


def _to_result(payload: dict, score: float, semantic: float, keyword: float) -> dict:
    return {
        "score": score,
        "semantic_score": semantic,
        "keyword_score": keyword,
        "chunk_id": payload.get("chunk_id"),
        "file_id": payload.get("file_id"),
        "doc_type": payload.get("doc_type"),
        "root": payload.get("root"),
        "path": payload.get("path"),
        "url": payload.get("url"),
        "title": payload.get("title"),
        "file_name": payload.get("file_name"),
        "breadcrumbs": payload.get("breadcrumbs", []),
        "section_heading": payload.get("section_heading"),
        "section_level": payload.get("section_level", 1),
        "section_depth": payload.get("section_depth", 1),
        "breadcrumb_levels": payload.get("breadcrumb_levels", []),
        "section_id": payload.get("section_id"),
        "section_ancestors": payload.get("section_ancestors", []),
        "section_titles": payload.get("section_titles", []),
        "section_depths": payload.get("section_depths", []),
        "section_numbers": payload.get("section_numbers", []),
        "section_title_clean": payload.get("section_title_clean"),
        "section_number": payload.get("section_number", ""),
        "raw_heading": payload.get("raw_heading"),
        "chunk_type": payload.get("chunk_type", "text"),
        "context_header": payload.get("context_header"),
        "char_count": payload.get("char_count", 0),
        "text": payload.get("text"),
        "caption": payload.get("caption"),
        "table_number": payload.get("table_number"),
    }


def dense_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    scope: str | None = None,
    paths: "str | list[str] | None" = None,
) -> list[dict]:
    """Pure semantic search (dense vectors only)."""
    query = query.strip()
    if not query:
        return []

    query_embedding = embed_query(query)
    hits = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        query_filter=_scope_filter(scope, paths),
    ).points

    return [_to_result(hit.payload or {}, hit.score, hit.score, 0.0) for hit in hits]


def vector_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    scope: str | None = None,
    paths: "str | list[str] | None" = None,
) -> list[dict]:
    """Native hybrid search: dense + sparse BM25 fused with RRF inside Qdrant.

    ``scope`` restricts results to one folder root (payload field ``root``);
    ``paths`` further restricts to one or many documents.
    """
    query = query.strip()
    if not query:
        return []

    qfilter = _scope_filter(scope, paths)
    query_embedding = embed_query(query)
    # Qdrant encodes the BM25 query text itself (fastembed), so no helper needed.
    query_sparse = models.Document(text=query, model=SPARSE_MODEL)
    # Fetch more than requested so dropping reference sections still leaves top_k.
    fetch_limit = max(top_k * FETCH_MULTIPLIER, top_k + FETCH_MIN_EXTRA)
    prefetch_limit = max(fetch_limit * 2, HYBRID_PREFETCH_LIMIT)

    # The fused RRF score is only a rank aggregate, so fetch the dense cosine
    # separately to keep a calibrated ``semantic_score`` for the relevance gate.
    dense_hits = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        using=DENSE_VECTOR_NAME,
        limit=prefetch_limit,
        query_filter=qfilter,
    ).points
    semantic_by_id = {
        (hit.payload or {}).get("chunk_id"): hit.score for hit in dense_hits
    }

    fused = get_client().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_embedding,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=qfilter,
            ),
            models.Prefetch(
                query=query_sparse,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=qfilter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch_limit,
        query_filter=qfilter,
    ).points

    # Down-weight bibliography-like chunks instead of dropping them: they stop
    # crowding the top for content queries, yet remain reachable when the query
    # is genuinely about references.
    reference_penalty = REFERENCE_PENALTY
    scored = []
    for hit in fused:
        payload = hit.payload or {}
        score = hit.score * (reference_penalty if _is_reference_like(payload) else 1.0)
        chunk_id = payload.get("chunk_id")
        scored.append((score, payload, semantic_by_id.get(chunk_id, 0.0)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_to_result(payload, score, semantic, 0.0) for score, payload, semantic in scored[:top_k]]


def section_chunks(
    scope: str | None,
    path: str,
    section_ids: str | list[str],
    limit: int = SECTION_CHUNKS_LIMIT,
    include_descendants: bool = False,
) -> list[dict]:
    """All indexed chunks of one or more sections, in document order.

    ``section_ids`` are the canonical materialized-path ids stored at ingestion
    (``s4``, ``s4.1``, with ``x`` fallbacks for unnumbered headings). With
    ``include_descendants=True`` the filter is applied to
    ``section_ancestors``, so a parent id returns its whole subtree (and a child
    id returns just that branch, including its table chunks). With
    ``include_descendants=False`` it filters the exact ``section_id`` — used by
    "small-to-big" expansion to pull a chunk's own section without exploding a
    parent into every descendant.
    """
    ids = [section_ids] if isinstance(section_ids, str) else list(section_ids or [])
    ids = [section_id for section_id in dict.fromkeys(ids) if section_id]
    if not path or not ids:
        return []

    key = "section_ancestors" if include_descendants else "section_id"
    match = (
        models.MatchValue(value=ids[0])
        if len(ids) == 1
        else models.MatchAny(any=ids)
    )
    must = [
        models.FieldCondition(key="path", match=models.MatchValue(value=path)),
        models.FieldCondition(key=key, match=match),
    ]
    if scope:
        must.append(
            models.FieldCondition(
                key="root", match=models.MatchValue(value=normalize_root(scope))
            )
        )

    points, _ = get_client().scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        scroll_filter=models.Filter(must=must),
    )

    def order(point) -> int:
        chunk_id = (point.payload or {}).get("chunk_id", "")
        try:
            return int(chunk_id.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return 0

    return [
        _to_result(point.payload or {}, 0.0, 0.0, 0.0)
        for point in sorted(points, key=order)
    ]


def document_outline(scope: str | None, path: str) -> dict:
    """A document's title and canonical section tree, in reading order.

    The tree lives in the Document Card, which is RAM-cached, so the normal turn
    does not scroll Qdrant. ``ensure_outline`` is the one-time migration path for
    cards created before the canonical tree existed.
    """
    if not path:
        return {"title": path, "sections": [], "tree": []}

    from app.capabilities.registry import ensure_outline
    from app.core.sections import flatten_sections

    card = ensure_outline(scope, path) if scope else {}
    tree = card.get("outline_tree") or []
    sections = [item["title"] for item in flatten_sections(tree) if item.get("title")]
    return {"title": card.get("title") or path, "sections": sections, "tree": tree}


def embed_query(query: str) -> list[float]:
    """Embed a query with the same provider/model used for the corpus."""
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    return embed_texts([query])[0]


if __name__ == "__main__":
    query = input("Query: ").strip()
    if not query:
        raise SystemExit("Query must not be empty.")

    results = vector_search(query, top_k=DEFAULT_TOP_K)
    print(json.dumps(results, indent=2, ensure_ascii=False))
