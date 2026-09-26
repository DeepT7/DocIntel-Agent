import re
import json
import atexit

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from app.core.config import (
    DEFAULT_TOP_K,
    QDRANT_PATH,
    QDRANT_COLLECTION,
)
from app.core.models import RetrievedChunk
from app.retrieval.embed_chunks import embed_texts

qdrant = QdrantClient(path=str(QDRANT_PATH))
COLLECTION_NAME = QDRANT_COLLECTION

# Close the local Qdrant client before interpreter shutdown. Otherwise
# QdrantClient.__del__ runs once sys.meta_path is gone and prints a spurious
# "ImportError: sys.meta_path is None, Python is likely shutting down".
atexit.register(qdrant.close)

# In-memory BM25 index built lazily from the stored Qdrant chunks.
_BM25_INDEX = None
_BM25_META: list = []


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _build_bm25_index():
    """Scroll every stored chunk from Qdrant and build a BM25 index over its text."""
    points = []
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break

    corpus = []
    meta = []
    for point in points:
        payload = point.payload or {}
        corpus.append(_tokenize(payload.get("text", "")))
        meta.append(payload)

    return BM25Okapi(corpus), meta


def _get_bm25_index():
    global _BM25_INDEX, _BM25_META
    if _BM25_INDEX is None:
        _BM25_INDEX, _BM25_META = _build_bm25_index()
    return _BM25_INDEX, _BM25_META


def vector_search(
        query: str,
        top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    query = query.strip()
    if not query:
        return []

    query_embedding = embed_query(query)

    # Query Qdrant for the top_k most similar chunks
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    )

    # Format the results into RetrievedChunk objects
    results = []
    for hit in search_result.points:
        payload = hit.payload or {}
        results.append({
            "score": hit.score,
            "semantic_score": hit.score,
            "keyword_score": 0.0,
            "chunk_id": payload.get("chunk_id"),
            "file_id": payload.get("file_id"),
            "doc_type": payload.get("doc_type"),
            "url": payload.get("url"),
            "title": payload.get("title"),
            "breadcrumbs": payload.get("breadcrumbs", []),
            "section_heading": payload.get("section_heading"),
            "char_count": payload.get("char_count", 0),
            "text": payload.get("text"),
        })

    return results


def hybrid_search(
        query: str,
        top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Combine dense Qdrant similarity with a BM25 keyword signal via score fusion."""
    query = query.strip()
    if not query:
        return []

    query_embedding = embed_query(query)
    candidate_k = max(top_k * 2, 10)

    vector_hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=candidate_k,
    )

    bm25, meta = _get_bm25_index()
    keyword_scores = bm25.get_scores(_tokenize(query))

    payload_by_id = {m.get("chunk_id"): m for m in meta}
    semantic_by_id = {}
    for hit in vector_hits.points:
        payload = hit.payload or {}
        semantic_by_id[payload.get("chunk_id")] = float(hit.score)

    keyword_by_id = {}
    for idx, score in enumerate(keyword_scores):
        if score > 0:
            keyword_by_id[meta[idx].get("chunk_id")] = float(score)

    # Normalize keyword scores to [0, 1] so they fuse on the same scale as cosine.
    max_keyword = max(keyword_by_id.values(), default=0.0)
    if max_keyword > 0:
        keyword_by_id = {pid: s / max_keyword for pid, s in keyword_by_id.items()}

    # Weighted fusion of the normalized signals.
    fused = {}
    for pid in set(semantic_by_id) | set(keyword_by_id):
        semantic = semantic_by_id.get(pid, 0.0)
        keyword = keyword_by_id.get(pid, 0.0)
        fused[pid] = 0.7 * semantic + 0.3 * keyword

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    results = []
    for pid, score in ranked:
        payload = payload_by_id.get(pid, {})
        results.append({
            "score": score,
            "semantic_score": semantic_by_id.get(pid, 0.0),
            "keyword_score": keyword_by_id.get(pid, 0.0),
            "chunk_id": payload.get("chunk_id"),
            "file_id": payload.get("file_id"),
            "doc_type": payload.get("doc_type"),
            "url": payload.get("url"),
            "title": payload.get("title"),
            "breadcrumbs": payload.get("breadcrumbs", []),
            "section_heading": payload.get("section_heading"),
            "char_count": payload.get("char_count", 0),
            "text": payload.get("text"),
        })

    return results


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
