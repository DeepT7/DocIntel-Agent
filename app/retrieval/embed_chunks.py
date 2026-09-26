"""Embed the chunk corpus and load it into Qdrant.

Two stages:
1. ``embed_corpus`` builds ``data/embeddings/corpus_embeddings.jsonl`` from
   ``data/chunks/corpus.jsonl``. It is incremental (a previous embedding is
   reused when the chunk text is byte-for-byte identical), batched, and
   retries on rate limits.
2. ``store_chunks_to_qdrant`` upserts those embeddings into the local Qdrant
   collection.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import (
    CHUNK_DIR,
    EMBEDDINGS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_COLLECTION,
)

load_dotenv()

COLLECTION_NAME = QDRANT_COLLECTION
BATCH_SIZE = 50
MAX_RETRIES = 6
BASE_DELAY = 5.0
TEXTS_PER_MINUTE = 100  # Gemini free-tier embed quota (counted per text, not per request)

_client = None


def _get_client() -> genai.Client:
    """Lazily create the Gemini client so storing to Qdrant needs no API key."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or API_KEY before embedding the corpus.")
        _client = genai.Client(api_key=api_key)
    return _client


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    text = str(error).upper()
    return any(token in text for token in ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE"))


def _embed_batch(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Embed a batch of texts, retrying with exponential backoff on transient errors."""
    client = _get_client()
    for attempt in range(MAX_RETRIES):
        try:
            contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]
            response = client.models.embed_content(model=model, contents=contents)
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(texts):
                raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
            return [list(item.values) for item in embeddings]
        except Exception as error:
            if attempt == MAX_RETRIES - 1 or not _is_retryable(error):
                raise
            delay = BASE_DELAY * (2**attempt)
            print(f"embed_batch retry {attempt + 1}/{MAX_RETRIES} after {delay:.0f}s: {error}")
            time.sleep(delay)

    raise RuntimeError("embed_batch failed after retries.")


def _openrouter_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before using the OpenRouter provider.")
    return api_key


def _embed_batch_openrouter(
    texts: list[str],
    model: str = OPENROUTER_EMBEDDING_MODEL,
) -> list[list[float]]:
    """Embed a batch through OpenRouter's OpenAI-compatible embeddings endpoint."""
    url = f"{OPENROUTER_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {_openrouter_key()}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": texts}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code in (429, 500, 502, 503, 504, 529):
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            data = response.json().get("data", [])
            if len(data) != len(texts):
                raise ValueError(f"Expected {len(texts)} embeddings, got {len(data)}")
            # OpenRouter returns an "index" per item; sort to be safe.
            data.sort(key=lambda item: item.get("index", 0))
            return [list(item["embedding"]) for item in data]
        except Exception as error:
            if attempt == MAX_RETRIES - 1 or not _is_retryable(error):
                raise
            delay = BASE_DELAY * (2**attempt)
            print(f"embed_batch(openrouter) retry {attempt + 1}/{MAX_RETRIES} after {delay:.0f}s: {error}")
            time.sleep(delay)

    raise RuntimeError("embed_batch(openrouter) failed after retries.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured provider."""
    if EMBEDDING_PROVIDER == "openrouter":
        return _embed_batch_openrouter(texts)
    return _embed_batch(texts)


def _record_model() -> str:
    """Model string stored in the embeddings file, used to scope the cache."""
    if EMBEDDING_PROVIDER == "openrouter":
        return OPENROUTER_EMBEDDING_MODEL
    return EMBEDDING_MODEL


def _load_existing_embeddings(path: Path, model: str) -> dict[str, list[float]]:
    """Map chunk text -> embedding from a previous run of the *same* model.

    Records produced by a different provider/model are ignored so we never mix
    vectors from different embedding spaces in one collection.
    """
    cached: dict[str, list[float]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            embedding = record.get("embedding")
            if text and embedding and record.get("model") == model:
                cached[text] = embedding
    return cached


_sent_window: list[tuple[float, int]] = []


def _wait_for_quota(count: int, limit: int = TEXTS_PER_MINUTE, window: float = 60.0) -> None:
    """Block until ``count`` more texts fit inside the rolling-minute quota."""
    while True:
        now = time.time()
        _sent_window[:] = [(t, n) for t, n in _sent_window if now - t < window]
        used = sum(n for _, n in _sent_window)
        if used + count <= limit:
            _sent_window.append((now, count))
            return
        sleep_for = window - (now - _sent_window[0][0]) + 1.0
        print(f"pacing: sleep {sleep_for:.0f}s (used {used}/{limit} in last minute)")
        time.sleep(sleep_for)


def _partial_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".partial")


def embed_corpus(
    chunk_path: Path = CHUNK_DIR / "corpus.jsonl",
    output_path: Path = EMBEDDINGS_PATH,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed chunks incrementally and write them to ``output_path``.

    Reuses embeddings from ``output_path`` and any in-progress ``.partial`` file,
    and checkpoints each batch to the partial file so a crash loses no work.
    Returns the number of chunks that required a fresh embedding.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_path(output_path)
    chunks = [json.loads(line) for line in chunk_path.open("r", encoding="utf-8") if line.strip()]
    record_model = _record_model()

    cached = _load_existing_embeddings(output_path, record_model)
    cached.update(_load_existing_embeddings(partial_path, record_model))
    pending: list[tuple[int, str]] = []
    embeddings: list[list[float] | None] = [None] * len(chunks)

    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if text in cached:
            embeddings[index] = cached[text]
        else:
            pending.append((index, text))

    print(f"provider={EMBEDDING_PROVIDER} model={record_model}")
    print(f"chunks={len(chunks)} reuse={len(chunks) - len(pending)} embed={len(pending)}")

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if EMBEDDING_PROVIDER != "openrouter":
            _wait_for_quota(len(batch))
        vectors = embed_texts([text for _, text in batch])
        with partial_path.open("a", encoding="utf-8") as handle:
            for (_, text), vector in zip(batch, vectors):
                handle.write(
                    json.dumps(
                        {"model": record_model, "text": text, "embedding": vector},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        for (index, _), vector in zip(batch, vectors):
            embeddings[index] = vector
        done = min(start + batch_size, len(pending))
        print(f"embedded {done}/{len(pending)}")

    embedded_at = datetime.now(timezone.utc).isoformat()
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk, vector in zip(chunks, embeddings):
            record = {
                "chunk_id": chunk.get("chunk_id"),
                "file_id": chunk.get("file_id"),
                "url": chunk.get("url"),
                "title": chunk.get("title"),
                "doc_type": chunk.get("doc_type"),
                "breadcrumbs": chunk.get("breadcrumbs", []),
                "section_heading": chunk.get("section_heading"),
                "char_count": chunk.get("char_count"),
                "model": record_model,
                "embedded_at": embedded_at,
                "text": chunk.get("text"),
                "embedding": vector,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if partial_path.exists():
        partial_path.unlink()
    print(f"wrote {len(chunks)} embeddings -> {output_path}")
    return len(pending)


def init_qdrant_client(recreate: bool = False) -> QdrantClient:
    client = QdrantClient(path=str(QDRANT_PATH))

    # Check if collection exists, if not create it
    collections = [c.name for c in client.get_collections().collections]
    if recreate and COLLECTION_NAME in collections:
        client.delete_collection(collection_name=COLLECTION_NAME)
        collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE,
            ),
        )

    return client


def store_chunks_to_qdrant(
    input_path: Path = EMBEDDINGS_PATH,
    recreate: bool = False,
) -> None:
    """Load precomputed chunk embeddings and upsert them into Qdrant."""
    qdrant_client = init_qdrant_client(recreate=recreate)

    with input_path.open("r", encoding="utf-8") as f:
        points = []
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            chunk = json.loads(line)

            embedding = chunk.get("embedding")
            if not embedding:
                continue

            # Prepare Qdrant point
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.get("chunk_id"),
                        "file_id": chunk.get("file_id"),
                        "url": chunk.get("url"),
                        "title": chunk.get("title"),
                        "doc_type": chunk.get("doc_type"),
                        "breadcrumbs": chunk.get("breadcrumbs") or [],
                        "section_heading": chunk.get("section_heading"),
                        "char_count": chunk.get("char_count"),
                        "text": chunk.get("text"),
                    },
                )
            )

            # Batch upsert every 100 points
            if len(points) >= 100:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

        if points:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("Successfully stored chunks into Qdrant collection:", COLLECTION_NAME)


if __name__ == "__main__":
    embed_corpus()
    store_chunks_to_qdrant(recreate=True)