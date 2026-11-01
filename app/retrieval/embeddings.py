"""Embedding client.

Batches text through the configured provider (Gemini or OpenRouter) with retry
and, for Gemini's free tier, per-minute pacing. Embeddings are deterministic, so
callers can cache them by text (see ``load_cached_embeddings``).
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests

from app.core.config import (
    EMBED_BASE_DELAY,
    EMBED_MAX_RETRIES,
    EMBED_REQUEST_TIMEOUT,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
    RETRYABLE_STATUS_CODES,
)

load_dotenv()

# Re-exported under the original names so the rest of this module does not
# change; the canonical values live in config.
MAX_RETRIES = EMBED_MAX_RETRIES
BASE_DELAY = EMBED_BASE_DELAY

_client = None


def _get_client() -> genai.Client:
    """Lazily create the Gemini client so other providers need no API key."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or API_KEY before using the Gemini provider.")
        _client = genai.Client(api_key=api_key)
    return _client


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in RETRYABLE_STATUS_CODES:
        return True
    text = str(error).upper()
    return any(token in text for token in ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE"))


def _embed_batch(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Embed a batch with Gemini, retrying with exponential backoff."""
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
            response = requests.post(url, headers=headers, json=payload, timeout=EMBED_REQUEST_TIMEOUT)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            data = response.json().get("data", [])
            if len(data) != len(texts):
                raise ValueError(f"Expected {len(texts)} embeddings, got {len(data)}")
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


def record_model() -> str:
    """Model string stored alongside embeddings, used to scope the cache."""
    if EMBEDDING_PROVIDER == "openrouter":
        return OPENROUTER_EMBEDDING_MODEL
    return EMBEDDING_MODEL


def load_cached_embeddings(path: Path, model: str | None = None) -> dict[str, list[float]]:
    """Map chunk text -> embedding from a previous run of the *same* model.

    Records produced by a different provider/model are ignored so we never mix
    vectors from different embedding spaces in one collection.
    """
    model = model or record_model()
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