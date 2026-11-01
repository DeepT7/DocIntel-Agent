"""Qdrant / vector store settings."""

from pathlib import Path

from pydantic import Field

from .base import ConfigGroup, env_str


class StoreSettings(ConfigGroup):
    # Embedded mode (path) allows only ONE process at a time. Set QDRANT_URL to
    # an HTTP endpoint to use a real Qdrant server instead, which allows many
    # processes to share the same data concurrently.
    qdrant_url: str = Field(default_factory=lambda: env_str("QDRANT_URL", default=""))
    qdrant_path: Path = Path("output/qdrant_db")
    collection: str = "docs_local"

    # Native hybrid search: one dense (semantic) and one sparse (BM25) vector
    # per point, fused with Reciprocal Rank Fusion inside Qdrant.
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    sparse_model: str = "Qdrant/bm25"
    hybrid_prefetch_limit: int = 20

    # Paging size when scrolling Qdrant (metadata-only reads).
    scroll_page_size: int = 256
    scroll_page_size_large: int = 1000

    # Retry before giving up on opening the embedded store (reload hand-over).
    open_retries: int = 5
    open_retry_delay: float = 0.6
