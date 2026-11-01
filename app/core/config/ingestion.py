"""Document ingestion, chunking and index schema versioning."""

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from .base import ConfigGroup


class IngestionSettings(ConfigGroup):
    # Chunking.
    chunk_target_chars: int = 1200
    chunk_max_chars: int = 1800
    supported_extensions: Annotated[set[str], NoDecode] = Field(
        default_factory=lambda: {
            ".md", ".markdown", ".txt", ".py", ".json",
            ".docx", ".pdf", ".html", ".htm",
        }
    )

    # Version bump forces a full re-index when extraction/chunking changes.
    index_version: str = "13"
    upsert_batch_size: int = 100

    # Auto-prune the embedding cache only when it has grown well past the index.
    cache_prune_ratio: float = 1.5
    cache_prune_min_extra: int = 200

    # Section outline depth used in contextual chunk headers.
    breadcrumb_max_depth: int = 3
    # Bumped when the canonical tree shape/metadata changes.
    outline_version: int = 2

    # File watcher + incremental indexer.
    watch_enabled: bool = True
    # Changes are coalesced for this long before a background re-index starts.
    watch_debounce_seconds: float = 1.0
    # Polling fallback when watchdog is not installed (or cannot watch the path).
    watch_poll_seconds: float = 4.0
    # Largest file (bytes) the on-demand content API will read in one call.
    content_max_bytes: int = 4_000_000

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def _parse_extensions(cls, value: object) -> object:
        # Accept "a,b,c" from an env var (NoDecode hands us the raw string).
        if isinstance(value, str):
            return {part.strip().lower() for part in value.split(",") if part.strip()}
        if isinstance(value, (list, tuple)):
            return {str(part).strip().lower() for part in value if str(part).strip()}
        return value
