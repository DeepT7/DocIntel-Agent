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
            ".md", ".markdown", ".txt", ".docx", ".pptx",
            ".pdf", ".html", ".htm",
        }
    )

    # Version bump forces a full re-index when extraction/chunking changes.
    index_version: str = "15"
    upsert_batch_size: int = 100

    # Auto-prune the embedding cache only when it has grown well past the index.
    cache_prune_ratio: float = 1.5
    cache_prune_min_extra: int = 200

    # Smart asset extraction (visual assets). Candidates smaller than these
    # thresholds, extremely thin, near-blank, or repeated template images are
    # rejected as visual junk (logos, icons, rules, banners).
    asset_min_bytes: int = 1024
    asset_min_width: int = 96
    asset_min_height: int = 96
    asset_min_area: int = 12_000
    asset_max_aspect_ratio: float = 8.0
    asset_min_ink_ratio: float = 0.005
    # PDF picture boxes are rendered at this DPI; tiny on-page boxes are skipped.
    asset_pdf_dpi: int = 150
    asset_pdf_min_display: float = 48.0
    asset_pdf_min_display_area: int = 4_000
    # Office pictures also carry their displayed size in the XML (inches).
    asset_office_min_display: float = 0.9
    asset_office_min_display_area: float = 0.8
    # An identical image used this many times (and not large) is a template logo.
    asset_repeat_count: int = 3
    asset_repeat_max_area: int = 200_000

    # Section outline depth used in contextual chunk headers.
    breadcrumb_max_depth: int = 3
    # Bumped when the canonical tree shape/metadata changes.
    outline_version: int = 3

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
