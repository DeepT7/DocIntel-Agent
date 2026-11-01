"""Retrieval ranking and citation-density heuristic settings."""

from .base import ConfigGroup


class RetrievalSettings(ConfigGroup):
    default_top_k: int = 5
    # Q&A benefits from more candidates than a plain lookup.
    ask_top_k: int = 8

    # Fusion fetch sizing multipliers.
    fetch_multiplier: int = 3
    fetch_min_extra: int = 10

    # RRF score penalty applied to bibliography-like chunks (kept, not dropped).
    reference_penalty: float = 0.4
    # Default chunk limit for whole-section reads.
    section_chunks_limit: int = 20

    # Citation-density heuristic for detecting bibliography-like sections.
    reference_min_text: int = 200
    reference_density_window: int = 500
    reference_density_threshold: float = 2.0
