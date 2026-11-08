"""Reasoning-graph caps, gating thresholds and section-resolution heuristics."""

from .base import ConfigGroup


class GraphSettings(ConfigGroup):
    # Section resolution is one fast call per focused document; cap the fan-out.
    max_section_docs: int = 3
    # Total chunks pulled from resolved sections across all documents.
    max_section_chunks: int = 24
    # Lower bound for the section cap once top_k is reserved.
    section_cap_floor: int = 6
    # Per-document chunk budget floor when splitting the section cap.
    section_per_doc_floor: int = 4

    # Small-to-big expansion (answer node widens each chunk to its section).
    section_expand_per_section: int = 4
    section_expand_max_chunks: int = 16

    # Skip LLM grading when the top dense score exceeds this.
    grade_skip_threshold: float = 0.5
    # Max query rewrites before falling back / going external.
    max_rewrites: int = 1
    # Per-chunk truncation (chars) fed to the grader.
    grade_chunk_chars: int = 700

    # Route: max rewritten sub-queries kept per turn.
    route_max_queries: int = 3
    # Section resolver: max canonical ids kept.
    section_ids_max: int = 4
    # Section-query heuristic: minimum title specificity before matching.
    section_query_min_title_chars: int = 12
    section_query_min_title_words: int = 2

    # Table hydration: inline rows injected into the answer context per table.
    table_max_rows: int = 40
    table_max_chars: int = 6000

    # Table tool: on-demand raw-table fetch before answer generation.
    table_tool_max_tables: int = 3
    table_tool_context_chars: int = 6000
