"""Capability-specific budgets: document cards, insights, workspace."""

from .base import ConfigGroup


class CardSettings(ConfigGroup):
    # Only this much text feeds the card prompt (cards summarize, not index).
    single_pass_chars: int = 14000
    max_keywords: int = 15
    # Filename hash truncation for registry/insight cache files.
    cache_hash_len: int = 8


class InsightSettings(ConfigGroup):
    # A document whose full text fits here is summarized in a single call.
    single_pass_chars: int = 14000
    # When map-reducing, chunks are packed into groups of about this size.
    map_group_chars: int = 6000


class WorkspaceSettings(ConfigGroup):
    max_files: int = 300
