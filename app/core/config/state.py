"""App-level and per-conversation state limits."""

from .base import ConfigGroup


class StateSettings(ConfigGroup):
    max_recent: int = 8
    max_session_messages: int = 20
    max_conversations: int = 20
    default_conversation: str = "default"
    # Number of documents stored as the "primary focus" after answering.
    primary_docs_max: int = 3
