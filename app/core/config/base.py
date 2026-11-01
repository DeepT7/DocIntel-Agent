"""Shared base types and helpers for the config package.

Each subsystem is a small :class:`ConfigGroup` (a plain pydantic model). The
top-level :class:`~app.core.config.settings.Settings` is a
``pydantic-settings`` ``BaseSettings`` that reads the ``DOCINTEL_`` environment.
Nested values use a double-underscore delimiter, e.g.::

    DOCINTEL_RETRIEVAL__DEFAULT_TOP_K=10

The helpers below keep the *legacy* (unprefixed) environment names working so an
existing ``.env`` does not break when a field moves into a group.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

# Load .env before the legacy-name helpers run, so old unprefixed keys placed in
# .env keep working. (DOCINTEL_* keys are read by pydantic-settings directly.)
# load_dotenv never overrides variables already present in the environment.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at import time
    pass

# Fixed endpoint, not a per-machine tunable.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ConfigGroup(BaseModel):
    """Base class for a group of related settings.

    ``extra="ignore"`` keeps a partially populated nested env dict valid even if
    it contains a key that is not a field of this group.
    """

    model_config = ConfigDict(extra="ignore")


def env_str(*names: str, default: str = "") -> str:
    """First non-empty value among ``names`` in the environment, else ``default``.

    Used to keep the legacy unprefixed env names (``QDRANT_URL``,
    ``EMBEDDING_PROVIDER``, ``OPENROUTER_*``) working alongside the new nested
    ``DOCINTEL_*`` scheme. A value supplied through the new scheme still wins,
    because it is applied by the settings source, not by the default factory.
    """
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def split_csv(value: object) -> object:
    """Turn a comma-separated string into a list; pass lists/others through."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value
