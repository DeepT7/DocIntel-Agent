"""Embedding provider, model, retry and pacing settings."""

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from .base import OPENROUTER_BASE_URL, ConfigGroup, env_str


class EmbeddingSettings(ConfigGroup):
    # Provider: "gemini" (google-genai) or "openrouter". Corpus and query
    # embeddings MUST use the same provider + model.
    provider: str = Field(
        default_factory=lambda: env_str("EMBEDDING_PROVIDER", default="openrouter")
    )
    model_name: str = "gemini-embedding-2"
    openrouter_base_url: str = OPENROUTER_BASE_URL
    openrouter_model: str = Field(
        default_factory=lambda: env_str(
            "OPENROUTER_EMBEDDING_MODEL", default="qwen/qwen3-embedding-8b"
        )
    )
    dim: int = 4096

    # Batch + retry/pacing for the embedding client.
    batch_size: int = 50
    max_retries: int = 6
    base_delay: float = 5.0
    request_timeout: int = 120

    # HTTP statuses treated as transient (retryable) across HTTP clients.
    retryable_status_codes: Annotated[set[int], NoDecode] = Field(
        default_factory=lambda: {429, 500, 502, 503, 504, 529}
    )

    @field_validator("retryable_status_codes", mode="before")
    @classmethod
    def _parse_status_codes(cls, value: object) -> object:
        if isinstance(value, str):
            return {int(part) for part in value.split(",") if part.strip()}
        if isinstance(value, (list, tuple)):
            return {int(part) for part in value}
        return value
