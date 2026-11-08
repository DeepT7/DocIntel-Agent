"""LLM models, generation retry behaviour, token budgets and history windows."""

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from .base import (
    OPENROUTER_BASE_URL,
    ConfigGroup,
    env_str,
    split_csv,
)

_DEFAULT_GENERATION_MODELS = (
    "z-ai/glm-4.7-flash,"
    "nvidia/nemotron-3-super-120b-a12b:free,"
    "qwen/qwen3-30b-a3b-instruct-2507"
)
_DEFAULT_UTILITY_MODELS = "qwen/qwen3-30b-a3b-instruct-2507"


class ModelSettings(ConfigGroup):
    # DEFAULT_MODEL stays Gemini-only: it powers external_search_node, which
    # relies on Google Search grounding that OpenRouter does not provide.
    default_model: str = "gemini-3.5-flash"

    # Answer generation provider: "gemini" or "openrouter".
    generation_provider: str = Field(
        default_factory=lambda: env_str("GENERATION_PROVIDER", default="openrouter")
    )
    openrouter_base_url: str = OPENROUTER_BASE_URL

    # Fallback chain, tried in order (each entry is a model id).
    openrouter_generation_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: split_csv(
            env_str(
                "OPENROUTER_GENERATION_MODELS", default=_DEFAULT_GENERATION_MODELS
            )
        )
    )
    # Fast, non-reasoning models for auxiliary tasks (condense, grade, rewrite).
    openrouter_utility_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: split_csv(
            env_str("OPENROUTER_UTILITY_MODELS", default=_DEFAULT_UTILITY_MODELS)
        )
    )

    @field_validator(
        "openrouter_generation_models", "openrouter_utility_models", mode="before"
    )
    @classmethod
    def _parse_models(cls, value: object) -> object:
        return split_csv(value)


class GenerationSettings(ConfigGroup):
    max_retries: int = 3
    timeout: int = 180
    backoff_base: float = 3.0
    empty_retry_delay: float = 2.0
    cooldown_seconds: int = 300


class TokenSettings(ConfigGroup):
    """Per-task ``max_tokens`` budgets.

    ``as_dict()`` preserves the historical ``MAX_TOKENS["answer"]`` access
    pattern while new code can use ``settings.tokens.answer``.
    """

    answer: int = 700
    # Thinking mode needs a larger output budget: hidden reasoning tokens are
    # billed inside the same completion and would otherwise starve the answer.
    answer_thinking: int = 2000
    no_context: int = 1200
    no_context_thinking: int = 2200
    chat_node: int = 200
    workspace_node: int = 800
    condense: int = 128
    classify_intent: int = 8
    resolve_section: int = 120
    grade: int = 256
    rewrite: int = 128
    route: int = 600
    table_tool: int = 200
    document_card: int = 700
    insight_single: int = 900
    insight_map: int = 400
    insight_reduce: int = 900

    def as_dict(self) -> dict[str, int]:
        return self.model_dump()


class HistorySettings(ConfigGroup):
    # Window sizes are counts of messages, not turns.
    max_messages: int = 6
    max_messages_route: int = 4
    max_messages_classify: int = 4
