"""The composed, typed settings tree.

Read the ``DOCINTEL_`` environment (and ``.env``) with a double-underscore
separator for nested fields::

    DOCINTEL_RETRIEVAL__DEFAULT_TOP_K=10
    DOCINTEL_GENERATION__MAX_RETRIES=5
    DOCINTEL_EMBEDDINGS__RETRYABLE_STATUS_CODES=429,500,503

Access it as ``settings.<group>.<field>`` (e.g.
``settings.retrieval.default_top_k``).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .capabilities import (
    CardSettings,
    InsightSettings,
    SuggestSettings,
    WorkspaceSettings,
)
from .desktop import DesktopSettings
from .embeddings import EmbeddingSettings
from .generation import (
    GenerationSettings,
    HistorySettings,
    ModelSettings,
    TokenSettings,
)
from .graph import GraphSettings
from .ingestion import IngestionSettings
from .paths import PathsSettings
from .retrieval import RetrievalSettings
from .state import StateSettings
from .store import StoreSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    models: ModelSettings = Field(default_factory=ModelSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    tokens: TokenSettings = Field(default_factory=TokenSettings)
    history: HistorySettings = Field(default_factory=HistorySettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)

    cards: CardSettings = Field(default_factory=CardSettings)
    insights: InsightSettings = Field(default_factory=InsightSettings)
    suggest: SuggestSettings = Field(default_factory=SuggestSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)

    state: StateSettings = Field(default_factory=StateSettings)
    desktop: DesktopSettings = Field(default_factory=DesktopSettings)
