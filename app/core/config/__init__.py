"""Central configuration package.

The typed, nested settings tree is the single source of truth. New code should
use it directly::

    from app.core.config import settings
    settings.retrieval.default_top_k

Every value can be overridden without editing code, via the ``DOCINTEL_``
environment (or ``.env``), using a double underscore for nesting::

    DOCINTEL_RETRIEVAL__DEFAULT_TOP_K=10
    DOCINTEL_GENERATION__MAX_RETRIES=5

The flat ``UPPER_CASE`` names at the bottom of this file are a
backward-compatibility facade for the existing imports across the codebase
(``from app.core.config import DEFAULT_TOP_K``). They merely mirror ``settings``
— they are not a second source of truth, and new code should prefer
``settings.<group>.<field>``.
"""

from .settings import Settings

# The single composed instance read by the application.
settings = Settings()

# --------------------------------------------------------------------------- #
# Backward-compatible flat aliases (existing imports). Prefer `settings.*`.
# --------------------------------------------------------------------------- #
# Paths & data
DOCS_DIR = settings.paths.docs_dir
INDEX_DIR = settings.paths.index_dir
STATE_PATH = settings.paths.state_path

# Ingestion & chunking
CHUNK_TARGET_CHARS = settings.ingestion.chunk_target_chars
CHUNK_MAX_CHARS = settings.ingestion.chunk_max_chars
SUPPORTED_EXTENSIONS = settings.ingestion.supported_extensions
INDEX_VERSION = settings.ingestion.index_version
UPSERT_BATCH_SIZE = settings.ingestion.upsert_batch_size
CACHE_PRUNE_RATIO = settings.ingestion.cache_prune_ratio
CACHE_PRUNE_MIN_EXTRA = settings.ingestion.cache_prune_min_extra
BREADCRUMB_MAX_DEPTH = settings.ingestion.breadcrumb_max_depth
OUTLINE_VERSION = settings.ingestion.outline_version
WATCH_ENABLED = settings.ingestion.watch_enabled
WATCH_DEBOUNCE_SECONDS = settings.ingestion.watch_debounce_seconds
WATCH_POLL_SECONDS = settings.ingestion.watch_poll_seconds
CONTENT_MAX_BYTES = settings.ingestion.content_max_bytes

# Embeddings
EMBEDDING_MODEL = settings.embeddings.model_name
EMBEDDING_PROVIDER = settings.embeddings.provider
OPENROUTER_BASE_URL = settings.embeddings.openrouter_base_url
OPENROUTER_EMBEDDING_MODEL = settings.embeddings.openrouter_model
EMBEDDING_DIM = settings.embeddings.dim
EMBED_BATCH_SIZE = settings.embeddings.batch_size
EMBED_MAX_RETRIES = settings.embeddings.max_retries
EMBED_BASE_DELAY = settings.embeddings.base_delay
EMBED_REQUEST_TIMEOUT = settings.embeddings.request_timeout
RETRYABLE_STATUS_CODES = settings.embeddings.retryable_status_codes

# Qdrant / vector store
QDRANT_URL = settings.store.qdrant_url
QDRANT_PATH = settings.store.qdrant_path
QDRANT_COLLECTION = settings.store.collection
DENSE_VECTOR_NAME = settings.store.dense_vector_name
SPARSE_VECTOR_NAME = settings.store.sparse_vector_name
SPARSE_MODEL = settings.store.sparse_model
HYBRID_PREFETCH_LIMIT = settings.store.hybrid_prefetch_limit
SCROLL_PAGE_SIZE = settings.store.scroll_page_size
SCROLL_PAGE_SIZE_LARGE = settings.store.scroll_page_size_large
STORE_OPEN_RETRIES = settings.store.open_retries
STORE_OPEN_RETRY_DELAY = settings.store.open_retry_delay

# Retrieval
DEFAULT_TOP_K = settings.retrieval.default_top_k
ASK_TOP_K = settings.retrieval.ask_top_k
FETCH_MULTIPLIER = settings.retrieval.fetch_multiplier
FETCH_MIN_EXTRA = settings.retrieval.fetch_min_extra
REFERENCE_PENALTY = settings.retrieval.reference_penalty
SECTION_CHUNKS_LIMIT = settings.retrieval.section_chunks_limit
REFERENCE_MIN_TEXT = settings.retrieval.reference_min_text
REFERENCE_DENSITY_WINDOW = settings.retrieval.reference_density_window
REFERENCE_DENSITY_THRESHOLD = settings.retrieval.reference_density_threshold

# Models & generation
DEFAULT_MODEL = settings.models.default_model
GENERATION_PROVIDER = settings.models.generation_provider
OPENROUTER_GENERATION_MODELS = settings.models.openrouter_generation_models
OPENROUTER_UTILITY_MODELS = settings.models.openrouter_utility_models
GENERATION_MAX_RETRIES = settings.generation.max_retries
GENERATION_TIMEOUT = settings.generation.timeout
GENERATION_BACKOFF_BASE = settings.generation.backoff_base
GENERATION_EMPTY_RETRY_DELAY = settings.generation.empty_retry_delay
MODEL_COOLDOWN_SECONDS = settings.generation.cooldown_seconds
MAX_TOKENS = settings.tokens.as_dict()
HISTORY_MAX_MESSAGES = settings.history.max_messages
HISTORY_MAX_MESSAGES_ROUTE = settings.history.max_messages_route
HISTORY_MAX_MESSAGES_CLASSIFY = settings.history.max_messages_classify

# Reasoning graph
MAX_SECTION_DOCS = settings.graph.max_section_docs
MAX_SECTION_CHUNKS = settings.graph.max_section_chunks
SECTION_CAP_FLOOR = settings.graph.section_cap_floor
SECTION_PER_DOC_FLOOR = settings.graph.section_per_doc_floor
SECTION_EXPAND_PER_SECTION = settings.graph.section_expand_per_section
SECTION_EXPAND_MAX_CHUNKS = settings.graph.section_expand_max_chunks
GRADE_SKIP_THRESHOLD = settings.graph.grade_skip_threshold
MAX_REWRITES = settings.graph.max_rewrites
GRADE_CHUNK_CHARS = settings.graph.grade_chunk_chars
ROUTE_MAX_QUERIES = settings.graph.route_max_queries
SECTION_IDS_MAX = settings.graph.section_ids_max
SECTION_QUERY_MIN_TITLE_CHARS = settings.graph.section_query_min_title_chars
SECTION_QUERY_MIN_TITLE_WORDS = settings.graph.section_query_min_title_words

# Capabilities
CARD_SINGLE_PASS_CHARS = settings.cards.single_pass_chars
CARD_MAX_KEYWORDS = settings.cards.max_keywords
CACHE_HASH_LEN = settings.cards.cache_hash_len
INSIGHT_SINGLE_PASS_CHARS = settings.insights.single_pass_chars
INSIGHT_MAP_GROUP_CHARS = settings.insights.map_group_chars
SUGGEST_MAX_CLIPBOARD_CHARS = settings.suggest.max_clipboard_chars
WORKSPACE_MAX_FILES = settings.workspace.max_files

# App state & sessions
MAX_RECENT = settings.state.max_recent
MAX_SESSION_MESSAGES = settings.state.max_session_messages
MAX_CONVERSATIONS = settings.state.max_conversations
DEFAULT_CONVERSATION = settings.state.default_conversation
PRIMARY_DOCS_MAX = settings.state.primary_docs_max

# Desktop app
DESKTOP_MAX_CLIPBOARD = settings.desktop.max_clipboard
DESKTOP_HOTKEY = settings.desktop.hotkey
QUICK_WINDOW_WIDTH = settings.desktop.quick_window_width
QUICK_WINDOW_HEIGHT = settings.desktop.quick_window_height
MAIN_WINDOW_WIDTH = settings.desktop.main_window_width
MAIN_WINDOW_HEIGHT = settings.desktop.main_window_height
