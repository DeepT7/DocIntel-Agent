from pathlib import Path
import os

DOCS_BASE_URL = "https://docs.langchain.com"
REFERENCE_BASE_URL = "https://reference.langchain.com"

CRAWL_ROOTS = [
    DOCS_BASE_URL,
]

SITEMAP_URLS = [
    f"{DOCS_BASE_URL}/sitemap.xml",
]

ALLOWED_DOMAINS = {
    "docs.langchain.com",
    "reference.langchain.com",
}

ALLOWED_PATH_PREFIXES = (
    "/oss/python/langgraph",
    "/python/langgraph",
)

EXCLUDED_PATH_HINTS = (
    "/changelog",
    "/release-notes",
    "/mcp",
    "/javascript",
    "/typescript",
)

SKIP_FILE_EXTENSIONS = {
    ".css",
    ".js",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".xml",
    ".txt",
    ".zip",
}

MAX_PAGES = 200
REQUEST_TIMEOUT = 20
USER_AGENT = "langgraph-rag-crawler/0.2"

CHUNK_TARGET_CHARS = 1200
CHUNK_MAX_CHARS = 1800

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"
CHUNK_DIR = DATA_DIR / "chunks"
EMBED_DIR = DATA_DIR / "embeddings"
GRAPH_DIR = DATA_DIR / "graph"
SAMPLES_DIR = DATA_DIR / "samples"

# Retrieval settings
EMBEDDINGS_PATH = EMBED_DIR / "corpus_embeddings.jsonl"
EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_TOP_K = 5

# Embedding provider: "gemini" (google-genai) or "openrouter".
# Corpus and query embeddings MUST use the same provider + model.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openrouter")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "google/gemini-embedding-2"
)
EMBEDDING_DIM = 3072

# Qdrant settings
QDRANT_PATH = Path("output/qdrant_db")
QDRANT_COLLECTION = "doc_chunks"

# Model settings
# DEFAULT_MODEL stays Gemini-only: it powers external_search_node, which relies
# on Google Search grounding that OpenRouter does not provide.
DEFAULT_MODEL = "gemini-3.5-flash"

# Answer generation provider: "gemini" (google-genai) or "openrouter".
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "openrouter")
# Comma-separated fallback chain, tried in order (each entry must end with a
# comma except the last). A reliable paid model comes first; free models are
# fallbacks; the last entry is an instruct model used as a safety net.
OPENROUTER_GENERATION_MODELS = [
    model.strip()
    for model in os.getenv(
        "OPENROUTER_GENERATION_MODELS",
        "z-ai/glm-4.7-flash,"
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "qwen/qwen3-30b-a3b-instruct-2507"
    ).split(",")
    if model.strip()
]
OPENROUTER_GENERATION_MODEL = OPENROUTER_GENERATION_MODELS[0]

# Fast, non-reasoning model for auxiliary LLM tasks (condense, grade, rewrite).
# Reasoning models spend seconds "thinking" before answering, which is wasteful
# for tasks that emit one short sentence or a tiny JSON object.
OPENROUTER_UTILITY_MODELS = [
    model.strip()
    for model in os.getenv(
        "OPENROUTER_UTILITY_MODELS",
        "qwen/qwen3-30b-a3b-instruct-2507"
    ).split(",")
    if model.strip()
]
