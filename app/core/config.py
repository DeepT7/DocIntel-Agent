from pathlib import Path
import os

# Chunking
CHUNK_TARGET_CHARS = 1200
CHUNK_MAX_CHARS = 1800

DATA_DIR = Path("data")

# Folder knowledge base (local documents)
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".py", ".json", ".docx", ".pdf", ".html", ".htm"}

# Retrieval settings
EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_TOP_K = 5

# Embedding provider: "gemini" (google-genai) or "openrouter".
# Corpus and query embeddings MUST use the same provider + model.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openrouter")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"
)
EMBEDDING_DIM = 4096

# Qdrant settings
# Embedded mode (path) allows only ONE process at a time. Set QDRANT_URL to an
# HTTP endpoint (e.g. http://localhost:6333) to use a real Qdrant server instead,
# which allows many processes to share the same data concurrently.
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_PATH = Path("output/qdrant_db")
QDRANT_COLLECTION = "docs_local"

# Native hybrid search: the collection holds one dense (semantic) vector and one
# sparse (BM25) vector per point. Fusion combines them with Reciprocal Rank
# Fusion inside Qdrant.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
SPARSE_MODEL = "Qdrant/bm25"  # FastEmbed sparse encoder
HYBRID_PREFETCH_LIMIT = 20    # candidates pulled per signal before fusion

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