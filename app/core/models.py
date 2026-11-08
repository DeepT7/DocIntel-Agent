from typing import TypedDict

try:  # Python 3.11+
    from typing import NotRequired
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from typing_extensions import NotRequired


class RetrievedChunk(TypedDict):
    score: float
    chunk_id: str
    file_id: str
    doc_type: str
    url: str
    title: str
    breadcrumbs: list[str]
    section_heading: str
    section_id: str
    section_ancestors: list[str]
    section_titles: list[str]
    section_depths: list[int]
    section_title_clean: str
    section_number: str
    chunk_type: str
    raw_heading: str
    char_count: int
    text: str
    # Raw-asset pointers (table_stub / image_stub chunks).
    caption: NotRequired[str | None]
    table_number: NotRequired[str | None]
    asset_id: NotRequired[str | None]
    asset_path: NotRequired[str | None]
    headers: NotRequired[list[str] | None]
    row_count: NotRequired[int | None]
    table_hydrated: NotRequired[bool]
    table_rows: NotRequired[int | None]


class ExternalSource(TypedDict):
    title: str
    url: str
    section_heading: str
    score: float
    breadcrumbs: list[str]
    doc_type: str
    text: str


class AnswerResult(TypedDict):
    query: str
    answer: str
    context: str
    sources: list[RetrievedChunk | ExternalSource]


class RAGState(TypedDict):
    query: str
    scope: NotRequired[str]
    doc_paths: NotRequired[list[str]]
    top_k: NotRequired[int]
    allow_external: NotRequired[bool]
    # Thinking mode: ask the generation model to reason before answering.
    thinking: NotRequired[bool]
    chat_history: NotRequired[list[dict]]
    intent: NotRequired[str]  # "chat" (small talk / meta) or "question"
    standalone_query: NotRequired[str]
    # path -> canonical section ids resolved by the self-query. A parent id
    # expands to its subtree in section_chunks.
    section_locks: NotRequired[dict[str, list[str]]]
    section_locked_chunk_ids: NotRequired[list[str]]
    queries: NotRequired[list[str]]  # router's sub-queries (multi-query retrieval)
    retrieved_chunks: NotRequired[list[RetrievedChunk]]
    relevant_chunks: NotRequired[list[RetrievedChunk]]
    graded_ok: NotRequired[bool]
    rewrite_count: NotRequired[int]
    context: NotRequired[str]
    external_context: NotRequired[str]
    answer: NotRequired[str]
    sources: NotRequired[list[RetrievedChunk | ExternalSource]]
    external_sources: NotRequired[list[ExternalSource]]
    retrieval_ok: NotRequired[bool]
    retrieval_reason: NotRequired[str]
    external_ok: NotRequired[bool]
    external_reason: NotRequired[str]
