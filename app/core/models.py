from typing import TypedDict, NotRequired


class ChunkRecord(TypedDict):
    chunk_id: str
    file_id: str
    url: str
    title: str
    source_domain: str
    doc_type: str
    breadcrumbs: list[str]
    section_heading: str
    section_anchor: str
    section_role: str
    priority_band: str
    importance_score: float
    reasons: list[str]
    text: str
    char_count: int


class EmbeddingRecord(TypedDict):
    chunk_id: str
    file_id: str
    url: str
    title: str
    doc_type: str
    breadcrumbs: list[str]
    section_heading: str
    char_count: int
    model: str
    embedded_at: str
    text: str
    embedding: list[float]


class RetrievedChunk(TypedDict):
    score: float
    chunk_id: str
    file_id: str
    doc_type: str
    url: str
    title: str
    breadcrumbs: list[str]
    section_heading: str
    char_count: int
    text: str


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
    chat_history: NotRequired[list[dict]]
    standalone_query: NotRequired[str]
    retrieved_chunks: NotRequired[list[RetrievedChunk]]
    context: NotRequired[str]
    external_context: NotRequired[str]
    answer: NotRequired[str]
    sources: NotRequired[list[RetrievedChunk | ExternalSource]]
    external_sources: NotRequired[list[ExternalSource]]
    retrieval_ok: NotRequired[bool]
    retrieval_reason: NotRequired[str]
    external_ok: NotRequired[bool]
    external_reason: NotRequired[str]
