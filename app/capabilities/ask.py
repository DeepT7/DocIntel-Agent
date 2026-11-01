"""Capability: grounded Q&A over the folder (multi-turn).

Wraps the existing LangGraph RAG graph (``run_rag``) and binds it to a scope, so
answers are retrieved only from the folder the user selected. Local-first by
default: when retrieval is weak it falls back to an honest "not enough support"
answer instead of sending the query to the web (opt in with ``allow_external``).
"""

from pathlib import Path

from app.core.config import (
    ASK_TOP_K,
    HISTORY_MAX_MESSAGES,
    MAX_TOKENS,
)
from app.core.prompts import NO_CONTEXT_SYSTEM, build_no_context_prompt
from app.core.scope import normalize_root
from app.reasoning.answer import _generate, format_history
from app.reasoning.rag_graph import run_rag


def ask(
    scope: str,
    query: str,
    chat_history: list[dict] | None = None,
    allow_external: bool = False,
    thinking: bool = False,
    doc_paths: list[str] | None = None,
    top_k: int | None = None,
    intent: str | None = None,
    queries: list[str] | None = None,
) -> dict:
    """Answer ``query`` from documents under ``scope``, keeping turn history.

    Pass ``doc_paths`` (file paths relative to the scope) to ask about one or
    several documents only — this is how the UI answers "in *this* PDF, ..." and
    supports cross-paper questions ("compare A and B").
    """
    query = (query or "").strip()
    if not query:
        return {"query": query, "scope": scope, "answer": "", "sources": []}

    root = normalize_root(scope)
    paths = [p for p in dict.fromkeys(doc_paths or []) if p] or None
    result = run_rag(
        query,
        chat_history=chat_history or [],
        scope=root,
        doc_paths=paths,
        top_k=top_k or ASK_TOP_K,
        allow_external=allow_external,
        thinking=thinking,
        intent=intent,
        queries=queries,
    )
    return {
        "query": query,
        "scope": root,
        "doc_paths": paths,
        "intent": result.get("intent", "question"),
        "standalone_query": result.get("standalone_query", query),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "section_locks": result.get("section_locks", {}),
        "section_locked_chunk_ids": result.get("section_locked_chunk_ids", []),
    }


def answer_without_context(
    scope: str,
    query: str,
    chat_history: list[dict] | None = None,
    notes: list[str] | None = None,
    files: list[str] | None = None,
    thinking: bool = False,
) -> dict:
    """Answer in "No-Context / Notes Only" mode (every source switched off).

    No retrieval happens and no document content is read: the model may use only
    the conversation history, the open notes, the workspace file names (metadata)
    and its general knowledge. The system prompt forbids pretending it read a file
    and forbids reconstructing one from the conversation. This is the deliberate
    alternative to silently searching the documents the user excluded.
    """
    query = (query or "").strip()
    root = normalize_root(scope)
    empty = {
        "query": query,
        "scope": root,
        "doc_paths": [],
        "intent": "no_context",
        "standalone_query": query,
        "answer": "",
        "sources": [],
    }
    if not query:
        return empty

    notes_text = "\n\n".join(
        str(note).strip() for note in (notes or []) if str(note).strip()
    )
    file_list = [str(path) for path in (files or []) if path]
    files_text = "\n".join(f"- {Path(path).name}" for path in file_list)
    answer = _generate(
        build_no_context_prompt(
            query,
            format_history(chat_history or [], max_messages=HISTORY_MAX_MESSAGES),
            files_text,
            len(file_list),
            notes_text,
        ),
        system=NO_CONTEXT_SYSTEM,
        max_tokens=MAX_TOKENS["no_context_thinking" if thinking else "no_context"],
        thinking=thinking,
    )
    return {**empty, "answer": answer}
