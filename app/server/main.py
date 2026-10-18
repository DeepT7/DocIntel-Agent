"""HTTP surface for the whole system (API-first).

Every capability is reachable over HTTP, and the UI is
only a client. This lets the Main window be an ordinary browser page while the same API can later serve a TypeScript frontend or the
ambient Quick window.

Run:  uvicorn app.server.main:app --reload --port 8756
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import app_state, jobs
from app.capabilities.ask import (
    answer_without_context as cap_answer_without_context,
    ask as cap_ask,
)
from app.capabilities.insights import get_insight, list_insights, summarize_source
from app.capabilities.outline import outline as cap_outline, section as cap_section
from app.capabilities.registry import (
    build_missing as cap_build_cards,
    get_card as cap_get_card,
    list_cards as cap_list_cards,
)
from app.capabilities.router import (
    resolve_scope as cap_resolve_scope,
    route as cap_route,
)
from app.capabilities.search import search as cap_search
from app.capabilities.sources import list_sources as cap_sources
from app.capabilities.suggest import suggest as cap_suggest
from app.capabilities.sync import sync as cap_sync

app = FastAPI(title="DocIntel", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ScopeRequest(BaseModel):
    path: str
    conversation_id: Optional[str] = None


class SyncRequest(BaseModel):
    scope: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    scope: Optional[str] = None
    path: Optional[str] = None
    paths: Optional[list[str]] = None
    top_k: int = 5


class SuggestRequest(BaseModel):
    query: str = ""
    clipboard: str = ""
    scope: Optional[str] = None


class AskRequest(BaseModel):
    query: str
    scope: Optional[str] = None
    conversation_id: Optional[str] = None
    path: Optional[str] = None  # restrict the answer to one document
    paths: Optional[list[str]] = None  # restrict to several documents (cross-paper)
    # Open notes supplied by the UI. Used only in No-Context / Notes Only mode
    # (every source switched off); ignored when documents are being searched.
    notes: Optional[list[str]] = None
    top_k: Optional[int] = None
    history: Optional[list[dict]] = None
    use_session: bool = True
    # Explicit switch: apply the sidebar source selection (server-side truth).
    # Distinguishes "no filter" from "the user unchecked everything".
    use_selection: bool = False
    allow_external: bool = False
    save: bool = True


class InsightRequest(BaseModel):
    path: str
    scope: Optional[str] = None
    force: bool = False
    background: bool = True


class WorkingSetRequest(BaseModel):
    paths: list[str] = []
    conversation_id: Optional[str] = None


class ClearRequest(BaseModel):
    conversation_id: Optional[str] = None


class SelectionRequest(BaseModel):
    paths: list[str] = []  # documents to EXCLUDE from the query scope


class RegistryRequest(BaseModel):
    scope: Optional[str] = None
    force: bool = False


def _resolve_scope(scope: Optional[str]) -> str | None:
    return scope or app_state.get_scope()


# Document scoping (which files a message is about) now lives in
# app.capabilities.router: one LLM turn-router replaces the old title matcher.


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/state")
def state(conversation_id: str = app_state.DEFAULT_CONVERSATION) -> dict:
    return app_state.snapshot(conversation_id)


@app.post("/api/scope")
def set_scope(request: ScopeRequest) -> dict:
    conv_id = request.conversation_id or app_state.DEFAULT_CONVERSATION
    app_state.set_scope(request.path, conv_id)
    return app_state.snapshot(conv_id)


@app.post("/api/sync")
def sync(request: SyncRequest) -> dict:
    scope = _resolve_scope(request.scope)
    if not scope:
        raise HTTPException(status_code=400, detail="no scope selected")
    return cap_sync(scope)


@app.get("/api/sources")
def sources(scope: Optional[str] = None) -> list[dict]:
    scope = _resolve_scope(scope)
    return cap_sources(scope) if scope else []


@app.get("/api/outline")
def outline(scope: Optional[str] = None, path: str = "") -> dict:
    """The canonical section tree (table of contents) of one document."""
    scope = _resolve_scope(scope)
    if not scope or not path:
        raise HTTPException(status_code=400, detail="scope and path are required")
    return cap_outline(scope, path)


@app.get("/api/section")
def section(
    scope: Optional[str] = None, path: str = "", section_id: str = ""
) -> list[dict]:
    """Read one or more whole sections by canonical id (comma-separated).

    A parent id (``s4``) includes its subtree; a child id (``s4.1``) includes
    that branch, including its table chunks.
    """
    scope = _resolve_scope(scope)
    section_ids = [item.strip() for item in section_id.split(",") if item.strip()]
    if not scope or not path or not section_ids:
        raise HTTPException(
            status_code=400, detail="scope, path and section_id are required"
        )
    return cap_section(scope, path, section_ids)


@app.post("/api/search")
def search(request: SearchRequest) -> list[dict]:
    scope = _resolve_scope(request.scope)
    if not scope:
        raise HTTPException(status_code=400, detail="no scope selected")
    paths = [p for p in dict.fromkeys([*(request.paths or []), *( [request.path] if request.path else [] )]) if p]
    return cap_search(scope, request.query, top_k=request.top_k, paths=paths or None)


@app.post("/api/suggest")
def suggest(request: SuggestRequest) -> dict:
    scope = _resolve_scope(request.scope)
    if not scope:
        raise HTTPException(status_code=400, detail="no scope selected")
    return cap_suggest(scope, request.query, clipboard=request.clipboard)


# --------------------------------------------------------------------------- #
# Q&A over the folder (LangGraph RAG, scoped, multi-turn)
# --------------------------------------------------------------------------- #
@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    scope = _resolve_scope(request.scope)
    if not scope:
        raise HTTPException(status_code=400, detail="no scope selected")
    conv_id = request.conversation_id or app_state.DEFAULT_CONVERSATION
    history = request.history
    if history is None and request.use_session:
        history = app_state.get_session(conv_id)
    else:
        history = history or []

    # Document scope for this turn. One LLM turn-router (Cấp độ 3) decides both
    # the intent and which files the message is about, by content (Document Cards),
    # not by matching file names. Precedence:
    #   explicit path(s) in the request -> exactly those
    #   sidebar source selection (L1)   -> the checked files
    #   router target_doc_ids           -> those documents
    #   otherwise                       -> None: search the whole folder
    requested = list(dict.fromkeys([*(request.paths or []), *( [request.path] if request.path else [] )]))
    sources = cap_sources(scope) if (request.use_session or request.use_selection) else []
    source_paths = {s["path"] for s in sources if s.get("path")}
    # Exact scope of the previous turn, minus files that no longer exist.
    last_used = [
        path for path in app_state.get_last_used_docs(conv_id) if path in source_paths
    ]

    # Explicit source selection (UI toggles). `use_selection` is an explicit
    # signal, so "everything unchecked" is a real state: switch to No-Context /
    # Notes Only instead of silently searching the excluded files.
    selected: list[str] | None = None
    no_sources = False
    if request.use_selection:
        excluded = set(app_state.get_excluded_docs())
        if excluded:
            selected = [s["path"] for s in sources if s.get("path") and s["path"] not in excluded]
            # Every source switched off -> No-Context / Notes Only mode. Explicit
            # request paths still win, so only enter it when none were given.
            no_sources = not selected and not requested

    if no_sources:
        # No document cards and no router: the no-context prompt handles chat,
        # workspace metadata, notes and general knowledge in one turn.
        routed = {"intent": "no_context", "scope_mode": "", "doc_paths": [], "queries": []}
    elif request.use_session:
        # The previous scope is passed explicitly; the model no longer has to
        # rediscover it from the last four messages.
        routed = cap_route(scope, request.query, history, current_docs=last_used)
    else:
        routed = {"intent": "question", "scope_mode": "", "doc_paths": [], "queries": []}
    intent = routed.get("intent", "question")
    scope_mode = routed.get("scope_mode", "")
    routed_docs = list(routed.get("doc_paths") or [])
    routed_queries = list(routed.get("queries") or [])

    working = app_state.get_active_docs(conv_id)   # documents named so far (UI chips)

    if requested:
        target = requested
        scope_mode = "EXPLICIT_DOCS"
    elif no_sources:
        target = None
        scope_mode = ""
    elif selected is not None:
        target = selected
        scope_mode = "EXPLICIT_DOCS"
    else:
        target = cap_resolve_scope(intent, scope_mode, routed_docs, last_used)
        if scope_mode == "INHERIT_PREVIOUS" and not target:
            scope_mode = "GLOBAL_FOLDER"  # nothing to inherit on the first turn
        elif not scope_mode:
            # Normalise a missing/invalid router mode to the effective one.
            if target and intent == "followup" and target == last_used:
                scope_mode = "INHERIT_PREVIOUS"
            elif target:
                scope_mode = "EXPLICIT_DOCS"
            else:
                scope_mode = "GLOBAL_FOLDER"

    inherited = (target or []) if (
        scope_mode == "INHERIT_PREVIOUS"
        or (intent == "followup" and not routed_docs and target and target == last_used)
    ) else []
    mentioned = requested or routed_docs or inherited
    if mentioned:
        app_state.set_active_docs([*working, *mentioned], conv_id)

    if no_sources:
        # No-Context / Notes Only: answer from history + open notes + file names +
        # general knowledge, and never touch the documents the user switched off.
        result = cap_answer_without_context(
            scope,
            request.query,
            chat_history=history,
            notes=request.notes,
            files=[s.get("path") for s in sources if s.get("path")],
        )
    else:
        result = cap_ask(
            scope,
            request.query,
            chat_history=history,
            allow_external=request.allow_external,
            doc_paths=target or None,
            top_k=request.top_k,
            intent=intent,
            queries=routed_queries or None,
        )

    # Remember the exact scope used this turn; the next follow-up inherits it.
    app_state.set_last_used_docs(target or [], conv_id)

    # Learn the focus from what actually answered, so the next follow-up sticks.
    if intent != "chat":
        answered = list(dict.fromkeys(
            s.get("path") for s in result.get("sources", []) if s.get("path")
        ))[:3]
        if answered:
            app_state.set_primary_docs(answered, conv_id)
        elif target:
            app_state.set_primary_docs(target, conv_id)

    if request.save:
        # One atomic write for the whole turn (question + answer).
        app_state.append_turns(
            conv_id,
            [("user", request.query), ("assistant", result.get("answer", ""))],
        )
    return {
        **result,
        "intent": result.get("intent", intent),
        "scope_mode": scope_mode,
        "conversation_id": conv_id,
        "working_set": app_state.get_active_docs(conv_id),
        "primary_docs": app_state.get_primary_docs(conv_id),
        "routed_queries": routed_queries,
    }


@app.get("/api/session")
def session(conversation_id: str = app_state.DEFAULT_CONVERSATION) -> list[dict]:
    return app_state.get_session(conversation_id)


@app.post("/api/session/clear")
def session_clear(request: ClearRequest | None = None) -> dict:
    conv_id = (request.conversation_id if request else None) or app_state.DEFAULT_CONVERSATION
    app_state.clear_session(conv_id)
    return {"session": [], "conversation_id": conv_id}


@app.post("/api/conversation/new")
def conversation_new() -> dict:
    """Create an empty conversation and return its id."""
    return {"conversation_id": app_state.new_conversation()}


@app.get("/api/conversations")
def conversations() -> list[dict]:
    """Recent conversations (id, message count, last update)."""
    return app_state.list_conversations()


# --------------------------------------------------------------------------- #
# Working set: the documents (papers) the conversation is about
# --------------------------------------------------------------------------- #
@app.get("/api/working-set")
def get_working_set(conversation_id: str = app_state.DEFAULT_CONVERSATION) -> dict:
    return {
        "conversation_id": conversation_id,
        "active_docs": app_state.get_active_docs(conversation_id),
        "primary_docs": app_state.get_primary_docs(conversation_id),
    }


@app.post("/api/working-set")
def set_working_set(request: WorkingSetRequest) -> dict:
    """Replace the working set (used by the header to unpin a document)."""
    conv_id = request.conversation_id or app_state.DEFAULT_CONVERSATION
    active = list(dict.fromkeys(p for p in request.paths if p))
    app_state.set_active_docs(active, conv_id)
    # Drop any pinned document from the current focus if it was just unpinned.
    primary = [p for p in app_state.get_primary_docs(conv_id) if p in active]
    app_state.set_primary_docs(primary, conv_id)
    return {"conversation_id": conv_id, "active_docs": active, "primary_docs": primary}


# --------------------------------------------------------------------------- #
# Source selection: the documents the user unchecked in the sidebar list
# --------------------------------------------------------------------------- #
@app.get("/api/selection")
def get_selection() -> dict:
    return {"excluded_docs": app_state.get_excluded_docs()}


@app.post("/api/selection")
def set_selection(request: SelectionRequest) -> dict:
    return {"excluded_docs": app_state.set_excluded_docs(request.paths)}


# --------------------------------------------------------------------------- #
# Insights (background-capable)
# --------------------------------------------------------------------------- #
@app.post("/api/insight")
def create_insight(request: InsightRequest) -> dict:
    scope = _resolve_scope(request.scope)
    if not scope:
        raise HTTPException(status_code=400, detail="no scope selected")
    if request.background:
        job_id = jobs.submit("insight", summarize_source, scope, request.path, request.force)
        return {"job_id": job_id, "status": "queued"}
    return summarize_source(scope, request.path, force=request.force)


@app.get("/api/insights")
def insights(scope: Optional[str] = None) -> list[dict]:
    scope = _resolve_scope(scope)
    return list_insights(scope) if scope else []


@app.get("/api/insight")
def insight(scope: Optional[str] = None, path: str = "") -> dict:
    scope = _resolve_scope(scope)
    if not scope or not path:
        raise HTTPException(status_code=400, detail="scope and path are required")
    found = get_insight(scope, path)
    if found is None:
        raise HTTPException(status_code=404, detail="no cached insight")
    return found


# --------------------------------------------------------------------------- #
# Document Registry: a content-based "card" per file (Cấp độ 2)
# --------------------------------------------------------------------------- #
@app.get("/api/registry")
def registry(scope: Optional[str] = None) -> list[dict]:
    scope = _resolve_scope(scope)
    return cap_list_cards(scope) if scope else []


@app.get("/api/card")
def card(scope: Optional[str] = None, path: str = "") -> dict:
    scope = _resolve_scope(scope)
    if not scope or not path:
        raise HTTPException(status_code=400, detail="scope and path are required")
    found = cap_get_card(scope, path)
    if found is None:
        raise HTTPException(status_code=404, detail="no card for this file")
    return found


@app.post("/api/registry")
def build_registry(request: RegistryRequest) -> dict:
    """Build cards for every source that lacks a fresh one (background job)."""
    scope = _resolve_scope(request.scope)
    if not scope:
        raise HTTPException(status_code=400, detail="no scope selected")
    job_id = jobs.submit("registry", cap_build_cards, scope, request.force)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job


@app.get("/api/jobs")
def job_list() -> list[dict]:
    return jobs.list_jobs()


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # Browsers request this automatically; answer 204 so it stops logging 404.
    return Response(status_code=204)
