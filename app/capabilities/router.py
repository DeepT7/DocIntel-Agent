"""Capability: turn router — intent + document scope + sub-queries (Cấp độ 3).

Replaces the old split of ``classify_intent`` (chat/followup/question/workspace)
and the title-matching ``_detect_docs`` with ONE model call. The model reads the
message, the Document Cards and the documents still open from the previous turn,
and returns:

    {"intent": ..., "scope_mode": ..., "target_doc_ids": [...], "rewritten_queries": [...]}

``scope_mode`` makes the *reason* for a scope explicit, so an empty
``target_doc_ids`` is no longer ambiguous:

    EXPLICIT_DOCS    message names/describes documents -> use target_doc_ids
    INHERIT_PREVIOUS continues the thread, no new doc  -> inherit current docs
    GLOBAL_FOLDER    whole-folder question / new topic -> search everything

Documents are chosen by *content* (file name + title + summary), not by matching
file names, so "the paper that imitates proprietary models" finds the right file
even when its title never appears. ``resolve_scope`` is the deterministic
guardrail: a follow-up that forgot its documents inherits the previous turn
instead of silently searching the whole folder.
"""

import json
import re

from app.core.config import OPENROUTER_UTILITY_MODELS
from app.core.prompts import build_turn_router_prompt
from app.reasoning.answer import _generate, classify_intent, format_history

VALID_INTENTS = {"chat", "followup", "question", "workspace"}
VALID_SCOPE_MODES = {"EXPLICIT_DOCS", "INHERIT_PREVIOUS", "GLOBAL_FOLDER"}


def _cards_for(scope: str) -> list[dict]:
    """The router's view of the folder: id + file name + title + summary.

    Deliberately the same minimal metadata a skill exposes (id/name/description):
    enough to tell what a file is about, without bloating the router prompt.
    """
    from app.capabilities.registry import list_cards

    cards = list_cards(scope)
    if cards:
        return [
            {
                "id": index,
                "path": card.get("path"),
                "file_name": card.get("file_name") or card.get("title"),
                "title": card.get("title") or "",
                "summary": (card.get("summary") or "").strip(),
            }
            for index, card in enumerate(cards)
        ]

    # No registry yet: fall back to bare file names.
    from app.capabilities.sources import list_sources

    return [
        {
            "id": index,
            "path": item.get("path"),
            "file_name": item.get("title") or item.get("path"),
            "title": item.get("title") or "",
            "summary": "",
        }
        for index, item in enumerate(list_sources(scope))
    ]


def _parse(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def route(
    scope: str,
    query: str,
    history: list[dict] | None = None,
    current_docs: list[str] | None = None,
) -> dict:
    """Return ``{"intent", "scope_mode", "doc_paths", "queries"}`` for one message.

    ``current_docs`` is the exact scope used by the previous turn (paths). It is
    injected into the prompt so the model does not have to rediscover it from the
    last four messages.
    """
    cards = _cards_for(scope)
    if not cards:
        return {
            "intent": classify_intent(query, history),
            "scope_mode": "GLOBAL_FOLDER",
            "doc_paths": [],
            "queries": [],
        }

    open_paths = {path for path in (current_docs or []) if path}
    open_cards = [card for card in cards if card.get("path") in open_paths]
    by_id = {card["id"]: card for card in cards}

    try:
        raw = _generate(
            build_turn_router_prompt(
                query,
                format_history(history or [], max_messages=4),
                cards,
                open_cards,
            ),
            models=OPENROUTER_UTILITY_MODELS,
            max_tokens=600,
        )
    except Exception as error:
        print(f"route failed, searching everything: {error}")
        return {
            "intent": "question",
            "scope_mode": "GLOBAL_FOLDER",
            "doc_paths": [],
            "queries": [],
        }

    data = _parse(raw)
    intent = data.get("intent") if data.get("intent") in VALID_INTENTS else "question"
    scope_mode = str(data.get("scope_mode") or "").strip().upper()
    if scope_mode not in VALID_SCOPE_MODES:
        scope_mode = ""  # resolve_scope applies the deterministic fallback

    doc_paths: list[str] = []
    if intent in ("question", "followup"):
        for doc_id in data.get("target_doc_ids") or []:
            card = by_id.get(doc_id) if isinstance(doc_id, int) else None
            path = card.get("path") if card else None
            if path and path not in doc_paths:
                doc_paths.append(path)

    queries = []
    if scope_mode in ("EXPLICIT_DOCS", "INHERIT_PREVIOUS") and (doc_paths or open_cards):
        queries = [
            str(item).strip()
            for item in (data.get("rewritten_queries") or [])
            if str(item).strip()
        ][:3]

    return {
        "intent": intent,
        "scope_mode": scope_mode,
        "doc_paths": doc_paths,
        "queries": queries,
    }


def resolve_scope(
    intent: str,
    scope_mode: str,
    routed_docs: list[str],
    current_docs: list[str],
) -> list[str] | None:
    """Deterministic scope resolution — the guardrail around the LLM router.

    - EXPLICIT_DOCS: use the router's docs. If it forgot them on a follow-up,
      inherit the previous turn instead of searching the whole folder.
    - INHERIT_PREVIOUS: use the previous turn's docs (None when nothing is open).
    - GLOBAL_FOLDER: search everything (None).
    - Missing/invalid mode: trust explicit docs, else inherit on a follow-up.
    """
    if intent in ("chat", "workspace"):
        return None
    if scope_mode == "EXPLICIT_DOCS":
        if routed_docs:
            return routed_docs
        return current_docs if intent == "followup" and current_docs else None
    if scope_mode == "INHERIT_PREVIOUS":
        return current_docs or None
    if scope_mode == "GLOBAL_FOLDER":
        return None
    # Router omitted or returned an invalid scope_mode.
    if routed_docs:
        return routed_docs
    if intent == "followup":
        return current_docs or None
    return None
