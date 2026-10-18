"""Shared app state for both windows (Quick and Main).

Two levels, so chats cannot contaminate each other:

- app / folder level: ``scope``, ``recent``, ``excluded_docs``
- conversation level: ``conversations[conv_id]`` with ``messages``,
  ``active_docs``, ``primary_docs``, ``last_used_docs`` and ``updated_at``

Only the app level is shared. Every mutable conversational value lives inside its
own conversation object; ``new_conversation()`` creates a fresh, empty one.

Graph isolation: nothing under ``app/reasoning`` reads this module. The server
injects the conversation's data into ``run_rag`` as arguments (see ``main.ask``),
so a graph turn can only see the conversation that started it.

Concurrency: FastAPI runs sync endpoints in a thread pool, so two turns can be
appended at once. Every read-modify-write goes through ``_mutate`` (a process-wide
RLock) and the file is replaced atomically, so concurrent requests cannot lose
each other's turns or leave a half-written JSON file. This assumes a single
uvicorn process; with multiple workers, use an OS-level file lock instead.
"""

import json
import os
import threading
import time
import uuid
from pathlib import Path

from app.core.scope import normalize_root

STATE_PATH = Path("data/app_state.json")
MAX_RECENT = 8
MAX_SESSION_MESSAGES = 20
MAX_CONVERSATIONS = 20
DEFAULT_CONVERSATION = "default"

# Serializes load -> modify -> save for every writer in this process.
_LOCK = threading.RLock()


def _new_conversation() -> dict:
    return {
        "messages": [],
        "active_docs": [],
        "primary_docs": [],
        "last_used_docs": [],
        "updated_at": time.time(),
    }


def _defaults() -> dict:
    """Fresh defaults (new lists each call — never share mutable defaults)."""
    return {
        "scope": None,
        "recent": [],
        "excluded_docs": [],
        "conversations": {DEFAULT_CONVERSATION: _new_conversation()},
    }


def _normalize(data: dict) -> dict:
    """Migrate the old flat schema and fill missing fields."""
    state = _defaults()
    for key in ("scope", "recent", "excluded_docs"):
        if key in data:
            state[key] = data[key]

    conversations = data.get("conversations")
    if not isinstance(conversations, dict):
        conversations = {}

    # Old flat state (session/active_docs/...) becomes the "default" chat.
    legacy = {
        "messages": data.get("session", []),
        "active_docs": data.get("active_docs", []),
        "primary_docs": data.get("primary_docs", []),
        "last_used_docs": data.get("last_used_docs", []),
    }
    if DEFAULT_CONVERSATION not in conversations and any(legacy.values()):
        conversations[DEFAULT_CONVERSATION] = {
            **_new_conversation(),
            **{key: value for key, value in legacy.items() if value},
        }

    for conv_id, conv in list(conversations.items()):
        base = _new_conversation()
        if isinstance(conv, dict):
            base.update({key: value for key, value in conv.items() if key in base})
        conversations[conv_id] = base

    conversations.setdefault(DEFAULT_CONVERSATION, _new_conversation())
    state["conversations"] = conversations
    return state


def load() -> dict:
    if not STATE_PATH.exists():
        return _defaults()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _defaults()
    return _normalize(data)


def save(state: dict) -> None:
    """Write the whole state atomically (readers never see a partial file)."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _mutate(fn):
    """Run ``fn(state)`` under the lock, persist, and return its result."""
    with _LOCK:
        state = load()
        result = fn(state)
        save(state)
        return result


def _evict(conversations: dict, keep: str) -> None:
    """Keep ``keep`` + the default chat + the most recently updated ones."""
    if len(conversations) <= MAX_CONVERSATIONS:
        return
    protected = {keep, DEFAULT_CONVERSATION}
    ordered = sorted(
        conversations.items(),
        key=lambda item: item[1].get("updated_at", 0),
        reverse=True,
    )
    for conv_id, _ in ordered:
        if len(conversations) <= MAX_CONVERSATIONS:
            break
        if conv_id not in protected:
            conversations.pop(conv_id, None)


def _conversation(state: dict, conversation_id: str) -> dict:
    """Get/create one conversation (mutates ``state``; call inside ``_mutate``)."""
    conv_id = conversation_id or DEFAULT_CONVERSATION
    conversations = state.setdefault("conversations", {})
    conv = conversations.get(conv_id)
    if not isinstance(conv, dict):
        conv = _new_conversation()
    else:
        base = _new_conversation()
        base.update({key: value for key, value in conv.items() if key in base})
        conv = base
    conversations[conv_id] = conv
    conv["updated_at"] = time.time()
    _evict(conversations, keep=conv_id)
    return conv


# --------------------------------------------------------------------------- #
# App / folder level (shared by every conversation)
# --------------------------------------------------------------------------- #
def set_scope(scope: str, conversation_id: str = DEFAULT_CONVERSATION) -> dict:
    def apply(state: dict) -> dict:
        root = normalize_root(scope)
        state["scope"] = root
        state["excluded_docs"] = []  # source selection is per-folder
        recent = [r for r in state.get("recent", []) if r != root]
        recent.insert(0, root)
        state["recent"] = recent[:MAX_RECENT]

        # Paths from the old folder are invalid in every conversation.
        for conv in state.get("conversations", {}).values():
            conv["active_docs"] = []
            conv["primary_docs"] = []
            conv["last_used_docs"] = []

        # The active chat starts fresh in the new folder (see IMP-003).
        conv = _conversation(state, conversation_id)
        conv["messages"] = []
        return state

    return _mutate(apply)


def get_scope() -> str | None:
    return load().get("scope")


def get_recent() -> list[str]:
    return load().get("recent", [])


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
def new_conversation() -> str:
    """Create and return a fresh, empty conversation id."""
    conv_id = uuid.uuid4().hex

    def apply(state: dict) -> str:
        conversations = state.setdefault("conversations", {})
        conversations[conv_id] = _new_conversation()
        _evict(conversations, keep=conv_id)
        return conv_id

    return _mutate(apply)


def list_conversations() -> list[dict]:
    state = load()
    items = [
        {
            "conversation_id": conv_id,
            "messages": len(conv.get("messages", [])),
            "updated_at": conv.get("updated_at"),
        }
        for conv_id, conv in state.get("conversations", {}).items()
    ]
    return sorted(items, key=lambda item: item.get("updated_at") or 0, reverse=True)


def _read_conversation(conversation_id: str) -> dict:
    state = load()
    conv = state.get("conversations", {}).get(conversation_id or DEFAULT_CONVERSATION)
    return conv if isinstance(conv, dict) else _new_conversation()


def get_session(conversation_id: str = DEFAULT_CONVERSATION) -> list[dict]:
    return _read_conversation(conversation_id).get("messages", [])


def append_turns(
    conversation_id: str, turns: list[tuple[str, str]]
) -> list[dict]:
    """Append several messages (e.g. user + assistant) in ONE write."""

    def apply(state: dict) -> list[dict]:
        conv = _conversation(state, conversation_id)
        messages = conv.get("messages", [])
        for role, content in turns:
            messages.append({"role": role, "content": content})
        conv["messages"] = messages[-MAX_SESSION_MESSAGES:]
        return conv["messages"]

    return _mutate(apply)


def append_turn(
    role: str, content: str, conversation_id: str = DEFAULT_CONVERSATION
) -> list[dict]:
    return append_turns(conversation_id, [(role, content)])


def clear_session(conversation_id: str = DEFAULT_CONVERSATION) -> None:
    def apply(state: dict) -> None:
        conv = _conversation(state, conversation_id)
        conv["messages"] = []
        conv["active_docs"] = []
        conv["primary_docs"] = []
        conv["last_used_docs"] = []

    _mutate(apply)


# --------------------------------------------------------------------------- #
# Conversation documents: working set + focus + inherited scope
# --------------------------------------------------------------------------- #
def _clean(paths: list[str]) -> list[str]:
    return [p for p in dict.fromkeys(paths or []) if p]


def get_active_docs(conversation_id: str = DEFAULT_CONVERSATION) -> list[str]:
    return _read_conversation(conversation_id).get("active_docs", [])


def set_active_docs(
    paths: list[str], conversation_id: str = DEFAULT_CONVERSATION
) -> None:
    def apply(state: dict) -> None:
        _conversation(state, conversation_id)["active_docs"] = _clean(paths)

    _mutate(apply)


def get_primary_docs(conversation_id: str = DEFAULT_CONVERSATION) -> list[str]:
    return _read_conversation(conversation_id).get("primary_docs", [])


def set_primary_docs(
    paths: list[str], conversation_id: str = DEFAULT_CONVERSATION
) -> None:
    def apply(state: dict) -> None:
        _conversation(state, conversation_id)["primary_docs"] = _clean(paths)

    _mutate(apply)


def get_last_used_docs(conversation_id: str = DEFAULT_CONVERSATION) -> list[str]:
    return _read_conversation(conversation_id).get("last_used_docs", [])


def set_last_used_docs(
    paths: list[str], conversation_id: str = DEFAULT_CONVERSATION
) -> None:
    def apply(state: dict) -> None:
        _conversation(state, conversation_id)["last_used_docs"] = _clean(paths)

    _mutate(apply)


def snapshot(conversation_id: str = DEFAULT_CONVERSATION) -> dict:
    """Flattened view for the UI (keeps the old /api/state response shape)."""
    state = load()
    conv = state.get("conversations", {}).get(conversation_id or DEFAULT_CONVERSATION)
    if not isinstance(conv, dict):
        conv = _new_conversation()
    return {
        "scope": state.get("scope"),
        "recent": state.get("recent", []),
        "excluded_docs": state.get("excluded_docs", []),
        "conversation_id": conversation_id or DEFAULT_CONVERSATION,
        "session": conv.get("messages", []),
        "active_docs": conv.get("active_docs", []),
        "primary_docs": conv.get("primary_docs", []),
        "last_used_docs": conv.get("last_used_docs", []),
    }


# --------------------------------------------------------------------------- #
# Explicit source selection (folder level)
# --------------------------------------------------------------------------- #
def get_excluded_docs() -> list[str]:
    return load().get("excluded_docs", [])


def set_excluded_docs(paths: list[str]) -> list[str]:
    def apply(state: dict) -> list[str]:
        state["excluded_docs"] = _clean(paths)
        return state["excluded_docs"]

    return _mutate(apply)
