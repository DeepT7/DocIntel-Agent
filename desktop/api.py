"""JS <-> Python bridge shared by both windows.

pywebview exposes this object to JavaScript as ``window.pywebview.api``. Both the
Quick window and the Main window call the *same* object, so scope, session and
index are shared by construction.

This module must not touch Qdrant directly; it only calls ``app.capabilities``.
"""

import pyperclip

from app import app_state
from app.capabilities.search import search as cap_search
from app.capabilities.sources import list_sources as cap_sources
from app.capabilities.suggest import suggest as cap_suggest
from app.capabilities.sync import sync as cap_sync
from app.core.config import DEFAULT_TOP_K, DESKTOP_MAX_CLIPBOARD

MAX_CLIPBOARD = DESKTOP_MAX_CLIPBOARD


class Api:
    def __init__(self) -> None:
        self.quick_window = None
        self.main_window = None
        self.quick_visible = False
        self._pending_for_main: dict | None = None

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        state = app_state.load()
        return {"scope": state.get("scope"), "recent": state.get("recent", [])}

    def set_scope(self, path: str) -> dict:
        if path:
            app_state.set_scope(path)
        return self.get_state()

    def pick_folder(self):
        import webview

        if self.quick_window is None:
            return None
        result = self.quick_window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        return self.set_scope(path)

    # ------------------------------------------------------------------ index
    def sync(self) -> dict:
        scope = app_state.get_scope()
        if not scope:
            return {"error": "no scope selected"}
        return cap_sync(scope)

    def sources(self) -> list[dict]:
        scope = app_state.get_scope()
        return cap_sources(scope) if scope else []

    # ------------------------------------------------------------------ query
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        scope = app_state.get_scope()
        if not scope:
            return []
        return cap_search(scope, query, top_k=top_k)

    def read_clipboard(self) -> str:
        try:
            return (pyperclip.paste() or "")[:MAX_CLIPBOARD]
        except Exception:
            return ""

    def suggest(self, query: str, clipboard: str = "") -> dict:
        scope = app_state.get_scope()
        if not scope:
            return {"error": "no scope selected", "suggestion": "", "sources": []}
        if not clipboard:
            clipboard = self.read_clipboard()
        return cap_suggest(scope, query, clipboard=clipboard)

    # ------------------------------------------------------- window controls
    def show_main(self, payload: dict | None = None) -> None:
        """Escalate the current request to the full window and focus it."""
        self._pending_for_main = payload or None
        if self.main_window is not None:
            self.main_window.show()
            self.main_window.restore()
        self.hide_quick()

    def get_pending(self):
        payload = self._pending_for_main
        self._pending_for_main = None
        return payload

    def hide_quick(self) -> None:
        if self.quick_window is not None:
            self.quick_window.hide()
        self.quick_visible = False