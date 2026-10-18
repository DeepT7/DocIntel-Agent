"""One process, two windows.

- Main window: full app (chat, sources, tasks).
- Quick window: small always-on-top popup toggled by a global hotkey.

Both are views over the same ``Api`` object, so scope/session/index are shared.
Run:  python -m desktop.app
"""

import threading
from pathlib import Path

import webview

from desktop.api import Api

UI_DIR = Path(__file__).resolve().parent / "ui"
HOTKEY = "<ctrl>+<alt>+space"


def _toggle_quick(quick: webview.Window, api: Api) -> None:
    if api.quick_visible:
        api.hide_quick()
    else:
        quick.show()
        api.quick_visible = True
        try:
            quick.evaluate_js("window.onShow && window.onShow()")
        except Exception:
            pass


def _start_hotkey(quick: webview.Window, api: Api) -> None:
    try:
        from pynput import keyboard
    except Exception as error:  # pragma: no cover - platform dependent
        print(f"hotkey disabled (pynput unavailable): {error}")
        return

    listener = keyboard.GlobalHotKeys({HOTKEY: lambda: _toggle_quick(quick, api)})
    listener.daemon = True
    listener.start()
    print(f"hotkey ready: {HOTKEY} toggles the Quick window")


def main() -> None:
    api = Api()

    quick = webview.create_window(
        "Trợ lý viết",
        url=str(UI_DIR / "quick.html"),
        width=470,
        height=580,
        on_top=True,
        frameless=True,
        resizable=False,
        hidden=True,
        js_api=api,
    )
    main_window = webview.create_window(
        "DocIntel — Trợ lý tài liệu",
        url=str(UI_DIR / "main.html"),
        width=1200,
        height=800,
        js_api=api,
        hidden=False,
    )

    api.quick_window = quick
    api.main_window = main_window

    def on_start():
        threading.Thread(target=_start_hotkey, args=(quick, api), daemon=True).start()

    webview.start(on_start)


if __name__ == "__main__":
    main()