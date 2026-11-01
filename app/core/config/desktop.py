"""Desktop (pywebview) window and bridge settings."""

from .base import ConfigGroup


class DesktopSettings(ConfigGroup):
    # Clipboard truncation for the desktop bridge. Distinct from the server-side
    # suggest prompt truncation in ``CapabilitySettings``.
    max_clipboard: int = 4000
    hotkey: str = "<ctrl>+<alt>+space"

    quick_window_width: int = 470
    quick_window_height: int = 580
    main_window_width: int = 1200
    main_window_height: int = 800
