"""Shared capabilities used by every surface (Quick window and Main window).

Surfaces must not import Qdrant or retrieval directly: they call these functions,
so the logic lives in one place regardless of the UI.
"""