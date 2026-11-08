"""Shared capabilities used by the HTTP API and UI surfaces.

Surfaces must not import Qdrant or retrieval directly: they call these functions,
so the logic lives in one place regardless of the UI.
"""