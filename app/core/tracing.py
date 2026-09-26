"""Optional Langfuse tracing helpers.

Tracing is opt-in: it is enabled only when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set, and not disabled via the official SDK switch
LANGFUSE_TRACING_ENABLED=false. When it is disabled every helper below is a
no-op, so the pipeline behaves exactly as if Langfuse were not installed.
"""

import contextlib
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def tracing_enabled() -> bool:
    # Official Langfuse SDK switch: only the literal "false" disables tracing.
    if (os.getenv("LANGFUSE_TRACING_ENABLED") or "true").lower() == "false":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


TRACING_ENABLED = tracing_enabled()


def observe(name: str, **kwargs):
    """Decorator that wraps a function in a Langfuse span (no-op if disabled).

    Example:
        @observe("retrieve_node")
        def retrieve_node(state): ...
    """
    if not TRACING_ENABLED:
        return lambda fn: fn
    try:
        from langfuse import observe as _lf_observe

        return _lf_observe(name=name, **kwargs)
    except Exception:
        return lambda fn: fn


def update_current_span(*, input: Any = None, output: Any = None, **metadata: Any) -> None:
    """Attach input/output/metadata to the span currently open on this call stack."""
    if not TRACING_ENABLED:
        return
    try:
        from langfuse import get_client

        fields: dict[str, Any] = {}
        if input is not None:
            fields["input"] = input
        if output is not None:
            fields["output"] = output
        clean_metadata = {k: v for k, v in metadata.items() if v is not None}
        if clean_metadata:
            fields["metadata"] = clean_metadata
        if fields:
            get_client().update_current_span(**fields)
    except Exception:
        pass


def set_trace_io(*, input: Any = None, output: Any = None) -> None:
    """Set the top-level input/output shown for the whole trace."""
    if not TRACING_ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().set_current_trace_io(input=input, output=output)
    except Exception:
        pass


def trace_context(**attributes: Any):
    """Group spans into one trace with session/user/tags (no-op if disabled).

    Usage:
        with trace_context(session_id="run-1", tags=["rag"]):
            graph.invoke(...)
    """
    if not TRACING_ENABLED:
        return contextlib.nullcontext()
    try:
        from langfuse import propagate_attributes

        return propagate_attributes(**attributes)
    except Exception:
        return contextlib.nullcontext()


def flush_traces() -> None:
    """Block until all buffered traces are sent (call before the process exits)."""
    if not TRACING_ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass