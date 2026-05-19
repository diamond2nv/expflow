#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow langfuse integration — trace query, cost analysis, session search.

All functions return JSON-serializable dicts.
Lazy imports langfuse SDK at call time.
"""

from typing import Any


def _get_client():
    """Lazy init of langfuse client."""
    from langfuse import Langfuse  # noqa: F401

    return Langfuse()


# ── Trace operations ──


def list_traces(
    limit: int = 100,
    user_id: str | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """List traces with optional filters.

    Args:
        limit: Max traces to return.
        user_id: Filter by user ID.
        tags: Filter by tags.
        session_id: Filter by session ID.

    Returns:
        List of trace dicts.
    """
    client = _get_client()
    kwargs: dict[str, Any] = {"limit": limit}
    if user_id is not None:
        kwargs["user_id"] = user_id
    if tags is not None:
        kwargs["tags"] = tags
    if session_id is not None:
        kwargs["session_id"] = session_id

    traces = client.api.trace.list(**kwargs)
    return [_serialize_trace(t) for t in traces]


def get_trace(trace_id: str) -> dict[str, Any]:
    """Get a single trace by ID.

    Args:
        trace_id: The langfuse trace ID.

    Returns:
        Trace dict.
    """
    client = _get_client()
    trace = client.api.trace.get(trace_id)
    return _serialize_trace(trace)


def get_trace_cost(trace_id: str) -> dict[str, Any]:
    """Get cost breakdown for a trace.

    Args:
        trace_id: The langfuse trace ID.

    Returns:
        Dict with trace_id, total_cost, usage.
    """
    client = _get_client()
    trace = client.api.trace.get(trace_id)

    cost = getattr(trace, "total_cost", 0) or 0
    usage = getattr(trace, "usage", {})

    return {
        "trace_id": trace.id,
        "total_cost": cost,
        "usage": usage if isinstance(usage, dict) else {},
    }


# ── Session operations ──


def list_sessions(limit: int = 100) -> list[dict[str, Any]]:
    """List sessions.

    Args:
        limit: Max sessions to return.

    Returns:
        List of session dicts.
    """
    client = _get_client()
    sessions = client.api.session.list(limit=limit)
    return [_serialize_session(s) for s in sessions]


def get_session(session_id: str) -> dict[str, Any]:
    """Get a single session by ID.

    Args:
        session_id: The langfuse session ID.

    Returns:
        Session dict.
    """
    client = _get_client()
    session = client.api.session.get(session_id)
    return _serialize_session(session)


# ── Metrics ──


def get_metrics(**filters: Any) -> dict[str, Any]:
    """Get aggregated usage/cost metrics.

    Args:
        **filters: Additional filter parameters.

    Returns:
        Metrics dict.
    """
    client = _get_client()
    result = client.api.metrics.get(query=filters)
    return dict(result) if isinstance(result, dict) else {"data": result}


# ── Serializers ──


def _serialize_trace(trace: Any) -> dict[str, Any]:
    return {
        "id": trace.id,
        "name": getattr(trace, "name", ""),
        "user_id": getattr(trace, "user_id", None),
        "tags": list(getattr(trace, "tags", []) or []),
        "session_id": getattr(trace, "session_id", None),
        "timestamp": str(getattr(trace, "timestamp", "")),
        "cost": getattr(trace, "cost", 0),
        "latency": getattr(trace, "latency", 0),
    }


def _serialize_session(session: Any) -> dict[str, Any]:
    return {
        "id": session.id,
        "user_id": getattr(session, "user_id", None),
        "created_at": str(getattr(session, "created_at", "")),
    }
