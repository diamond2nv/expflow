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


def _get_clearml_task():
    """Lazy import of clearml Task class.

    Returns the clearml Task class for clearml-object operations.
    The import is deferred to call time so that callers can mock it
    without triggering numpy re-import (a known issue in test isolation).
    """
    from clearml import Task  # noqa: F401

    return Task


def _clearml_get_task(task_id: str) -> Any:
    """Load a clearml Task by ID via lazy import.

    Wrapped as a separate function so tests can patch just this one call
    instead of the entire clearml module (avoiding numpy re-import crash).
    """
    clearml_task = _get_clearml_task()
    return clearml_task.get_task(task_id=task_id)


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


# ── Session ID resolution ──


def _resolve_session_id(
    session_id: str | None = None,
    parent_task_id: str | None = None,
) -> str:
    """Resolve a Langfuse session_id using a three-tier fallback strategy.

    Tier 1 — explicit: caller provided a session_id string.
    Tier 2 — inheritance: parent clearml Task has ``expflow:langfuse_session_id``
             in its metadata (to keep child experiments under the same session).
    Tier 3 — auto-generate: expflow produces a snowflake ID prefixed ``exp:snow_``.

    Args:
        session_id: Optional caller-provided session_id (Tier 1).
        parent_task_id: Optional clearml parent Task ID to check for inherited
            session_id (Tier 2).

    Returns:
        A non-empty session_id string.
    """
    # Tier 1: explicit
    if session_id and session_id.strip():
        return session_id.strip()

    # Tier 2: inherit from parent clearml Task metadata
    if parent_task_id:
        try:
            task = _clearml_get_task(parent_task_id)
            meta = task.get_metadata() or {}
            inherited = meta.get("expflow:langfuse_session_id", "")
            if inherited and isinstance(inherited, str) and inherited.strip():
                return inherited.strip()
        except Exception:
            pass  # Non-critical — fall through to Tier 3

    # Tier 3: auto-generate snowflake ID
    from expflow_pde.snowflake import snowflake_session_id

    return snowflake_session_id()


# ── clearml → Langfuse trace sync ──


def trace_experiment(
    task_id: str,
    trace_name: str | None = None,
    session_id: str | None = None,
    user_id: str = "expflow",
    parent_trace_id: str | None = None,
    parent_task_id: str | None = None,
) -> dict[str, Any]:
    """Sync a clearml Task's metrics and hyperparams to Langfuse as a trace.

    Reads the clearml Task identified by task_id, extracts its final scalars
    and hyperparameters, and writes them to Langfuse. This bridges the gap
    between the execution plane (clearml) and the observability plane (Langfuse),
    allowing you to see experiment results alongside Hermes Agent LLM traces.

    Session ID resolution follows a three-tier fallback:
      1. Explicit ``session_id`` parameter (highest priority).
      2. Inherit ``expflow:langfuse_session_id`` from ``parent_task_id``'s clearml
         metadata — keeps child experiments under the same Langfuse session.
      3. Auto-generate a snowflake ID (``exp:snow_<id>``) — every experiment
         always gets a session, even standalone runs.

    If parent_trace_id is provided, the created trace links back to the
    Hermes Agent trace that triggered this experiment (bidirectional linking).

    Args:
        task_id: clearml Task ID to sync.
        trace_name: Name for the Langfuse trace (default: "clearml:<task_id>").
        session_id: Optional Langfuse session ID (Tier 1 — overrides fallback).
        user_id: Langfuse user ID (default: "expflow").
        parent_trace_id: Optional parent Langfuse trace ID (from Hermes decision).
        parent_task_id: Optional clearml parent Task ID to check for inherited
            session_id (Tier 2).

    Returns:
        Dict with langfuse_trace_id, task_id, status.

    Raises:
        ValueError: If clearml Task not found or Langfuse write fails.
    """
    client = _get_client()

    # 1. Load clearml task
    try:
        task = _clearml_get_task(task_id)
    except Exception as e:
        raise ValueError(f"clearml Task not found: {task_id}") from e

    # 2. Extract metadata
    task_name = getattr(task, "name", task_id)
    project = getattr(task, "project", "")
    tags = list(getattr(task, "get_tags", lambda: [])() or [])

    # 3. Extract hyperparameters
    try:
        params = task.get_parameters()
        if isinstance(params, dict):
            params = {k: str(v) for k, v in params.items() if not k.startswith("_")}
        else:
            params = {}
    except Exception:
        params = {}

    # 4. Extract final scalars
    try:
        scalars = task.get_last_scalar_metrics()
        metrics_out: dict[str, float] = {}
        for group, metrics in scalars.items():
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "last" in metric_data:
                    try:
                        metrics_out[f"{group}/{metric_name}"] = float(metric_data["last"])
                    except (ValueError, TypeError):
                        pass
    except Exception:
        metrics_out = {}

    # 5. Build status summary
    status = getattr(task, "status", "unknown")

    # 6. Build metadata — include parent trace link
    metadata: dict[str, Any] = {
        "source": "expflow",
        "task_id": task_id,
        "project": project,
        "task_name": task_name,
    }
    if parent_trace_id:
        metadata["parent_trace_id"] = parent_trace_id

    # 7. Resolve session_id (three-tier fallback) and write to Langfuse
    resolved_session = _resolve_session_id(
        session_id=session_id,
        parent_task_id=parent_task_id,
    )
    trace_name_actual = trace_name or f"clearml:{task_id[:12]}"
    trace = client.trace(
        name=trace_name_actual,
        input={
            "task_id": task_id,
            "task_name": task_name,
            "project": project,
            "tags": tags,
            "hyperparams": params,
        },
        output={
            "status": status,
            "metrics": metrics_out,
        },
        session_id=resolved_session,
        user_id=user_id,
        metadata=metadata,
    )

    trace_id = getattr(trace, "id", str(trace))

    # 8. Write Langfuse IDs back to clearml Task metadata (reverse link)
    try:
        task.set_metadata("expflow:langfuse_trace_id", trace_id)
        task.set_metadata("expflow:langfuse_session_id", resolved_session)
        if parent_trace_id:
            task.set_metadata("expflow:langfuse_parent_trace_id", parent_trace_id)
    except Exception:
        pass  # Non-critical — metadata write failure shouldn't fail the sync

    return {
        "langfuse_trace_id": trace_id,
        "task_id": task_id,
        "task_name": task_name,
        "project": project,
        "status": status,
        "metrics_count": len(metrics_out),
        "params_count": len(params),
        "session_id": resolved_session,
        "parent_trace_id": parent_trace_id or "",
    }


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
