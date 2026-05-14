#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow langfuse CLI sub-commands — lazy imports of langfuse at call time."""

from typing import Optional

import typer

langfuse_app = typer.Typer(
    name="langfuse",
    help="Interact with Langfuse observability platform",
    no_args_is_help=True,
)


def get_langfuse_app() -> typer.Typer:
    return langfuse_app


# ── Trace commands ──


@langfuse_app.command("traces")
def list_traces_cmd(
    limit: int = typer.Option(100, "--limit", "-l", help="Max traces"),
    user_id: Optional[str] = typer.Option(None, "--user-id", "-u", help="Filter by user ID"),
    tags: Optional[str] = typer.Option(
        None, "--tags", "-t", help="Filter by tags (comma-separated)"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Filter by session ID"
    ),
) -> None:
    """List Langfuse traces."""
    from expflow.langfuse import list_traces

    tag_list = tags.split(",") if tags else None
    traces = list_traces(limit=limit, user_id=user_id, tags=tag_list, session_id=session_id)

    if not traces:
        print("No traces found.")
        return

    print(f"{'ID':<24} {'NAME':<30} {'USER':<16} {'COST':<10}")
    print("-" * 80)
    for t in traces:
        cost = f"${t['cost']:.4f}" if t.get("cost") else "-"
        uid = t.get("user_id", "-") or "-"
        print(f"{t['id']:<24} {t['name']:<30} {uid:<16} {cost:<10}")


@langfuse_app.command("trace")
def get_trace_cmd(
    trace_id: str = typer.Argument(..., help="Trace ID"),
) -> None:
    """Show details for a trace."""
    from expflow.langfuse import get_trace

    t = get_trace(trace_id)
    print(f"ID:        {t['id']}")
    print(f"Name:      {t['name']}")
    print(f"User:      {t.get('user_id', '-')}")
    print(f"Tags:      {', '.join(t.get('tags', [])) or '(none)'}")
    print(f"Session:   {t.get('session_id', '-')}")
    print(f"Cost:      ${t.get('cost', 0):.4f}")
    print(f"Latency:   {t.get('latency', 0):.2f}s")


@langfuse_app.command("trace-cost")
def trace_cost_cmd(
    trace_id: str = typer.Argument(..., help="Trace ID"),
) -> None:
    """Show cost breakdown for a trace."""
    from expflow.langfuse import get_trace_cost

    c = get_trace_cost(trace_id)
    print(f"Trace:     {c['trace_id']}")
    print(f"Total:     ${c['total_cost']:.4f}")
    if c.get("usage"):
        print(f"Usage:     {c['usage']}")


# ── Session commands ──


@langfuse_app.command("sessions")
def list_sessions_cmd(
    limit: int = typer.Option(100, "--limit", "-l", help="Max sessions"),
) -> None:
    """List Langfuse sessions."""
    from expflow.langfuse import list_sessions

    sessions = list_sessions(limit=limit)

    if not sessions:
        print("No sessions found.")
        return

    print(f"{'ID':<24} {'USER':<16} {'CREATED':<30}")
    print("-" * 70)
    for s in sessions:
        uid = s.get("user_id", "-") or "-"
        print(f"{s['id']:<24} {uid:<16} {s.get('created_at', '-'):<30}")


@langfuse_app.command("session")
def get_session_cmd(
    session_id: str = typer.Argument(..., help="Session ID"),
) -> None:
    """Show details for a session."""
    from expflow.langfuse import get_session

    s = get_session(session_id)
    print(f"ID:        {s['id']}")
    print(f"User:      {s.get('user_id', '-')}")
    print(f"Created:   {s.get('created_at', '-')}")


# ── Metrics ──


@langfuse_app.command("metrics")
def metrics_cmd() -> None:
    """Show aggregated usage/cost metrics."""
    from expflow.langfuse import get_metrics

    m = get_metrics()
    for k, v in m.items():
        print(f"{k}: {v}")
