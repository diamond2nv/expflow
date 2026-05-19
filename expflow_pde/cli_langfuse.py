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
    from expflow_pde.langfuse import list_traces

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
    from expflow_pde.langfuse import get_trace

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
    from expflow_pde.langfuse import get_trace_cost

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
    from expflow_pde.langfuse import list_sessions

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
    from expflow_pde.langfuse import get_session

    s = get_session(session_id)
    print(f"ID:        {s['id']}")
    print(f"User:      {s.get('user_id', '-')}")
    print(f"Created:   {s.get('created_at', '-')}")


# ── Metrics ──


@langfuse_app.command("metrics")
def metrics_cmd() -> None:
    """Show aggregated usage/cost metrics."""
    from expflow_pde.langfuse import get_metrics

    m = get_metrics()
    for k, v in m.items():
        print(f"{k}: {v}")


# ── clearml → Langfuse sync ──


@langfuse_app.command("trace-experiment")
def trace_experiment_cmd(
    task_id: str = typer.Argument(..., help="clearml Task ID to sync to Langfuse"),
    trace_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Langfuse trace name (default: clearml:<task_id>)"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session", "-s", help="Langfuse session ID (Tier 1 — overrides fallback)"
    ),
    user_id: str = typer.Option("expflow", "--user", "-u", help="Langfuse user ID"),
    parent_trace: Optional[str] = typer.Option(
        None, "--parent-trace", help="Parent Langfuse trace ID (from Hermes Agent decision)"
    ),
    parent_task: Optional[str] = typer.Option(
        None, "--parent-task-id", help="clearml parent Task ID (Tier 2 — inherit its session_id)"
    ),
) -> None:
    """Sync a clearml experiment to Langfuse as a trace.

    Reads the clearml Task's final metrics and hyperparameters and writes
    them to Langfuse, bridging the execution and observability planes.

    Session ID resolution follows a three-tier fallback:
      1. --session (explicit) — highest priority.
      2. --parent-task-id — inherit expflow:langfuse_session_id from parent
         clearml Task's metadata.
      3. Auto-generate a snowflake ID (exp:snow_<id>) — every experiment
         always gets a session, even standalone runs.

    Use --parent-trace to link this experiment to a Hermes Agent decision trace.

    Examples:

        # Sync a specific experiment (auto-generates session_id)
        expflow langfuse trace-experiment a1b2c3d4e5f6

        # Group under a session, linking to Hermes decision
        expflow langfuse trace-experiment a1b2c3d4e5f6 \\
            --session pdebench:hpo_burgers_v2 \\
            --parent-trace lf_abc123

        # Inherit session from parent clearml Task
        expflow langfuse trace-experiment child_task_id \\
            --parent-task-id parent_task_id
    """
    from expflow_pde.langfuse import trace_experiment

    result = trace_experiment(
        task_id=task_id,
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        parent_trace_id=parent_trace,
        parent_task_id=parent_task,
    )

    print("Experiment synced to Langfuse:")
    print(f"  Langfuse trace: {result['langfuse_trace_id']}")
    print(f"  clearml task:   {result['task_id']}")
    print(f"  Task name:      {result['task_name']}")
    print(f"  Project:        {result['project']}")
    print(f"  Status:         {result['status']}")
    print(f"  Metrics:        {result['metrics_count']}")
    print(f"  Hyperparams:    {result['params_count']}")
    if result.get("session_id"):
        print(f"  Session:        {result['session_id']}")
    if result.get("parent_trace_id"):
        print(f"  Parent trace:   {result['parent_trace_id']}")
