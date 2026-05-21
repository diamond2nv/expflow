#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow dispatch CLI — dispatch database management commands."""

from __future__ import annotations

from typing import Optional

import typer

dispatch_app = typer.Typer(
    name="dispatch",
    help="Manage the local experiment dispatch database (SQLite)",
    no_args_is_help=True,
)


@dispatch_app.command("status")
def status_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment ID (exp:snow_<id>)"),
) -> None:
    """Show detailed status for a single experiment."""
    from expflow_pde.dispatch_db import DispatchDB

    db = DispatchDB()
    exp = db.get_experiment(experiment_id)
    if exp is None:
        print(f"Experiment not found: {experiment_id}")
        raise typer.Exit(code=1)

    print(f"Experiment:  {exp['id']}")
    print(f"Status:      {exp['status']}")
    print(f"FSM State:   {exp.get('fsm_state', '?')}")
    print(f"Script:      {exp['script']}")
    print(f"Queue:       {exp.get('queue', '?')}")
    print(f"Project:     {exp.get('project', '?')}")
    print(f"Created:     {exp.get('created_at', '?')}")
    print(f"Updated:     {exp.get('updated_at', '?')}")
    if exp.get("started_at"):
        print(f"Started:     {exp['started_at']}")
    if exp.get("completed_at"):
        print(f"Completed:   {exp['completed_at']}")
    if exp.get("clearml_task_id"):
        print(f"ClearML:     {exp['clearml_task_id']}")
    if exp.get("best_value") is not None:
        print(f"Best Value:  {exp['best_value']}")
    if exp.get("error_message"):
        print(f"Error:       {exp['error_message'][:200]}")
    if exp.get("parent_id"):
        print(f"Parent:      {exp['parent_id']}")
        print(f"Root:        {exp['root_id']}")


@dispatch_app.command("tree")
def tree_cmd(
    root_id: str = typer.Argument(..., help="Root experiment ID to display tree for"),
) -> None:
    """Display an experiment's branching tree."""
    from expflow_pde.dispatch_db import DispatchDB

    db = DispatchDB()
    tree = db.get_experiment_tree(root_id)
    if not tree:
        print(f"No experiments found for root: {root_id}")
        raise typer.Exit(code=1)

    _print_tree(tree, indent=0)


@dispatch_app.command("list")
def list_cmd(
    limit: int = typer.Option(20, "--limit", "-l", help="Max experiments to show"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project"),
) -> None:
    """List recent experiments from the dispatch database."""
    from expflow_pde.dispatch_db import DispatchDB

    db = DispatchDB()
    experiments = db.query_recent(limit=limit, status=status, project=project)
    if not experiments:
        print("No experiments found.")
        return

    print(f"{'ID':<32} {'STATUS':<12} {'SCRIPT':<24} {'QUEUE':<12} {'CREATED'}")
    print("-" * 96)
    for exp in experiments:
        exp_id = exp["id"][:30]
        scr = (exp.get("script") or "?")[:22]
        que = (exp.get("queue") or "?")[:10]
        created = (exp.get("created_at") or "?")[:19]
        print(f"{exp_id:<32} {exp['status']:<12} {scr:<24} {que:<12} {created}")


@dispatch_app.command("archive")
def archive_cmd(
    before_date: str = typer.Argument(
        ..., help="Archive experiments completed before this date (YYYY-MM-DD)"
    ),
    archive_path: Optional[str] = typer.Option(
        None, "--archive-path", "-o", help="Custom archive DB path"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Only count, don't move"),
) -> None:
    """Archive completed experiments to a separate SQLite file."""
    from expflow_pde.dispatch_db import DispatchDB

    db = DispatchDB()

    if dry_run:
        exp_ids = _find_archivable(db, before_date)
        if exp_ids:
            print(f"Would archive {len(exp_ids)} experiment(s) completed before {before_date}:")
            for eid in exp_ids[:10]:
                print(f"  - {eid}")
            if len(exp_ids) > 10:
                print(f"  ... and {len(exp_ids) - 10} more")
        else:
            print(f"No experiments to archive before {before_date}.")
        return

    result = db.archive(before_date, archive_path=archive_path)
    if result["moved_count"] > 0:
        print(f"Archived {result['moved_count']} experiment(s) to {result['archive_path']}")
    else:
        print(f"No experiments to archive before {before_date}.")


@dispatch_app.command("stats")
def stats_cmd() -> None:
    """Show dispatch database statistics."""
    from expflow_pde.dispatch_db import DispatchDB

    db = DispatchDB()
    s = db.stats()

    print(f"Dispatch Database:  {s['db_path']}")
    print(f"Total Experiments:  {s['total_experiments']}")
    print(f"Total Metrics:      {s['total_metrics']}")
    print(f"Audit Entries:      {s['total_audit_entries']}")
    print(f"DB Size:            {_fmt_size(s['db_size_bytes'])}")
    print()
    if s["by_status"]:
        print("By Status:")
        for status, count in sorted(s["by_status"].items()):
            print(f"  {status:<12} {count}")


@dispatch_app.command("audit-log")
def audit_log_cmd(
    experiment_id: Optional[str] = typer.Option(
        None, "--experiment", "-e", help="Filter by experiment ID"
    ),
    event_type: Optional[str] = typer.Option(
        None, "--event-type", "-t", help="Filter by event type"
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="Max entries to show"),
) -> None:
    """Show audit log entries."""
    from expflow_pde.dispatch_db import DispatchDB

    db = DispatchDB()
    entries = db.get_audit_log(
        experiment_id=experiment_id,
        event_type=event_type,
        limit=limit,
    )
    if not entries:
        print("No audit entries found.")
        return

    print(f"{'TIME':<22} {'EVENT':<20} {'EXPERIMENT':<24} {'DETAIL'}")
    print("-" * 110)
    for e in entries:
        ts = (e.get("created_at") or "?")[:19]
        evt = e.get("event_type", "?")[:18]
        eid = (e.get("experiment_id") or "system")[:22]
        detail = e.get("detail_json", "")
        if detail and len(detail) > 40:
            detail = detail[:37] + "..."
        print(f"{ts:<22} {evt:<20} {eid:<24} {detail}")


# ── Helpers ──


def _print_tree(node: dict, indent: int = 0) -> None:
    """Recursively print an experiment tree."""
    prefix = "  " * indent
    status = node.get("status", "?")
    script = node.get("script", "?")[:30]
    exp_id = node.get("id", "?")[:24]
    print(f"{prefix}├─ [{status}] {script}  ({exp_id})")
    for child in node.get("children", []):
        _print_tree(child, indent + 1)


def _find_archivable(db, before_date: str) -> list[str]:
    """Return IDs of experiments that would be archived (dry-run)."""
    with db._read_tx() as conn:
        rows = conn.execute(
            "SELECT id FROM experiments "
            "WHERE status IN ('completed', 'failed', 'cancelled', 'pruned') "
            "AND completed_at < ?",
            (before_date,),
        ).fetchall()
    return [r[0] for r in rows]


def _fmt_size(bytes_: int) -> str:
    """Format byte size to human-readable."""
    if bytes_ < 1024:
        return f"{bytes_} B"
    if bytes_ < 1024**2:
        return f"{bytes_ / 1024:.1f} KB"
    return f"{bytes_ / (1024 ** 2):.1f} MB"
