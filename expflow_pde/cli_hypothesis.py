#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow hypothesis CLI — record and track experiment hypotheses."""

import typer
from typing import Optional

hypothesis_app = typer.Typer(
    name="hypothesis",
    help="Record, track, and review experiment hypotheses (negative result logging)",
    no_args_is_help=True,
)


def get_hypothesis_app() -> typer.Typer:
    return hypothesis_app


@hypothesis_app.command("list")
def list_cmd(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter: proposed, accepted, rejected, inconclusive",
    ),
) -> None:
    """List all recorded hypotheses."""
    from expflow_pde.hypothesis import list_hypotheses

    hyps = list_hypotheses(status=status)
    if not hyps:
        print("  No hypotheses recorded.")
        return

    status_icons = {
        "proposed": "  ?",
        "accepted": "  V",
        "rejected": "  X",
        "inconclusive": "  ~",
    }
    print(f"  {'ID':<30} {'Status':<14} {'Hypothesis':<50}")
    print(f"  {'-' * 94}")
    for h in hyps:
        hid = h.get("id", "?")
        st = h.get("status", "?")
        icon = status_icons.get(st, "  ?")
        hyp_text = h.get("hypothesis", "?")[:48]
        print(f"  {hid:<30} {icon} {st:<12} {hyp_text:<50}")


@hypothesis_app.command("show")
def show_cmd(
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID"),
) -> None:
    """Show details of a single hypothesis."""
    from expflow_pde.hypothesis import show_hypothesis

    h = show_hypothesis(hypothesis_id)
    if h is None:
        print(f"  Unknown hypothesis: {hypothesis_id}")
        raise typer.Exit(code=1)

    print(f"  Hypothesis: {h.get('id')}")
    print(f"  {'─' * 60}")
    print(f"  Statement: {h.get('hypothesis')}")
    print(f"  Rationale: {h.get('rationale')}")
    print(f"  Status:    {h.get('status')}")
    print(f"  Created:   {h.get('created')}")
    if h.get("closed"):
        print(f"  Closed:    {h.get('closed')}")
    if h.get("suggested_params"):
        print(f"  Params:    {h.get('suggested_params')}")
    if h.get("origin_task_id"):
        print(f"  Origin:    {h.get('origin_task_id')}")
    if h.get("evidence"):
        print(f"  Evidence:  {h.get('evidence')}")
    if h.get("evidence_task_id"):
        print(f"  Evidence task: {h.get('evidence_task_id')}")


@hypothesis_app.command("close")
def close_cmd(
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID"),
    status: str = typer.Option(
        ..., "--status", "-s",
        help="Outcome: accepted, rejected, inconclusive",
    ),
    evidence: str = typer.Option(
        ..., "--evidence", "-e",
        help="What the experiment showed",
    ),
    evidence_task_id: Optional[str] = typer.Option(
        None, "--task", "-t",
        help="clearml task ID with evidence",
    ),
) -> None:
    """Close a hypothesis with experimental evidence."""
    from expflow_pde.hypothesis import close_hypothesis

    result = close_hypothesis(
        hypothesis_id=hypothesis_id,
        status=status,
        evidence=evidence,
        evidence_task_id=evidence_task_id,
    )
    if result is None:
        print(f"  Unknown hypothesis: {hypothesis_id}")
        raise typer.Exit(code=1)
    print(f"  Closed {hypothesis_id} as {status}")


@hypothesis_app.command("record")
def record_cmd(
    hypothesis: str = typer.Argument(..., help="The hypothesis statement"),
    rationale: str = typer.Option(
        ..., "--rationale", "-r",
        help="Why this hypothesis makes sense",
    ),
    suggested_param: Optional[list[str]] = typer.Option(
        None, "--param", "-p",
        help="Suggested params in key=val format (repeatable)",
    ),
    origin_task_id: Optional[str] = typer.Option(
        None, "--origin", "-o",
        help="clearml task ID that inspired this",
    ),
) -> None:
    """Record a new hypothesis before running an experiment."""
    from expflow_pde.hypothesis import record_hypothesis

    params = {}
    if suggested_param:
        for p in suggested_param:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v

    result = record_hypothesis(
        hypothesis=hypothesis,
        rationale=rationale,
        suggested_params=params or None,
        origin_task_id=origin_task_id,
    )
    print(f"  Recorded hypothesis: {result['id']}")
    print(f"  Status: proposed")


@hypothesis_app.command("rejected")
def rejected_cmd() -> None:
    """List all rejected (negative result) hypotheses."""
    from expflow_pde.hypothesis import get_rejected_directions

    hyps = get_rejected_directions()
    if not hyps:
        print("  No rejected hypotheses. All directions are still open.")
        return

    print(f"  Rejected hypotheses ({len(hyps)}):")
    for h in hyps:
        print(f"    X {h.get('hypothesis', '?')[:70]}")
        print(f"      Evidence: {h.get('evidence', '?')}")


@hypothesis_app.command("open")
def open_cmd() -> None:
    """List all unresolved (proposed) hypotheses."""
    from expflow_pde.hypothesis import get_open_hypotheses

    hyps = get_open_hypotheses()
    if not hyps:
        print("  No open hypotheses.")
        return

    print(f"  Open hypotheses ({len(hyps)}):")
    for h in hyps:
        print(f"    ? {h.get('id', '?')}: {h.get('hypothesis', '?')[:60]}")
