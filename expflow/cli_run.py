#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow run CLI — experiment dispatch sub-commands."""

from typing import Optional

import typer

run_app = typer.Typer(
    name="run",
    help="Submit and manage experiments",
    no_args_is_help=True,
)


def get_run_app() -> typer.Typer:
    return run_app


@run_app.command("submit")
def submit_cmd(
    command: str = typer.Argument(..., help="Command to execute"),
    queue: str = typer.Option("default", "--queue", "-q", help="Target queue"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    project: str = typer.Option("expflow", "--project", "-p", help="Project name"),
) -> None:
    """Submit an experiment."""
    from expflow.dispatcher import dispatch_experiment

    tag_list = tags.split(",") if tags else None
    result = dispatch_experiment(command, queue=queue, tags=tag_list, project=project)
    print(f"Experiment submitted: {result['experiment_id']}")
    print(f"  Queue:     {result['queue']}")
    print(f"  Status:    {result['status']}")
    print(f"  Timestamp: {result['timestamp']}")


@run_app.command("list")
def list_cmd() -> None:
    """List all experiments."""
    from expflow.dispatcher import list_experiments

    exps = list_experiments()

    if not exps:
        print("No experiments.")
        return

    print(f"{'ID':<12} {'STATUS':<14} {'QUEUE':<12} {'COMMAND':<40}")
    print("-" * 78)
    for e in exps:
        cmd_short = e["command"][:37] + "..." if len(e["command"]) > 40 else e["command"]
        print(f"{e['experiment_id']:<12} {e['status']:<14} {e['queue']:<12} {cmd_short:<40}")


@run_app.command("status")
def status_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
) -> None:
    """Show experiment status."""
    from expflow.dispatcher import get_experiment_status

    result = get_experiment_status(experiment_id)
    if "error" in result:
        print(f"Experiment '{experiment_id}' not found.")
        return

    print(f"ID:        {result['experiment_id']}")
    print(f"Status:    {result['status']}")
    print(f"Queue:     {result['queue']}")
    print(f"Command:   {result['command']}")
    print(f"Tags:      {', '.join(result.get('tags', [])) or '-'}")


@run_app.command("cancel")
def cancel_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
) -> None:
    """Cancel an experiment."""
    from expflow.dispatcher import cancel_experiment

    result = cancel_experiment(experiment_id)
    if "error" in result:
        print(f"Experiment '{experiment_id}' not found.")
        return

    print(f"Experiment '{result['experiment_id']}' {result['status']}.")
