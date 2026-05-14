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
    command: str = typer.Argument(..., help="Command or script path to execute"),
    queue: str = typer.Option("default", "--queue", "-q", help="Target queue"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    project: str = typer.Option("expflow", "--project", "-p", help="Project name"),
    use_clearml: bool = typer.Option(
        False, "--use-clearml", help="Use clearml Task.create() with auto-injected Task.init"
    ),
    arg: Optional[list[str]] = typer.Option(
        None, "--arg", "-a", help="Script argument, e.g. --arg lr=0.001 --arg epochs=100"
    ),
) -> None:
    """Submit an experiment.

    Use --use-clearml to wrap a Python script with automatic Task.init injection,
    so the script runs via clearml-agent without any code modification.
    """
    from expflow.dispatcher import dispatch_experiment

    tag_list = tags.split(",") if tags else None
    script_args: dict[str, str] = {}
    if arg:
        for a in arg:
            if "=" in a:
                k, v = a.split("=", 1)
                script_args[k] = v

    result = dispatch_experiment(
        command,
        queue=queue,
        tags=tag_list,
        project=project,
        use_clearml_task=use_clearml,
        script_args=script_args if script_args else None,
    )
    print(f"Experiment submitted: {result['experiment_id']}")
    print(f"  Queue:     {result['queue']}")
    print(f"  Status:    {result['status']}")
    print(f"  Timestamp: {result['timestamp']}")
    if "clearml_task_id" in result and result["clearml_task_id"]:
        print(f"  clearml Task: {result['clearml_task_id']}")


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
