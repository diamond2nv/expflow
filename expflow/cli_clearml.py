#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow clearml CLI sub-commands — lazy imports of clearml SDK at call time."""

from typing import Optional

import typer

clearml_app = typer.Typer(
    name="clearml",
    help="Interact with ClearML experiment management",
    no_args_is_help=True,
)


def get_clearml_app() -> typer.Typer:
    """Return the clearml sub-command group."""
    return clearml_app


# ── Task commands ──


@clearml_app.command("tasks")
def list_tasks_cmd(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project name"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Filter by task name"),
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status (comma-separated)"
    ),
) -> None:
    """List ClearML tasks."""
    from expflow.clearml import list_tasks

    status_list = status.split(",") if status else None
    tasks = list_tasks(
        project_name=project,
        task_name=name,
        status=status_list,
    )

    if not tasks:
        print("No tasks found.")
        return

    print(f"{'ID':<24} {'NAME':<30} {'PROJECT':<20} {'STATUS':<12}")
    print("-" * 86)
    for t in tasks:
        print(f"{t['id']:<24} {t['name']:<30} {t['project']:<20} {t['status']:<12}")


@clearml_app.command("task")
def get_task_cmd(
    task_id: str = typer.Argument(..., help="Task ID"),
) -> None:
    """Show details for a single task."""
    from expflow.clearml import get_task

    t = get_task(task_id)
    print(f"ID:       {t['id']}")
    print(f"Name:     {t['name']}")
    print(f"Project:  {t['project']}")
    print(f"Status:   {t['status']}")
    print(f"Tags:     {', '.join(t['tags']) if t['tags'] else '(none)'}")
    print(f"Iterations: {t['last_iteration']}")


# ── Queue commands ──


@clearml_app.command("enqueue")
def enqueue_cmd(
    task_id: str = typer.Argument(..., help="Task ID to enqueue"),
    queue: str = typer.Option("default", "--queue", "-q", help="Queue name"),
) -> None:
    """Enqueue a task to a queue."""
    from expflow.clearml import enqueue_task

    result = enqueue_task(task_id, queue_name=queue)
    print(f"Task {result['task_id']} enqueued to '{result['queue']}' (status: {result['status']})")


@clearml_app.command("dequeue")
def dequeue_cmd(
    task_id: str = typer.Argument(..., help="Task ID to dequeue"),
) -> None:
    """Dequeue a task."""
    from expflow.clearml import dequeue_task

    result = dequeue_task(task_id)
    print(f"Task {result['task_id']} dequeued (status: {result['status']})")


@clearml_app.command("queues")
def list_queues_cmd() -> None:
    """List all available queues."""
    from expflow.clearml import list_queues

    queues = list_queues()

    if not queues:
        print("No queues found.")
        return

    print(f"{'ID':<24} {'NAME':<20}")
    print("-" * 44)
    for q in queues:
        print(f"{q['id']:<24} {q['name']:<20}")


# ── Dataset commands ──


@clearml_app.command("dataset-register")
def dataset_register_cmd(
    name: str = typer.Argument(..., help="Dataset name"),
    version: str = typer.Option("1.0", "--version", "-v", help="Dataset version"),
    path: str = typer.Option(..., "--path", "-p", help="Dataset file path"),
    compliance: str = typer.Option(
        "allowed",
        "--compliance",
        "-c",
        help="Compliance status: allowed or forbidden",
    ),
) -> None:
    """Register a PDEBench dataset with compliance annotation."""
    from expflow.clearml import register_dataset

    result = register_dataset(
        name=name,
        version=version,
        path=path,
        compliance=compliance,  # type: ignore — validated by register_dataset
    )
    print(f"Dataset registered: {result['name']} v{result['version']}")
    print(f"  ID:         {result['id']}")
    print(f"  Compliance: {result['compliance']}")
    print(f"  Path:       {result['path']}")


@clearml_app.command("dataset-list")
def dataset_list_cmd(
    name_filter: Optional[str] = typer.Option(
        None, "--name", "-n", help="Filter by dataset name (substring)"
    ),
    compliance_filter: Optional[str] = typer.Option(
        None,
        "--compliance",
        "-c",
        help="Filter by compliance status: allowed or forbidden",
    ),
) -> None:
    """List registered datasets with compliance info."""
    from expflow.clearml import list_datasets

    datasets = list_datasets(
        name_filter=name_filter,
        compliance_filter=compliance_filter,  # type: ignore
    )

    if not datasets:
        print("No datasets found.")
        return

    print(f"{'ID':<24} {'NAME':<30} {'VERSION':<10} {'COMPLIANCE':<12}")
    print("-" * 76)
    for ds in datasets:
        comp = ds.get("compliance", "?") or "-"
        print(f"{ds['id']:<24} {ds['name']:<30} {ds['version']:<10} {comp:<12}")
