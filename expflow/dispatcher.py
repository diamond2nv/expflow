#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow experiment dispatcher — submit, list, track experiments.

Manages experiment lifecycle with in-memory registry plus optional clearml/OpenCode delegation.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

# In-memory experiment registry (volatile — for CLI demo/testing)
# In production, experiments live in clearml tasks
_experiments: dict[str, dict[str, Any]] = {}


def dispatch_experiment(
    command: str,
    queue: str = "default",
    tags: list[str] | None = None,
    project: str = "expflow",
    use_clearml_task: bool = False,
    script_args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Submit an experiment for execution.

    When use_clearml_task=True and command is a Python script path,
    uses clearml Task.create() to wrap the script with Task.init auto-injection.

    Args:
        command: The shell command or script path to run.
        queue: Target clearml queue (default: 'default').
        tags: Optional tags for the experiment.
        project: Project name for clearml task grouping.
        use_clearml_task: When True, uses clearml Task.create() to wrap the
            script instead of running it directly.
        script_args: Dict of hyperparameters to pass to the script
            (e.g. {'lr': '0.001', 'epochs': '100'}).

    Returns:
        Dict with experiment_id, status, queue, tags, timestamp, and
        clearml_task_id if use_clearml_task=True.
    """
    experiment_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": "dispatched",
        "queue": queue,
        "command": command,
        "project": project,
        "tags": tags or [],
        "timestamp": now,
    }

    if use_clearml_task and is_python_script(command):
        clearml_result = _submit_via_clearml_task(
            script_path=command,
            project=project,
            queue=queue,
            tags=tags,
            script_args=script_args,
        )
        record["status"] = "queued"
        record["clearml_task_id"] = clearml_result.get("task_id", "")
        record["clearml_pipeline_id"] = clearml_result.get("pipeline_id", "")

    _experiments[experiment_id] = record
    return dict(record)


def is_python_script(command: str) -> bool:
    """Check if a command string is a Python script path."""
    return command.endswith(".py") and " " not in command.strip()


def _submit_via_clearml_task(
    script_path: str,
    project: str,
    queue: str,
    tags: list[str] | None = None,
    script_args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Submit a Python script via clearml Task.create() with auto-injected Task.init.

    The script is patched to include Task.init() automatically, so users don't
    need to modify their training scripts.
    """
    try:
        from clearml import Task
    except ImportError:
        return {"error": "clearml not installed"}

    task = Task.create(
        project_name=project,
        task_name=script_path,
        repo=script_path,
        script=script_path,
        add_task_init_call=True,  # Auto-inject Task.init into the script
    )

    # Override script arguments via argparse
    if script_args:
        argparse_args = [(f"--{k}", v) for k, v in script_args.items()]
        task.set_parameters({"Args": argparse_args})

    # Enqueue
    if queue:
        Task.enqueue(task=task, queue_name=queue)

    return {
        "task_id": task.id,
        "queue": queue,
    }


def list_experiments() -> list[dict[str, Any]]:
    """List all dispatched experiments.

    Returns:
        List of experiment records.
    """
    return list(_experiments.values())


def get_experiment_status(experiment_id: str) -> dict[str, Any]:
    """Get status of a specific experiment.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Experiment record or error dict.
    """
    record = _experiments.get(experiment_id)
    if record is None:
        return {"experiment_id": experiment_id, "error": "not found"}
    return dict(record)


def cancel_experiment(experiment_id: str) -> dict[str, Any]:
    """Cancel a running experiment.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Dict confirming cancellation or error.
    """
    record = _experiments.get(experiment_id)
    if record is None:
        return {"experiment_id": experiment_id, "error": "not found"}

    record["status"] = "cancelling"
    _experiments[experiment_id] = record
    return {"experiment_id": experiment_id, "status": "cancelling"}
