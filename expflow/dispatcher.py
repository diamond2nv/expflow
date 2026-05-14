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
) -> dict[str, Any]:
    """Submit an experiment for execution.

    Args:
        command: The shell command to run.
        queue: Target clearml queue (default: 'default').
        tags: Optional tags for the experiment.
        project: Project name for clearml task grouping.

    Returns:
        Dict with experiment_id, status, queue, tags, timestamp.
    """
    experiment_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    record = {
        "experiment_id": experiment_id,
        "status": "dispatched",
        "queue": queue,
        "command": command,
        "project": project,
        "tags": tags or [],
        "timestamp": now,
    }
    _experiments[experiment_id] = record
    return dict(record)


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
