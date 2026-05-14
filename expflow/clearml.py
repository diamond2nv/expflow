#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow clearml integration — Task CRUD, queue management, dataset compliance.

All functions return JSON-serializable dicts, not clearml SDK objects.
"""

from typing import Any, Literal

# Import clearml lazily so missing dependency fails at call time, not import time.

# ── Internal helpers ──


def _get_task_module():
    """Lazy import of clearml to avoid import-time dependency."""
    import clearml  # noqa: F401 — triggers Task/Queue/Dataset availability
    from clearml import Task  # noqa: F811

    return Task


def _get_queue_module():
    import clearml
    from clearml import Queue  # noqa: F811

    return clearml, Queue


def _get_dataset_module():
    import clearml
    from clearml import Dataset  # noqa: F811

    return clearml, Dataset


def _serialize_task(task: Any) -> dict[str, Any]:
    """Convert a clearml Task object to a serializable dict."""
    return {
        "id": task.id,
        "name": task.name,
        "project": task.project,
        "status": task.status,
        "tags": list(task.get_tags() or []),
        "last_iteration": getattr(task, "last_iteration", 0),
    }


def _serialize_queue(queue: Any) -> dict[str, Any]:
    """Convert a clearml Queue object to a serializable dict."""
    return {
        "id": queue.id,
        "name": queue.name,
    }


# ── Task operations ──


def list_tasks(
    project_name: str | None = None,
    task_name: str | None = None,
    tags: list[str] | None = None,
    status: list[str] | None = None,
) -> list[dict[str, Any]]:
    """List clearml tasks, return serializable dicts.

    Args:
        project_name: Filter by project name (partial match).
        task_name: Filter by task name (partial match).
        tags: Filter by tags (tasks must have ALL specified tags).
        status: Filter by status values (e.g. ['completed', 'failed']).

    Returns:
        List of task dicts with id, name, project, status, tags, last_iteration.
    """
    task_cls = _get_task_module()  # noqa: N806
    kwargs: dict[str, Any] = {}
    if project_name is not None:
        kwargs["project_name"] = project_name
    if task_name is not None:
        kwargs["task_name"] = task_name
    if tags is not None:
        kwargs["tags"] = tags
    if status is not None:
        kwargs["status"] = status

    tasks = task_cls.get_tasks(**kwargs)
    return [_serialize_task(t) for t in tasks]


def get_task(task_id: str) -> dict[str, Any]:
    """Get a single task by ID.

    Args:
        task_id: The clearml task ID.

    Returns:
        Task dict with id, name, project, status, tags, last_iteration.
    """
    task_cls = _get_task_module()  # noqa: N806
    task = task_cls.get_task(task_id=task_id)
    return _serialize_task(task)


def enqueue_task(task_id: str, queue_name: str = "default") -> dict[str, Any]:
    """Enqueue a task to a clearml queue.

    Args:
        task_id: The clearml task ID to enqueue.
        queue_name: Queue name (default: 'default').

    Returns:
        Dict with task_id, queue, status.
    """
    task_cls = _get_task_module()  # noqa: N806
    task = task_cls.get_task(task_id=task_id)
    task.enqueue(queue_name=queue_name)
    return {
        "task_id": task.id,
        "queue": queue_name,
        "status": task.status,
    }


def dequeue_task(task_id: str) -> dict[str, Any]:
    """Dequeue a task from its current queue.

    Args:
        task_id: The clearml task ID to dequeue.

    Returns:
        Dict with task_id, status.
    """
    task_cls = _get_task_module()  # noqa: N806
    task = task_cls.get_task(task_id=task_id)
    task.dequeue()
    return {
        "task_id": task.id,
        "status": task.status,
    }


# ── Queue operations ──


def list_queues() -> list[dict[str, Any]]:
    """List all available queues.

    Returns:
        List of queue dicts with id, name.
    """
    _, queue_cls = _get_queue_module()  # noqa: N806
    queues = queue_cls.get_queues()
    return [_serialize_queue(q) for q in queues]


def get_queue_status(queue_name: str) -> dict[str, Any]:
    """Get detailed status of a specific queue.

    Args:
        queue_name: The queue name.

    Returns:
        Dict with queue id, name, entries (list of queued task IDs).
    """
    _, queue_cls = _get_queue_module()  # noqa: N806
    queue = queue_cls.get_queue(queue_name=queue_name)
    result = _serialize_queue(queue)
    result["entries"] = getattr(queue, "entries", [])
    return result


# ── Dataset operations ──

COMPLIANCE_METADATA_KEY = "expflow:compliance"
_COMPLIANCE_VALUES = frozenset(["allowed", "forbidden"])


def register_dataset(
    name: str,
    version: str,
    path: str,
    compliance: Literal["allowed", "forbidden"],
    **metadata: Any,
) -> dict[str, Any]:
    """Register a dataset with compliance annotation.

    Args:
        name: Dataset name.
        version: Dataset version string.
        path: Local/remote path to dataset files.
        compliance: 'allowed' (competition-legal) or 'forbidden' (not allowed).
        **metadata: Additional metadata to attach.

    Returns:
        Dict with id, name, version, compliance, path.

    Raises:
        ValueError: If compliance is not 'allowed' or 'forbidden'.
    """
    if compliance not in _COMPLIANCE_VALUES:
        raise ValueError(
            f"compliance must be one of {sorted(_COMPLIANCE_VALUES)}, got '{compliance}'"
        )

    _, dataset_cls = _get_dataset_module()  # noqa: N806
    ds = dataset_cls.create(
        dataset_name=name,
        dataset_version=version,
        dataset_project="expflow",
        dataset_tags=[f"expflow:compliance={compliance}"],
    )

    # Set compliance as metadata for queryability
    ds.set_metadata(COMPLIANCE_METADATA_KEY, compliance)

    # Set additional metadata
    for k, v in metadata.items():
        ds.set_metadata(f"expflow:{k}", str(v))

    # Set the dataset files path (sync from local if path exists)
    ds.set_metadata("expflow:source_path", path)

    return {
        "id": ds.id,
        "name": name,
        "version": version,
        "compliance": compliance,
        "path": path,
    }


def list_datasets(
    name_filter: str | None = None,
    compliance_filter: Literal["allowed", "forbidden"] | None = None,
) -> list[dict[str, Any]]:
    """List registered datasets with compliance info.

    Args:
        name_filter: Optional name substring filter.
        compliance_filter: Optional compliance status filter.

    Returns:
        List of dataset dicts with id, name, version, compliance.
    """
    _, dataset_cls = _get_dataset_module()  # noqa: N806
    datasets = dataset_cls.list_datasets()

    result = []
    for ds in datasets:
        compliance = ds.get_metadata(COMPLIANCE_METADATA_KEY)

        # Apply filters
        if compliance_filter is not None and compliance != compliance_filter:
            continue
        if name_filter is not None and name_filter not in ds.name:
            continue

        result.append(
            {
                "id": ds.id,
                "name": ds.name,
                "version": ds.version,
                "compliance": compliance,
            }
        )

    return result
