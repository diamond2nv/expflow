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


COMPLIANCE_METADATA_KEY = "expflow:compliance"
_COMPLIANCE_VALUES = frozenset(["allowed", "forbidden"])


def annotate_compliance(
    dataset_id: str,
    compliance: Literal["allowed", "forbidden"],
) -> dict[str, Any]:
    """Tag an existing dataset with compliance metadata.

    Args:
        dataset_id: The dataset ID to annotate.
        compliance: 'allowed' or 'forbidden'.

    Returns:
        Dict with id, compliance.
    """
    if compliance not in _COMPLIANCE_VALUES:
        raise ValueError(
            f"compliance must be one of {sorted(_COMPLIANCE_VALUES)}, got '{compliance}'"
        )
    _, dataset_cls = _get_dataset_module()
    ds = dataset_cls.get(dataset_id=dataset_id)
    ds.set_metadata(COMPLIANCE_METADATA_KEY, compliance)
    return {"id": ds.id, "compliance": compliance}


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
    _, dataset_cls = _get_dataset_module()
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


# ── Dataset upload / download / lineage ──


def dataset_upload(
    local_path: str,
    dataset_name: str,
    dataset_project: str = "PDEBench",
    version: str | None = None,
    parent_dataset_ids: list[str] | None = None,
    compliance: Literal["allowed", "forbidden"] | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Upload local files to clearml Fileserver and register as a Dataset.

    Wraps the clearml SDK chain: Dataset.create() -> add_files() -> upload() -> finalize().

    Args:
        local_path: Path to local file(s) or folder to upload.
        dataset_name: Dataset name in clearml.
        dataset_project: Project name (default: PDEBench).
        version: Semantic version string (auto-increments if None).
        parent_dataset_ids: List of parent dataset IDs for lineage inheritance.
        compliance: 'allowed' (competition-legal) or 'forbidden'.
        description: Optional human-readable description.
        tags: Optional list of tags.
        extra_metadata: Optional dict of additional metadata key-value pairs.

    Returns:
        Dict with id, name, version, compliance, uri, file_count, total_bytes.
    """
    _, dataset_cls = _get_dataset_module()

    ds = dataset_cls.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=version,
        parent_datasets=parent_dataset_ids,
        description=description or "",
    )

    # Add files
    ds.add_files(path=local_path)

    # Set compliance metadata
    if compliance is not None:
        ds.set_metadata(COMPLIANCE_METADATA_KEY, compliance)

    # Set tags
    if tags:
        for tag in tags:
            ds.set_metadata(f"expflow:tag:{tag}", "true")

    # Set extra metadata
    if extra_metadata:
        for k, v in extra_metadata.items():
            ds.set_metadata(f"expflow:{k}", v)

    # Set source path
    ds.set_metadata("expflow:source_path", local_path)

    # Upload to clearml fileserver (default)
    ds.upload()

    # Finalize (close, immutable)
    ds.finalize()

    return {
        "id": ds.id,
        "name": dataset_name,
        "version": version or ds.version,
        "compliance": compliance,
        "uri": f"clearml://datasets/{ds.id}",
    }


def dataset_download(
    target_folder: str,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    dataset_project: str | None = "PDEBench",
    dataset_version: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download a Dataset from clearml Fileserver to local folder.

    Wraps clearml SDK: Dataset.get() -> get_mutable_local_copy().

    Args:
        target_folder: Local path to download to.
        dataset_id: Dataset ID (mutually exclusive with name/project).
        dataset_name: Dataset name for lookup (requires project).
        dataset_project: Project name (default: PDEBench).
        dataset_version: Specific version (None = latest).
        overwrite: Whether to overwrite existing target folder contents.

    Returns:
        Dict with id, name, version, local_path.

    Raises:
        ValueError: If neither dataset_id nor (dataset_name + project) is provided.
    """
    if not dataset_id and not (dataset_name and dataset_project):
        raise ValueError("Provide either dataset_id or dataset_name + dataset_project")
    _, dataset_cls = _get_dataset_module()

    ds = dataset_cls.get(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=dataset_version,
        only_completed=True,
    )

    local_path = ds.get_mutable_local_copy(
        target_folder=target_folder,
        overwrite=overwrite,
    )

    return {
        "id": ds.id,
        "name": getattr(ds, "name", dataset_name or ""),
        "version": getattr(ds, "version", dataset_version or ""),
        "local_path": str(local_path),
    }


def dataset_lineage(
    dataset_id: str,
    depth: int = 10,
) -> list[dict[str, Any]]:
    """Trace dataset lineage by recursively following the parent chain.

    Args:
        dataset_id: Starting dataset ID.
        depth: Maximum recursion depth (default: 10).

    Returns:
        List of dicts from oldest to newest, each with
        id, name, version, compliance, parent_id.
    """
    _, dataset_cls = _get_dataset_module()

    lineage: list[dict[str, Any]] = []
    current_id: str | None = dataset_id

    for _ in range(depth):
        if not current_id:
            break
        try:
            ds = dataset_cls.get(dataset_id=current_id)
        except Exception:
            break

        parent_id: str | None = getattr(ds, "parent", None)
        if parent_id and isinstance(parent_id, str):
            parent_id = parent_id
        elif hasattr(ds, "parent") and ds.parent is not None:
            parent_id = str(ds.parent)
        else:
            parent_id = None

        entry = {
            "id": ds.id,
            "name": getattr(ds, "name", ""),
            "version": getattr(ds, "version", ""),
            "compliance": (
                ds.get_metadata(COMPLIANCE_METADATA_KEY) if hasattr(ds, "get_metadata") else None
            ),
            "parent_id": parent_id,
        }
        lineage.append(entry)
        current_id = parent_id

    # Reverse: oldest first
    lineage.reverse()
    return lineage


# ── Model operations ──


def model_list(
    project_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    only_published: bool = False,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """List registered models (checkpoints) from clearml Model store.

    Wraps clearml SDK: Model.query_models().

    Args:
        project_name: Filter by project name.
        tags: Filter by tags (OR within list, supports __$and/__$not).
        metadata: Filter by metadata key-value pairs.
        only_published: Only published models.
        max_results: Maximum number of results.

    Returns:
        List of model dicts with id, name, project, tags, created, uri, task_id, framework.
    """
    _import_model_module()
    from clearml import Model  # noqa: F811

    models = Model.query_models(
        project_name=project_name,
        tags=tags,
        metadata=metadata,
        only_published=only_published,
        max_results=max_results,
    )

    result = []
    for m in models:
        result.append(
            {
                "id": m.id,
                "name": getattr(m, "name", ""),
                "project": getattr(m, "project", ""),
                "tags": list(getattr(m, "tags", []) or []),
                "created": str(getattr(m, "created", "")),
                "uri": getattr(m, "uri", ""),
                "task_id": getattr(m, "task_id", ""),
                "framework": getattr(m, "framework", ""),
            }
        )
    return result


def model_upload(
    local_path: str,
    task_id: str,
    framework: str = "PyTorch",
    model_name: str | None = None,
    upload_uri: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Upload a local model checkpoint to clearml Model store.

    Wraps clearml SDK: OutputModel(task=task) -> update_weights().

    Args:
        local_path: Path to the model weights file.
        task_id: ID of the clearml task producing this model.
        framework: Framework name (default: PyTorch).
        model_name: Optional human-readable name.
        upload_uri: Override upload destination (default: clearml fileserver).
        tags: Optional tags.

    Returns:
        Dict with id, name, task_id, uri.
    """
    _import_model_module()
    from clearml import (  # noqa: F811
        OutputModel,  # noqa: F811
        Task,
    )

    task = Task.get_task(task_id=task_id)

    output_model = OutputModel(
        task=task,
        framework=framework,
    )
    if model_name:
        output_model.name = model_name
    if tags:
        output_model.set_metadata("expflow:tags", ",".join(tags))

    output_model.update_weights(
        weights_filename=local_path,
        upload_uri=upload_uri,
        update_comment=f"Uploaded via expflow from {local_path}",
    )

    return {
        "id": output_model.id,
        "name": model_name or Task.get_task(task_id=task_id).name + "_model",
        "task_id": task_id,
        "uri": getattr(output_model, "uri", ""),
    }


def _import_model_module() -> None:
    """Lazy import of clearml model modules."""
    import clearml  # noqa: F401
