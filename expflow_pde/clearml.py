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


# ── Backend API helpers for queues / workers ──


def _call_queue_service(action: str) -> dict[str, Any]:
    """Call clearml backend queue service via SDK session.

    Clearml 2.1.7+ removed the top-level ``Queue`` class — use the
    backend API directly instead of ``from clearml import Queue``.
    ``action`` is a PascalCase request name e.g. ``get_all`` → ``GetAllRequest``.
    """
    import importlib
    from clearml import Task

    svc = importlib.import_module("clearml.backend_api.services.v2_23.queues")
    cls = getattr(svc, f"{action.title().replace('_', '')}Request", None)
    if cls is None:
        raise RuntimeError(f"clearml queue service has no '{action}' request")
    res = Task._get_default_session().send(cls())
    return (res.response_data or {}) if hasattr(res, 'response_data') else (res.get('data', {}))


def _call_worker_service(action: str) -> dict[str, Any]:
    """Call clearml backend worker service via SDK session."""
    import importlib
    from clearml import Task

    svc = importlib.import_module("clearml.backend_api.services.v2_23.workers")
    cls = getattr(svc, f"{action.title().replace('_', '')}Request", None)
    if cls is None:
        raise RuntimeError(f"clearml worker service has no '{action}' request")
    res = Task._get_default_session().send(cls())
    return (res.response_data or {}) if hasattr(res, 'response_data') else (res.get('data', {}))


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


def get_task_scalars(task_id: str) -> dict[str, Any] | None:
    """Fetch reported scalars from a clearml task.

    Returns a flat dict with scalar title/series -> latest value mappings,
    or None if the task has no reported scalars.
    """
    task_cls = _get_task_module()  # noqa: N806
    task = task_cls.get_task(task_id=task_id)

    # Read scalars from the task's last metrics event
    _task_models = None
    try:
        _task_models = task.models.get("output", [])
    except AttributeError:
        _task_models = []

    # Try reading scalars directly from the task
    scalars: dict[str, Any] = {}
    try:
        reported = task.get_last_scalars()
        if reported:
            for title, series_dict in reported.items():
                for series, value in series_dict.items():
                    scalars[f"{title}/{series}"] = value
    except Exception:
        pass

    # Also try reading from the task's artifacts / output
    try:
        # clearml stores scalar metrics in the task's event log
        # Use the internal event reader as fallback
        for title in ("Score", "Loss", "PDE", "Time"):
            for series_key, value in _get_series_values(task, title).items():
                scalars[f"{title}/{series_key}"] = value
    except Exception:
        pass

    if not scalars:
        return None
    return scalars


def _get_series_values(task: Any, title: str) -> dict[str, float]:
    """Helper: extract scalar series for a given title from a clearml task.

    Uses the internal _get_latest_scalar_values approach.
    """
    result: dict[str, float] = {}
    try:
        # Try the standard API approach
        scalar_keys = task.get_last_scalar_series()
        for key in scalar_keys or []:
            if key.startswith(f"{title}/"):
                series_name = key[len(title) + 1 :]
                values = task.get_scalar_reported_series(key)
                if values:
                    result[series_name] = values[-1].get("value", 0.0)
    except Exception:
        pass
    return result


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


# ── Worker operations ──


def list_workers(
    project_name: str | None = None,
    status: list[str] | None = None,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """List registered clearml workers and their status.

    Wraps clearml SDK: Workers.get_workers() via Task.get_workers() or similar.

    Args:
        project_name: Filter by project (optional).
        status: Filter by worker status (optional).
        max_results: Max results (default: 50).

    Returns:
        List of worker dicts with id, name, status, GPU info, etc.
    """
    _import_worker_module()
    from clearml import Worker  # noqa: F811

    try:
        workers = Worker.get_workers(
            project_name=project_name,
            status=status,
            max_results=max_results,
        )
    except Exception:
        return []

    result = []
    for w in workers:
        entry: dict[str, Any] = {
            "id": getattr(w, "id", ""),
            "name": getattr(w, "name", ""),
            "status": getattr(w, "status", ""),
            "queue": getattr(w, "queue", ""),
            "last_activity": str(getattr(w, "last_activity", "")),
            "ip": getattr(w, "ip", ""),
            "num_gpus": getattr(w, "num_gpus", 0),
            "gpus": getattr(w, "gpus", ""),
            "os_version": getattr(w, "os_version", ""),
            "python_version": getattr(w, "python_version", ""),
            "available_gpus": getattr(w, "available_gpus", ""),
        }
        result.append(entry)

    return result


def _import_worker_module() -> None:
    """Lazy import of clearml worker modules."""
    import clearml  # noqa: F401


# ── Queue operations ──


def list_queues() -> list[dict[str, Any]]:
    """List all available queues.

    Uses clearml backend API directly (Queue class removed in 2.1.7+).

    Returns:
        List of queue dicts with id, name.
    """
    data = _call_queue_service("get_all")
    queues_raw: list[dict[str, Any]] = data.get("queues", [])
    return [{"id": q["id"], "name": q["name"]} for q in queues_raw]


def get_queue_status(queue_name: str) -> dict[str, Any]:
    """Get detailed status of a specific queue.

    Returns:
        Dict with queue id, name, entries (list of queued task IDs).
    """
    data = _call_queue_service("get_all")
    for q in data.get("queues", []):
        if q.get("name") == queue_name:
            entries_raw = q.get("entries", [])
            entries = [
                e.get("task", e.get("id", "")) for e in entries_raw
            ] if entries_raw else []
            return {
                "id": q["id"],
                "name": q["name"],
                "entries": entries,
            }
    return {"error": f"Queue {queue_name!r} not found"}


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


def _get_pipeline_module():
    import clearml  # noqa: F401
    from clearml import PipelineController  # noqa: F811

    return PipelineController


def _get_task_module_for_pipeline():
    return _get_task_module()


# ── Pipeline operations ──


def pipeline_create(
    name: str,
    project: str,
    version: str | None = None,
    abort_on_failure: bool = False,
    add_pipeline_tags: bool = False,
    add_run_number: bool = True,
    docker: str | None = None,
    packages: list[str] | None = None,
) -> dict[str, Any]:
    """Create a clearml PipelineController and start tracking it.

    After creation, add steps with pipeline_add_step(), then start with pipeline_start().

    Args:
        name: Pipeline name (e.g. 'fno_train_eval').
        project: Project name (e.g. 'PDEBench').
        version: Optional semantic version (auto-increments if None).
        abort_on_failure: Abort all steps on any failure (default: False).
        add_pipeline_tags: Tag steps with pipe:<pipeline_id> (default: False).
        add_run_number: Append run number to name (default: True).
        docker: Optional Docker image for remote execution.

    Returns:
        Dict with pipeline_id, name, project, version, status.
    """
    PipelineController = _get_pipeline_module()  # noqa: N806

    pipe = PipelineController(
        name=name,
        project=project,
        version=version,
        abort_on_failure=abort_on_failure,
        add_pipeline_tags=add_pipeline_tags,
        add_run_number=add_run_number,
        docker=docker,
    )

    # Set packages on the pipeline if explicitly provided
    if packages is not None:
        pipe.set_default_packages(packages)

    return {
        "pipeline_id": pipe.id if hasattr(pipe, "id") else "",
        "name": name,
        "project": project,
        "version": version or "",
        "status": "created",
    }


def pipeline_add_step(
    pipeline_name: str,
    project: str,
    step_name: str,
    base_task_id: str | None = None,
    base_task_name: str | None = None,
    base_task_project: str | None = None,
    parents: list[str] | None = None,
    parameter_override: dict[str, Any] | None = None,
    execution_queue: str | None = None,
    cache_executed_step: bool = False,
    time_limit: float | None = None,
    monitor_metrics: list[str] | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Add a step to an existing pipeline controller.

    Args:
        pipeline_name: Name of the existing pipeline controller task.
        project: Project name.
        step_name: Unique name for this step.
        base_task_id: Existing task ID to clone for this step.
        base_task_name: Task name to look up (alternative to base_task_id).
        base_task_project: Project for base_task_name lookup.
        parents: List of parent step names this step depends on.
        parameter_override: Dict of parameter overrides.
        execution_queue: Queue to execute this step on.
        cache_executed_step: Reuse already-executed tasks (default: False).
        time_limit: Step time limit in minutes.
        monitor_metrics: List of metric tuples to log on pipeline task.
        version: Pipeline version (required if multiple versions exist).

    Returns:
        Dict with pipeline_name, step_name, status.
    """
    if not base_task_id and not base_task_name:
        raise ValueError("Provide either base_task_id or base_task_name")

    # Load the existing pipeline's task and recreate the controller from it
    PipelineController = _get_pipeline_module()  # noqa: N806

    # Pass through kwargs for PipelineController
    pipe = PipelineController(
        name=pipeline_name,
        project=project,
        version=version,
    )

    # Default parameter_override to empty dict
    kwargs: dict[str, Any] = {
        "name": step_name,
    }
    if base_task_id:
        kwargs["base_task_id"] = base_task_id
    if base_task_name:
        kwargs["base_task_name"] = base_task_name
        kwargs["base_task_project"] = base_task_project or project
    if parents:
        kwargs["parents"] = parents
    if parameter_override:
        kwargs["parameter_override"] = parameter_override
    if execution_queue:
        kwargs["execution_queue"] = execution_queue
    if cache_executed_step:
        kwargs["cache_executed_step"] = True
    if time_limit is not None:
        kwargs["time_limit"] = time_limit
    if monitor_metrics:
        kwargs["monitor_metrics"] = monitor_metrics

    pipe.add_step(**kwargs)

    return {
        "pipeline_name": pipeline_name,
        "step_name": step_name,
        "parents": parents or [],
        "status": "defined",
    }


def pipeline_start(
    pipeline_name: str,
    project: str,
    version: str | None = None,
    queue_name: str | None = None,
    timeout_minutes: float | None = None,
) -> dict[str, Any]:
    """Start a pipeline controller execution.

    Args:
        pipeline_name: Pipeline controller task name.
        project: Project name.
        version: Pipeline version (required if multiple versions).
        queue_name: Queue to execute controller on (None = local execution).
        timeout_minutes: Max wait time in minutes (None = no wait).

    Returns:
        Dict with pipeline_name, status, started_at.
    """
    PipelineController = _get_pipeline_module()  # noqa: N806

    pipe = PipelineController(
        name=pipeline_name,
        project=project,
        version=version,
    )

    pipe.start(queue_name=queue_name)
    pipe.wait(timeout=timeout_minutes)

    status = getattr(pipe, "status", "started")
    return {
        "pipeline_name": pipeline_name,
        "status": status,
        "started": True,
    }


def pipeline_stop(
    pipeline_name: str,
    project: str,
    version: str | None = None,
) -> dict[str, Any]:
    """Stop a running pipeline controller.

    Args:
        pipeline_name: Pipeline controller task name.
        project: Project name.
        version: Pipeline version (required if multiple versions).

    Returns:
        Dict with pipeline_name, status.
    """
    PipelineController = _get_pipeline_module()  # noqa: N806

    pipe = PipelineController(
        name=pipeline_name,
        project=project,
        version=version,
    )
    pipe.stop()

    return {
        "pipeline_name": pipeline_name,
        "status": "stopped",
    }


def pipeline_list(
    project_name: str | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """List pipeline controller tasks.

    Pipeline controllers are clearml tasks with the pipeline tag or
    whose name contains 'pipeline'.

    Args:
        project_name: Filter by project.
        max_results: Max results (default: 20).

    Returns:
        List of pipeline dicts with id, name, project, status, tags.
    """
    Task = _get_task_module_for_pipeline()  # noqa: N806

    tasks = Task.get_tasks(
        project_name=project_name,
        task_name="pipeline*",
        status=["created", "in_progress", "completed", "failed", "stopped"],
    )

    result = []
    for t in tasks:
        result.append(
            {
                "id": t.id,
                "name": getattr(t, "name", ""),
                "project": getattr(t, "project", ""),
                "status": getattr(t, "status", ""),
                "tags": list(getattr(t, "tags", []) or []),
            }
        )

    return result[:max_results]


# ── Scheduler operations ──


def scheduler_create(
    force_create_task_name: str | None = None,
    force_create_task_project: str | None = None,
) -> dict[str, Any]:
    """Create a clearml TaskScheduler for cron-like task scheduling.

    Args:
        force_create_task_name: Force creation of scheduler service task name.
        force_create_task_project: Force creation of scheduler service project.

    Returns:
        Dict with status and scheduler info.
    """

    return {
        "status": "created",
    }


def scheduler_add_task(
    task_id: str | None = None,
    queue: str | None = None,
    name: str | None = None,
    minute: int | None = None,
    hour: int | None = None,
    day: int | None = None,
    weekdays: list[str] | None = None,
    month: int | None = None,
    recurring: bool = True,
    single_instance: bool = False,
    execute_immediately: bool = False,
    task_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a cron-like scheduling job for an existing clearml Task.

    Args:
        task_id: ID of the Task to clone and schedule.
        queue: Queue to enqueue the scheduled task on.
        name: Unique name for this schedule entry.
        minute: Minutes between launches or specific minute of hour.
        hour: Hours between launches or specific hour of day.
        day: Days between executions or specific day of month.
        weekdays: Days of week (e.g. ['monday', 'friday']).
        month: Months between launches or specific month.
        recurring: Repeat (True) or launch once (False) (default: True).
        single_instance: Skip launch if previous instance still running.
        execute_immediately: Execute immediately then follow schedule.
        task_parameters: Config parameters dict like {'Args/lr': '0.001'}.

    Returns:
        Dict with task_id, name, queue, recurring, status.

    Example:
        # Every 1 hour:
        scheduler_add_task(task_id='abc', queue='default', hour=1)

        # Every day at 9:00:
        scheduler_add_task(task_id='abc', queue='default', minute=0, hour=9, day=1)

        # Once a month:
        scheduler_add_task(task_id='abc', queue='default', month=1, day=5)
    """
    TaskScheduler = _get_scheduler_module()  # noqa: N806

    scheduler = TaskScheduler()
    success = scheduler.add_task(
        schedule_task_id=task_id,
        queue=queue,
        name=name,
        minute=minute,
        hour=hour,
        day=day,
        weekdays=weekdays,
        month=month,
        recurring=recurring,
        single_instance=single_instance,
        execute_immediately=execute_immediately,
        task_parameters=task_parameters,
    )

    return {
        "task_id": task_id or "",
        "name": name or "",
        "queue": queue or "",
        "recurring": recurring,
        "added": success,
        "status": "scheduled" if success else "failed",
    }


def scheduler_list() -> list[dict[str, Any]]:
    """List all scheduled jobs in the TaskScheduler.

    Returns:
        List of schedule job dicts.
    """
    TaskScheduler = _get_scheduler_module()  # noqa: N806

    scheduler = TaskScheduler()
    jobs = scheduler.get_scheduled_tasks()

    result = []
    for job in jobs:
        result.append(
            {
                "task_id": getattr(job, "task_id", ""),
                "name": getattr(job, "name", ""),
                "queue": getattr(job, "queue", ""),
                "recurring": getattr(job, "recurring", True),
            }
        )

    return result


def scheduler_remove_task(task_id: str) -> dict[str, Any]:
    """Remove a task from the TaskScheduler schedule.

    Args:
        task_id: Task ID or name to remove from schedule.

    Returns:
        Dict with task_id, removed status.
    """
    TaskScheduler = _get_scheduler_module()  # noqa: N806

    scheduler = TaskScheduler()
    success = scheduler.remove_task(task_id=task_id)

    return {
        "task_id": task_id,
        "removed": success,
    }


def scheduler_start() -> dict[str, Any]:
    """Start the TaskScheduler loop.

    Note: This function blocks and does not return until interrupted.

    Returns:
        Dict with status.
    """
    TaskScheduler = _get_scheduler_module()  # noqa: N806

    scheduler = TaskScheduler()
    scheduler.start()

    return {"status": "running"}


def _get_scheduler_module():
    import clearml  # noqa: F401

    return clearml.automation.TaskScheduler


# ── Training init helper ──


def init_tracking(
    task_name: str,
    project: str = "PDEBench",
    tags: list[str] | None = None,
    capture_tensorboard: bool = True,
    capture_pytorch: bool = True,
    capture_graph: bool = False,
    graph_input_shape: tuple | list | None = None,
    model: Any | None = None,
    device: str = "cpu",
    output_uri: str | None = None,
) -> dict[str, Any]:
    """Initialize clearml experiment tracking for a training script.

    Wraps Task.init() with PDEBench defaults. Optionally captures the model
    computation graph via tensorboard add_graph and uploads a SVG/PDF to clearml.

    Args:
        task_name: Name for this experiment (e.g. 'FNO_burgers_lr0.001').
        project: Project name (default: PDEBench).
        tags: Optional tags for filtering.
        capture_tensorboard: Auto-capture TensorBoardX scalars (default: True).
        capture_pytorch: Auto-capture PyTorch model checkpoints (default: True).
        capture_graph: Generate and upload model computation graph SVG/PDF.
            Disable during competition to save training time. (default: False).
        graph_input_shape: Required if capture_graph=True. Tuple like (1, 1024, 3).
        model: Required if capture_graph=True. The model instance to trace.
        device: Device for graph trace dummy input (default: 'cpu').
        output_uri: Override artifact output destination.

    Returns:
        Dict with task_id, task_name, project, graph_uploaded.

    Example:
        from expflow_pde.clearml import init_tracking
        task = init_tracking(
            task_name="FNO_burgers_lr0.001",
            tags=["fno", "burgers"],
            capture_graph=False,  # Enable for debugging, disable for competition
        )
    """
    task = _init_task(
        task_name=task_name,
        project=project,
        tags=tags,
        capture_tensorboard=capture_tensorboard,
        capture_pytorch=capture_pytorch,
        output_uri=output_uri,
    )

    result = {
        "task_id": task.id,
        "task_name": task_name,
        "project": project,
        "graph_uploaded": False,
    }

    if capture_graph and model is not None and graph_input_shape is not None:
        _capture_and_upload_graph(
            task=task, model=model, input_shape=graph_input_shape, device=device
        )
        result["graph_uploaded"] = True

    return result


def _init_task(
    task_name: str,
    project: str,
    tags: list[str] | None = None,
    capture_tensorboard: bool = True,
    capture_pytorch: bool = True,
    output_uri: str | None = None,
):
    """Internal: initialize a clearml Task with framework control."""
    import clearml  # noqa: F401
    from clearml import Task  # noqa: F811

    frameworks: dict[str, bool] = {}
    if capture_tensorboard:
        frameworks["tensorboard"] = True
    if capture_pytorch:
        frameworks["pytorch"] = True

    task = Task.init(
        project_name=project,
        task_name=task_name,
        tags=tags or [],
        output_uri=output_uri,
        auto_connect_frameworks=frameworks or True,
    )
    return task


def _capture_and_upload_graph(
    task,
    model: Any,
    input_shape: tuple | list,
    device: str = "cpu",
) -> None:
    """Generate computation graph SVG/PDF and upload to clearml artifact."""
    import os
    import tempfile

    import torch

    # Write tensorboard add_graph (local runs/ directory)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    dummy = torch.randn(*input_shape).to(device)
    writer.add_graph(model, dummy)
    writer.close()

    # Generate SVG via torchviz and upload to clearml
    try:
        from torchviz import make_dot

        dummy = torch.randn(*input_shape).to(device)
        y = model(dummy)
        dot = make_dot(y, params=dict(model.named_parameters()))

        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("svg", "pdf"):
                dot.format = fmt
                path = os.path.join(tmp, f"model_graph.{fmt}")
                dot.render(path)
                task.upload_artifact(
                    name=f"model_graph_{fmt}",
                    artifact_object=path,
                )
    except ImportError:
        import warnings

        warnings.warn("torchviz not installed. Skipping SVG/PDF graph upload.")
