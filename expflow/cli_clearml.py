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


@clearml_app.command("compare")
def compare_cmd(
    task_id_a: str = typer.Argument(..., help="First task ID"),
    task_id_b: str = typer.Argument(..., help="Second task ID"),
) -> None:
    """Compare two tasks side by side."""
    from expflow.compare import compare_tasks

    result = compare_tasks(task_id_a, task_id_b)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    a, b = result["a"], result["b"]
    print(f"Comparison: {a['name']} vs {b['name']}")
    print()
    print(f"{'Field':<20} {'Task A':<30} {'Task B':<30}")
    print("-" * 80)
    print(f"{'Status':<20} {a['status']:<30} {b['status']:<30}")
    print(f"{'Project':<20} {a['project']:<30} {b['project']:<30}")
    print(
        f"{'Tags':<20} {', '.join(a.get('tags', []))[:27]:<30} {', '.join(b.get('tags', []))[:27]:<30}"
    )


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
    """[DEPRECATED] Register a PDEBench dataset with compliance annotation.

    Use `dataset-upload` or `annotate-compliance` instead.
    """
    print(
        "  [WARN] 'dataset-register' is deprecated. Use 'dataset-upload' to upload,"
        " or 'annotate-compliance' for metadata-only tagging."
    )
    # annotate_compliance requires a dataset_id, not a name/path
    # This is a best-effort: look up dataset by name or warn
    from expflow.clearml import annotate_compliance, list_datasets

    datasets = list_datasets(name_filter=name)
    ds = next((d for d in datasets if d["name"] == name), None)
    if not ds:
        print(f"  ERROR: No dataset found with name '{name}'. Use 'dataset-upload' first.")
        raise typer.Exit(code=1)

    result = annotate_compliance(
        dataset_id=ds["id"],
        compliance=compliance,  # type: ignore
    )
    print(f"Dataset registered: {ds['name']} v{ds.get('version', '?')}")
    print(f"  ID:         {result['id']}")
    print(f"  Compliance: {result['compliance']}")
    print(f"  Path:       {path}")


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


# ── Dataset upload / download / lineage commands ──


@clearml_app.command("dataset-upload")
def dataset_upload_cmd(
    local_path: str = typer.Argument(..., help="Path to local file or folder to upload"),
    dataset_name: str = typer.Argument(..., help="Dataset name in clearml"),
    project: str = typer.Option("PDEBench", "--project", "-p", help="Project name"),
    version: Optional[str] = typer.Option(
        None, "--version", "-v", help="Semantic version (auto if omitted)"
    ),
    parent_ids: Optional[str] = typer.Option(
        None, "--parent-ids", help="Comma-separated parent dataset IDs"
    ),
    compliance: Optional[str] = typer.Option(
        None, "--compliance", "-c", help="Compliance: allowed or forbidden"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Dataset description"
    ),
) -> None:
    """Upload local files to clearml Fileserver and register as a Dataset."""
    from expflow.clearml import dataset_upload

    parent_list = parent_ids.split(",") if parent_ids else None
    result = dataset_upload(
        local_path=local_path,
        dataset_name=dataset_name,
        dataset_project=project,
        version=version,
        parent_dataset_ids=parent_list,
        compliance=compliance,  # type: ignore
        description=description,
    )
    print(f"Dataset uploaded: {result['name']} v{result['version']}")
    print(f"  ID:         {result['id']}")
    print(f"  Compliance: {result.get('compliance', '-')}")
    print(f"  URI:        {result['uri']}")


@clearml_app.command("dataset-download")
def dataset_download_cmd(
    target_folder: str = typer.Argument(..., help="Local folder to download to"),
    dataset_id: Optional[str] = typer.Option(None, "--id", help="Dataset ID"),
    dataset_name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset name"),
    dataset_project: Optional[str] = typer.Option(
        "PDEBench", "--project", "-p", help="Project name"
    ),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Specific version"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing target folder"),
) -> None:
    """Download a Dataset from clearml Fileserver to a local folder."""
    from expflow.clearml import dataset_download

    result = dataset_download(
        target_folder=target_folder,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=version,
        overwrite=overwrite,
    )
    print(f"Dataset downloaded: {result['name']} v{result['version']}")
    print(f"  ID:         {result['id']}")
    print(f"  Local path: {result['local_path']}")


@clearml_app.command("dataset-lineage")
def dataset_lineage_cmd(
    dataset_id: str = typer.Argument(..., help="Dataset ID to trace lineage from"),
    depth: int = typer.Option(10, "--depth", "-d", help="Max recursion depth"),
) -> None:
    """Trace dataset lineage via parent chain."""
    from expflow.clearml import dataset_lineage

    lineage = dataset_lineage(dataset_id=dataset_id, depth=depth)
    if not lineage:
        print(f"No lineage found for dataset {dataset_id}")
        return

    print(f"Lineage for dataset {dataset_id}:")
    print()
    print(f"{'ID':<24} {'NAME':<30} {'VERSION':<10} {'COMPLIANCE':<12}")
    print("-" * 76)
    for entry in lineage:
        comp = entry.get("compliance") or "-"
        print(f"{entry['id']:<24} {entry['name']:<30} {entry['version']:<10} {comp:<12}")


# ── Model commands ──


@clearml_app.command("model-list")
def model_list_cmd(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project name"),
    only_published: bool = typer.Option(False, "--published", help="Only published models"),
    max_results: int = typer.Option(20, "--max", "-m", help="Max results"),
) -> None:
    """List registered checkpoint models."""
    from expflow.clearml import model_list

    models = model_list(
        project_name=project,
        only_published=only_published,
        max_results=max_results,
    )
    if not models:
        print("No models found.")
        return

    print(f"{'ID':<24} {'NAME':<30} {'FRAMEWORK':<14} {'PROJECT':<20}")
    print("-" * 88)
    for m in models:
        print(
            f"{m['id']:<24} {m['name']:<30} {m.get('framework', '-'):<14} {m.get('project', '-'):<20}"
        )


@clearml_app.command("model-upload")
def model_upload_cmd(
    local_path: str = typer.Argument(..., help="Path to model weights file"),
    task_id: str = typer.Argument(..., help="ID of the task producing this model"),
    framework: str = typer.Option("PyTorch", "--framework", "-f", help="Framework name"),
    model_name: Optional[str] = typer.Option(None, "--name", "-n", help="Model display name"),
) -> None:
    """Upload a model checkpoint to clearml Model store."""
    from expflow.clearml import model_upload

    result = model_upload(
        local_path=local_path,
        task_id=task_id,
        framework=framework,
        model_name=model_name,
    )
    print(f"Model uploaded: {result['name']}")
    print(f"  ID:     {result['id']}")
    print(f"  Task:   {result['task_id']}")
    print(f"  URI:    {result.get('uri', '-')}")


# ── Pipeline commands ──


@clearml_app.command("pipeline-create")
def pipeline_create_cmd(
    name: str = typer.Argument(..., help="Pipeline name"),
    project: str = typer.Option("PDEBench", "--project", "-p", help="Project name"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Semantic version"),
    abort_on_failure: bool = typer.Option(False, "--abort-on-failure", help="Abort on failure"),
) -> None:
    """Create a pipeline controller."""
    from expflow.clearml import pipeline_create

    result = pipeline_create(
        name=name,
        project=project,
        version=version,
        abort_on_failure=abort_on_failure,
    )
    print(f"Pipeline created: {result['name']}")
    print(f"  Project: {result['project']}")
    print(f"  Version: {result['version']}")


@clearml_app.command("pipeline-add-step")
def pipeline_add_step_cmd(
    pipeline_name: str = typer.Argument(..., help="Pipeline name"),
    step_name: str = typer.Argument(..., help="Step name"),
    project: str = typer.Option("PDEBench", "--project", "-p", help="Project name"),
    base_task_id: Optional[str] = typer.Option(None, "--task-id", help="Base task ID to clone"),
    base_task_name: Optional[str] = typer.Option(None, "--task-name", "-n", help="Base task name"),
    parents: Optional[str] = typer.Option(
        None, "--parents", help="Comma-separated parent step names"
    ),
    execution_queue: Optional[str] = typer.Option(None, "--queue", "-q", help="Execution queue"),
) -> None:
    """Add a step to an existing pipeline."""
    from expflow.clearml import pipeline_add_step

    parent_list = parents.split(",") if parents else None
    result = pipeline_add_step(
        pipeline_name=pipeline_name,
        project=project,
        step_name=step_name,
        base_task_id=base_task_id,
        base_task_name=base_task_name,
        parents=parent_list,
        execution_queue=execution_queue,
    )
    print(f"Step added: {result['step_name']} -> {result['pipeline_name']}")
    print(f"  Parents: {result['parents']}")
    print(f"  Status:  {result['status']}")


@clearml_app.command("pipeline-start")
def pipeline_start_cmd(
    pipeline_name: str = typer.Argument(..., help="Pipeline name"),
    project: str = typer.Option("PDEBench", "--project", "-p", help="Project name"),
    queue: Optional[str] = typer.Option(None, "--queue", "-q", help="Execution queue"),
) -> None:
    """Start a pipeline controller."""
    from expflow.clearml import pipeline_start

    result = pipeline_start(
        pipeline_name=pipeline_name,
        project=project,
        queue_name=queue,
    )
    print(f"Pipeline started: {result['pipeline_name']}")
    print(f"  Status: {result['status']}")


@clearml_app.command("pipeline-stop")
def pipeline_stop_cmd(
    pipeline_name: str = typer.Argument(..., help="Pipeline name"),
    project: str = typer.Option("PDEBench", "--project", "-p", help="Project name"),
) -> None:
    """Stop a running pipeline."""
    from expflow.clearml import pipeline_stop

    result = pipeline_stop(
        pipeline_name=pipeline_name,
        project=project,
    )
    print(f"Pipeline stopped: {result['pipeline_name']}")
    print(f"  Status: {result['status']}")


@clearml_app.command("pipeline-list")
def pipeline_list_cmd(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project"),
    max_results: int = typer.Option(20, "--max", "-m", help="Max results"),
) -> None:
    """List pipeline controller tasks."""
    from expflow.clearml import pipeline_list

    pipelines = pipeline_list(
        project_name=project,
        max_results=max_results,
    )

    if not pipelines:
        print("No pipelines found.")
        return

    print(f"{'ID':<24} {'NAME':<30} {'STATUS':<14} {'PROJECT':<20}")
    print("-" * 88)
    for p in pipelines:
        print(f"{p['id']:<24} {p['name']:<30} {p['status']:<14} {p['project']:<20}")
