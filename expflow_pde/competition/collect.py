"""Collect output artifacts from a completed clearml pipeline.

After a training+eval pipeline completes on a remote clearml agent (e.g. 5090),
the eval script should upload result files (pred.hdf5, time.csv) as clearml
artifacts via task.upload_artifact(). This module downloads them.

Usage:
    from expflow_pde.competition.collect import collect_pipeline_artifacts

    manifest = collect_pipeline_artifacts(
        pipeline_id="pipe_abc123",
        output_dir="~/.competition/artifacts",
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def collect_pipeline_artifacts(
    pipeline_id: str,
    output_dir: str = "~/.competition/artifacts",
    step_names: list[str] | None = None,
) -> dict[str, Any]:
    """Download artifacts from all steps of a completed pipeline.

    Expects each pipeline step to have uploaded artifacts via
    task.upload_artifact(name, file_object) in the training/eval scripts.

    Args:
        pipeline_id: clearml pipeline controller task ID.
        output_dir: Local directory to write artifacts to.
        step_names: Only download from specific step names (e.g. ['eval']).
            If None, downloads from all steps.

    Returns:
        Manifest dict with:
            pipeline_id: str
            steps: list of step results (name, task_id, artifacts, output_dir)
            total_artifacts: int
            output_base: str
    """
    from clearml import Task

    output_base = Path(output_dir).expanduser()
    output_base.mkdir(parents=True, exist_ok=True)

    # Get pipeline controller
    try:
        pipe_task = Task.get_task(task_id=pipeline_id)
    except Exception as e:
        return {"pipeline_id": pipeline_id, "error": f"Cannot find pipeline: {e}", "artifacts": []}

    # Get pipeline model
    from expflow_pde.clearml import _get_pipeline_module

    pipeline_cls = _get_pipeline_module()
    try:
        pipe = pipeline_cls(
            name=pipe_task.name,
            project=pipe_task.project if hasattr(pipe_task, "project") else "",
            version=pipe_task.version if hasattr(pipe_task, "version") else "0.1.0",
        )
    except Exception as e:
        return {"pipeline_id": pipeline_id, "error": f"Cannot create pipeline controller: {e}"}

    step_results: list[dict[str, Any]] = []
    total_artifacts = 0

    # List available steps
    try:
        available_steps = pipe.steps if hasattr(pipe, "steps") else pipe.get("steps", {})
    except Exception:
        available_steps = {}

    step_filter = set(step_names) if step_names else None

    for step_name, step_info in available_steps.items():
        if step_filter and step_name not in step_filter:
            continue

        step_task_id = None
        if isinstance(step_info, dict):
            step_task_id = step_info.get("id") or step_info.get("task_id")
        elif hasattr(step_info, "id"):
            step_task_id = step_info.id

        if not step_task_id:
            continue

        try:
            step_task = Task.get_task(task_id=step_task_id)
        except Exception:
            continue

        step_output = output_base / step_name
        step_output.mkdir(parents=True, exist_ok=True)

        step_artifacts: list[dict[str, Any]] = []
        try:
            arts = step_task.artifacts
            for art_name, art_obj in arts.items():
                try:
                    local_path = art_obj.get_local_copy(
                        os.path.join(str(step_output), art_name)
                    )
                    step_artifacts.append({
                        "name": art_name,
                        "local_path": str(local_path),
                        "size_bytes": os.path.getsize(local_path) if os.path.isfile(local_path) else 0,
                    })
                    total_artifacts += 1
                except Exception as e:
                    step_artifacts.append({
                        "name": art_name,
                        "error": str(e),
                    })
        except Exception:
            pass

        step_results.append({
            "step_name": step_name,
            "task_id": step_task_id,
            "artifacts": step_artifacts,
            "local_dir": str(step_output),
        })

    # Write manifest
    manifest: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "steps": step_results,
        "total_artifacts": total_artifacts,
        "output_base": str(output_base),
    }

    manifest_path = output_base / "collect_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    return manifest


def collect_task_artifacts(
    task_id: str,
    output_dir: str = "~/.competition/artifacts",
    label: str = "artifacts",
) -> dict[str, Any]:
    """Download artifacts from a single clearml task.

    Useful for direct task-to-local download without pipeline wrapping.

    Args:
        task_id: clearml task ID.
        output_dir: Local directory to write artifacts to.
        label: Subdirectory name within output_dir.

    Returns:
        Dict with task_id, artifacts list.
    """
    from clearml import Task

    output_base = Path(output_dir).expanduser() / label
    output_base.mkdir(parents=True, exist_ok=True)

    try:
        task = Task.get_task(task_id=task_id)
    except Exception as e:
        return {"task_id": task_id, "error": str(e), "artifacts": []}

    artifacts_list: list[dict[str, Any]] = []

    try:
        arts = task.artifacts
        for art_name, art_obj in arts.items():
            try:
                local_path = art_obj.get_local_copy(
                    os.path.join(str(output_base), art_name)
                )
                artifacts_list.append({
                    "name": art_name,
                    "local_path": str(local_path),
                    "size_bytes": os.path.getsize(local_path) if os.path.isfile(local_path) else 0,
                })
            except Exception as e:
                artifacts_list.append({
                    "name": art_name,
                    "error": str(e),
                })
    except Exception:
        pass

    return {"task_id": task_id, "artifacts": artifacts_list}
