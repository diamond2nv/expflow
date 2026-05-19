#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow experiment dispatcher — submit, list, track experiments.

Supports three submission modes:
1. default — shell-level experiment tracking with in-memory registry
2. clearml — submits via clearml Task.create() + enqueue to queue
3. worktree — creates git worktree branch + clearml Task.enqueue
4. stash — lighter alternative to worktree (git stash + branch)
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

# In-memory experiment registry (volatile — for CLI demo/testing)
# In production, experiments live in clearml tasks + persistent .jsonl
_experiments: dict[str, dict[str, Any]] = {}


# ── Persisted experiment registry ──


def _registry_path() -> str:
    """Get the experiment registry file path."""
    expflow_dir = os.path.expanduser("~/.expflow")
    os.makedirs(expflow_dir, exist_ok=True)
    return os.path.join(expflow_dir, "experiments.jsonl")


def _load_registry() -> list[dict[str, Any]]:
    """Load all experiment records from persistent storage."""
    rpath = _registry_path()
    if not os.path.isfile(rpath):
        return []
    records = []
    with open(rpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _save_to_registry(record: dict[str, Any]) -> None:
    """Append a record to the persistent registry."""
    rpath = _registry_path()
    with open(rpath, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Also keep in-memory
    _experiments[record["experiment_id"]] = record


# ── Main dispatch ──


def dispatch_experiment(
    command: str,
    queue: str = "default",
    tags: Optional[list[str]] = None,
    project: str = "expflow",
    use_clearml_task: bool = False,
    use_worktree: bool = False,
    use_stash: bool = False,
    include_files: Optional[list[str]] = None,
    script_args: Optional[dict[str, str]] = None,
    git_push: bool = True,
) -> dict[str, Any]:
    """Submit an experiment for execution.

    Args:
        command: The shell command or script path to run.
        queue: Target clearml queue name.
        tags: Optional experiment tags.
        project: ClearML project name.
        use_clearml_task: Submit via clearml Task.create() + enqueue.
        use_worktree: Create git worktree branch for clean submission.
        use_stash: Use git stash (lighter than worktree).
        include_files: Files to include in worktree (worktree mode only).
        script_args: Dict of hyperparameters (e.g. {'lr': '0.001'}).
        git_push: Whether to git push before enqueue (default: True).

    Returns:
        Dict with experiment metadata.
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

    if use_worktree:
        # Git worktree mode: creates isolated branch, then clearml Task
        clearml_result = _submit_via_worktree(
            script_path=command,
            project=project,
            queue=queue,
            tags=tags,
            include_files=include_files,
            script_args=script_args,
            git_push=git_push,
        )
        record["status"] = "queued"
        record["method"] = "worktree"
        record.update(clearml_result)
    elif use_stash:
        # Git stash mode: lighter than worktree
        clearml_result = _submit_via_stash(
            script_path=command,
            project=project,
            queue=queue,
            tags=tags,
            script_args=script_args,
            git_push=git_push,
        )
        record["status"] = "queued"
        record["method"] = "stash"
        record.update(clearml_result)
    elif use_clearml_task and is_python_script(command):
        # Standard clearml task submission
        clearml_result = _submit_via_clearml_task(
            script_path=command,
            project=project,
            queue=queue,
            tags=tags,
            script_args=script_args,
            git_push=git_push,
        )
        record["status"] = "queued"
        record["method"] = "clearml"
        record.update(clearml_result)
    else:
        record["method"] = "shell"

    _save_to_registry(record)
    return dict(record)


def is_python_script(command: str) -> bool:
    """Check if a command string is a Python script path."""
    return command.endswith(".py") and " " not in command.strip()


# ── Submission methods ──


def _submit_via_clearml_task(
    script_path: str,
    project: str,
    queue: str,
    tags: Optional[list[str]] = None,
    script_args: Optional[dict[str, str]] = None,
    git_push: bool = True,
) -> dict[str, Any]:
    """Submit a Python script via clearml Task.create() with auto-injected Task.init.

    The script is patched to include Task.init() automatically.
    """
    try:
        from clearml import Task
    except ImportError:
        return {"error": "clearml not installed"}

    # Auto git push if enabled (ensures agent can clone fresh code)
    if git_push:
        _auto_git_push(script_path)

    task = Task.create(
        project_name=project,
        task_name=script_path,
        repo=script_path,
        script=script_path,
        add_task_init_call=True,
    )

    if script_args:
        argparse_args = [(f"--{k}", v) for k, v in script_args.items()]
        task.set_parameters({"Args": argparse_args})

    if queue:
        Task.enqueue(task=task, queue_name=queue)

    return {
        "task_id": task.id,
        "clearml_task_id": task.id,
        "queue": queue,
    }


def _submit_via_worktree(
    script_path: str,
    project: str,
    queue: str,
    tags: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    script_args: Optional[dict[str, str]] = None,
    git_push: bool = True,
) -> dict[str, Any]:
    """Submit via git worktree + clearml task.

    Creates a temporary git worktree branch with selected files,
    pushes the branch, then creates a clearml Task pointing to that branch.
    The worktree is cleaned up after the task is enqueued.
    """
    try:
        from clearml import Task
    except ImportError:
        return {"error": "clearml not installed"}

    from expflow_pde.worktree import cleanup_worktree, create_experiment_worktree

    # 1. Create worktree
    wt_result = create_experiment_worktree(
        include_files=include_files,
        skip_push=not git_push,
    )

    branch = wt_result["branch"]
    commit_hash = wt_result["commit_hash"]

    # 2. Create clearml Task pointing to experiment branch
    task = Task.create(
        project_name=project,
        task_name=f"{os.path.basename(script_path)} ({branch})",
        branch=branch,
        script=script_path,
        add_task_init_call=True,
    )

    if script_args:
        argparse_args = [(f"--{k}", v) for k, v in script_args.items()]
        task.set_parameters({"Args": argparse_args})

    if tags:
        task.set_tags(tags)

    # 3. Enqueue
    if queue:
        Task.enqueue(task=task, queue_name=queue)

    # 4. Cleanup worktree (after enqueue, agent clones branch independently)
    repo_path = None
    try:
        from expflow_pde.worktree import _find_git_root

        repo_path = _find_git_root()
    except RuntimeError:
        pass

    if repo_path:
        cleanup_worktree(branch, wt_result["worktree_path"], repo_path)

    return {
        "task_id": task.id,
        "clearml_task_id": task.id,
        "branch": branch,
        "commit_hash": commit_hash,
        "queue": queue,
    }


def _submit_via_stash(
    script_path: str,
    project: str,
    queue: str,
    tags: Optional[list[str]] = None,
    script_args: Optional[dict[str, str]] = None,
    git_push: bool = True,
) -> dict[str, Any]:
    """Submit via git stash + clearml task (lighter than worktree).

    Stashes WIP changes, creates a temporary branch, pushes it,
    creates a clearml Task, then restores WIP via stash pop.
    """
    try:
        from clearml import Task
    except ImportError:
        return {"error": "clearml not installed"}

    from expflow_pde.worktree import submit_via_stash as _stash_and_branch

    # 1. Stash + branch
    stash_result = _stash_and_branch(
        script_path=script_path,
        skip_push=not git_push,
    )

    branch = stash_result["branch"]
    commit_hash = stash_result["commit_hash"]

    # 2. Create clearml Task
    task = Task.create(
        project_name=project,
        task_name=f"{os.path.basename(script_path)} ({branch})",
        branch=branch,
        script=script_path,
        add_task_init_call=True,
    )

    if script_args:
        argparse_args = [(f"--{k}", v) for k, v in script_args.items()]
        task.set_parameters({"Args": argparse_args})

    if tags:
        task.set_tags(tags)

    if queue:
        Task.enqueue(task=task, queue_name=queue)

    return {
        "clearml_task_id": task.id,
        "branch": branch,
        "commit_hash": commit_hash,
        "queue": queue,
    }


def _auto_git_push(script_path: str) -> None:
    """Auto git commit + push if there are uncommitted changes.

    Only commits the specific script file and any explicitly staged files.
    This is a best-effort operation — silent on failures.
    """
    import subprocess
    import sys

    try:
        repo_dir = os.path.dirname(os.path.abspath(script_path))
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=10,
        )
        if not result.stdout.strip():
            return  # clean, nothing to push

        # Stage and commit
        subprocess.run(
            ["git", "add", script_path],
            capture_output=True,
            cwd=repo_dir,
            timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", "expflow: auto-push for experiment submission"],
            capture_output=True,
            cwd=repo_dir,
            timeout=10,
        )
        subprocess.run(
            ["git", "push"],
            capture_output=True,
            cwd=repo_dir,
            timeout=30,
        )
    except Exception as exc:
        # Best-effort only — don't fail the submission
        print(f"  [expflow] git push skipped: {exc}", file=sys.stderr)


# ── Query methods ──


def list_experiments(include_registry: bool = True) -> list[dict[str, Any]]:
    """List all dispatched experiments.

    Args:
        include_registry: Include persisted experiments from ~/.expflow/
                          in addition to in-memory ones.

    Returns:
        List of experiment records (most recent first).
    """
    records: list[dict[str, Any]] = list(_experiments.values())

    if include_registry:
        persisted = _load_registry()
        existing_ids = {r["experiment_id"] for r in records}
        for r in persisted:
            if r["experiment_id"] not in existing_ids:
                records.append(r)

    # Sort by timestamp descending
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records


def get_experiment_status(experiment_id: str) -> dict[str, Any]:
    """Get status of a specific experiment.

    Checks in-memory first, then persisted registry, then clearml API.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Experiment record or error dict.
    """
    # Check in-memory
    record = _experiments.get(experiment_id)
    if record:
        return dict(record)

    # Check persistent registry
    for r in _load_registry():
        if r.get("experiment_id") == experiment_id:
            return dict(r)

    return {"experiment_id": experiment_id, "error": "not found"}


def cancel_experiment(experiment_id: str) -> dict[str, Any]:
    """Cancel a running experiment.

    Args:
        experiment_id: The experiment ID.

    Returns:
        Dict confirming cancellation or error.
    """
    record = _experiments.get(experiment_id)
    if record is None:
        # Check persistent
        for r in _load_registry():
            if r.get("experiment_id") == experiment_id:
                record = r
                break
    if record is None:
        return {"experiment_id": experiment_id, "error": "not found"}

    # Try to dequeue from clearml if there's a task_id
    task_id = record.get("clearml_task_id")
    if task_id:
        try:
            from clearml import Task as ClearmlTask

            task = ClearmlTask.get_task(task_id=task_id)
            task.dequeue()
            task.mark_stopped()
            record["status"] = "cancelled"
        except Exception as exc:
            return {"experiment_id": experiment_id, "error": str(exc)}
    else:
        record["status"] = "cancelled"
        _experiments[experiment_id] = record

    return {"experiment_id": experiment_id, "status": "cancelled"}
