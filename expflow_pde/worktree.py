#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow git worktree management — create experiment branches for remote dispatch.

Manages temporary git worktrees so users can submit experiments with WIP code
without committing unfinished changes to the main branch.

Flow:
    1. git branch exp_<id> from main
    2. git worktree add /tmp/expflow-<id> exp_<id>
    3. Copy selected files (or all) into worktree
    4. git commit + git push -u origin exp_<id>
    5. clearml Task.create(branch=exp_<id>)
    6. cleanup: git worktree remove + branch -D + remote delete
"""

import os
import shutil
import subprocess
import uuid

import expflow_pde.config as config

# ── helpers ──


def _find_git_root(path: str | None = None) -> str:
    """Find the git repo root from cwd or given path."""
    cwd = path or os.getcwd()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Not a git repository: {cwd}")
    return result.stdout.strip()


def _run_git(args: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    return subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=check,
        timeout=30,
    )


def _run_quiet(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run a command silently — ignore errors."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)


# ── public API ──


def create_experiment_worktree(
    repo_path: str | None = None,
    include_files: list[str] | None = None,
    branch_prefix: str = "exp",
    worktree_base: str | None = None,
    skip_push: bool = False,
) -> dict:
    """Create a temporary worktree for experiment submission.

    Args:
        repo_path: Git repo path (default: cwd).
        include_files: Only copy these files (default: all except .git/__pycache__/).
        branch_prefix: Branch name prefix (default: 'exp').
        worktree_base: Worktree parent dir (default: /tmp/expflow/).
        skip_push: Skip git push (for testing).

    Returns:
        Dict with 'worktree_path', 'branch', 'commit_hash'.

    Raises:
        RuntimeError: If git operations fail.
    """
    cfg = config.load_config()
    repo = repo_path or _find_git_root()
    exp_id = uuid.uuid4().hex[:8]
    branch = f"{branch_prefix}_{exp_id}"
    base = worktree_base or cfg.get("worktree.dir", "/tmp/expflow")
    wt_path = os.path.join(base, f"{branch_prefix}_{exp_id}")

    # Ensure base dir exists
    os.makedirs(base, exist_ok=True)

    # 1. Create branch from main (force clean if exists from prior failed attempt)
    _run_quiet(["git", "branch", "-D", branch], cwd=repo)
    _run_git(["branch", branch, "main"], cwd=repo)

    # 2. Add worktree
    _run_git(["worktree", "add", wt_path, branch], cwd=repo)

    try:
        # 3. Copy files
        if not os.path.isdir(wt_path):
            os.makedirs(wt_path, exist_ok=True)

        if include_files:
            for f in include_files:
                src = os.path.join(repo, f)
                dst = os.path.join(wt_path, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        else:
            # Copy all except git/pycache
            for item in os.listdir(repo):
                if item in (
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    ".ruff_cache",
                    "checkpoints",
                    "data_old",
                    "data_new2",
                    "venv",
                    ".venv",
                ):
                    continue
                src = os.path.join(repo, item)
                dst = os.path.join(wt_path, item)
                if os.path.isdir(src):
                    shutil.copytree(
                        src,
                        dst,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(
                            "__pycache__", ".git", "*.pyc", "checkpoints", "data_*"
                        ),
                    )
                else:
                    shutil.copy2(src, dst)

        # 4. Commit
        _run_git(["add", "-A"], cwd=wt_path)
        result = _run_git(["diff", "--cached", "--quiet"], cwd=wt_path, check=False)
        if result.returncode != 0:
            _run_git(
                ["commit", "-m", f"expflow: {branch}\n\nExperiment snapshot for remote dispatch."],
                cwd=wt_path,
            )
        else:
            # No changes — create empty commit
            _run_git(
                ["commit", "--allow-empty", "-m", f"expflow: {branch} (empty snapshot)"],
                cwd=wt_path,
            )

        # 5. Push
        if not skip_push:
            _run_git(["push", "-u", "origin", branch], cwd=wt_path)

        # 6. Get commit hash
        commit = _run_git(["rev-parse", "HEAD"], cwd=wt_path).stdout.strip()[:8]

        # 7. Record in registry for later cleanup
        _record_worktree(repo, branch, wt_path)

        return {
            "worktree_path": wt_path,
            "branch": branch,
            "commit_hash": commit,
            "experiment_id": exp_id,
        }

    except Exception:
        # Cleanup on failure
        _run_quiet(["git", "worktree", "remove", "--force", wt_path], cwd=repo)
        _run_quiet(["git", "branch", "-D", branch], cwd=repo)
        raise


def cleanup_worktree(
    branch: str,
    worktree_path: str,
    repo_path: str | None = None,
    skip_remote: bool = False,
) -> dict:
    """Remove a worktree and its local/remote branches.

    Args:
        branch: Branch name to delete.
        worktree_path: Worktree directory to remove.
        repo_path: Git repo path (default: auto-detect).
        skip_remote: Skip remote branch deletion.

    Returns:
        Dict with 'branch', 'status'.
    """
    repo = repo_path or _find_git_root()

    _run_quiet(["git", "worktree", "remove", "--force", worktree_path], cwd=repo)
    _run_quiet(["git", "branch", "-D", branch], cwd=repo)
    if not skip_remote:
        _run_quiet(["git", "push", "origin", "--delete", branch], cwd=repo)

    # Remove from registry
    _remove_worktree_record(branch)

    return {"branch": branch, "status": "cleaned"}


def list_worktrees(repo_path: str | None = None) -> list[dict]:
    """List expflow-managed worktrees.

    Returns:
        List of dicts with 'branch', 'worktree_path', 'created_at'.
    """
    records = _load_worktree_registry(repo_path)
    # Verify each worktree still exists
    result = []
    for r in records:
        if os.path.isdir(r.get("worktree_path", "")):
            result.append(r)
    return result


def cleanup_all_stale(repo_path: str | None = None, max_age_hours: int = 24) -> list[dict]:
    """Clean all stale worktrees older than max_age_hours.

    Args:
        repo_path: Git repo path.
        max_age_hours: Max age in hours before cleanup (default: 24).

    Returns:
        List of cleaned worktree dicts.
    """
    import time

    repo = repo_path or _find_git_root()
    cleaned = []
    for wt in list_worktrees(repo):
        created = wt.get("created_at", 0)
        if created and (time.time() - created) > max_age_hours * 3600:
            result = cleanup_worktree(wt["branch"], wt["worktree_path"], repo)
            cleaned.append(result)
    return cleaned


# ── persistence: ~/.expflow/worktrees.jsonl ──


def _registry_path(repo_path: str) -> str:
    """Get the worktree registry file path."""
    repo = repo_path or _find_git_root()
    expflow_dir = os.path.join(os.path.dirname(repo), ".expflow")
    os.makedirs(expflow_dir, exist_ok=True)
    return os.path.join(expflow_dir, "worktrees.jsonl")


def _record_worktree(repo: str, branch: str, wt_path: str) -> None:
    """Append a worktree record to the registry."""
    import json
    import time

    record = {
        "branch": branch,
        "worktree_path": wt_path,
        "created_at": time.time(),
    }
    rpath = _registry_path(repo)
    with open(rpath, "a") as f:
        f.write(json.dumps(record) + "\n")


def _remove_worktree_record(branch: str) -> None:
    """Remove a worktree record from the registry by branch name."""
    import json

    repo = _find_git_root()
    rpath = _registry_path(repo)
    if not os.path.isfile(rpath):
        return
    records = []
    with open(rpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("branch") != branch:
                records.append(rec)
    with open(rpath, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_worktree_registry(repo_path: str | None = None) -> list[dict]:
    """Load all worktree records from the registry."""
    import json

    repo = repo_path or _find_git_root()
    rpath = _registry_path(repo)
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


# ── stash mode (lighter alternative) ──


def submit_via_stash(
    script_path: str,
    repo_path: str | None = None,
    skip_push: bool = False,
) -> dict:
    """Create a temporary experiment branch using git stash (lighter than worktree).

    Use when there are few changes and you're confident stash pop won't conflict.
    """
    repo = repo_path or _find_git_root()
    exp_id = uuid.uuid4().hex[:8]
    branch = f"exp_{exp_id}"

    # 1. Stash current WIP
    stash_result = _run_git(["stash", "push", "-m", f"expflow: {exp_id}"], cwd=repo, check=False)

    try:
        # 2. Create branch from current HEAD (clean after stash)
        _run_quiet(["git", "branch", "-D", branch], cwd=repo)
        _run_git(["checkout", "-b", branch], cwd=repo)

        # 3. Commit
        result = _run_git(["diff", "--cached", "--quiet"], cwd=repo, check=False)
        if result.returncode == 0:
            _run_git(
                ["commit", "--allow-empty", "-m", f"expflow: {branch} (current HEAD snapshot)"],
                cwd=repo,
            )
        else:
            _run_git(["commit", "-m", f"expflow: {branch}"], cwd=repo)

        # 4. Push
        if not skip_push:
            _run_git(["push", "-u", "origin", branch], cwd=repo)

        commit = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()[:8]

        return {
            "branch": branch,
            "commit_hash": commit,
            "experiment_id": exp_id,
            "method": "stash",
        }
    finally:
        # Always restore original branch + unstash
        _run_quiet(["git", "checkout", "-"], cwd=repo)  # checkout previous branch
        if stash_result.returncode == 0:
            _run_quiet(["git", "stash", "pop"], cwd=repo)
        # Cleanup remote branch if we pushed but something went wrong after
        if not skip_push:
            _run_quiet(["git", "push", "origin", "--delete", branch], cwd=repo)
        _run_quiet(["git", "branch", "-D", branch], cwd=repo)
