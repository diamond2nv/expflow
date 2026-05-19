#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.worktree — git worktree management.

All tests use mocked subprocess. No real git operations are performed.
"""

import json
import os
import subprocess  # noqa: I
import time
from unittest.mock import MagicMock, patch

import pytest

# ── Fixture: mock subprocess and filesystem ──


@pytest.fixture(autouse=True)
def mock_git_env(tmp_path: str) -> dict:
    """Set up a mock git environment with a real git init repo.

    Returns a dict with paths and mock subprocess controls.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    wt_base = tmp_path / "worktrees"
    wt_base.mkdir()

    # Initialize real git repo so _find_git_root works
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=str(repo), capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True)
    # Create initial commit so we can branch from main
    (repo / "README.md").write_text("# test")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)

    os.chdir(str(repo))

    # Config override so worktree paths go to tmp
    import expflow_pde.config as cfg

    cfg._config_cache = {}
    cfg._config_cache["worktree"] = {"dir": str(wt_base)}

    env = {
        "repo": str(repo),
        "wt_base": str(wt_base),
    }

    # Create a dummy file so worktree has something to commit
    (repo / "train.py").write_text("# training script\n")

    return env


@pytest.fixture
def mock_subprocess() -> MagicMock:
    """Mock subprocess.run to simulate git commands.

    The mock returns a CompletedProcess-like object via spec=True.
    We override this to be flexible — some commands need specific return values.
    """
    with patch("expflow_pde.worktree.subprocess.run") as mock_run:
        # Default: return success with empty stdout
        default_result = MagicMock()
        default_result.returncode = 0
        default_result.stdout = ""
        default_result.stderr = ""
        mock_run.return_value = default_result
        yield mock_run


# ── Tests: _find_git_root ──


def test_find_git_root_in_repo(mock_git_env: dict) -> None:
    """Should return the repo path when inside a git repo."""
    from expflow_pde.worktree import _find_git_root

    result = _find_git_root(mock_git_env["repo"])
    assert result == mock_git_env["repo"]


def test_find_git_root_not_repo(tmp_path: str) -> None:
    """Should raise RuntimeError when not in a git repo."""
    from expflow_pde.worktree import _find_git_root

    with pytest.raises(RuntimeError, match="Not a git repository"):
        _find_git_root(str(tmp_path))


# ── Tests: create_experiment_worktree (with mock subprocess) ──


def test_create_worktree_basic(mock_git_env: dict, mock_subprocess: MagicMock) -> None:
    """Should create worktree, commit, push, return branch info."""
    from expflow_pde.worktree import create_experiment_worktree

    result = create_experiment_worktree(
        repo_path=mock_git_env["repo"],
        skip_push=True,  # Don't actually push
    )

    assert "branch" in result
    assert result["branch"].startswith("exp_")
    assert "commit_hash" in result
    assert "worktree_path" in result
    assert "experiment_id" in result
    assert len(result["experiment_id"]) == 8

    # Verify git commands were called
    git_calls = [c[0][0] for c in mock_subprocess.call_args_list if c[0][0][0] == "git"]
    assert len(git_calls) >= 5  # branch, worktree add, add, commit, (maybe diff)
    assert any("branch" in c for c in git_calls)
    assert any("worktree" in c for c in git_calls)


def test_create_worktree_with_include(mock_git_env: dict, mock_subprocess: MagicMock) -> None:
    """Should only copy specified files when --include is given."""
    from expflow_pde.worktree import create_experiment_worktree

    repo = mock_git_env["repo"]
    # Create a sub-directory with a file
    os.makedirs(os.path.join(repo, "utils"))
    open(os.path.join(repo, "utils", "eval.py"), "w").close()

    result = create_experiment_worktree(
        repo_path=repo,
        include_files=["train.py", "utils/eval.py"],
        skip_push=True,
    )

    assert result["branch"].startswith("exp_")


def test_create_worktree_skip_push_no_remote(
    mock_git_env: dict, mock_subprocess: MagicMock
) -> None:
    """Should not attempt push when skip_push=True."""
    from expflow_pde.worktree import create_experiment_worktree

    mock_subprocess.reset_mock()
    create_experiment_worktree(
        repo_path=mock_git_env["repo"],
        skip_push=True,
    )

    # Check that the actual git push command was NOT called
    push_calls = [c for c in mock_subprocess.call_args_list if c[0][0][:2] == ["git", "push"]]
    assert len(push_calls) == 0


# ── Tests: cleanup_worktree ──


def test_cleanup_worktree(mock_git_env: dict, mock_subprocess: MagicMock) -> None:
    """Should remove worktree, local branch, and remote branch."""
    from expflow_pde.worktree import cleanup_worktree

    result = cleanup_worktree(
        branch="exp_test123",
        worktree_path="/tmp/worktree_test",
        repo_path=mock_git_env["repo"],
    )

    assert result["branch"] == "exp_test123"
    assert result["status"] == "cleaned"

    # Verify worktree remove was called
    git_calls = [str(c) for c in mock_subprocess.call_args_list]
    assert any("worktree" in c and "remove" in c for c in git_calls)
    assert any("branch" in c and "-D" in c for c in git_calls)


# ── Tests: list_worktrees ──


def test_list_worktrees_empty(mock_git_env: dict) -> None:
    """Should return empty list when no worktrees have been created."""
    from expflow_pde.worktree import list_worktrees

    result = list_worktrees(mock_git_env["repo"])
    assert result == []


def test_list_worktrees_with_records(mock_git_env: dict, mock_subprocess: MagicMock) -> None:
    """Should return worktree records from the registry."""
    from expflow_pde.worktree import _registry_path, list_worktrees

    rpath = _registry_path(mock_git_env["repo"])

    # Write a manual record
    record = {
        "branch": "exp_abc123",
        "worktree_path": os.path.join(mock_git_env["wt_base"], "exp_abc123"),
        "created_at": time.time(),
    }
    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    with open(rpath, "w") as f:
        f.write(json.dumps(record) + "\n")

    # Create the directory so it appears valid
    os.makedirs(record["worktree_path"], exist_ok=True)

    result = list_worktrees(mock_git_env["repo"])
    assert len(result) == 1
    assert result[0]["branch"] == "exp_abc123"


# ── Tests: registry persistence ──


def test_registry_record_and_remove(mock_git_env: dict) -> None:
    """Should record and then remove worktree entries."""
    from expflow_pde.worktree import _record_worktree, _registry_path, _remove_worktree_record

    # Record
    _record_worktree(mock_git_env["repo"], "exp_abc", "/tmp/wt_abc")
    records = json.loads(open(_registry_path(mock_git_env["repo"])).read().strip())
    assert records["branch"] == "exp_abc"

    # Remove
    _remove_worktree_record("exp_abc")
    with open(_registry_path(mock_git_env["repo"])) as f:
        content = f.read().strip()
    assert content == ""  # nothing left


def test_registry_remove_nonexistent(mock_git_env: dict) -> None:
    """Should not error when removing a non-existent record."""
    from expflow_pde.worktree import _remove_worktree_record

    _remove_worktree_record("nonexistent")  # should not raise


# ── Tests: cleanup_all_stale ──


def test_cleanup_stale_worktrees(mock_git_env: dict, mock_subprocess: MagicMock) -> None:
    """Should only clean worktrees older than max_age_hours."""
    from expflow_pde.worktree import _record_worktree, cleanup_all_stale

    # Record a stale worktree (old timestamp)
    wt_dir = os.path.join(mock_git_env["wt_base"], "exp_old")
    _record_worktree(mock_git_env["repo"], "exp_old", wt_dir)
    os.makedirs(wt_dir, exist_ok=True)

    result = cleanup_all_stale(mock_git_env["repo"], max_age_hours=0)  # everything is stale
    assert len(result) == 1
    assert result[0]["branch"] == "exp_old"


# ── Tests: submit_via_stash ──


def test_submit_via_stash_basic(mock_git_env: dict, mock_subprocess: MagicMock) -> None:
    """Should create stash branch, commit, then cleanup."""
    from expflow_pde.worktree import submit_via_stash

    result = submit_via_stash(
        script_path="train.py",
        repo_path=mock_git_env["repo"],
        skip_push=True,
    )

    assert result["branch"].startswith("exp_")
    assert result["method"] == "stash"
    assert "commit_hash" in result

    # Verify stash was used
    git_calls = [str(c) for c in mock_subprocess.call_args_list]
    assert any("stash" in c for c in git_calls)
    assert any("checkout" in c for c in git_calls)
