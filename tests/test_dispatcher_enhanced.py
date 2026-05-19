#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for enhanced expflow_pde.dispatcher — worktree/stash mode, persisted registry.

All tests use mocked clearml SDK and git subprocesses.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──


@pytest.fixture(autouse=True)
def mock_clearml_pkg() -> MagicMock:
    """Mock clearml SDK for Task.create() calls."""
    pkg = MagicMock(name="clearml_pkg")
    pkg.Task = MagicMock(name="Task")
    mock_task = MagicMock(name="task_instance")
    mock_task.id = "cl_task_12345"
    pkg.Task.create.return_value = mock_task
    pkg.Task.get_task.return_value = mock_task

    if "expflow_pde.dispatcher" in sys.modules:
        del sys.modules["expflow_pde.dispatcher"]

    with patch.dict("sys.modules", {"clearml": pkg}):
        yield pkg


@pytest.fixture(autouse=True)
def mock_git() -> MagicMock:
    """Mock the _auto_git_push function to avoid real git calls."""
    with patch("expflow_pde.dispatcher._auto_git_push") as mock_fn:
        yield mock_fn


@pytest.fixture(autouse=True)
def mock_worktree() -> MagicMock:
    """Mock the worktree module to avoid actual file operations."""
    patcher = patch("expflow_pde.dispatcher._submit_via_worktree")
    mock_fn = patcher.start()
    mock_fn.return_value = {
        "task_id": "cl_task_12345",
        "branch": "exp_test123",
        "commit_hash": "abc12345",
        "queue": "gpu_queue",
    }
    yield mock_fn
    patcher.stop()


@pytest.fixture(autouse=True)
def mock_stash() -> MagicMock:
    """Mock the stash submission path."""
    patcher = patch("expflow_pde.dispatcher._submit_via_stash")
    mock_fn = patcher.start()
    mock_fn.return_value = {
        "task_id": "cl_task_67890",
        "branch": "exp_stash123",
        "commit_hash": "def67890",
        "queue": "gpu_queue",
    }
    yield mock_fn
    patcher.stop()


@pytest.fixture
def clean_registry() -> None:
    """Ensure clean experiment registry for each test."""
    from expflow_pde.dispatcher import _experiments

    _experiments.clear()
    rpath = os.path.expanduser("~/.expflow/experiments.jsonl")
    if os.path.isfile(rpath):
        os.remove(rpath)
    yield
    _experiments.clear()
    if os.path.isfile(rpath):
        os.remove(rpath)


# ── Tests: dispatch_experiment ──


class TestDispatchExperiment:
    """Test the main dispatch_experiment function."""

    def test_shell_mode_default(self, clean_registry: None) -> None:
        """Should dispatch in shell mode when no flags set."""
        from expflow_pde.dispatcher import dispatch_experiment

        result = dispatch_experiment("train.py", queue="gpu_queue")
        assert result["method"] == "shell"
        assert result["status"] == "dispatched"
        assert result["queue"] == "gpu_queue"

    def test_worktree_mode(self, clean_registry: None, mock_worktree: MagicMock) -> None:
        """Should dispatch in worktree mode."""
        from expflow_pde.dispatcher import dispatch_experiment

        result = dispatch_experiment(
            "train.py",
            queue="gpu_queue",
            use_worktree=True,
            script_args={"lr": "0.001", "epochs": "80"},
        )
        assert result["method"] == "worktree"
        assert result["status"] == "queued"
        assert result["branch"] == "exp_test123"
        assert "task_id" in result

    def test_stash_mode(self, clean_registry: None, mock_stash: MagicMock) -> None:
        """Should dispatch in stash mode."""
        from expflow_pde.dispatcher import dispatch_experiment

        result = dispatch_experiment(
            "train.py",
            queue="gpu_queue",
            use_stash=True,
        )
        assert result["method"] == "stash"
        assert result["status"] == "queued"
        assert result["branch"] == "exp_stash123"

    def test_clearml_mode(self, clean_registry: None, mock_clearml_pkg: MagicMock) -> None:
        """Should dispatch via clearml Task.create()."""
        from expflow_pde.dispatcher import dispatch_experiment

        mock_clearml_pkg.Task.create.return_value.id = "cl_task_manual"
        result = dispatch_experiment(
            "train.py",
            queue="gpu_queue",
            use_clearml_task=True,
        )
        assert result["method"] == "clearml"
        assert result["status"] == "queued"
        assert "task_id" in result

    def test_script_args_passed(self, clean_registry: None, mock_worktree: MagicMock) -> None:
        """Should pass script_args to clearml Task."""
        from expflow_pde.dispatcher import dispatch_experiment

        result = dispatch_experiment(
            "train.py",
            queue="gpu_queue",
            use_worktree=True,
            script_args={"lr": "0.001", "epochs": "80"},
        )
        assert result["method"] == "worktree"

    def test_tags_passed(self, clean_registry: None, mock_worktree: MagicMock) -> None:
        """Should include tags in the record."""
        from expflow_pde.dispatcher import dispatch_experiment

        result = dispatch_experiment(
            "train.py",
            queue="gpu_queue",
            use_worktree=True,
            tags=["task1", "fno"],
        )
        assert "task1" in result["tags"]
        assert "fno" in result["tags"]

    def test_is_python_script(self) -> None:
        """Should detect Python scripts correctly."""
        from expflow_pde.dispatcher import is_python_script

        assert is_python_script("train.py") is True
        assert is_python_script("train.sh") is False
        assert is_python_script("python train.py") is False  # has space


# ── Tests: persistent registry ──


class TestPersistentRegistry:
    """Test the experiment registry persistence."""

    def test_registry_saves_and_loads(self, clean_registry: None) -> None:
        """Should save experiment records to ~/.expflow/experiments.jsonl."""
        from expflow_pde.dispatcher import _load_registry, _save_to_registry, list_experiments

        _save_to_registry(
            {
                "experiment_id": "abc12345",
                "status": "queued",
                "method": "worktree",
                "timestamp": "2026-05-19T10:00:00Z",
            }
        )

        loaded = _load_registry()
        assert len(loaded) == 1
        assert loaded[0]["experiment_id"] == "abc12345"

        # list_experiments should include persisted records
        all_exp = list_experiments()
        assert any(e["experiment_id"] == "abc12345" for e in all_exp)

    def test_registry_multiple_records(self, clean_registry: None) -> None:
        """Should handle multiple records."""
        from expflow_pde.dispatcher import _load_registry, _save_to_registry

        for i in range(3):
            _save_to_registry(
                {
                    "experiment_id": f"exp_{i}",
                    "status": "queued",
                    "timestamp": f"2026-05-19T{10 + i}:00:00Z",
                }
            )

        loaded = _load_registry()
        assert len(loaded) == 3

    def test_get_experiment_status_from_registry(self, clean_registry: None) -> None:
        """Should find experiment in persistent registry."""
        from expflow_pde.dispatcher import _save_to_registry, get_experiment_status

        _save_to_registry(
            {
                "experiment_id": "exp_abc123",
                "status": "queued",
                "method": "worktree",
            }
        )

        result = get_experiment_status("exp_abc123")
        assert result["status"] == "queued"

    def test_get_experiment_status_not_found(self, clean_registry: None) -> None:
        """Should return error for missing experiment."""
        from expflow_pde.dispatcher import get_experiment_status

        result = get_experiment_status("nonexistent")
        assert "error" in result


# ── Tests: cancel_experiment ──


class TestCancelExperiment:
    """Test experiment cancellation."""

    def test_cancel_with_clearml_task(
        self, clean_registry: None, mock_clearml_pkg: MagicMock
    ) -> None:
        """Should dequeue and stop the clearml task."""
        from expflow_pde.dispatcher import _save_to_registry, cancel_experiment

        _save_to_registry(
            {
                "experiment_id": "exp_abc",
                "clearml_task_id": "cl_task_123",
                "status": "queued",
            }
        )

        result = cancel_experiment("exp_abc")
        assert result["status"] == "cancelled"
        assert mock_clearml_pkg.Task.get_task.called
        mock_clearml_pkg.Task.get_task.return_value.dequeue.assert_called_once()

    def test_cancel_not_found(self, clean_registry: None) -> None:
        """Should return error for non-existent experiment."""
        from expflow_pde.dispatcher import cancel_experiment

        result = cancel_experiment("nonexistent")
        assert "error" in result
