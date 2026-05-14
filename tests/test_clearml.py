#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow.clearml — Task listing, queue management, dataset compliance.

All tests use mocked clearml SDK. No real clearml server needed.

Mock strategy: We patch 'clearml' in sys.modules BEFORE importing expflow.clearml.
This lets the lazy import in clearml.py receive a mock clearml package directly.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import _make_mock_queue, _make_mock_task

# ── Helpers ──


def _make_mock_ds(ds_id: str, name: str, version: str, compliance: str | None) -> MagicMock:
    """Create a mock clearml Dataset with realistic attribute access."""
    ds = MagicMock(name=f"Dataset({name})")
    ds.id = ds_id
    ds.name = name
    ds.version = version
    ds.get_metadata.return_value = compliance
    return ds


# ── Fixture: mock clearml package ──


@pytest.fixture(autouse=True)
def mock_clearml_pkg() -> MagicMock:
    """Replace 'clearml' in sys.modules with a mock before any imports."""
    pkg = MagicMock(name="clearml_pkg")
    pkg.Task = MagicMock(name="Task")
    pkg.Queue = MagicMock(name="Queue")
    pkg.Dataset = MagicMock(name="Dataset")

    if "expflow.clearml" in sys.modules:
        del sys.modules["expflow.clearml"]

    with patch.dict("sys.modules", {"clearml": pkg}):
        yield pkg

    if "expflow.clearml" in sys.modules:
        del sys.modules["expflow.clearml"]


# ══════════════════════════════════════════════════════════════
# list_tasks
# ══════════════════════════════════════════════════════════════


class TestListTasks:
    """list_tasks() — list clearml tasks with optional filters."""

    def test_list_tasks_no_filter_returns_all(self, mock_clearml_pkg):
        """Without filters, list_tasks() returns all tasks as dicts."""
        mock_tasks = [
            _make_mock_task("t1", "run_a", "project_x", "completed"),
            _make_mock_task("t2", "run_b", "project_x", "failed"),
            _make_mock_task("t3", "run_c", "project_y", "running"),
        ]
        mock_clearml_pkg.Task.get_tasks.return_value = mock_tasks

        from expflow.clearml import list_tasks

        result = list_tasks()

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {
            "id": "t1",
            "name": "run_a",
            "project": "project_x",
            "status": "completed",
            "tags": [],
            "last_iteration": 100,
        }
        assert result[2]["status"] == "running"

    def test_list_tasks_filters_by_project_name(self, mock_clearml_pkg):
        """Passing project_name filters results."""
        mock_clearml_pkg.Task.get_tasks.side_effect = lambda **kwargs: (
            [_make_mock_task("t2", "b", "proj_y", "completed")]
            if kwargs.get("project_name") == "proj_y"
            else []
        )

        from expflow.clearml import list_tasks

        result = list_tasks(project_name="proj_y")

        assert len(result) == 1
        assert result[0]["project"] == "proj_y"

    def test_list_tasks_serializable_dict_not_mock(self, mock_clearml_pkg):
        """Return value must be plain dicts (JSON-serializable), not mock objects."""
        mock_clearml_pkg.Task.get_tasks.return_value = [
            _make_mock_task("t1", "a", "p", "completed")
        ]

        from expflow.clearml import list_tasks

        result = list_tasks()

        import json

        json.dumps(result)  # Must not raise TypeError

    def test_list_tasks_empty_result(self, mock_clearml_pkg):
        """No tasks returns empty list."""
        mock_clearml_pkg.Task.get_tasks.return_value = []

        from expflow.clearml import list_tasks

        result = list_tasks()

        assert result == []


# ══════════════════════════════════════════════════════════════
# list_tasks — 参数转发
# ══════════════════════════════════════════════════════════════


class TestListTasksArgs:
    """Verify list_tasks() forwards arguments correctly to SDK."""

    def test_passes_project_name(self, mock_clearml_pkg):
        mock_clearml_pkg.Task.get_tasks.return_value = []
        from expflow.clearml import list_tasks

        list_tasks(project_name="my_project")
        mock_clearml_pkg.Task.get_tasks.assert_called_with(project_name="my_project")

    def test_passes_task_name(self, mock_clearml_pkg):
        from expflow.clearml import list_tasks

        list_tasks(task_name="my_task")
        mock_clearml_pkg.Task.get_tasks.assert_called_with(task_name="my_task")

    def test_passes_tags(self, mock_clearml_pkg):
        from expflow.clearml import list_tasks

        list_tasks(tags=["important"])
        mock_clearml_pkg.Task.get_tasks.assert_called_with(tags=["important"])

    def test_passes_status(self, mock_clearml_pkg):
        from expflow.clearml import list_tasks

        list_tasks(status=["completed"])
        mock_clearml_pkg.Task.get_tasks.assert_called_with(status=["completed"])

    def test_no_args_passes_no_kwargs(self, mock_clearml_pkg):
        mock_clearml_pkg.Task.get_tasks.return_value = []
        from expflow.clearml import list_tasks

        list_tasks()
        mock_clearml_pkg.Task.get_tasks.assert_called_with()


# ══════════════════════════════════════════════════════════════
# get_task
# ══════════════════════════════════════════════════════════════


class TestGetTask:
    """get_task() — get single task by ID."""

    def test_get_task_returns_serialized_dict(self, mock_clearml_pkg):
        mock_task = _make_mock_task("t1", "my_task", "proj", "completed")
        mock_clearml_pkg.Task.get_task.return_value = mock_task

        from expflow.clearml import get_task

        result = get_task("t1")

        assert result["id"] == "t1"
        assert result["name"] == "my_task"
        assert result["status"] == "completed"

    def test_get_task_passes_id(self, mock_clearml_pkg):
        mock_clearml_pkg.Task.get_task.return_value = _make_mock_task("t1")
        from expflow.clearml import get_task

        get_task("t1")
        mock_clearml_pkg.Task.get_task.assert_called_with(task_id="t1")


# ══════════════════════════════════════════════════════════════
# enqueue_task / dequeue_task
# ══════════════════════════════════════════════════════════════


class TestQueueOperations:
    """enqueue_task() and dequeue_task() operations."""

    def test_enqueue_task_returns_serialized_result(self, mock_clearml_pkg):
        mock_task = _make_mock_task("t1", "a", "p", "queued")
        mock_clearml_pkg.Task.get_task.return_value = mock_task

        from expflow.clearml import enqueue_task

        result = enqueue_task("t1", queue_name="default")

        assert result["task_id"] == "t1"
        assert result["queue"] == "default"
        assert result["status"] == "queued"

    def test_enqueue_task_enqueue_called(self, mock_clearml_pkg):
        mock_task = _make_mock_task("t1", "a", "p", "created")
        mock_clearml_pkg.Task.get_task.return_value = mock_task

        from expflow.clearml import enqueue_task

        enqueue_task("t1")

        mock_task.enqueue.assert_called_once_with(queue_name="default")

    def test_dequeue_task_marks_dequeued(self, mock_clearml_pkg):
        mock_task = _make_mock_task("t1", "a", "p", "created")
        mock_clearml_pkg.Task.get_task.return_value = mock_task

        from expflow.clearml import dequeue_task

        result = dequeue_task("t1")

        mock_task.dequeue.assert_called_once()
        assert result["task_id"] == "t1"
        assert result["status"] == "created"


# ══════════════════════════════════════════════════════════════
# list_queues / get_queue_status
# ══════════════════════════════════════════════════════════════


class TestListQueues:
    """list_queues() — retrieve available queues."""

    def test_list_queues_returns_serialized_dicts(self, mock_clearml_pkg):
        mock_qs = [_make_mock_queue("q1", "default"), _make_mock_queue("q2", "gpu_queue")]
        mock_clearml_pkg.Queue.get_queues.return_value = mock_qs

        from expflow.clearml import list_queues

        result = list_queues()

        assert len(result) == 2
        assert result[0] == {"id": "q1", "name": "default"}
        assert result[1] == {"id": "q2", "name": "gpu_queue"}

    def test_get_queue_status_returns_dict(self, mock_clearml_pkg):
        mock_q = _make_mock_queue("q1", "default")
        mock_q.entries = []
        mock_clearml_pkg.Queue.get_queue.return_value = mock_q

        from expflow.clearml import get_queue_status

        result = get_queue_status("default")

        assert result["id"] == "q1"
        assert result["name"] == "default"
        assert "entries" in result

    def test_get_queue_status_passes_name(self, mock_clearml_pkg):
        mock_clearml_pkg.Queue.get_queue.return_value = _make_mock_queue("q1", "default")
        from expflow.clearml import get_queue_status

        get_queue_status("gpu_queue")
        mock_clearml_pkg.Queue.get_queue.assert_called_with(queue_name="gpu_queue")


# ══════════════════════════════════════════════════════════════
# Dataset compliance
# ══════════════════════════════════════════════════════════════


class TestDatasetCompliance:
    """register_dataset() and list_datasets() with compliance annotation."""

    def test_register_dataset_adds_compliance_tag(self, mock_clearml_pkg):
        mock_ds = MagicMock()
        mock_ds.id = "ds1"
        mock_ds.name = "burgers_nu0.001"
        mock_ds.version = "1.0"
        mock_clearml_pkg.Dataset.create.return_value = mock_ds

        from expflow.clearml import register_dataset

        result = register_dataset(
            name="burgers_nu0.001",
            version="1.0",
            path="/data/burgers.hdf5",
            compliance="allowed",
            description="Burgers equation 1D",
        )

        mock_ds.set_metadata.assert_any_call("expflow:compliance", "allowed")
        assert result["id"] == "ds1"
        assert result["compliance"] == "allowed"

    def test_register_dataset_rejects_invalid_compliance(self, mock_clearml_pkg):
        from expflow.clearml import register_dataset

        with pytest.raises(ValueError, match="compliance"):
            register_dataset(
                name="bad",
                version="1.0",
                path="/x",
                compliance="unknown",
            )

    def test_list_datasets_returns_compliance_info(self, mock_clearml_pkg):
        mock_ds_1 = _make_mock_ds("ds1", "allowed_ds", "1.0", "allowed")
        mock_ds_2 = _make_mock_ds("ds2", "forbidden_ds", "2.0", None)
        mock_clearml_pkg.Dataset.list_datasets.return_value = [mock_ds_1, mock_ds_2]

        from expflow.clearml import list_datasets

        result = list_datasets()

        assert len(result) == 2
        assert result[0]["compliance"] == "allowed"
        assert result[1]["compliance"] is None

    def test_list_datasets_filters_by_compliance(self, mock_clearml_pkg):
        mock_ds_list = [
            _make_mock_ds(f"ds{i}", f"ds_{i}", "1.0", v)
            for i, v in enumerate(["allowed", "forbidden", "allowed", None])
        ]
        mock_clearml_pkg.Dataset.list_datasets.return_value = mock_ds_list

        from expflow.clearml import list_datasets

        result = list_datasets(compliance_filter="allowed")

        assert len(result) == 2
        assert all(r["compliance"] == "allowed" for r in result)

    def test_list_datasets_filters_by_name(self, mock_clearml_pkg):
        mock_ds_list = [
            _make_mock_ds("ds1", "burgers_nu0.001", "1.0", "allowed"),
            _make_mock_ds("ds2", "ns_eq_re100", "1.0", "allowed"),
        ]
        mock_clearml_pkg.Dataset.list_datasets.return_value = mock_ds_list

        from expflow.clearml import list_datasets

        result = list_datasets(name_filter="burgers")

        assert len(result) == 1
        assert result[0]["name"] == "burgers_nu0.001"

    def test_list_datasets_empty_result(self, mock_clearml_pkg):
        mock_clearml_pkg.Dataset.list_datasets.return_value = []

        from expflow.clearml import list_datasets

        result = list_datasets()

        assert result == []


# ══════════════════════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════════════════════


class TestClearmlErrors:
    """Error handling for clearml SDK failures."""

    def test_get_task_raises_on_missing(self, mock_clearml_pkg):
        mock_clearml_pkg.Task.get_task.side_effect = ValueError("Task not found")

        from expflow.clearml import get_task

        with pytest.raises(ValueError, match="Task not found"):
            get_task("nonexistent")
