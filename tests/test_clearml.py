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
    pkg.Model = MagicMock(name="Model")
    pkg.OutputModel = MagicMock(name="OutputModel")

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
    """annotate_compliance() and list_datasets() with compliance annotation."""

    def test_annotate_compliance_sets_metadata(self, mock_clearml_pkg):
        mock_ds = MagicMock()
        mock_ds.id = "ds1"
        mock_clearml_pkg.Dataset.get.return_value = mock_ds

        from expflow.clearml import annotate_compliance

        result = annotate_compliance(
            dataset_id="ds1",
            compliance="allowed",
        )

        mock_ds.set_metadata.assert_called_with("expflow:compliance", "allowed")
        assert result["id"] == "ds1"
        assert result["compliance"] == "allowed"

    def test_annotate_compliance_rejects_invalid(self, mock_clearml_pkg):
        from expflow.clearml import annotate_compliance

        with pytest.raises(ValueError, match="compliance"):
            annotate_compliance(
                dataset_id="ds1",
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


# ══════════════════════════════════════════════════════════════
# Dataset upload / download / lineage
# ══════════════════════════════════════════════════════════════


class TestDatasetUpload:
    """dataset_upload() — upload local files to clearml Fileserver."""

    def test_upload_creates_dataset_and_adds_files(self, mock_clearml_pkg):
        mock_ds = MagicMock()
        mock_ds.id = "ds_uploaded"
        mock_ds.version = "1.0"
        mock_clearml_pkg.Dataset.create.return_value = mock_ds

        from expflow.clearml import dataset_upload

        result = dataset_upload(
            local_path="/data/burgers.hdf5",
            dataset_name="burgers_nu0.001",
        )

        mock_clearml_pkg.Dataset.create.assert_called_with(
            dataset_name="burgers_nu0.001",
            dataset_project="PDEBench",
            dataset_version=None,
            parent_datasets=None,
            description="",
        )
        mock_ds.add_files.assert_called_with(path="/data/burgers.hdf5")
        mock_ds.upload.assert_called_once()
        mock_ds.finalize.assert_called_once()
        assert result["id"] == "ds_uploaded"
        assert result["name"] == "burgers_nu0.001"

    def test_upload_with_compliance_and_parents(self, mock_clearml_pkg):
        mock_ds = MagicMock()
        mock_ds.id = "ds2"
        mock_ds.version = "2.0"
        mock_clearml_pkg.Dataset.create.return_value = mock_ds

        from expflow.clearml import dataset_upload

        result = dataset_upload(
            local_path="/data/new.hdf5",
            dataset_name="burgers_synthetic",
            version="2.0",
            parent_dataset_ids=["ds_parent"],
            compliance="forbidden",
            description="Synthetic long-time data",
            tags=["synthetic", "long-time"],
        )

        mock_clearml_pkg.Dataset.create.assert_called_with(
            dataset_name="burgers_synthetic",
            dataset_project="PDEBench",
            dataset_version="2.0",
            parent_datasets=["ds_parent"],
            description="Synthetic long-time data",
        )
        mock_ds.set_metadata.assert_any_call("expflow:compliance", "forbidden")
        assert result["compliance"] == "forbidden"


class TestDatasetDownload:
    """dataset_download() — download from Fileserver to local."""

    def test_download_by_id(self, mock_clearml_pkg):
        mock_ds = MagicMock()
        mock_ds.id = "ds1"
        mock_ds.name = "test_ds"
        mock_ds.version = "1.0"
        mock_ds.get_mutable_local_copy.return_value = "/tmp/download/test_ds"
        mock_clearml_pkg.Dataset.get.return_value = mock_ds

        from expflow.clearml import dataset_download

        result = dataset_download(
            target_folder="/tmp/download",
            dataset_id="ds1",
        )

        mock_clearml_pkg.Dataset.get.assert_called_with(
            dataset_id="ds1",
            dataset_name=None,
            dataset_project="PDEBench",
            dataset_version=None,
            only_completed=True,
        )
        mock_ds.get_mutable_local_copy.assert_called_with(
            target_folder="/tmp/download",
            overwrite=False,
        )
        assert result["local_path"] == "/tmp/download/test_ds"

    def test_download_requires_id_or_name(self, mock_clearml_pkg):
        from expflow.clearml import dataset_download

        with pytest.raises(ValueError, match="Provide either dataset_id"):
            dataset_download(target_folder="/tmp/download")


class TestDatasetLineage:
    """dataset_lineage() — trace parent chain."""

    def test_lineage_single_level(self, mock_clearml_pkg):
        mock_ds = MagicMock()
        mock_ds.id = "child"
        mock_ds.name = "child_ds"
        mock_ds.version = "2.0"
        mock_ds.parent = None
        mock_ds.get_metadata.return_value = "allowed"
        mock_clearml_pkg.Dataset.get.return_value = mock_ds

        from expflow.clearml import dataset_lineage

        lineage = dataset_lineage(dataset_id="child")

        assert len(lineage) == 1
        assert lineage[0]["id"] == "child"
        assert lineage[0]["compliance"] == "allowed"
        assert lineage[0]["parent_id"] is None

    def test_lineage_multi_level(self, mock_clearml_pkg):
        grandparent = MagicMock()
        grandparent.id = "gp"
        grandparent.name = "grandparent"
        grandparent.version = "0.5"
        grandparent.parent = None
        grandparent.get_metadata.return_value = "allowed"

        parent = MagicMock()
        parent.id = "p"
        parent.name = "parent"
        parent.version = "1.0"
        parent.parent = "gp"
        parent.get_metadata.return_value = "allowed"

        child = MagicMock()
        child.id = "c"
        child.name = "child"
        child.version = "2.0"
        child.parent = "p"
        child.get_metadata.return_value = "forbidden"

        mock_clearml_pkg.Dataset.get.side_effect = [child, parent, grandparent]

        from expflow.clearml import dataset_lineage

        lineage = dataset_lineage(dataset_id="c")

        assert len(lineage) == 3
        assert lineage[0]["id"] == "gp"
        assert lineage[1]["id"] == "p"
        assert lineage[2]["id"] == "c"
        # Compliance inherited
        assert lineage[0]["compliance"] == "allowed"
        assert lineage[2]["compliance"] == "forbidden"


# ══════════════════════════════════════════════════════════════
# Model operations
# ══════════════════════════════════════════════════════════════


class TestModelList:
    """model_list() — query models from clearml Model store."""

    def test_model_list_returns_serialized(self, mock_clearml_pkg):
        mock_model = MagicMock()
        mock_model.id = "model_1"
        mock_model.name = "fno_burgers"
        mock_model.project = "PDEBench"
        mock_model.tags = ["fno", "pde"]
        mock_model.created = "2026-01-01"
        mock_model.uri = "clearml://models/model_1"
        mock_model.task_id = "task_1"
        mock_model.framework = "PyTorch"

        mock_clearml_pkg.Model.query_models.return_value = [mock_model]

        from expflow.clearml import model_list

        models = model_list(project_name="PDEBench", max_results=10)

        assert len(models) == 1
        assert models[0]["id"] == "model_1"
        assert models[0]["name"] == "fno_burgers"
        assert models[0]["framework"] == "PyTorch"

    def test_model_list_empty(self, mock_clearml_pkg):
        mock_clearml_pkg.Model.query_models.return_value = []

        from expflow.clearml import model_list

        models = model_list()
        assert models == []


class TestModelUpload:
    """model_upload() — upload checkpoint to Model store."""

    def test_model_upload_creates_output_model(self, mock_clearml_pkg):
        mock_task = MagicMock()
        mock_task.name = "train_fno"
        mock_clearml_pkg.Task.get_task.return_value = mock_task

        mock_output = MagicMock()
        mock_output.id = "model_out_1"
        mock_output.uri = "clearml://fileserver/models/model_out_1"
        mock_clearml_pkg.OutputModel.return_value = mock_output

        from expflow.clearml import model_upload

        result = model_upload(
            local_path="checkpoint_50.pt",
            task_id="task_train_1",
            model_name="fno_epoch50",
        )

        mock_clearml_pkg.OutputModel.assert_called_with(
            task=mock_task,
            framework="PyTorch",
        )
        mock_output.update_weights.assert_called()
        assert result["id"] == "model_out_1"
        assert result["task_id"] == "task_train_1"


# ══════════════════════════════════════════════════════════════
# Pipeline operations
# ══════════════════════════════════════════════════════════════


class TestPipelineCreate:
    """pipeline_create() — create a clearml PipelineController."""

    def test_pipeline_create_returns_serialized(self, mock_clearml_pkg):
        """Creating a pipeline returns plain dict."""
        mock_pipe = MagicMock()
        mock_pipe.id = "pipe_1"
        mock_clearml_pkg.PipelineController.return_value = mock_pipe

        from expflow.clearml import pipeline_create

        result = pipeline_create(
            name="test_pipeline",
            project="PDEBench",
        )

        assert result["name"] == "test_pipeline"
        assert result["project"] == "PDEBench"
        assert result["status"] == "created"

    def test_pipeline_create_passes_project(self, mock_clearml_pkg):
        """PipelineController created with correct project."""
        mock_clearml_pkg.PipelineController.return_value = MagicMock()

        from expflow.clearml import pipeline_create

        pipeline_create(name="pipe1", project="MyProject")

        mock_clearml_pkg.PipelineController.assert_called_with(
            name="pipe1",
            project="MyProject",
            version=None,
            abort_on_failure=False,
            add_pipeline_tags=False,
            add_run_number=True,
            docker=None,
        )

    def test_pipeline_create_with_options(self, mock_clearml_pkg):
        """Pipeline with all options forwarded."""
        mock_clearml_pkg.PipelineController.return_value = MagicMock()

        from expflow.clearml import pipeline_create

        pipeline_create(
            name="opt_pipe",
            project="Proj",
            version="2.0",
            abort_on_failure=True,
            docker="python:3.11",
        )

        _, kwargs = mock_clearml_pkg.PipelineController.call_args
        assert kwargs["version"] == "2.0"
        assert kwargs["abort_on_failure"] is True
        assert kwargs["docker"] == "python:3.11"


class TestPipelineAddStep:
    """pipeline_add_step() — add a step to a pipeline controller."""

    def test_add_step_returns_serialized(self, mock_clearml_pkg):
        """Adding a step via base_task_id returns plain dict."""
        mock_pipe = MagicMock()
        mock_clearml_pkg.PipelineController.return_value = mock_pipe

        from expflow.clearml import pipeline_add_step

        result = pipeline_add_step(
            pipeline_name="pipe1",
            project="PDEBench",
            step_name="train",
            base_task_id="task_1",
        )

        assert result["step_name"] == "train"
        assert result["pipeline_name"] == "pipe1"
        assert result["status"] == "defined"
        mock_pipe.add_step.assert_called_once_with(
            name="train",
            base_task_id="task_1",
        )

    def test_add_step_with_parents(self, mock_clearml_pkg):
        """Step with parents gets dependency chain."""
        mock_pipe = MagicMock()
        mock_clearml_pkg.PipelineController.return_value = mock_pipe

        from expflow.clearml import pipeline_add_step

        result = pipeline_add_step(
            pipeline_name="pipe1",
            project="PDEBench",
            step_name="validate",
            base_task_id="task_2",
            parents=["train"],
            execution_queue="gpu_queue",
        )

        assert result["parents"] == ["train"]
        mock_pipe.add_step.assert_called_once_with(
            name="validate",
            base_task_id="task_2",
            parents=["train"],
            execution_queue="gpu_queue",
        )

    def test_add_step_requires_task_id_or_name(self, mock_clearml_pkg):
        """Providing neither base_task_id nor base_task_name raises ValueError."""
        from expflow.clearml import pipeline_add_step

        with pytest.raises(ValueError, match="Provide either"):
            pipeline_add_step(
                pipeline_name="pipe1",
                project="PDEBench",
                step_name="bad_step",
            )


class TestPipelineStart:
    """pipeline_start() — execute a pipeline controller."""

    def test_pipeline_start_calls_start_and_wait(self, mock_clearml_pkg):
        """Start triggers pipe.start() and pipe.wait()."""
        mock_pipe = MagicMock()
        mock_clearml_pkg.PipelineController.return_value = mock_pipe

        from expflow.clearml import pipeline_start

        result = pipeline_start(
            pipeline_name="pipe1",
            project="PDEBench",
            queue_name="default",
            timeout_minutes=60,
        )

        mock_pipe.start.assert_called_once_with(queue_name="default")
        mock_pipe.wait.assert_called_once_with(timeout=60)
        assert result["pipeline_name"] == "pipe1"
        assert result["started"] is True


class TestPipelineStop:
    """pipeline_stop() — stop a running pipeline."""

    def test_pipeline_stop_calls_stop(self, mock_clearml_pkg):
        """Stop triggers pipe.stop()."""
        mock_pipe = MagicMock()
        mock_clearml_pkg.PipelineController.return_value = mock_pipe

        from expflow.clearml import pipeline_stop

        result = pipeline_stop(
            pipeline_name="pipe1",
            project="PDEBench",
        )

        mock_pipe.stop.assert_called_once()
        assert result["status"] == "stopped"


class TestPipelineList:
    """pipeline_list() — query pipeline tasks."""

    def test_pipeline_list_returns_serialized(self, mock_clearml_pkg):
        """List returns plain dicts from Task.get_tasks."""
        mock_task = MagicMock()
        mock_task.id = "pipe_1"
        mock_task.name = "training_pipeline"
        mock_task.project = "PDEBench"
        mock_task.status = "completed"
        mock_task.tags = ["pipeline"]
        mock_clearml_pkg.Task.get_tasks.return_value = [mock_task]

        from expflow.clearml import pipeline_list

        pipelines = pipeline_list(project_name="PDEBench")

        assert len(pipelines) == 1
        assert pipelines[0]["id"] == "pipe_1"
        assert pipelines[0]["name"] == "training_pipeline"
        assert pipelines[0]["status"] == "completed"

    def test_pipeline_list_empty(self, mock_clearml_pkg):
        """No pipelines returns empty list."""
        mock_clearml_pkg.Task.get_tasks.return_value = []

        from expflow.clearml import pipeline_list

        pipelines = pipeline_list()

        assert pipelines == []


# ══════════════════════════════════════════════════════════════
# init_tracking
# ══════════════════════════════════════════════════════════════


class TestInitTracking:
    """init_tracking() — Task.init wrapper with graph capture support."""

    def test_init_tracking_returns_task_info(self, mock_clearml_pkg):
        """Basic init returns task_id, task_name, project."""
        mock_task = MagicMock()
        mock_task.id = "task_init_1"
        mock_clearml_pkg.Task.init.return_value = mock_task

        from expflow.clearml import init_tracking

        result = init_tracking(
            task_name="test_run",
            project="PDEBench",
        )

        assert result["task_id"] == "task_init_1"
        assert result["task_name"] == "test_run"
        assert result["project"] == "PDEBench"
        assert result["graph_uploaded"] is False

    def test_init_tracking_passes_frameworks(self, mock_clearml_pkg):
        """Task.init called with auto_connect_frameworks dict."""
        mock_clearml_pkg.Task.init.return_value = MagicMock()

        from expflow.clearml import init_tracking

        init_tracking(task_name="t", project="P")

        _, kwargs = mock_clearml_pkg.Task.init.call_args
        assert kwargs["project_name"] == "P"
        assert kwargs["task_name"] == "t"
        assert kwargs["auto_connect_frameworks"] == {"tensorboard": True, "pytorch": True}

    def test_init_tracking_disabled_frameworks(self, mock_clearml_pkg):
        """capture_tensorboard=False omits tensorboard from frameworks."""
        mock_clearml_pkg.Task.init.return_value = MagicMock()

        from expflow.clearml import init_tracking

        init_tracking(task_name="t", project="P", capture_tensorboard=False)

        _, kwargs = mock_clearml_pkg.Task.init.call_args
        fw = kwargs["auto_connect_frameworks"]
        assert "tensorboard" not in fw
        assert fw["pytorch"] is True

    def test_init_tracking_graph_disabled_by_default(self, mock_clearml_pkg):
        """capture_graph=False (default) skips graph upload."""
        mock_task = MagicMock()
        mock_task.id = "t1"
        mock_clearml_pkg.Task.init.return_value = mock_task

        from expflow.clearml import init_tracking

        result = init_tracking(task_name="t", project="P", capture_graph=False)
        assert result["graph_uploaded"] is False
