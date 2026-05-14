#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow CLI — clearml sub-commands via CliRunner.

Mock clearml SDK using sys.modules patch to verify CLI output format.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from expflow.cli import app
from tests.helpers import _make_mock_queue, _make_mock_task

runner = CliRunner()


# ── Fixture: mock clearml before CLI import ──


@pytest.fixture(autouse=True)
def mock_clearml_cli() -> MagicMock:
    """Mock clearml package before CLI import for CliRunner."""
    pkg = MagicMock(name="clearml_pkg")
    pkg.Task = MagicMock(name="Task")
    pkg.Queue = MagicMock(name="Queue")
    pkg.Dataset = MagicMock(name="Dataset")

    # Clear cached module so lazy imports in cli_clearml.py get the mock
    for mod in ["expflow.clearml", "expflow.cli_clearml"]:
        if mod in sys.modules:
            del sys.modules[mod]

    with patch.dict("sys.modules", {"clearml": pkg}):
        yield pkg


# ══════════════════════════════════════════════════════════════
# Basic CLI commands
# ══════════════════════════════════════════════════════════════


class TestVersionInfo:
    """version and info commands."""

    def test_version_output(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.stdout.startswith("expflow v0.1.0")

    def test_info_output(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Platform:" in result.stdout
        assert "Python:" in result.stdout


# ══════════════════════════════════════════════════════════════
# clearml tasks
# ══════════════════════════════════════════════════════════════


class TestClearmlTasksCmd:
    """expflow clearml tasks sub-command."""

    def test_tasks_list(self, mock_clearml_cli):
        mock_clearml_cli.Task.get_tasks.return_value = [
            _make_mock_task("abc123", "run_1", "project_a", "completed"),
            _make_mock_task("def456", "run_2", "project_b", "running"),
        ]

        # Force re-import after mock setup
        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "tasks"])
        assert result.exit_code == 0
        assert "abc123" in result.stdout
        assert "run_1" in result.stdout
        assert "def456" in result.stdout
        assert "running" in result.stdout

    def test_tasks_empty(self, mock_clearml_cli):
        mock_clearml_cli.Task.get_tasks.return_value = []

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "tasks"])
        assert result.exit_code == 0
        assert "No tasks found." in result.stdout

    def test_tasks_with_filter(self, mock_clearml_cli):
        mock_clearml_cli.Task.get_tasks.return_value = [
            _make_mock_task("abc123", "run_1", "project_a", "completed"),
        ]

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "tasks", "--project", "project_a"])
        assert result.exit_code == 0
        assert "abc123" in result.stdout


# ══════════════════════════════════════════════════════════════
# clearml task
# ══════════════════════════════════════════════════════════════


class TestClearmlTaskCmd:
    """expflow clearml task <id> sub-command."""

    def test_task_detail(self, mock_clearml_cli):
        mock_clearml_cli.Task.get_task.return_value = _make_mock_task(
            "abc123", "run_1", "project_a", "failed", tags=["test", "gpu"]
        )

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "task", "abc123"])
        assert result.exit_code == 0
        assert "abc123" in result.stdout
        assert "run_1" in result.stdout
        assert "failed" in result.stdout
        assert "test, gpu" in result.stdout


# ══════════════════════════════════════════════════════════════
# clearml enqueue / dequeue
# ══════════════════════════════════════════════════════════════


class TestClearmlQueueCmd:
    """expflow clearml enqueue and dequeue."""

    def test_enqueue_output(self, mock_clearml_cli):
        mock_task = _make_mock_task("abc123", "run_1", "p", "queued")
        mock_clearml_cli.Task.get_task.return_value = mock_task

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "enqueue", "abc123"])
        assert result.exit_code == 0
        assert "abc123" in result.stdout
        assert "queued" in result.stdout

    def test_dequeue_output(self, mock_clearml_cli):
        mock_task = _make_mock_task("abc123", "run_1", "p", "created")
        mock_clearml_cli.Task.get_task.return_value = mock_task

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "dequeue", "abc123"])
        assert result.exit_code == 0
        assert "abc123" in result.stdout
        assert "dequeued" in result.stdout


# ══════════════════════════════════════════════════════════════
# clearml queues
# ══════════════════════════════════════════════════════════════


class TestClearmlQueuesCmd:
    """expflow clearml queues sub-command."""

    def test_queues_list(self, mock_clearml_cli):
        mock_clearml_cli.Queue.get_queues.return_value = [
            _make_mock_queue("q1", "default"),
            _make_mock_queue("q2", "gpu_queue"),
        ]

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "queues"])
        assert result.exit_code == 0
        assert "q1" in result.stdout
        assert "default" in result.stdout
        assert "gpu_queue" in result.stdout

    def test_queues_empty(self, mock_clearml_cli):
        mock_clearml_cli.Queue.get_queues.return_value = []

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "queues"])
        assert result.exit_code == 0
        assert "No queues found." in result.stdout


# ══════════════════════════════════════════════════════════════
# clearml dataset-register / dataset-list
# ══════════════════════════════════════════════════════════════


class TestClearmlDatasetCmd:
    """expflow clearml dataset-register and dataset-list."""

    def test_dataset_register_output(self, mock_clearml_cli):
        mock_ds = MagicMock()
        mock_ds.id = "ds1"
        mock_ds.name = "burgers_nu0.001"
        mock_ds.version = "1.0"
        mock_clearml_cli.Dataset.create.return_value = mock_ds

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(
            app,
            [
                "clearml",
                "dataset-register",
                "burgers_nu0.001",
                "--version",
                "1.0",
                "--path",
                "/data/burgers.hdf5",
                "--compliance",
                "allowed",
            ],
        )
        assert result.exit_code == 0
        assert "burgers_nu0.001" in result.stdout
        assert "allowed" in result.stdout
        assert "/data/burgers.hdf5" in result.stdout

    def test_dataset_list_output(self, mock_clearml_cli):
        def _make_ds(ds_id, name, version, compliance):
            ds = MagicMock()
            ds.id = ds_id
            ds.name = name
            ds.version = version
            ds.get_metadata.return_value = compliance
            return ds

        mock_clearml_cli.Dataset.list_datasets.return_value = [
            _make_ds("ds1", "burgers_nu0.001", "1.0", "allowed"),
            _make_ds("ds2", "ns_eq_re100", "1.0", "forbidden"),
        ]

        for mod in ["expflow.clearml", "expflow.cli_clearml"]:
            if mod in sys.modules:
                del sys.modules[mod]

        result = runner.invoke(app, ["clearml", "dataset-list"])
        assert result.exit_code == 0
        assert "burgers_nu0.001" in result.stdout
        assert "allowed" in result.stdout
        assert "ns_eq_re100" in result.stdout
        assert "forbidden" in result.stdout
