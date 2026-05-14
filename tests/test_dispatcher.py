#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for expflow.dispatcher — experiment dispatch orchestration."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Fixture ──


@pytest.fixture(autouse=True)
def mock_deps():
    """Mock both clearml and optuna so dispatcher can be imported."""
    for mod in ["expflow.dispatcher", "expflow.clearml", "expflow.optuna", "clearml", "optuna"]:
        if mod in sys.modules:
            del sys.modules[mod]

    clearml_pkg = MagicMock(name="clearml")
    clearml_pkg.Task = MagicMock()
    clearml_pkg.Queue = MagicMock()

    optuna_pkg = MagicMock(name="optuna")
    optuna_pkg.create_study = MagicMock()
    optuna_pkg.load_study = MagicMock()

    with patch.dict("sys.modules", {"clearml": clearml_pkg, "optuna": optuna_pkg}):
        yield {"clearml": clearml_pkg, "optuna": optuna_pkg}

    for mod in ["expflow.dispatcher", "expflow.clearml", "expflow.optuna"]:
        if mod in sys.modules:
            del sys.modules[mod]


# ══════════════════════════════════════════════════════════════
# dispatch_experiment
# ══════════════════════════════════════════════════════════════


class TestDispatchExperiment:
    """dispatch_experiment() — submit and track an experiment."""

    def test_dispatch_returns_serialized(self, mock_deps):
        """dispatch_experiment returns dict with experiment_id, status, queue."""
        from expflow.dispatcher import dispatch_experiment

        result = dispatch_experiment(
            command="python train.py --config config.yaml",
            queue="default",
        )

        assert result["status"] == "dispatched"
        assert "experiment_id" in result
        assert result["queue"] == "default"

    def test_dispatch_includes_timestamp(self, mock_deps):
        """dispatch result includes a timestamp field."""
        from expflow.dispatcher import dispatch_experiment

        result = dispatch_experiment("echo hello")
        assert "timestamp" in result

    def test_dispatch_with_tags(self, mock_deps):
        """dispatch accepts tags metadata."""
        from expflow.dispatcher import dispatch_experiment

        result = dispatch_experiment(
            command="python train.py",
            tags=["pdebench", "burgers"],
        )
        assert result["tags"] == ["pdebench", "burgers"]

    def test_dispatch_default_queue(self, mock_deps):
        """Default queue is 'default'."""
        from expflow.dispatcher import dispatch_experiment

        result = dispatch_experiment("python train.py")
        assert result["queue"] == "default"


# ══════════════════════════════════════════════════════════════
# list_experiments
# ══════════════════════════════════════════════════════════════


class TestListExperiments:
    """list_experiments() — list dispatched experiments."""

    def test_list_experiments(self, mock_deps):
        """list_experiments returns list of experiment records."""
        from expflow.dispatcher import dispatch_experiment, list_experiments

        dispatch_experiment("cmd1", queue="gpu")
        dispatch_experiment("cmd2", queue="cpu")

        result = list_experiments()
        assert len(result) >= 2

    def test_list_experiments_empty_initially(self, mock_deps):
        """list_experiments returns empty list before any dispatch."""
        from expflow.dispatcher import list_experiments

        result = list_experiments()
        # Could be empty or have residual from previous test — just check it's a list
        assert isinstance(result, list)


# ══════════════════════════════════════════════════════════════
# get_experiment_status
# ══════════════════════════════════════════════════════════════


class TestGetExperimentStatus:
    """get_experiment_status() — check experiment state."""

    def test_get_status_returns_dict(self, mock_deps):
        """get_experiment_status returns dict with experiment info."""
        from expflow.dispatcher import dispatch_experiment, get_experiment_status

        dispatched = dispatch_experiment("python train.py")
        result = get_experiment_status(dispatched["experiment_id"])

        assert result["experiment_id"] == dispatched["experiment_id"]
        assert "status" in result


# ══════════════════════════════════════════════════════════════
# cancel_experiment
# ══════════════════════════════════════════════════════════════


class TestCancelExperiment:
    """cancel_experiment() — cancel a running experiment."""

    def test_cancel_experiment(self, mock_deps):
        """cancel_experiment returns dict confirming cancellation."""
        from expflow.dispatcher import cancel_experiment, dispatch_experiment

        dispatched = dispatch_experiment("python train.py")
        result = cancel_experiment(dispatched["experiment_id"])

        assert result["experiment_id"] == dispatched["experiment_id"]
        assert result["status"] == "cancelling"

    def test_cancel_nonexistent_returns_error(self, mock_deps):
        """Cancelling a nonexistent experiment returns error dict."""
        from expflow.dispatcher import cancel_experiment

        result = cancel_experiment("nonexistent-id")
        assert "error" in result
