#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.pipeline — ExperimentPipeline high-level class.

Uses mocking to avoid clearml SDK dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config cache between tests."""
    from expflow_pde import config

    config._config_cache.clear()
    yield


@pytest.fixture
def mock_clearml():
    """Mock clearml module functions that pipeline.py imports at call time.

    Uses individual patch() calls because patch.multiple() with keyword-arg
    MagicMock objects returns an empty dict.
    """
    m_create = MagicMock(
        return_value={
            "pipeline_id": "pipe_abc123",
            "name": "pipeline_test",
            "project": "PDEBench",
            "version": "0.1.0",
            "status": "created",
        }
    )
    m_add = MagicMock(
        return_value={
            "pipeline_name": "pipeline_test",
            "step_name": "train",
            "parents": [],
            "status": "defined",
        }
    )
    m_start = MagicMock(
        return_value={
            "pipeline_name": "pipeline_test",
            "status": "started",
            "started": True,
        }
    )

    patcher_create = patch("expflow_pde.clearml.pipeline_create", m_create)
    patcher_add = patch("expflow_pde.clearml.pipeline_add_step", m_add)
    patcher_start = patch("expflow_pde.clearml.pipeline_start", m_start)

    patcher_create.start()
    patcher_add.start()
    patcher_start.start()
    yield {"create": m_create, "add": m_add, "start": m_start}
    patcher_start.stop()
    patcher_add.stop()
    patcher_create.stop()


@pytest.fixture
def mock_clearml_with_eval():
    """Mock clearml with a smarter add_step that tracks step names."""
    call_log = []

    def add_side_effect(**kwargs):
        step_name = kwargs.get("step_name", "train")
        parents = kwargs.get("parents", [])
        entry = {"step_name": step_name, "parents": parents, "kwargs": kwargs}
        call_log.append(entry)
        return {
            "pipeline_name": kwargs.get("pipeline_name", "pipeline_test"),
            "step_name": step_name,
            "parents": parents,
            "status": "defined",
        }

    m_create = MagicMock(
        return_value={
            "pipeline_id": "pipe_abc123",
            "name": "pipeline_test",
            "project": "PDEBench",
            "version": "0.1.0",
            "status": "created",
        }
    )
    m_add = MagicMock(side_effect=add_side_effect)
    m_start = MagicMock(
        return_value={
            "pipeline_name": "pipeline_test",
            "status": "started",
            "started": True,
        }
    )

    patcher_create = patch("expflow_pde.clearml.pipeline_create", m_create)
    patcher_add = patch("expflow_pde.clearml.pipeline_add_step", m_add)
    patcher_start = patch("expflow_pde.clearml.pipeline_start", m_start)

    patcher_create.start()
    patcher_add.start()
    patcher_start.start()
    yield {"create": m_create, "add": m_add, "start": m_start, "call_log": call_log}
    patcher_start.stop()
    patcher_add.stop()
    patcher_create.stop()


# ── Tests: ExperimentPipeline class ──


class TestExperimentPipeline:
    """Tests for the ExperimentPipeline orchestrator."""

    def test_train_only_submit(self, mock_clearml):
        """Submit a train-only pipeline."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline(project="PDEBench", queue="gpu_queue")
        result = ep.train_val_submit(
            train_script="train_task1.py",
            train_params={"epochs": 80, "lr": 0.001},
        )

        assert result["name"] == "pipeline_test"
        assert result["project"] == "PDEBench"
        assert result["queue"] == "gpu_queue"
        assert result["status"] == "started"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["name"] == "train"

        # Verify clearml create was called
        mock_clearml["create"].assert_called_once()

        # Verify add_step was called for train
        mock_clearml["add"].assert_called_once()
        call_kwargs = mock_clearml["add"].call_args[1]
        assert call_kwargs["step_name"] == "train"
        assert call_kwargs["base_task_name"] == "train_task1.py"
        assert call_kwargs["execution_queue"] == "gpu_queue"

        # Verify parameter override
        params = call_kwargs.get("parameter_override", {})
        assert params.get("Args", {}).get("--epochs") == "80"
        assert params.get("Args", {}).get("--lr") == "0.001"

        # Verify start was called
        mock_clearml["start"].assert_called_once()

    def test_train_eval_submit(self, mock_clearml_with_eval):
        """Submit a train -> eval pipeline."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline(project="PDEBench", queue="gpu_queue")
        result = ep.train_val_submit(
            train_script="train_task1.py",
            train_params={"epochs": 80},
            eval_script="eval_task1.py",
            eval_params={"sub_step": 5, "val_only": True},
        )

        assert len(result["steps"]) == 2
        assert result["steps"][0]["name"] == "train"
        assert result["steps"][1]["name"] == "eval"

        # Verify add_step was called twice (train + eval)
        assert mock_clearml_with_eval["add"].call_count == 2

        # Check call log for eval step
        call_log = mock_clearml_with_eval["call_log"]
        assert len(call_log) == 2
        eval_call = call_log[1]
        assert eval_call["step_name"] == "eval"
        assert eval_call["parents"] == ["train"]

        # Check parameter override
        eval_kwargs = eval_call["kwargs"]
        params = eval_kwargs.get("parameter_override", {})
        assert params.get("Args", {}).get("--sub_step") == "5"
        assert params.get("Args", {}).get("--val_only") == "True"

    def test_custom_pipeline_name(self, mock_clearml):
        """Submit with custom pipeline name."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline()
        result = ep.train_val_submit(
            train_script="train_task1.py",
            pipeline_name="my_custom_pipeline",
        )

        # The name from the mock is always "pipeline_test" since
        # we mocked pipeline_create's return value
        assert result["name"] == "pipeline_test"

    def test_custom_version(self, mock_clearml):
        """Submit with custom version."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline()
        ep.train_val_submit(
            train_script="train_task1.py",
            version="2.0.0",
        )

        # Verify version was passed to pipeline_create
        create_kwargs = mock_clearml["create"].call_args[1]
        assert create_kwargs["version"] == "2.0.0"

    def test_no_params(self, mock_clearml):
        """Submit with no parameters at all."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline()
        result = ep.train_val_submit(
            train_script="train_task1.py",
        )

        assert result["status"] == "started"
        assert len(result["steps"]) == 1
        mock_clearml["create"].assert_called_once()
        mock_clearml["start"].assert_called_once()

    def test_last_result_property(self, mock_clearml):
        """Verify last_result returns the most recent result."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline()
        assert ep.last_result is None

        result = ep.train_val_submit(train_script="train_task1.py")
        assert ep.last_result == result

    def test_docker_passed_through(self, mock_clearml):
        """Docker image should be passed to pipeline_create."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline(
            project="PDEBench",
            docker="nvcr.io/nvidia/pytorch:24.01-py3",
        )
        ep.train_val_submit(train_script="train_task1.py")

        create_kwargs = mock_clearml["create"].call_args[1]
        assert create_kwargs["docker"] == "nvcr.io/nvidia/pytorch:24.01-py3"

    def test_abort_on_failure_override(self, mock_clearml):
        """Instance-level abort_on_failure should be passed to create."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline(abort_on_failure=True)
        ep.train_val_submit(train_script="train_task1.py")

        create_kwargs = mock_clearml["create"].call_args[1]
        assert create_kwargs["abort_on_failure"] is True

    def test_timeout_passed_to_start(self, mock_clearml):
        """Timeout minutes should be passed to pipeline_start."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline()
        ep.train_val_submit(train_script="train_task1.py", timeout=30.0)

        start_kwargs = mock_clearml["start"].call_args[1]
        assert start_kwargs["timeout_minutes"] == 30.0

    def test_no_train_params(self, mock_clearml):
        """When no train_params, no parameter_override should be set."""
        from expflow_pde.pipeline import ExperimentPipeline

        ep = ExperimentPipeline()
        ep.train_val_submit(train_script="train_task1.py")

        add_kwargs = mock_clearml["add"].call_args[1]
        assert add_kwargs.get("parameter_override") is None


# ── Tests: CLI commands ──


class TestPipelineCLI:
    """Tests for the pipeline submit CLI command."""

    def test_submit_via_cli(self, mock_clearml):
        """Test pipeline submit CLI triggers the correct flow."""
        from typer.testing import CliRunner

        from expflow_pde.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "pipeline",
                "submit",
                "train_task1.py",
                "--queue",
                "gpu_queue",
                "--project",
                "PDEBench",
                "--train-param",
                "epochs=80",
            ],
        )

        assert result.exit_code == 0
        assert "Pipeline submitted" in result.stdout
        # Mock returns the name "pipeline_test"
        assert "pipeline_test" in result.stdout

    def test_submit_with_eval_via_cli(self, mock_clearml):
        """Test pipeline submit with eval script via CLI."""
        from typer.testing import CliRunner

        from expflow_pde.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "pipeline",
                "submit",
                "train_task1.py",
                "--eval-script",
                "eval_task1.py",
                "--train-param",
                "epochs=80",
                "--eval-param",
                "sub_step=5",
            ],
        )

        assert result.exit_code == 0
        assert "Steps:    2" in result.stdout

    def test_submit_help(self):
        """Pipeline submit help should show usage."""
        import sys as _sys

        for _m in list(_sys.modules.keys()):
            if _m.startswith("expflow_pde") or _m == "typer":
                del _sys.modules[_m]

        from typer.testing import CliRunner

        from expflow_pde.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["pipeline", "submit", "--help"])

        assert result.exit_code == 0
        assert "fast train -> eval pipeline" in result.stdout
        assert "--eval-script" in result.stdout
        assert "--train-param" in result.stdout
        assert "--skip" in result.stdout

    def test_submit_full_help(self):
        """Pipeline submit-full help should show usage."""
        import sys as _sys

        for _m in list(_sys.modules.keys()):
            if _m.startswith("expflow_pde") or _m == "typer":
                del _sys.modules[_m]

        from typer.testing import CliRunner

        from expflow_pde.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["pipeline", "submit-full", "--help"])

        assert result.exit_code == 0
        assert "HPO -> train -> eval pipeline" in result.stdout
        assert "--trials" in result.stdout
        assert "--parallel" in result.stdout
        assert "--metric" in result.stdout
        assert "--pruner" in result.stdout

    def test_submit_full_via_cli(self, mock_clearml):
        """Test pipeline submit-full CLI triggers HPO mode."""
        from typer.testing import CliRunner

        from expflow_pde.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "pipeline",
                "submit-full",
                "train_task1.py",
                "--queue",
                "default",
                "--trials",
                "30",
                "--parallel",
                "4",
                "--eval-script",
                "eval_task1.py",
            ],
        )

        assert result.exit_code == 0
        assert "mode: full" in result.stdout
        assert "Steps:" in result.stdout
        assert "HPO:" in result.stdout
