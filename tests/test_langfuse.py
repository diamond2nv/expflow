#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.langfuse — Langfuse tracsintegration.

Tests patch thin wrapper `_clearml_get_task` and `_get_client` instead of
the real clearml/langfuse modules, to avoid numpy re-import crashes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_config():
    from expflow_pde import config

    config._config_cache.clear()


@pytest.fixture
def mock_clearml_task():
    task = MagicMock()
    task.id = "abc123def456"
    task.name = "test_experiment"
    task.project = "PDEBench"
    task.status = "completed"
    task.get_tags = MagicMock(return_value=["task1", "sub_step5"])
    task.get_parameters = MagicMock(
        return_value={
            "Args/--lr": "0.001",
            "Args/--epochs": "80",
            "Args/--batch_size": "64",
            "_hidden": "should_be_skipped",
        }
    )
    task.get_last_scalar_metrics = MagicMock(
        return_value={
            "Score": {
                "seg_total": {"last": 57.09},
                "seg1": {"last": 33.64},
            },
            "PDE": {
                "pde_mean": {"last": 18.29},
            },
        }
    )
    return task


@pytest.fixture
def mock_langfuse_client():
    client = MagicMock()
    trace_obj = MagicMock()
    trace_obj.id = "lf_trace_xyz789"
    client.trace = MagicMock(return_value=trace_obj)
    return client


# ── Tests: trace_experiment ──


class TestTraceExperiment:
    """Tests for trace_experiment function.

    Patches _clearml_get_task (a thin wrapper that lazy-imports clearml)
    and _get_client (lazy-imports langfuse) to avoid actual imports.
    """

    def test_trace_experiment_syncs_task(self, mock_clearml_task, mock_langfuse_client):
        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=mock_clearml_task),
        ):
            from expflow_pde.langfuse import trace_experiment

            result = trace_experiment(task_id="abc123def456")

            assert result["langfuse_trace_id"] == "lf_trace_xyz789"
            assert result["task_id"] == "abc123def456"
            assert result["task_name"] == "test_experiment"
            assert result["project"] == "PDEBench"
            assert result["status"] == "completed"
            assert result["metrics_count"] == 3
            assert result["params_count"] == 3

            mock_langfuse_client.trace.assert_called_once()
            ck = mock_langfuse_client.trace.call_args[1]
            assert ck["name"] == "clearml:abc123def456"
            assert ck["input"]["project"] == "PDEBench"
            assert ck["input"]["tags"] == ["task1", "sub_step5"]
            assert ck["output"]["metrics"]["Score/seg_total"] == 57.09
            assert ck["output"]["status"] == "completed"
            assert ck["metadata"]["source"] == "expflow"

    def test_trace_experiment_custom_name(self, mock_clearml_task, mock_langfuse_client):
        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=mock_clearml_task),
        ):
            from expflow_pde.langfuse import trace_experiment

            trace_experiment(task_id="abc123def456", trace_name="my_trace")
            ck = mock_langfuse_client.trace.call_args[1]
            assert ck["name"] == "my_trace"

    def test_trace_experiment_session_id(self, mock_clearml_task, mock_langfuse_client):
        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=mock_clearml_task),
        ):
            from expflow_pde.langfuse import trace_experiment

            trace_experiment(task_id="abc123def456", session_id="pdebench:hpo_v2")
            ck = mock_langfuse_client.trace.call_args[1]
            assert ck["session_id"] == "pdebench:hpo_v2"

    def test_trace_experiment_parent_trace(self, mock_clearml_task, mock_langfuse_client):
        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=mock_clearml_task),
        ):
            from expflow_pde.langfuse import trace_experiment

            result = trace_experiment(
                task_id="abc123def456",
                parent_trace_id="lf_hermes_decision_xyz",
            )
            assert result["parent_trace_id"] == "lf_hermes_decision_xyz"
            ck = mock_langfuse_client.trace.call_args[1]
            assert ck["metadata"]["parent_trace_id"] == "lf_hermes_decision_xyz"

    def test_trace_experiment_task_not_found(self):
        with patch("expflow_pde.langfuse._clearml_get_task", side_effect=Exception("Not found")):
            from expflow_pde.langfuse import trace_experiment

            with pytest.raises(ValueError, match="clearml Task not found"):
                trace_experiment(task_id="nonexistent")

    def test_trace_experiment_empty_metrics(self, mock_langfuse_client):
        empty_task = MagicMock()
        empty_task.id = "empty123"
        empty_task.name = "empty_task"
        empty_task.project = "PDEBench"
        empty_task.status = "failed"
        empty_task.get_tags = MagicMock(return_value=[])
        empty_task.get_parameters = MagicMock(return_value={})
        empty_task.get_last_scalar_metrics = MagicMock(return_value={})

        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=empty_task),
        ):
            from expflow_pde.langfuse import trace_experiment

            result = trace_experiment(task_id="empty123")
            assert result["status"] == "failed"
            assert result["metrics_count"] == 0
            assert result["params_count"] == 0


class TestTraceExperimentCLI:
    def test_help(self):
        import sys as _sys

        for _m in list(_sys.modules.keys()):
            if _m.startswith("expflow_pde") or _m == "typer":
                del _sys.modules[_m]

        from typer.testing import CliRunner

        from expflow_pde.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["langfuse", "trace-experiment", "--help"])
        assert result.exit_code == 0
        assert "Sync a clearml experiment" in result.stdout
        assert "--parent-trace" in result.stdout
        assert "--parent-task-id" in result.stdout


# ── Tests: _resolve_session_id (three-tier fallback) ──


class TestResolveSessionID:
    """Three-tier fallback for session_id resolution."""

    def test_explicit_wins(self, mock_langfuse_client, mock_clearml_task):
        """Tier 1: explicit session_id overrides everything."""
        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=mock_clearml_task),
        ):
            from expflow_pde.langfuse import trace_experiment

            trace_experiment(
                task_id="abc123def456",
                session_id="my_session",
                parent_task_id="parent_task_id",
            )
            ck = mock_langfuse_client.trace.call_args[1]
            assert ck["session_id"] == "my_session"

    def test_inherit_from_parent(self, mock_langfuse_client):
        """Tier 2: inherit expflow:langfuse_session_id from parent clearml Task."""
        parent_task = MagicMock()
        parent_task.get_metadata = MagicMock(
            return_value={"expflow:langfuse_session_id": "inherited_session_abc"}
        )

        from expflow_pde.langfuse import _resolve_session_id

        with patch("expflow_pde.langfuse._clearml_get_task", return_value=parent_task):
            sid = _resolve_session_id(session_id=None, parent_task_id="parent123")
            assert sid == "inherited_session_abc"

    def test_auto_generate_snowflake(self, mock_langfuse_client):
        """Tier 3: auto-generate snowflake ID when nothing else is available."""
        from expflow_pde.langfuse import _resolve_session_id

        with patch(
            "expflow_pde.snowflake.snowflake_session_id", return_value="exp:snow_9876543210"
        ):
            sid = _resolve_session_id(session_id=None, parent_task_id=None)
            assert sid == "exp:snow_9876543210"
            assert sid.startswith("exp:snow_")

    def test_empty_explicit_falls_through(self, mock_langfuse_client):
        """Empty string or whitespace-only explicit session falls through to auto."""
        from expflow_pde.langfuse import _resolve_session_id

        with patch("expflow_pde.snowflake.snowflake_session_id", return_value="exp:snow_1111"):
            sid = _resolve_session_id(session_id="", parent_task_id=None)
            assert sid == "exp:snow_1111"

            sid = _resolve_session_id(session_id="   ", parent_task_id=None)
            assert sid == "exp:snow_1111"

    def test_parent_metadata_missing_falls_through(self, mock_langfuse_client):
        """Parent Task exists but has no expflow:langfuse_session_id -> Tier 3."""
        parent_task = MagicMock()
        parent_task.get_metadata = MagicMock(return_value={})

        from expflow_pde.langfuse import _resolve_session_id

        with (
            patch("expflow_pde.langfuse._clearml_get_task", return_value=parent_task),
            patch("expflow_pde.snowflake.snowflake_session_id", return_value="exp:snow_2222"),
        ):
            sid = _resolve_session_id(session_id=None, parent_task_id="parent123")
            assert sid == "exp:snow_2222"

    def test_parent_not_found_falls_through(self, mock_langfuse_client):
        """Parent Task load fails -> Tier 3."""
        from expflow_pde.langfuse import _resolve_session_id

        with (
            patch("expflow_pde.langfuse._clearml_get_task", side_effect=Exception("boom")),
            patch("expflow_pde.snowflake.snowflake_session_id", return_value="exp:snow_3333"),
        ):
            sid = _resolve_session_id(session_id=None, parent_task_id="parent123")
            assert sid == "exp:snow_3333"

    def test_trace_experiment_auto_session(self, mock_clearml_task, mock_langfuse_client):
        """Integration: trace_experiment without session_id produces non-empty snowflake session."""
        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=mock_clearml_task),
            patch("expflow_pde.snowflake.snowflake_session_id", return_value="exp:snow_5555"),
        ):
            from expflow_pde.langfuse import trace_experiment

            result = trace_experiment(task_id="abc123def456")
            assert result["session_id"] == "exp:snow_5555"
            assert result["session_id"] != ""
            ck = mock_langfuse_client.trace.call_args[1]
            assert ck["session_id"] == "exp:snow_5555"

    def test_metadata_writes_session_id(self, mock_clearml_task, mock_langfuse_client):
        """Metadata is always written back to clearml Task."""
        with (
            patch("expflow_pde.langfuse._get_client", return_value=mock_langfuse_client),
            patch("expflow_pde.langfuse._clearml_get_task", return_value=mock_clearml_task),
            patch("expflow_pde.snowflake.snowflake_session_id", return_value="exp:snow_6666"),
        ):
            from expflow_pde.langfuse import trace_experiment

            trace_experiment(task_id="abc123def456")
            mock_clearml_task.set_metadata.assert_any_call(
                "expflow:langfuse_session_id", "exp:snow_6666"
            )
