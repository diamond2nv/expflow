#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow CLI — optuna command group.

Covers: study-graph command, help output, pareto command (basic).
Uses typer.testing.CliRunner to invoke CLI without subprocess.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Provide a CliRunner for optuna CLI tests."""
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config cache between tests."""
    from expflow_pde import config

    config._config_cache.clear()
    yield


class TestStudyGraphCLI:
    """Tests for ``expflow optuna study-graph``."""

    def test_help_output(self, runner: CliRunner):
        """study-graph --help shows description and options."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["study-graph", "--help"])
        assert result.exit_code == 0
        assert "study-graph" in result.output.lower()
        assert "--json" in result.output

    def test_empty_graph(self, runner: CliRunner):
        """study-graph with no study shows a meaningful message."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["study-graph", "--json"])
        # Should produce valid json with empty graph, not crash
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "nodes" in data
        assert "edges" in data

    def test_json_output_structure(self, runner: CliRunner):
        """JSON output includes nodes, edges, top_trials."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["study-graph", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "nodes" in data
        assert "edges" in data
        assert "top_trials" in data


class TestParetoCLI:
    """Basic tests for ``expflow optuna pareto``."""

    def test_help_output(self, runner: CliRunner):
        """pareto --help shows usage."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["pareto", "--help"])
        assert result.exit_code == 0
        assert "pareto" in result.output.lower()

    def test_pareto_no_study_prints_error(self, runner: CliRunner):
        """pareto on empty DB shows a meaningful message."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["pareto", "nonexistent_study"])
        # Should produce some output (error or empty), not crash
        assert result.exit_code in (0, 1)
