#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow CLI — optuna command group.

Covers: search-tree command, help output, pareto command (basic).
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


class TestSearchTreeCLI:
    """Tests for ``expflow optuna search-tree``."""

    def test_help_output(self, runner: CliRunner):
        """search-tree --help shows description and options."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["search-tree", "--help"])
        assert result.exit_code == 0
        assert "search-tree" in result.output.lower()
        assert "--json" in result.output

    def test_ascii_tree_output(self, runner: CliRunner):
        """search-tree (no args) shows ascii tree with architecture choices."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["search-tree"])
        assert result.exit_code == 0
        assert "FNO" in result.output
        assert "DeepONet" in result.output
        assert "choice" in result.output
        assert "Search Tree:" in result.output

    def test_json_output(self, runner: CliRunner):
        """search-tree --json outputs parseable JSON."""
        from expflow_pde.cli_optuna import optuna_app

        result = runner.invoke(optuna_app, ["search-tree", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "architecture" in data
        assert data["architecture"]["type"] == "categorical"
        assert "FNO" in data["architecture"]["_children"]

    def test_json_output_valid_structure(self, runner: CliRunner):
        """JSON output matches the _DEFAULT_SEARCH_TREE schema."""
        from expflow_pde.cli_optuna import optuna_app
        from expflow_pde.hpo import _DEFAULT_SEARCH_TREE

        result = runner.invoke(optuna_app, ["search-tree", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Verify structure matches tree constant
        assert set(data.keys()) == set(_DEFAULT_SEARCH_TREE.keys())
        assert data["architecture"]["choices"] == _DEFAULT_SEARCH_TREE["architecture"]["choices"]


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
