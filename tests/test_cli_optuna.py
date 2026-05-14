#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI tests for expflow optuna sub-commands via CliRunner.

Mock optuna SDK using sys.modules patch to verify CLI output format.
"""

import sys
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from expflow.cli import app

runner = CliRunner()


# ── Helpers ──


def _make_mock_study(
    study_id: int = 1,
    name: str = "test_study",
    direction: str = "minimize",
) -> MagicMock:
    """Create a mock optuna Study."""
    study = MagicMock(name=f"Study({name})")
    study._study_id = study_id
    study.study_name = name
    study.direction.name = direction.upper()
    study.best_trial = MagicMock()
    study.best_trial.number = 0
    study.best_trial.params = {"lr": 0.001, "epochs": 100}
    study.best_trial.value = 0.05
    study.trials = [
        MagicMock(number=0, params={"lr": 0.01}, value=0.1),
        MagicMock(number=1, params={"lr": 0.001}, value=0.05),
    ]
    return study


def _mock_optuna_pkg() -> MagicMock:
    pkg = MagicMock(name="optuna_pkg")
    pkg.create_study = MagicMock()
    pkg.delete_study = MagicMock()
    pkg.load_study = MagicMock()
    pkg.get_all_study_summaries = MagicMock()
    pkg.visualization = MagicMock()
    return pkg


# ── Fixture ──


def _setup_mock_optuna():
    """Setup mock optuna package in sys.modules before CLI import."""
    pkg = _mock_optuna_pkg()

    for mod in ["expflow.optuna", "expflow.cli_optuna"]:
        if mod in sys.modules:
            del sys.modules[mod]

    return patch.dict("sys.modules", {"optuna": pkg}), pkg


# ══════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════


class TestOptunaCli:
    """expflow optuna sub-commands."""

    def test_create_study(self):
        mock_pkg = _mock_optuna_pkg()
        mock_pkg.create_study.return_value = _make_mock_study(1, "hpo_test", "minimize")

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(
                app,
                [
                    "optuna",
                    "create-study",
                    "hpo_test",
                ],
            )

        assert result.exit_code == 0
        assert "hpo_test" in result.stdout
        assert "MINIMIZE" in result.stdout

    def test_create_study_maximize(self):
        mock_pkg = _mock_optuna_pkg()
        mock_pkg.create_study.return_value = _make_mock_study(1, "hpo_max", "maximize")

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(
                app,
                [
                    "optuna",
                    "create-study",
                    "hpo_max",
                    "--direction",
                    "maximize",
                ],
            )

        assert result.exit_code == 0
        assert "MAXIMIZE" in result.stdout

    def test_list_studies(self):
        mock_pkg = _mock_optuna_pkg()
        sum1 = MagicMock()
        sum1._study_id = 1
        sum1.study_name = "study_a"
        sum1.direction.name = "MINIMIZE"
        sum1.best_trial = MagicMock(number=3, value=0.05)

        sum2 = MagicMock()
        sum2._study_id = 2
        sum2.study_name = "study_b"
        sum2.direction.name = "MAXIMIZE"
        sum2.best_trial = None

        mock_pkg.get_all_study_summaries.return_value = [sum1, sum2]

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(app, ["optuna", "studies"])

        assert result.exit_code == 0
        assert "study_a" in result.stdout
        assert "study_b" in result.stdout
        assert "0.050000" in result.stdout

    def test_list_studies_empty(self):
        mock_pkg = _mock_optuna_pkg()
        mock_pkg.get_all_study_summaries.return_value = []

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(app, ["optuna", "studies"])

        assert result.exit_code == 0
        assert "No studies found." in result.stdout

    def test_get_study(self):
        mock_pkg = _mock_optuna_pkg()
        mock_pkg.load_study.return_value = _make_mock_study(1, "my_study", "minimize")

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(app, ["optuna", "study", "my_study"])

        assert result.exit_code == 0
        assert "my_study" in result.stdout
        assert "MINIMIZE" in result.stdout
        assert "0.05" in result.stdout

    def test_delete_study(self):
        mock_pkg = _mock_optuna_pkg()

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(app, ["optuna", "delete-study", "my_study"])

        assert result.exit_code == 0
        assert "deleted" in result.stdout

    def test_ask_trial(self):
        mock_pkg = _mock_optuna_pkg()
        mock_study = _make_mock_study(1, "my_study", "minimize")
        mock_trial = MagicMock()
        mock_trial.number = 3
        mock_trial.params = {"lr": 0.01, "epochs": 200}
        mock_study.ask.return_value = mock_trial
        mock_pkg.load_study.return_value = mock_study

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(app, ["optuna", "ask", "my_study"])

        assert result.exit_code == 0
        assert "Trial #3" in result.stdout
        assert "lr: 0.01" in result.stdout

    def test_tell_trial(self):
        mock_pkg = _mock_optuna_pkg()
        mock_study = _make_mock_study(1, "my_study", "minimize")
        mock_pkg.load_study.return_value = mock_study

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with patch.dict("sys.modules", {"optuna": mock_pkg}):
            result = runner.invoke(
                app,
                [
                    "optuna",
                    "tell",
                    "my_study",
                    "3",
                    "0.05",
                ],
            )

        assert result.exit_code == 0
        assert "Trial #3" in result.stdout
        assert "0.05" in result.stdout

    def test_plot(self):
        import os
        import tempfile

        mock_pkg = _mock_optuna_pkg()
        mock_study = _make_mock_study(1, "my_study", "minimize")
        mock_pkg.load_study.return_value = mock_study
        mock_fig = MagicMock()
        mock_pkg.visualization.plot_optimization_history.return_value = mock_fig

        for mod in ["expflow.optuna", "expflow.cli_optuna"]:
            if mod in sys.modules:
                del sys.modules[mod]

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            with patch.dict("sys.modules", {"optuna": mock_pkg}):
                result = runner.invoke(
                    app,
                    [
                        "optuna",
                        "plot",
                        "my_study",
                        "--output",
                        output_path,
                    ],
                )

            assert result.exit_code == 0
            assert output_path in result.stdout
        finally:
            os.unlink(output_path)
