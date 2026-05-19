#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for expflow.optuna — Study CRUD, trial ask/tell, visualization.

All tests use mocked optuna SDK. No real optuna storage needed.

Mock strategy: patch 'optuna' in sys.modules before importing expflow.optuna.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Mock optuna helpers ──


def _make_mock_study(
    study_id: int = 1,
    name: str = "test_study",
    direction: str = "minimize",
) -> MagicMock:
    """Create a mock optuna Study with realistic attribute access."""
    study = MagicMock(name=f"Study({name})")
    study._study_id = study_id
    study.study_name = name
    study.direction.name = direction.upper()
    study.best_trial = MagicMock()
    study.best_trial.number = 0
    study.best_trial.params = {"lr": 0.001, "epochs": 100}
    study.best_trial.value = 0.05
    study.trials = [study.best_trial]
    return study


def _mock_optuna_pkg() -> MagicMock:
    """Create a mock 'optuna' package with create_study, delete_study, etc."""
    pkg = MagicMock(name="optuna_pkg")
    pkg.create_study = MagicMock()
    pkg.delete_study = MagicMock()
    pkg.load_study = MagicMock()
    pkg.get_all_study_summaries = MagicMock()
    pkg.visualization = MagicMock()
    return pkg


# ══════════════════════════════════════════════════════════════
# Fixture
# ══════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def mock_optuna_pkg() -> MagicMock:
    """Replace 'optuna' in sys.modules with a mock before imports."""
    pkg = _mock_optuna_pkg()

    for mod in ["expflow.optuna", "optuna"]:
        if mod in sys.modules:
            del sys.modules[mod]

    with patch.dict("sys.modules", {"optuna": pkg}):
        yield pkg

    for mod in ["expflow.optuna", "optuna"]:
        if mod in sys.modules:
            del sys.modules[mod]


# ══════════════════════════════════════════════════════════════
# Test: create_study
# ══════════════════════════════════════════════════════════════


class TestCreateStudy:
    """create_study() — create a new optuna study."""

    def test_create_study_returns_serialized(self, mock_optuna_pkg):
        """create_study returns dict with id, name, direction."""
        mock_study = _make_mock_study(1, "hpo_burgers", "minimize")
        mock_optuna_pkg.create_study.return_value = mock_study

        from expflow_pde.optuna import create_study

        result = create_study("hpo_burgers")

        assert result["study_id"] == 1
        assert result["name"] == "hpo_burgers"
        assert result["direction"] == "MINIMIZE"

    def test_create_study_passes_name(self, mock_optuna_pkg):
        """create_study forwards study_name to optuna."""
        mock_optuna_pkg.create_study.return_value = _make_mock_study(1, "test")

        from expflow_pde.optuna import create_study

        create_study("my_study", storage="sqlite:///optuna.db")

        mock_optuna_pkg.create_study.assert_called_with(
            study_name="my_study",
            storage="sqlite:///optuna.db",
            direction="minimize",
        )

    def test_create_study_maximize_direction(self, mock_optuna_pkg):
        """create_study with direction='maximize'."""
        mock_study = _make_mock_study(1, "max_study", "maximize")
        mock_optuna_pkg.create_study.return_value = mock_study

        from expflow_pde.optuna import create_study

        result = create_study("max_study", direction="maximize")

        assert result["direction"] == "MAXIMIZE"
        mock_optuna_pkg.create_study.assert_called_with(
            study_name="max_study",
            storage=None,
            direction="maximize",
        )


# ══════════════════════════════════════════════════════════════
# Test: list_studies
# ══════════════════════════════════════════════════════════════


class TestListStudies:
    """list_studies() — list all studies."""

    def test_list_studies_returns_serialized(self, mock_optuna_pkg):
        """list_studies returns list of study summaries."""
        mock_summary_1 = MagicMock()
        mock_summary_1._study_id = 1
        mock_summary_1.study_name = "study_a"
        mock_summary_1.direction.name = "MINIMIZE"
        mock_summary_1.best_trial = MagicMock(number=5, value=0.03)

        mock_summary_2 = MagicMock()
        mock_summary_2._study_id = 2
        mock_summary_2.study_name = "study_b"
        mock_summary_2.direction.name = "MAXIMIZE"
        mock_summary_2.best_trial = None

        mock_optuna_pkg.get_all_study_summaries.return_value = [mock_summary_1, mock_summary_2]

        from expflow_pde.optuna import list_studies

        result = list_studies()

        assert len(result) == 2
        assert result[0] == {
            "study_id": 1,
            "name": "study_a",
            "direction": "MINIMIZE",
            "best_value": 0.03,
            "best_trial": 5,
        }
        assert result[1]["best_value"] is None

    def test_list_studies_empty(self, mock_optuna_pkg):
        """No studies returns empty list."""
        mock_optuna_pkg.get_all_study_summaries.return_value = []

        from expflow_pde.optuna import list_studies

        assert list_studies() == []


# ══════════════════════════════════════════════════════════════
# Test: get_study
# ══════════════════════════════════════════════════════════════


class TestGetStudy:
    """get_study() — get study details."""

    def test_get_study_returns_serialized(self, mock_optuna_pkg):
        """get_study returns dict with full study info."""
        mock_study = _make_mock_study(1, "my_study", "minimize")
        mock_study.trials = [
            MagicMock(number=0, params={"lr": 0.01}, value=0.1),
            MagicMock(number=1, params={"lr": 0.001}, value=0.05),
        ]
        mock_optuna_pkg.load_study.return_value = mock_study

        from expflow_pde.optuna import get_study

        result = get_study("my_study")

        assert result["study_id"] == 1
        assert result["name"] == "my_study"
        assert result["direction"] == "MINIMIZE"
        assert result["best_trial"] == {
            "number": 0,
            "params": {"lr": 0.001, "epochs": 100},
            "value": 0.05,
        }
        assert len(result["trials"]) == 2

    def test_get_study_passes_storage(self, mock_optuna_pkg):
        """get_study forwards storage to optuna.load_study."""
        mock_optuna_pkg.load_study.return_value = _make_mock_study(1, "s")

        from expflow_pde.optuna import get_study

        get_study("my_study", storage="sqlite:///optuna.db")

        mock_optuna_pkg.load_study.assert_called_with(
            study_name="my_study",
            storage="sqlite:///optuna.db",
        )


# ══════════════════════════════════════════════════════════════
# Test: delete_study
# ══════════════════════════════════════════════════════════════


class TestDeleteStudy:
    """delete_study() — delete a study."""

    def test_delete_study_calls_optuna(self, mock_optuna_pkg):
        """delete_study forwards to optuna.delete_study."""
        from expflow_pde.optuna import delete_study

        delete_study("my_study")

        mock_optuna_pkg.delete_study.assert_called_with(
            study_name="my_study",
            storage=None,
        )

    def test_delete_study_returns_serialized(self, mock_optuna_pkg):
        """delete_study returns dict confirming deletion."""
        from expflow_pde.optuna import delete_study

        result = delete_study("my_study")

        assert result["status"] == "deleted"
        assert result["study_name"] == "my_study"


# ══════════════════════════════════════════════════════════════
# Test: ask_trial
# ══════════════════════════════════════════════════════════════


class TestAskTrial:
    """ask_trial() — get next trial parameters."""

    def test_ask_trial_returns_serialized(self, mock_optuna_pkg):
        """ask_trial returns dict with params."""
        mock_study = _make_mock_study(1, "my_study", "minimize")
        mock_trial = MagicMock()
        mock_trial.number = 3
        mock_trial.params = {"lr": 0.01, "epochs": 200}
        mock_study.ask.return_value = mock_trial
        mock_optuna_pkg.load_study.return_value = mock_study

        from expflow_pde.optuna import ask_trial

        result = ask_trial("my_study")

        assert result["trial_number"] == 3
        assert result["params"] == {"lr": 0.01, "epochs": 200}


# ══════════════════════════════════════════════════════════════
# Test: tell_trial
# ══════════════════════════════════════════════════════════════


class TestTellTrial:
    """tell_trial() — report trial result."""

    def test_tell_trial_returns_serialized(self, mock_optuna_pkg):
        """tell_trial returns dict confirming report."""
        mock_study = _make_mock_study(1, "my_study", "minimize")
        mock_optuna_pkg.load_study.return_value = mock_study

        from expflow_pde.optuna import tell_trial

        result = tell_trial("my_study", trial_number=3, value=0.05)

        mock_study.tell.assert_called_with(3, 0.05)
        assert result["status"] == "reported"
        assert result["trial_number"] == 3
        assert result["value"] == 0.05


# ══════════════════════════════════════════════════════════════
# Test: plot_study
# ══════════════════════════════════════════════════════════════


class TestPlotStudy:
    """plot_study() — generate optimization visualization."""

    def test_plot_study_history(self, mock_optuna_pkg):
        """plot_study('my_study', 'history') calls plotly and saves to file."""
        import os
        import tempfile

        mock_study = _make_mock_study(1, "my_study", "minimize")
        mock_optuna_pkg.load_study.return_value = mock_study

        # Mock plotly figure
        mock_fig = MagicMock()
        mock_optuna_pkg.visualization.plot_optimization_history.return_value = mock_fig

        from expflow_pde.optuna import plot_study

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            result = plot_study(
                "my_study",
                plot_type="history",
                output_path=output_path,
            )

            assert result["status"] == "saved"
            assert result["output_path"] == output_path
            assert result["plot_type"] == "history"
            mock_fig.write_html.assert_called_with(output_path)
        finally:
            os.unlink(output_path)

    def test_plot_study_unknown_type(self, mock_optuna_pkg):
        """Unknown plot type raises ValueError."""
        from expflow_pde.optuna import plot_study

        with pytest.raises(ValueError, match="Unknown plot type"):
            plot_study("my_study", plot_type="unknown")


# ══════════════════════════════════════════════════════════════
# Test: Error handling
# ══════════════════════════════════════════════════════════════


class TestOptunaErrors:
    """Error propagation from optuna SDK."""

    def test_create_study_raises(self, mock_optuna_pkg):
        """create_study propagates optuna errors."""
        mock_optuna_pkg.create_study.side_effect = ValueError("Study 'my_study' already exists.")

        from expflow_pde.optuna import create_study

        with pytest.raises(ValueError, match="already exists"):
            create_study("my_study")

    def test_get_nonexistent_study_raises(self, mock_optuna_pkg):
        """get_study propagates KeyError for nonexistent study."""
        mock_optuna_pkg.load_study.side_effect = KeyError("my_study")

        from expflow_pde.optuna import get_study

        with pytest.raises(KeyError):
            get_study("nonexistent")
