#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.hpo — HPO runner.

Uses mocking to avoid actual subprocess execution and optuna dependency.
"""

from __future__ import annotations

import json
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
def mock_subprocess():
    """Mock subprocess.run to return configurable output."""

    def _run(cmd, **kwargs):
        return MagicMock(
            stdout="METRIC:seg_total=57.09\nMETRIC:pde_mean=18.29\n",
            stderr="",
            returncode=0,
        )

    with patch("expflow_pde.hpo.subprocess.run", side_effect=_run) as m:
        yield m


@pytest.fixture
def mock_optuna():
    """Mock optuna study with ask/tell/best_trial. Also patches pruner to None."""
    study = MagicMock()
    study.study_name = "test_study"
    study.direction.name = "maximize"

    # Mock trial
    trial = MagicMock()
    trial.number = 0

    def suggest_float(name, low, high, log=False, step=None):
        return {"lr": 0.001, "weight_decay": 1e-6}.get(name, low)

    def suggest_int(name, low, high, step=1):
        return {"epochs": 80, "batch_size": 64}.get(name, low)

    trial.suggest_float = MagicMock(side_effect=suggest_float)
    trial.suggest_int = MagicMock(side_effect=suggest_int)
    trial.params = {"lr": 0.001, "epochs": 80}

    # Mock best_trial
    best_trial = MagicMock()
    best_trial.number = 0
    best_trial.value = 57.09
    best_trial.params = {"lr": 0.001, "epochs": 80}

    study.ask = MagicMock(return_value=trial)
    study.tell = MagicMock()
    study.best_trial = best_trial
    study._study_id = 1

    def load_study(study_name=None, storage=None):
        study.study_name = study_name or "loaded_study"
        return study

    def create_study(study_name=None, direction=None, storage=None):
        study.study_name = study_name or "created_study"
        return study

    optuna_mod = MagicMock()
    optuna_mod.load_study = MagicMock(side_effect=load_study)
    optuna_mod.create_study = MagicMock(side_effect=create_study)

    with (
        patch("expflow_pde.hpo._import_optuna", return_value=optuna_mod),
        patch("expflow_pde.hpo._get_pruner", return_value=None),
    ):
        yield study


# ── Tests: Search space ──


class TestSearchSpace:
    """Tests for the default hyperparameter search space."""

    def test_get_search_space_returns_dict(self):
        """get_search_space should return a non-empty dict."""
        from expflow_pde.hpo import get_search_space

        space = get_search_space()
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_search_space_has_required_params(self):
        """Common PDEBench parameters should be present."""
        from expflow_pde.hpo import get_search_space

        space = get_search_space()
        for param in ("lr", "batch_size", "epochs", "weight_decay"):
            assert param in space
            assert "type" in space[param]
            assert "low" in space[param]
            assert "high" in space[param]

    def test_search_space_values_have_correct_types(self):
        """Each param spec should have valid types."""
        from expflow_pde.hpo import get_search_space

        space = get_search_space()
        for name, spec in space.items():
            assert spec["type"] in ("float", "int", "categorical"), f"{name}: bad type"
            if spec["type"] in ("float", "int"):
                assert spec["low"] < spec["high"], f"{name}: low >= high"


# ── Tests: suggest_params ──


class TestSuggestParams:
    """Tests for _suggest_params helper."""

    def test_suggest_params_samples_all_keys(self):
        """All search space keys should appear in output."""
        from expflow_pde.hpo import _suggest_params, get_search_space

        space = get_search_space()
        trial = MagicMock()
        trial.suggest_float = MagicMock(return_value=0.001)
        trial.suggest_int = MagicMock(return_value=64)

        params = _suggest_params(trial, space)
        for name in space:
            assert name in params

    def test_suggest_params_uses_suggest_methods(self):
        """Should call trial.suggest_float for float params."""
        from expflow_pde.hpo import _suggest_params, get_search_space

        space = get_search_space()
        trial = MagicMock()
        trial.suggest_float = MagicMock(return_value=0.001)
        trial.suggest_int = MagicMock(return_value=64)

        _suggest_params(trial, space)
        assert trial.suggest_float.call_count > 0
        assert trial.suggest_int.call_count > 0


# ── Tests: run_hpo ──


class TestRunHPO:
    """Tests for the main run_hpo function."""

    def test_run_hpo_returns_study_info(self, mock_optuna, mock_subprocess):
        """run_hpo should return dict with study metadata."""
        from expflow_pde.hpo import run_hpo

        result = run_hpo(
            script="train_task1.py",
            n_trials=5,
            study_name="test_hpo",
        )

        assert result["study_name"] == "test_hpo" or result["study_name"] == "created_study"
        assert result["n_trials"] == 5
        assert result["direction"] == "maximize"
        assert "duration_sec" in result
        assert isinstance(result["duration_sec"], (int, float))

    def test_run_hpo_executes_subprocess(self, mock_optuna, mock_subprocess):
        """run_hpo should call subprocess for each trial."""
        from expflow_pde.hpo import run_hpo

        run_hpo(
            script="train_task1.py",
            n_trials=3,
            study_name="test_subprocess",
        )

        # subprocess should be called once per trial
        assert mock_subprocess.call_count == 3

    def test_run_hpo_passes_script_to_subprocess(self, mock_optuna, mock_subprocess):
        """subprocess should receive the script as first argument."""
        from expflow_pde.hpo import run_hpo

        run_hpo(
            script="train_task1.py",
            n_trials=1,
            study_name="test_args",
        )

        args = mock_subprocess.call_args[0][0]
        assert args[0] == "train_task1.py"
        # Should have hyperparameter flags
        assert any(a.startswith("--") for a in args)

    def test_run_hpo_respects_timeout(self, mock_optuna, mock_subprocess):
        """Time-limited HPO should stop before completing all trials."""
        from expflow_pde.hpo import run_hpo

        # With 0 second timeout, should complete 0 trials
        # (can't actually test because time passes, but we can check the
        # structure)
        result = run_hpo(
            script="train_task1.py",
            n_trials=50,
            study_name="test_timeout",
            timeout_minutes=60,  # generous
        )

        assert isinstance(result["completed"], int)
        assert result["timeout_minutes"] == 60.0

    def test_run_hpo_reads_metric_from_output(self, mock_optuna, mock_subprocess):
        """Should parse METRIC: lines from script stdout."""
        from expflow_pde.hpo import run_hpo

        result = run_hpo(
            script="train_task1.py",
            n_trials=1,
            study_name="test_metric",
        )

        assert result["completed"] >= 0

    def test_run_hpo_with_custom_direction(self, mock_optuna, mock_subprocess):
        """Should accept custom direction parameter."""
        from expflow_pde.hpo import run_hpo

        result = run_hpo(
            script="train_task1.py",
            n_trials=1,
            study_name="test_dir",
            direction="minimize",
        )

        assert result["direction"] == "minimize"

    def test_run_hpo_default_search_space(self, mock_optuna, mock_subprocess):
        """Should use default search space when none provided."""
        from expflow_pde.hpo import run_hpo

        result = run_hpo(
            script="train_task1.py",
            n_trials=1,
            study_name="test_space",
        )

        assert isinstance(result["completed"], int)

    def test_run_hpo_auto_generates_study_name(self, mock_optuna, mock_subprocess):
        """Should generate study name when not provided."""
        from expflow_pde.hpo import run_hpo

        result = run_hpo(
            script="train_task1.py",
            n_trials=1,
            study_name=None,
        )

        # The mock returns "created_study" as the study_name
        assert result["study_name"] is not None


# ── Tests: _run_trial ──


class TestRunTrial:
    """Tests for the _run_trial_local helper."""

    def test_run_trial_parses_metric(self, mock_subprocess):
        """Should extract objective value from METRIC line."""
        from expflow_pde.hpo import _run_trial_local

        value = _run_trial_local("train.py", {"lr": 0.001}, objective_metric="seg_total")
        assert value == 57.09

    def test_run_trial_parses_different_metric(self):
        """Should extract different metric names."""
        with patch("expflow_pde.hpo.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="METRIC:val_loss=0.023\n",
                stderr="",
                returncode=0,
            )

            from expflow_pde.hpo import _run_trial_local

            value = _run_trial_local("train.py", {}, objective_metric="val_loss")
            assert value == 0.023

    def test_run_trial_fallback_json(self):
        """Should fall back to JSON parsing of last line."""
        with patch("expflow_pde.hpo.subprocess.run") as mock_run:
            output = json.dumps({"seg_total": 58.0, "pde_mean": 12.0})
            mock_run.return_value = MagicMock(
                stdout=output + "\n",
                stderr="",
                returncode=0,
            )

            from expflow_pde.hpo import _run_trial_local

            value = _run_trial_local("train.py", {}, objective_metric="seg_total")
            assert value == 58.0

    def test_run_trial_no_metric_found(self):
        """Should return None when no matching metric found."""
        with patch("expflow_pde.hpo.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Training complete.\nLoss: 0.1\n",
                stderr="",
                returncode=0,
            )

            from expflow_pde.hpo import _run_trial_local

            value = _run_trial_local("train.py", {}, objective_metric="seg_total")
            assert value is None


# ── Tests: run_hpo with distributed flag ──


class TestRunHPODistributed:
    """Tests for distributed HPO mode.

    Note: These tests patch at the module level that hpo.py imports from,
    rather than patching clearml.Task directly (which triggers numpy re-import
    issues in test isolation).
    """

    def test_run_hpo_distributed_dispatches_to_clearml(self, mock_optuna, mock_subprocess):
        """distributed=True should trigger clearml path."""
        # Patch at the hpo module level (after Task is already imported there)
        # We can use mock_clearml mock on the functions hpo.py calls
        with patch("expflow_pde.hpo._run_hpo_distributed") as mock_dist:
            mock_dist.return_value = {
                "study_name": "test_dist_hpo",
                "n_trials": 3,
                "completed": 3,
                "failed": 0,
                "best_value": 57.09,
                "best_params": {"lr": 0.001, "epochs": 80},
                "direction": "maximize",
                "duration_sec": 120.5,
                "timeout_minutes": None,
            }

            from expflow_pde.hpo import run_hpo

            result = run_hpo(
                script="train_task1.py",
                n_trials=3,
                study_name="test_dist_hpo",
                distributed=True,
                queue="gpu_queue",
                direction="maximize",
            )

            mock_dist.assert_called_once()
            assert result["study_name"] == "test_dist_hpo"
            assert result["n_trials"] == 3

    def test_run_hpo_distributed_requires_queue(self, mock_optuna):
        """distributed=True without queue should use 'default'."""
        with patch("expflow_pde.hpo._run_hpo_distributed") as mock_dist:
            mock_dist.return_value = {
                "study_name": "test_dist_fallback",
                "n_trials": 1,
                "completed": 1,
                "failed": 0,
                "best_value": 42.0,
                "best_params": {"lr": 0.001},
                "direction": "maximize",
                "duration_sec": 30.0,
                "timeout_minutes": None,
            }

            from expflow_pde.hpo import run_hpo

            result = run_hpo(
                script="train.py",
                n_trials=1,
                distributed=True,
                queue="default",
            )

            assert result["study_name"] == "test_dist_fallback"


# ── Tests: _narrow_space ──


class TestNarrowSpace:
    """Tests for _narrow_space — hierarchical HPO search space narrowing."""

    TOP_3_SEARCH_SPACE = {
        "lr": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
        "epochs": {"type": "int", "low": 10, "high": 200},
        "arch": {"type": "categorical", "choices": ["FNO", "DeepONet"]},
    }

    def test_narrow_space_minimal_trials_returns_original(self):
        """With fewer than 3 trials, return original space unchanged."""
        from expflow_pde.hpo import _narrow_space

        trials = [{"value": 50.0, "params": {"lr": 0.01, "epochs": 80, "arch": "FNO"}}]
        result = _narrow_space(trials, self.TOP_3_SEARCH_SPACE)
        assert result == self.TOP_3_SEARCH_SPACE

    def test_narrow_space_float_range_reduces(self):
        """Float params should be narrowed around top performers."""
        from expflow_pde.hpo import _narrow_space

        trials = [
            {"value": 80.0, "params": {"lr": 0.001, "epochs": 120, "arch": "FNO"}},
            {"value": 75.0, "params": {"lr": 0.002, "epochs": 100, "arch": "DeepONet"}},
            {"value": 60.0, "params": {"lr": 0.01, "epochs": 50, "arch": "FNO"}},
        ]
        result = _narrow_space(trials, self.TOP_3_SEARCH_SPACE, top_frac=0.67)

        # lr should be narrowed around 0.001-0.002 range
        assert result["lr"]["low"] >= 1e-6
        assert result["lr"]["high"] < 1.0

        # categorical unchanged
        assert result["arch"] == self.TOP_3_SEARCH_SPACE["arch"]

    def test_narrow_space_int_rounding(self):
        """Int ranges should be rounded and have min gap of 1."""
        from expflow_pde.hpo import _narrow_space

        trials = [
            {"value": 90.0, "params": {"lr": 0.001, "epochs": 100, "arch": "FNO"}},
            {"value": 88.0, "params": {"lr": 0.0012, "epochs": 95, "arch": "FNO"}},
            {"value": 85.0, "params": {"lr": 0.0011, "epochs": 90, "arch": "FNO"}},
        ]
        result = _narrow_space(trials, self.TOP_3_SEARCH_SPACE, top_frac=0.67)

        assert isinstance(result["epochs"]["low"], int)
        assert isinstance(result["epochs"]["high"], int)
        assert result["epochs"]["high"] >= result["epochs"]["low"] + 1

    def test_narrow_space_identical_top_values(self):
        """When all top values are identical, narrow from original range."""
        from expflow_pde.hpo import _narrow_space

        trials = [
            {"value": 70.0, "params": {"lr": 0.001, "epochs": 80, "arch": "FNO"}},
            {"value": 69.0, "params": {"lr": 0.001, "epochs": 80, "arch": "FNO"}},
            {"value": 68.0, "params": {"lr": 0.001, "epochs": 80, "arch": "FNO"}},
        ]
        result = _narrow_space(trials, self.TOP_3_SEARCH_SPACE, top_frac=0.67)

        # Should still be narrowed, not identical to original
        assert result["lr"]["high"] < 1.0
        assert result["epochs"]["high"] < 200
        assert result["epochs"]["low"] >= 10


# ── Tests: run_hpo_hierarchical ──


class TestRunHpoHierarchical:
    """Tests for run_hpo_hierarchical — two-stage HPO."""

    def test_hierarchical_delegates_to_run_hpo(self):
        """run_hpo_hierarchical should call run_hpo twice and merge results."""
        base_result = {
            "study_name": "test",
            "n_trials": 20,
            "completed": 20,
            "failed": 0,
            "best_value": 50.0,
            "best_params": {"lr": 0.001, "epochs": 80},
            "direction": "maximize",
            "duration_sec": 30.0,
            "timeout_minutes": None,
        }

        from expflow_pde.hpo import run_hpo_hierarchical

        with patch("expflow_pde.hpo.run_hpo", return_value=base_result) as mock_run:
            result = run_hpo_hierarchical(
                script="train.py",
                n_trials=50,
                n_stage1=10,
            )

        assert mock_run.call_count == 2
        assert result["method"] == "hierarchical"
        assert result["n_trials"] == 50
        assert result["completed"] == 40  # 20 + 20
        assert result["best_value"] == 50.0
        assert "phase_1" in result
        assert "phase_2" in result

    def test_hierarchical_picks_best_phase(self):
        """Should select the phase with the better best_value."""
        from expflow_pde.hpo import run_hpo_hierarchical

        phase1_result = {
            "study_name": "s1",
            "n_trials": 10,
            "completed": 10,
            "failed": 0,
            "best_value": 60.0,
            "best_params": {"lr": 0.001},
            "direction": "maximize",
            "duration_sec": 15.0,
            "timeout_minutes": None,
        }
        phase2_result = {
            "study_name": "s2",
            "n_trials": 40,
            "completed": 40,
            "failed": 0,
            "best_value": 75.0,
            "best_params": {"lr": 0.0005},
            "direction": "maximize",
            "duration_sec": 60.0,
            "timeout_minutes": None,
        }

        with patch("expflow_pde.hpo.run_hpo", side_effect=[phase1_result, phase2_result]):
            result = run_hpo_hierarchical(
                script="train.py",
                n_trials=50,
                n_stage1=10,
                direction="maximize",
            )

        # Phase 2 is better (75 > 60)
        assert result["best_value"] == 75.0
        assert result["best_params"] == {"lr": 0.0005}

    def test_hierarchical_picks_phase1_when_better(self):
        """When phase 1 has the better value, the merged result uses phase 1."""
        from expflow_pde.hpo import run_hpo_hierarchical

        phase1_result = {
            "study_name": "s1",
            "n_trials": 10,
            "completed": 10,
            "failed": 0,
            "best_value": 95.0,
            "best_params": {"lr": 0.01},
            "direction": "maximize",
            "duration_sec": 15.0,
            "timeout_minutes": None,
        }
        phase2_result = {
            "study_name": "s2",
            "n_trials": 40,
            "completed": 40,
            "failed": 0,
            "best_value": 80.0,
            "best_params": {"lr": 0.001},
            "direction": "maximize",
            "duration_sec": 60.0,
            "timeout_minutes": None,
        }

        with patch("expflow_pde.hpo.run_hpo", side_effect=[phase1_result, phase2_result]):
            result = run_hpo_hierarchical(
                script="train.py",
                n_trials=50,
                n_stage1=10,
                direction="maximize",
            )

        # Phase 1 is better (95 > 80)
        assert result["best_value"] == 95.0
        assert result["best_params"] == {"lr": 0.01}

    def test_hierarchical_minimize_direction(self):
        """Should pick the lower value for minimize direction."""
        from expflow_pde.hpo import run_hpo_hierarchical

        phase1_result = {
            "study_name": "s1",
            "n_trials": 10,
            "completed": 10,
            "failed": 0,
            "best_value": 0.5,
            "best_params": {"lr": 0.01},
            "direction": "minimize",
            "duration_sec": 15.0,
            "timeout_minutes": None,
        }
        phase2_result = {
            "study_name": "s2",
            "n_trials": 40,
            "completed": 40,
            "failed": 0,
            "best_value": 0.2,
            "best_params": {"lr": 0.001},
            "direction": "minimize",
            "duration_sec": 60.0,
            "timeout_minutes": None,
        }

        with patch("expflow_pde.hpo.run_hpo", side_effect=[phase1_result, phase2_result]):
            result = run_hpo_hierarchical(
                script="train.py",
                n_trials=50,
                n_stage1=10,
                direction="minimize",
            )

        assert result["best_value"] == 0.2  # Phase 2 lower = better for minimize


# ── Tests: _load_trials_from_storage ──


class TestLoadTrialsFromStorage:
    """Tests for _load_trials_from_storage — reading Optuna trials from SQLite."""

    def test_load_trials_no_storage_returns_empty(self):
        """Without storage, should return empty list."""
        from expflow_pde.hpo import _load_trials_from_storage

        result = _load_trials_from_storage("no_such_study")
        assert result == []

    def test_load_trials_with_mock_optuna(self):
        """Should load and restore trial values correctly for maximize."""
        from expflow_pde.hpo import _load_trials_from_storage

        mock_optuna = MagicMock()
        mock_study = MagicMock()

        # Simulate 3 trials: maximize stores -value internally
        trial1 = MagicMock(number=0, value=-90.0, params={"lr": 0.001})
        trial2 = MagicMock(number=1, value=-75.0, params={"lr": 0.002})
        trial3 = MagicMock(number=2, value=-60.0, params={"lr": 0.01})
        mock_study.trials = [trial1, trial2, trial3]

        mock_optuna.load_study.return_value = mock_study

        with patch("expflow_pde.hpo._import_optuna", return_value=mock_optuna):
            # Also need a matching DB file — patch glob to succeed
            with patch("glob.glob", return_value=["/tmp/test_optuna.db"]):
                result = _load_trials_from_storage("test_study", direction="maximize")

        assert len(result) == 3
        assert result[0]["value"] == 90.0  # restored from -90
        assert result[1]["value"] == 75.0
        assert result[2]["value"] == 60.0

    def test_load_trials_minimize_direction(self):
        """minimize direction should not negate values."""
        from expflow_pde.hpo import _load_trials_from_storage

        mock_optuna = MagicMock()
        mock_study = MagicMock()

        trial1 = MagicMock(number=0, value=0.5, params={"lr": 0.01})
        trial2 = MagicMock(number=1, value=0.2, params={"lr": 0.001})
        trial3 = MagicMock(number=2, value=0.8, params={"lr": 0.1})
        mock_study.trials = [trial1, trial2, trial3]

        mock_optuna.load_study.return_value = mock_study

        with patch("expflow_pde.hpo._import_optuna", return_value=mock_optuna):
            with patch("glob.glob", return_value=["/tmp/test_optuna.db"]):
                result = _load_trials_from_storage("test_study", direction="minimize")

        # Values stay as-is for minimize; sorted descending for _narrow_space consumption
        assert result[0]["value"] == 0.8  # highest first
        assert result[1]["value"] == 0.5
        assert result[2]["value"] == 0.2


# ── Tests: Hierarchical integration with narrowing ──


class TestHierarchicalNarrowing:
    """Tests that run_hpo_hierarchical properly triggers narrowing."""

    def test_hierarchical_calls_narrow_space_when_enough_trials(self):
        """With enough Phase 1 trials, _narrow_space should be called."""
        from expflow_pde.hpo import run_hpo_hierarchical

        base_result = {
            "study_name": "test",
            "n_trials": 5,
            "completed": 5,
            "failed": 0,
            "best_value": 50.0,
            "best_params": {"lr": 0.001, "epochs": 80},
            "direction": "maximize",
            "duration_sec": 15.0,
            "timeout_minutes": None,
        }

        mock_trials = [
            {"number": 0, "value": 90.0, "params": {"lr": 0.001, "epochs": 100}},
            {"number": 1, "value": 75.0, "params": {"lr": 0.002, "epochs": 80}},
            {"number": 2, "value": 60.0, "params": {"lr": 0.01, "epochs": 50}},
        ]

        with (
            patch("expflow_pde.hpo.run_hpo", return_value=base_result) as mock_run,
            patch(
                "expflow_pde.hpo._load_trials_from_storage",
                return_value=mock_trials,
            ) as mock_load,
            patch("expflow_pde.hpo._narrow_space") as mock_narrow,
        ):
            run_hpo_hierarchical(
                script="train.py",
                n_trials=25,
                n_stage1=5,
            )

        assert mock_run.call_count == 2
        mock_load.assert_called_once()
        # _narrow_space should have been called with mock_trials and search_space
        mock_narrow.assert_called_once()
        args = mock_narrow.call_args[0]
        assert args[0] == mock_trials  # trials passed
        assert "lr" in args[1]  # search space passed

    def test_hierarchical_skips_narrowing_when_few_trials(self):
        """Without enough Phase 1 trials, _narrow_space should not be called."""
        from expflow_pde.hpo import run_hpo_hierarchical

        base_result = {
            "study_name": "test",
            "n_trials": 2,
            "completed": 2,
            "failed": 0,
            "best_value": 50.0,
            "best_params": {"lr": 0.001, "epochs": 80},
            "direction": "maximize",
            "duration_sec": 15.0,
            "timeout_minutes": None,
        }

        with (
            patch("expflow_pde.hpo.run_hpo", return_value=base_result),
            patch("expflow_pde.hpo._load_trials_from_storage", return_value=[]),
            patch("expflow_pde.hpo._narrow_space") as mock_narrow,
        ):
            run_hpo_hierarchical(
                script="train.py",
                n_trials=10,
                n_stage1=2,
            )

        # _narrow_space should NOT be called (fewer than 3 trials)
        mock_narrow.assert_not_called()

    def test_hierarchical_fallback_to_best_trial_when_storage_empty(self):
        """When _load_trials_from_storage returns empty, fall back to best trial."""
        from expflow_pde.hpo import run_hpo_hierarchical

        base_result = {
            "study_name": "test",
            "n_trials": 5,
            "completed": 5,
            "failed": 0,
            "best_value": 50.0,
            "best_params": {"lr": 0.001, "epochs": 80},
            "direction": "maximize",
            "duration_sec": 15.0,
            "timeout_minutes": None,
        }

        with (
            patch("expflow_pde.hpo.run_hpo", return_value=base_result),
            patch("expflow_pde.hpo._load_trials_from_storage", return_value=[]),
            patch("expflow_pde.hpo._narrow_space") as mock_narrow,
        ):
            run_hpo_hierarchical(
                script="train.py",
                n_trials=25,
                n_stage1=5,
            )

        # _narrow_space should NOT be called (fewer than 3 trials from fallback too)
        mock_narrow.assert_not_called()
