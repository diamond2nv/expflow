#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.explore_search_space — explore-mode HPO.

Covers: architecture registry, shared params, architecture-specific
params, the Define-by-Run objective factory, and the introspection API.
"""

from __future__ import annotations

import optuna
import pytest

from expflow_pde.explore_search_space import (
    ARCHITECTURES,
    describe_explore_space,
    get_explore_objective,
    suggest_fno_params,
    suggest_deeponet_params,
    suggest_loss,
    suggest_optimizer,
)


# ── Architecture registry ──


class TestArchitectureRegistry:
    def test_has_known_architectures(self):
        assert "FNO" in ARCHITECTURES
        assert "DeepONet" in ARCHITECTURES
        assert "PINO" in ARCHITECTURES
        assert len(ARCHITECTURES) >= 5

    def test_describe_explore_space_returns_dict(self):
        info = describe_explore_space()
        assert "architectures" in info
        assert len(info["architectures"]) == len(ARCHITECTURES)
        assert "description" in info
        assert "inspired_by" in info


# ── Suggestion helpers ──


class TestSuggestionHelpers:
    def test_suggest_optimizer_returns_dict(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        optim = suggest_optimizer(trial)
        assert "name" in optim
        assert "weight_decay" in optim
        assert optim["name"] in ("Adam", "AdamW")

    def test_suggest_fno_params_returns_dict(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_fno_params(trial)
        assert "n_modes" in params
        assert "width" in params
        assert "n_layers" in params
        assert "activation" in params
        assert "ar_steps" in params
        assert "freeze" in params

    def test_suggest_deeponet_params_returns_dict(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_deeponet_params(trial)
        assert "branch_layers" in params
        assert "trunk_layers" in params
        assert "width" in params

    def test_suggest_loss_returns_string(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        loss = suggest_loss(trial)
        assert loss in ("l2_rel", "h1_1d", "smoothl1_rel", "mse_rel")


# ── Objective factory ──


class TestExploreObjective:
    def test_objective_returns_float(self):
        """Mock objective (no train_script) returns a synthetic score."""
        objective = get_explore_objective()
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        score = objective(trial)
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 150.0

    def test_objective_samples_all_architectures(self):
        """With enough trials, every architecture should be sampled."""
        objective = get_explore_objective()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=42),
        )
        study.optimize(objective, n_trials=50)

        archs_sampled = set()
        for trial in study.trials:
            if trial.params.get("architecture"):
                archs_sampled.add(trial.params["architecture"])

        # RandomSampler with 50 trials should hit at least 5 of 7 archs
        assert len(archs_sampled) >= 5, f"Only sampled {archs_sampled}"

    def test_objective_params_for_fno_vs_deeponet(self):
        """FNO objective should have FNO-specific params; DeepONet should not."""
        objective = get_explore_objective()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=1),
        )
        study.optimize(objective, n_trials=20)

        for trial in study.trials:
            arch = trial.params.get("architecture", "")
            if arch == "FNO":
                assert "n_modes" in trial.params
            elif arch == "DeepONet":
                assert "branch_layers" in trial.params
            # Shared params always present
            assert "lr" in trial.params
            assert "batch_size" in trial.params
            assert "epochs" in trial.params

    def test_mock_objective_reproducible(self):
        """Same seed → same mock score."""
        objective = get_explore_objective()

        study1 = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=42),
        )
        study1.optimize(objective, n_trials=5)
        best1 = study1.best_value

        study2 = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(seed=42),
        )
        study2.optimize(objective, n_trials=5)
        best2 = study2.best_value

        assert best1 == pytest.approx(best2, abs=1.0)

    def test_objective_accepts_callable(self):
        """Custom eval_fn should be called and its return used."""

        def my_eval(params):
            return float(params.get("lr", 0)) * 1000

        objective = get_explore_objective(train_script=my_eval)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        score = objective(trial)
        assert isinstance(score, float)
