#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for expflow.audit — result validation, compliance checking, reporting."""

import sys

import pytest

# ── Fixture ──


@pytest.fixture(autouse=True)
def no_external_deps():
    """Clean sys.modules so audit module is fresh each test."""
    for mod in list(sys.modules.keys()):
        if mod.startswith("expflow_pde.audit") or mod.startswith("expflow_pde.report"):
            del sys.modules[mod]
    yield


# ══════════════════════════════════════════════════════════════
# validate_experiment
# ══════════════════════════════════════════════════════════════


class TestValidateExperiment:
    """validate_experiment() — check experiment reproducibility."""

    def test_validate_returns_checklist(self):
        """validate_experiment returns a dict with check items."""
        from expflow_pde.audit import validate_experiment

        result = validate_experiment(
            experiment_id="exp_123",
            config_snapshot={"lr": 0.001, "epochs": 100},
            metrics={"final_loss": 0.05, "accuracy": 0.95},
        )

        assert result["experiment_id"] == "exp_123"
        assert "checks" in result
        assert isinstance(result["checks"], list)
        assert len(result["checks"]) > 0

    def test_validate_has_timestamp(self):
        """validate result includes timestamp."""
        from expflow_pde.audit import validate_experiment

        result = validate_experiment("exp_123", {}, {})
        assert "timestamp" in result

    def test_validate_config_presence(self):
        """Validate checks if config is present."""
        from expflow_pde.audit import validate_experiment

        result = validate_experiment("exp_123", {"lr": 0.001}, {})
        config_check = next(
            (c for c in result["checks"] if "config" in c.get("name", "").lower()),
            None,
        )
        assert config_check is not None
        assert config_check["passed"] is True

    def test_validate_empty_config_fails(self):
        """Empty config fails the config check."""
        from expflow_pde.audit import validate_experiment

        result = validate_experiment("exp_123", {}, {"final_loss": 0.05})
        config_check = next(
            (c for c in result["checks"] if "config" in c.get("name", "").lower()),
            None,
        )
        assert config_check is not None
        assert config_check["passed"] is False

    def test_validate_metrics_presence(self):
        """Validate checks if metrics are present."""
        from expflow_pde.audit import validate_experiment

        result = validate_experiment("exp_123", {"lr": 0.001}, {"final_loss": 0.05})
        metrics_check = next(
            (c for c in result["checks"] if "metric" in c.get("name", "").lower()),
            None,
        )
        assert metrics_check is not None
        assert metrics_check["passed"] is True

    def test_validate_empty_metrics_fails(self):
        """Empty metrics fails the metrics check."""
        from expflow_pde.audit import validate_experiment

        result = validate_experiment("exp_123", {"lr": 0.001}, {})
        metrics_check = next(
            (c for c in result["checks"] if "metric" in c.get("name", "").lower()),
            None,
        )
        assert metrics_check is not None
        assert metrics_check["passed"] is False


# ══════════════════════════════════════════════════════════════
# check_dataset_compliance
# ══════════════════════════════════════════════════════════════


class TestCheckDatasetCompliance:
    """check_dataset_compliance() — dataset provenance & compliance check."""

    def test_check_compliance_allowed(self):
        """Allowed dataset passes."""
        from expflow_pde.audit import check_dataset_compliance

        result = check_dataset_compliance(
            dataset_name="burgers_nu0.001",
            compliance="allowed",
        )
        assert result["compliant"] is True

    def test_check_compliance_forbidden(self):
        """Forbidden dataset fails."""
        from expflow_pde.audit import check_dataset_compliance

        result = check_dataset_compliance(
            dataset_name="synthetic_burgers",
            compliance="forbidden",
        )
        assert result["compliant"] is False

    def test_check_compliance_unknown(self):
        """Unknown compliance raises ValueError."""
        from expflow_pde.audit import check_dataset_compliance

        with pytest.raises(ValueError, match="compliance"):
            check_dataset_compliance("ds", "invalid")

    def test_check_compliance_returns_metadata(self):
        """Result includes dataset name and compliance."""
        from expflow_pde.audit import check_dataset_compliance

        result = check_dataset_compliance("burgers_nu0.001", "allowed")
        assert result["dataset_name"] == "burgers_nu0.001"
        assert result["compliance"] == "allowed"


# ══════════════════════════════════════════════════════════════
# generate_report
# ══════════════════════════════════════════════════════════════


class TestGenerateReport:
    """generate_report() — auto-generated experiment report."""

    def test_generate_report_returns_dict(self):
        """generate_report returns a dict with markdown content."""
        from expflow_pde.audit import generate_report

        result = generate_report(
            experiment_id="exp_123",
            config={"lr": 0.001},
            metrics={"final_loss": 0.05},
        )

        assert "markdown" in result
        assert "experiment_id" in result
        assert "exp_123" in result["markdown"]

    def test_report_includes_metrics_table(self):
        """Report markdown includes a metrics table."""
        from expflow_pde.audit import generate_report

        result = generate_report(
            "exp_123",
            {"lr": 0.001},
            {"final_loss": 0.05, "accuracy": 0.95},
        )
        assert "|" in result["markdown"]  # table separator
        assert "final_loss" in result["markdown"]
        assert "accuracy" in result["markdown"]

    def test_report_includes_validation(self):
        """Report includes validation results."""
        from expflow_pde.audit import generate_report

        result = generate_report("exp_123", {"lr": 0.001}, {"final_loss": 0.05})
        assert "config" in result["markdown"].lower()


# ══════════════════════════════════════════════════════════════
# validate_competition_rules
# ══════════════════════════════════════════════════════════════


class TestValidateCompetitionRules:
    """validate_competition_rules() — PDEBench competition rule checking."""

    def test_all_rules_pass(self):
        """All metrics and sub_step present and within threshold."""
        from expflow_pde.audit import validate_competition_rules

        result = validate_competition_rules(
            task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5},
            task_params={"Args/--sub_step": "5"},
        )
        assert result["all_pass"] is True
        assert len(result["checks"]) == 4

    def test_pde_mean_exceeds_threshold(self):
        """pde_mean > 18.09 gate fails."""
        from expflow_pde.audit import validate_competition_rules

        result = validate_competition_rules(
            task_metrics={"seg_total": 57.09, "pde_mean": 20.0, "train_time_min": 45.5},
            task_params={"Args/--sub_step": "5"},
        )
        assert result["all_pass"] is False
        pde_check = next(c for c in result["checks"] if c["name"] == "pde_mean")
        assert pde_check["passed"] is False

    def test_train_time_exceeds_limit(self):
        """train_time_min > 60 fails."""
        from expflow_pde.audit import validate_competition_rules

        result = validate_competition_rules(
            task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 75.0},
            task_params={"Args/--sub_step": "5"},
        )
        assert result["all_pass"] is False
        time_check = next(c for c in result["checks"] if c["name"] == "train_time_min")
        assert time_check["passed"] is False

    def test_missing_metric_fails(self):
        """Missing metric makes all_pass False."""
        from expflow_pde.audit import validate_competition_rules

        result = validate_competition_rules(
            task_metrics={"seg_total": 57.09},
            task_params={"Args/--sub_step": "5"},
        )
        assert result["all_pass"] is False
        missing = next(c for c in result["checks"] if c["value"] is None)
        assert missing["passed"] is False

    def test_missing_sub_step_fails(self):
        """No sub_step in params makes that check fail."""
        from expflow_pde.audit import validate_competition_rules

        result = validate_competition_rules(
            task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5},
            task_params=None,
        )
        assert result["all_pass"] is False
        sub_check = next(c for c in result["checks"] if c["name"] == "sub_step")
        assert sub_check["passed"] is False

    def test_sub_step_zero_fails(self):
        """sub_step=0 fails."""
        from expflow_pde.audit import validate_competition_rules

        result = validate_competition_rules(
            task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5},
            task_params={"Args/--sub_step": "0"},
        )
        assert result["all_pass"] is False
        sub_check = next(c for c in result["checks"] if c["name"] == "sub_step")
        assert sub_check["passed"] is False

    def test_empty_params_falls_through(self):
        """Empty params dict also fails sub_step check."""
        from expflow_pde.audit import validate_competition_rules

        result = validate_competition_rules(
            task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5},
            task_params={},
        )
        sub_check = next(c for c in result["checks"] if c["name"] == "sub_step")
        assert sub_check["passed"] is False

    def test_result_includes_metrics(self):
        """Result includes original metrics dict."""
        from expflow_pde.audit import validate_competition_rules

        metrics = {"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5}
        result = validate_competition_rules(task_metrics=metrics, task_params=None)
        assert result["metrics"] == metrics
