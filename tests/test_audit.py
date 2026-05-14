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
        if mod.startswith("expflow.audit") or mod.startswith("expflow.report"):
            del sys.modules[mod]
    yield


# ══════════════════════════════════════════════════════════════
# validate_experiment
# ══════════════════════════════════════════════════════════════


class TestValidateExperiment:
    """validate_experiment() — check experiment reproducibility."""

    def test_validate_returns_checklist(self):
        """validate_experiment returns a dict with check items."""
        from expflow.audit import validate_experiment

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
        from expflow.audit import validate_experiment

        result = validate_experiment("exp_123", {}, {})
        assert "timestamp" in result

    def test_validate_config_presence(self):
        """Validate checks if config is present."""
        from expflow.audit import validate_experiment

        result = validate_experiment("exp_123", {"lr": 0.001}, {})
        config_check = next(
            (c for c in result["checks"] if "config" in c.get("name", "").lower()),
            None,
        )
        assert config_check is not None
        assert config_check["passed"] is True

    def test_validate_empty_config_fails(self):
        """Empty config fails the config check."""
        from expflow.audit import validate_experiment

        result = validate_experiment("exp_123", {}, {"final_loss": 0.05})
        config_check = next(
            (c for c in result["checks"] if "config" in c.get("name", "").lower()),
            None,
        )
        assert config_check is not None
        assert config_check["passed"] is False

    def test_validate_metrics_presence(self):
        """Validate checks if metrics are present."""
        from expflow.audit import validate_experiment

        result = validate_experiment("exp_123", {"lr": 0.001}, {"final_loss": 0.05})
        metrics_check = next(
            (c for c in result["checks"] if "metric" in c.get("name", "").lower()),
            None,
        )
        assert metrics_check is not None
        assert metrics_check["passed"] is True

    def test_validate_empty_metrics_fails(self):
        """Empty metrics fails the metrics check."""
        from expflow.audit import validate_experiment

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
        from expflow.audit import check_dataset_compliance

        result = check_dataset_compliance(
            dataset_name="burgers_nu0.001",
            compliance="allowed",
        )
        assert result["compliant"] is True

    def test_check_compliance_forbidden(self):
        """Forbidden dataset fails."""
        from expflow.audit import check_dataset_compliance

        result = check_dataset_compliance(
            dataset_name="synthetic_burgers",
            compliance="forbidden",
        )
        assert result["compliant"] is False

    def test_check_compliance_unknown(self):
        """Unknown compliance raises ValueError."""
        from expflow.audit import check_dataset_compliance

        with pytest.raises(ValueError, match="compliance"):
            check_dataset_compliance("ds", "invalid")

    def test_check_compliance_returns_metadata(self):
        """Result includes dataset name and compliance."""
        from expflow.audit import check_dataset_compliance

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
        from expflow.audit import generate_report

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
        from expflow.audit import generate_report

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
        from expflow.audit import generate_report

        result = generate_report("exp_123", {"lr": 0.001}, {"final_loss": 0.05})
        assert "config" in result["markdown"].lower()
