#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.metrics — metric registry and PDEBench metrics."""

from __future__ import annotations

import torch

from expflow_pde.metrics import (
    STANDARD_METRICS,
    compute_rel_mse,
    get_metric_info,
    get_registered_metrics,
    report_standard,
    set_metric_threshold,
    validate_metric_threshold,
)

# ── Metric Registry ──


class TestMetricRegistry:
    def test_registry_has_core_metrics(self):
        """Core competition metrics must exist."""
        for name in ("seg_total", "seg1", "seg2", "seg3"):
            assert name in STANDARD_METRICS, f"Missing: {name}"

    def test_registry_has_pdebench_6(self):
        """All 6 PDEBench error metrics must be registered."""
        for name in ("val_rmse", "val_nrmse", "val_max_err", "val_bd_err", "val_csv_err"):
            assert name in STANDARD_METRICS, f"Missing: {name}"

    def test_registry_has_fourier_bands(self):
        for name in ("val_fourier_low", "val_fourier_mid", "val_fourier_high"):
            assert name in STANDARD_METRICS

    def test_registry_has_loss_metrics(self):
        for name in ("val_lprel", "val_h1rel"):
            assert name in STANDARD_METRICS

    def test_get_registered_metrics_returns_copy(self):
        d = get_registered_metrics()
        assert "seg_total" in d
        # Verify it's a copy
        d["fake"] = {}
        assert "fake" not in STANDARD_METRICS

    def test_get_metric_info(self):
        info = get_metric_info("seg_total")
        assert info is not None
        assert info["group"] == "Score"
        assert info["higher_is_better"] is True
        assert get_metric_info("nonexistent") is None

    def test_all_metrics_have_required_keys(self):
        for name, info in STANDARD_METRICS.items():
            assert "type" in info, f"{name} missing type"
            assert "group" in info, f"{name} missing group"
            assert "higher_is_better" in info, f"{name} missing higher_is_better"


# ── Report Standard ──


class TestReportStandard:
    def test_report_known_metrics_no_task(self):
        """report_standard returns dict for known metrics."""
        result = report_standard(
            task=None,
            seg_total=57.0,
            val_rmse=0.05,
        )
        assert result["seg_total"] == 57.0
        assert result["val_rmse"] == 0.05

    def test_report_skips_unknown_metric(self):
        """Unknown metrics are silently skipped."""
        result = report_standard(
            task=None,
            seg_total=57.0,
            made_up_metric=42.0,
        )
        assert "seg_total" in result
        assert "made_up_metric" not in result


# ── Thresholds ──


class TestMetricThresholds:
    def test_validate_passing(self):
        """Value below threshold = pass."""
        assert validate_metric_threshold("pde_mean", 10.0) is True

    def test_validate_failing(self):
        """Value above threshold = fail."""
        assert validate_metric_threshold("pde_mean", 20.0) is False

    def test_validate_unknown_metric(self):
        """Unknown metric always passes."""
        assert validate_metric_threshold("nonexistent", 999.0) is True

    def test_set_custom_threshold(self):
        set_metric_threshold("val_mse", 0.1)
        assert validate_metric_threshold("val_mse", 0.05) is True
        assert validate_metric_threshold("val_mse", 0.2) is False


# ── compute_rel_mse ──


class TestComputeRelMSE:
    def test_perfect_prediction(self):
        pred = torch.randn(4, 64, 10, 1)
        target = pred.clone()
        rel = compute_rel_mse(pred, target)
        assert rel.shape == (4,)
        assert torch.allclose(rel, torch.zeros(4), atol=1e-6)

    def test_random_prediction(self):
        pred = torch.randn(4, 64, 1)
        target = torch.randn(4, 64, 1)
        rel = compute_rel_mse(pred, target)
        assert rel.shape == (4,)
        assert torch.all(rel >= 0)
        assert torch.isfinite(rel).all()

    def test_zero_target(self):
        """rel_mse when target is all zeros → returns finite value."""
        pred = torch.randn(4, 64, 1)
        target = torch.zeros(4, 64, 1)
        rel = compute_rel_mse(pred, target, eps=1e-7)
        assert rel.shape == (4,)
        assert torch.isfinite(rel).all()

    def test_single_sample(self):
        pred = torch.randn(1, 64, 1)
        target = torch.randn(1, 64, 1)
        rel = compute_rel_mse(pred, target)
        assert rel.shape == (1,)

    def test_multi_channel(self):
        pred = torch.randn(4, 64, 3)
        target = torch.randn(4, 64, 3)
        rel = compute_rel_mse(pred, target)
        assert rel.shape == (4,)
