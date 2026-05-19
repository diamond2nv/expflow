#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.metrics — standardized metric registry.

Covers:
- STANDARD_METRICS registry structure
- get_registered_metrics() / get_metric_info()
- report_standard() with and without task
- validate_metric_threshold()
- Error cases (unknown metric)
"""

import pytest

from expflow_pde.metrics import (
    STANDARD_METRICS,
    get_metric_info,
    get_registered_metrics,
    report_standard,
    validate_metric_threshold,
)


class TestRegistry:
    """STANDARD_METRICS structure."""

    def test_registered_count(self):
        assert len(STANDARD_METRICS) >= 12

    def test_each_metric_has_required_keys(self):
        for name, info in STANDARD_METRICS.items():
            assert "type" in info, f"{name} missing 'type'"
            assert "group" in info, f"{name} missing 'group'"
            assert isinstance(info.get("type"), str)
            assert isinstance(info.get("group"), str)

    def test_groups_exist(self):
        groups = {info["group"] for info in STANDARD_METRICS.values()}
        assert "Score" in groups
        assert "Loss" in groups
        assert "PDE" in groups
        assert "Time" in groups
        assert "Model" in groups
        assert "Training" in groups

    def test_seg_total_is_primary(self):
        info = STANDARD_METRICS["seg_total"]
        assert info["higher_is_better"] is True

    def test_pde_mean_has_threshold(self):
        info = STANDARD_METRICS["pde_mean"]
        assert info["threshold"] == 18.09
        assert info["higher_is_better"] is False

    def test_train_time_min_has_threshold(self):
        info = STANDARD_METRICS["train_time_min"]
        assert info["threshold"] == 60
        assert info["higher_is_better"] is False


class TestGetRegisteredMetrics:
    """get_registered_metrics() and get_metric_info()."""

    def test_returns_copy_not_reference(self):
        r1 = get_registered_metrics()
        r2 = get_registered_metrics()
        assert r1 == r2
        r1["new_key"] = {}
        assert "new_key" not in r2  # not mutated

    def test_get_metric_info_known(self):
        info = get_metric_info("seg_total")
        assert info is not None
        assert info["group"] == "Score"

    def test_get_metric_info_unknown(self):
        assert get_metric_info("nonexistent") is None


class TestReportStandard:
    """report_standard() helper."""

    def test_reports_correct_metrics(self):
        result = report_standard(seg_total=57.09, pde_mean=20.29)
        assert result == {"seg_total": 57.09, "pde_mean": 20.29}

    def test_reports_with_task(self):
        """When task is provided, it calls report_scalar but returns dict."""

        # In unit tests without a real clearml Task, task=None means no reporting.
        # Passing a dict with report_scalar method simulates a task.
        class MockTask:
            last_call: tuple | None = None

            def report_scalar(self, title, series, value, iteration=0):
                self.last_call = (title, series, value, iteration)

        mock_task = MockTask()
        result = report_standard(task=mock_task, seg_total=57.09, pde_mean=20.29)
        assert result == {"seg_total": 57.09, "pde_mean": 20.29}
        # Should have called report_scalar for pde_mean
        assert mock_task.last_call is not None
        assert mock_task.last_call[1] == "pde_mean"  # series name

    def test_raises_on_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            report_standard(seg_total=57.09, unknown_metric=42)

    def test_raises_with_helpful_message(self):
        with pytest.raises(ValueError) as exc:
            report_standard(seg_total=57.09, foobar=42)
        assert "seg_total" in str(exc.value)  # lists registered metrics


class TestValidateMetricThreshold:
    """validate_metric_threshold() helper."""

    def test_pde_mean_below_threshold_passes(self):
        result = validate_metric_threshold("pde_mean", 17.5)
        assert result["passed"] is True
        assert result["threshold"] == 18.09

    def test_pde_mean_above_threshold_fails(self):
        result = validate_metric_threshold("pde_mean", 20.0)
        assert result["passed"] is False

    def test_train_time_below_60_passes(self):
        result = validate_metric_threshold("train_time_min", 45)
        assert result["passed"] is True

    def test_train_time_above_60_fails(self):
        result = validate_metric_threshold("train_time_min", 75)
        assert result["passed"] is False

    def test_metric_with_no_threshold(self):
        result = validate_metric_threshold("seg_total", 57.09)
        assert result["passed"] is True
        assert result["threshold"] is None

    def test_unknown_metric_returns_pass(self):
        result = validate_metric_threshold("unknown_metric", 42)
        assert result["passed"] is True
        assert result["detail"] == "Unknown metric"
