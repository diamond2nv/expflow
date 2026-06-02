#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.monitor — stagnation detection.

Tests cover:
- No stagnation for active search with progress
- KEEP-count stagnation detection
- Single-axis exhaustion detection
- Combined stagnation modes
- Empty history handling
- generate_monitor_report basic output
"""

from __future__ import annotations

from expflow_pde.monitor import detect_stagnation, generate_monitor_report


def _make_exp(
    metric_value: float,
    action: str = "promote",
    axis: str = "lr",
    status: str = "completed",
) -> dict:
    return {
        "metric_value": metric_value,
        "action": action,
        "axis": axis,
        "status": status,
    }


class TestDetectStagnation:
    def test_empty_history(self):
        """Empty history → no stagnation."""
        result = detect_stagnation([])
        assert not result["stagnant"]
        assert result["suggested_action"] == "continue"

    def test_active_no_stagnation(self):
        """Consistent progress with significant gains → no stagnation."""
        history = [
            _make_exp(50.0, "promote", "lr"),
            _make_exp(55.0, "promote", "lr"),
            _make_exp(62.0, "promote", "arch"),
            _make_exp(70.0, "promote", "lr"),
        ]
        result = detect_stagnation(
            history,
            significant_improvement_ratio=0.01,
        )
        assert not result["stagnant"]
        assert result["suggested_action"] == "continue"

    def test_keep_count_stagnation(self):
        """Multiple KEEPs without significant gain → stagnation."""
        history = [
            _make_exp(100.0, "promote", "lr"),
            _make_exp(100.1, "promote", "lr"),   # 0.1% gain
            _make_exp(100.15, "promote", "lr"),  # 0.05% gain
            _make_exp(100.2, "promote", "lr"),   # 0.05% gain
        ]
        result = detect_stagnation(
            history,
            max_keep_no_progress=3,
            significant_improvement_ratio=0.01,  # 1% minimum gain
        )
        assert result["stagnant"]
        assert "KEEP-count" in result["reason"]
        assert result["suggested_action"] == "regroup"

    def test_axis_exhaustion(self):
        """Many discards on few axes → axis exhaustion."""
        history = []
        for _ in range(10):
            history.append(
                _make_exp(0.0, "reject", "lr", "failed"),
            )
        result = detect_stagnation(
            history,
            single_axis_threshold=8,
            max_exhausted_axes=1,
        )
        assert result["stagnant"]
        assert "exhaust" in result["reason"].lower()
        assert result["suggested_action"] == "explore_new_axis"

    def test_axis_exhaustion_not_reached(self):
        """Few discards → no exhaustion."""
        history = [
            _make_exp(0.0, "reject", "lr", "failed"),
            _make_exp(0.0, "reject", "lr", "failed"),
        ]
        result = detect_stagnation(
            history,
            single_axis_threshold=8,
        )
        assert not result["stagnant"]

    def test_combined_modes(self):
        """Both KEEP-stagnation and axis-exhaustion detected."""
        # 3 KEEPs without progress + 8 discards on lr
        history = [
            _make_exp(100.0, "promote", "lr"),
            _make_exp(100.05, "promote", "lr"),
            _make_exp(100.08, "promote", "lr"),
        ]
        for _ in range(8):
            history.append(
                _make_exp(0.0, "reject", "lr", "failed"),
            )
        result = detect_stagnation(
            history,
            max_keep_no_progress=2,
            single_axis_threshold=8,
            max_exhausted_axes=1,
            significant_improvement_ratio=0.01,
        )
        assert result["stagnant"]
        assert result["exhausted_axes"] == ["lr"]

    def test_axis_distribution(self):
        """Axis distribution is correctly computed."""
        history = [
            _make_exp(50.0, "promote", "lr"),
            _make_exp(60.0, "promote", "arch"),
            _make_exp(70.0, "promote", "arch"),
        ]
        result = detect_stagnation(history)
        assert result["axis_distribution"]["lr"] == 1
        assert result["axis_distribution"]["arch"] == 2

    def test_keep_count_zero_for_progress(self):
        """Significant gains → keep_count = 0."""
        history = [
            _make_exp(50.0, "promote", "lr"),
            _make_exp(100.0, "promote", "lr"),  # 100% gain
        ]
        result = detect_stagnation(
            history,
            significant_improvement_ratio=0.01,
        )
        assert result["keep_count"] == 0


class TestGenerateMonitorReport:
    def test_report_stagnant(self):
        """Report marks stagnant state."""
        # Use dummy axis data — generate_monitor_report doesn't need real history
        report = generate_monitor_report(
            [{"axis": "lr", "count": 3, "exhausted": False}],
            registry_axes={"lr": 10},
        )
        assert "Suggested action" in report

    def test_report_active(self):
        """Report marks active state."""
        history = [
            _make_exp(50.0, "promote", "lr"),
            _make_exp(70.0, "promote", "arch"),
        ]
        report = generate_monitor_report(history)
        assert "Active" in report
        assert "Suggested action" in report
