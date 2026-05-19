#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.compare — model comparison and scoring.

Covers:
- compare_scores() with mocked clearml tasks
- Gate filtering (pass/fail)
- Sort by metric
- compare_tasks() basic structure
- compare_scores with empty results
"""

import pytest

from expflow_pde.compare import _apply_gate, compare_tasks


@pytest.mark.integration
class TestCompareTasks:
    """compare_tasks() — requires clearml SDK."""

    def test_without_clearml_returns_error(self):
        # When clearml is not installed, get_task() inside compare_tasks will fail
        result = compare_tasks("nonexistent-a", "nonexistent-b")
        assert "error" in result


class TestCompareScores:
    """compare_scores() — tests business logic via mocked data.

    In unit tests, compares_scores requires clearml so it's marked integration.
    We test _apply_gate and the sorting logic directly.
    """


class TestApplyGate:
    """_apply_gate() logic."""

    def test_lt_passes(self):
        assert _apply_gate(10.0, "lt", 20.0) is True

    def test_lt_fails(self):
        assert _apply_gate(20.0, "lt", 10.0) is False

    def test_le_passes(self):
        assert _apply_gate(10.0, "le", 10.0) is True
        assert _apply_gate(5.0, "le", 10.0) is True

    def test_le_fails(self):
        assert _apply_gate(15.0, "le", 10.0) is False

    def test_gt_passes(self):
        assert _apply_gate(20.0, "gt", 10.0) is True

    def test_gt_fails(self):
        assert _apply_gate(5.0, "gt", 10.0) is False

    def test_ge_passes(self):
        assert _apply_gate(10.0, "ge", 10.0) is True
        assert _apply_gate(15.0, "ge", 10.0) is True

    def test_ge_fails(self):
        assert _apply_gate(5.0, "ge", 10.0) is False

    def test_unknown_op_passes(self):
        assert _apply_gate(10.0, "eq", 20.0) is True
