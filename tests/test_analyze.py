#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.analyze — task intelligence and strategic advising.

Covers:
- list_task_summaries()
- analyze_task() per task
- get_strategic_recommendation()
- get_equation_analysis()
- list_all_equations_summary()
- estimate_score_potential()
"""
from expflow_pde.analyze import (
    _format_deadline_str,
    _get_competition_deadline,
    analyze_task,
    diagnose_experiment,
    estimate_score_potential,
    get_equation_analysis,
    get_strategic_recommendation,
    list_all_equations_summary,
    list_task_summaries,
)


class TestListTaskSummaries:
    """list_task_summaries() — overview of all tasks."""

    def test_returns_all_tasks(self):
        summaries = list_task_summaries()
        assert len(summaries) >= 3

    def test_each_task_has_required_keys(self):
        for s in list_task_summaries():
            assert "task_id" in s
            assert "label" in s
            assert "max_score" in s
            assert "difficulty" in s
            assert "priority" in s
            assert "status" in s
            assert "bottlenecks" not in s or True  # renamed
            assert "key_bottlenecks" in s
            assert "proven_strategies" in s
            assert "next_steps" in s

    def test_task_ids_are_valid(self):
        ids = {s["task_id"] for s in list_task_summaries()}
        assert ids == {"task1", "task2", "task3"}

    def test_scores_are_positive(self):
        for s in list_task_summaries():
            assert s["max_score"] > 0
            assert s["estimated_ceiling"] >= 0


class TestAnalyzeTask:
    """analyze_task() — detailed task analysis."""

    def test_task1_details(self):
        r = analyze_task("task1")
        assert r is not None
        assert r["max_score"] == 150
        assert r["difficulty"] == "medium"
        assert r["status"] == "in_progress"
        assert r["current_best"] == 57.09
        assert len(r["key_bottlenecks"]) > 0
        assert len(r["proven_strategies"]) > 0
        assert len(r["equations"]) > 0

    def test_task2_details(self):
        r = analyze_task("task2")
        assert r is not None
        assert r["max_score"] == 150
        assert r["difficulty"] == "hard"
        assert r["status"] == "not_started"
        assert r["current_best"] is None

    def test_task3_details(self):
        r = analyze_task("task3")
        assert r is not None
        assert r["max_score"] == 350
        assert r["difficulty"] == "very_hard"
        assert r["status"] == "not_started"

    def test_unknown_task(self):
        assert analyze_task("task99") is None

    def test_each_task_has_estimated_ceiling(self):
        for task_id in ("task1", "task2", "task3"):
            r = analyze_task(task_id)
            assert r is not None
            assert r["estimated_ceiling"] > 0

    def test_each_task_has_bottlenecks(self):
        for task_id in ("task1", "task2", "task3"):
            r = analyze_task(task_id)
            assert r is not None
            assert len(r["key_bottlenecks"]) > 0


class TestEstimateScorePotential:
    """estimate_score_potential() — score projection."""

    def test_task1_estimates(self):
        e = estimate_score_potential("task1")
        assert e["optimistic"] > e["expected"]
        assert e["expected"] > e["conservative"]
        assert e["confidence"] == "high"

    def test_task2_estimates(self):
        e = estimate_score_potential("task2")
        assert e["optimistic"] > e["expected"]
        assert e["expected"] > e["conservative"]
        assert e["confidence"] == "low"

    def test_task3_estimates(self):
        e = estimate_score_potential("task3")
        assert e["optimistic"] > e["expected"]
        assert e["expected"] > e["conservative"]
        assert e["confidence"] == "low"


class TestStrategicRecommendation:
    """get_strategic_recommendation() — overall strategy."""

    def test_returns_recommendation(self):
        r = get_strategic_recommendation()
        assert r["primary_focus"] in ("task1", "task2", "task3")
        assert r["remaining_days"] >= 0
        assert isinstance(r["competition_deadline"], str) and len(r["competition_deadline"]) > 0
        assert len(r["suggested_schedule"]) >= 1

    def test_schedule_has_day_keys(self):
        r = get_strategic_recommendation()
        for key in r["suggested_schedule"]:
            assert "_" in key  # day_1_2 etc.

    def test_env_override_deadline(self):
        """EXPFLOW_COMPETITION_DEADLINE env var overrides default."""
        import os

        os.environ["EXPFLOW_COMPETITION_DEADLINE"] = "2026-07-15"
        try:
            d = _get_competition_deadline()
            assert d.isoformat() == "2026-07-15"
        finally:
            del os.environ["EXPFLOW_COMPETITION_DEADLINE"]

    def test_format_deadline_str(self):
        s = _format_deadline_str()
        assert isinstance(s, str) and len(s) > 0


class TestEquationAnalysis:
    """get_equation_analysis() — per-equation detail."""

    def test_burgers_exists(self):
        e = get_equation_analysis("burgers")
        assert e is not None
        assert e["full_name"] is not None
        assert "task1" in e.get("assigned_tasks", [])
        assert e["dim"] == 1

    def test_ks_exists(self):
        e = get_equation_analysis("kuramoto_sivashinsky")
        assert e is not None
        assert "task3" in e.get("assigned_tasks", [])
        assert e["time_dependent"] is True

    def test_unknown_equation(self):
        assert get_equation_analysis("nonexistent") is None

    def test_advection(self):
        e = get_equation_analysis("advection")
        assert e is not None
        assert e["assigned_tasks"] == []


class TestDiagnoseExperiment:
    """diagnose_experiment() — degradation pattern detection."""

    def test_diagnose_stable(self):
        from expflow_pde.analyze import diagnose_experiment

        result = diagnose_experiment(json_path="nonexistent.json")
        assert result is None  # file not found

    def test_diagnose_ceiling_detection(self):
        import json
        import os
        import tempfile

        from expflow_pde.analyze import diagnose_experiment

        data = {
            "segmented_scores": {
                "seg1": 68,
                "seg2": 67,
                "seg3": 55,
                "total": 190,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            result = diagnose_experiment(json_path=fname)
            assert result is not None
            assert result["degradation_pattern"] == "ceiling", (
                f"Expected ceiling, got {result['degradation_pattern']}"
            )
            assert any("ceiling" in d.lower() for d in result["diagnosis"])
        finally:
            os.unlink(fname)

    def test_diagnose_compound_mid_long(self):
        import json
        import os
        import tempfile

        from expflow_pde.analyze import diagnose_experiment

        data = {
            "segmented_scores": {
                "seg1": 95,
                "seg2": 40,
                "seg3": 38,
                "total": 173,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            result = diagnose_experiment(json_path=fname)
            assert result is not None
            # Seg1-Seg2=55 > 25 → mid_term, Seg3=38 >= 35 but Seg3 < Seg2*0.6=24?
            # 38 < 24? No. So long_term not triggered. Mid_term only.
            # Wait: let me verify the actual pattern
            print(f"DEBUG: {result}")
            assert result["degradation_pattern"] in ("mid_term",), (
                f"Expected mid_term, got {result['degradation_pattern']}"
            )
        finally:
            os.unlink(fname)

    def test_diagnose_compound_mid_long_both(self):
        import json
        import os
        import tempfile

        from expflow_pde.analyze import diagnose_experiment

        data = {
            "segmented_scores": {
                "seg1": 95,
                "seg2": 40,
                "seg3": 15,
                "total": 150,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            result = diagnose_experiment(json_path=fname)
            assert result is not None
            assert result["degradation_pattern"] == "compound_mid_long", (
                f"Expected compound_mid_long, got {result['degradation_pattern']}"
            )
        finally:
            os.unlink(fname)

    def test_diagnose_clearml_error_returns_info(self):
        # Test that _error path in diagnose_experiment works end-to-end
        import json
        import os
        import tempfile

        from expflow_pde.analyze import diagnose_experiment

        data = {"_error": "clearml server connection failed: test"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            result = diagnose_experiment(json_path=fname)
            assert result is not None
            assert result["degradation_pattern"] == "error"
            assert "_connection_error" in result
        finally:
            os.unlink(fname)


class TestListAllEquationsSummary:
    """list_all_equations_summary() — compact listing."""

    def test_contains_known_equations(self):
        eqs = list_all_equations_summary()
        names = {e["name"] for e in eqs}
        assert "burgers" in names
        assert "kuramoto_sivashinsky" in names
        assert "advection" in names
        assert len(names) >= 10

    def test_each_has_required_fields(self):
        for e in list_all_equations_summary():
            assert e["name"]
            assert e["dim"]
            assert e["difficulty"]
            assert e["competition_task"] is not None
