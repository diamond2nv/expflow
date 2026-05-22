#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary collision tests using the DummyExperimentGame + repair + state.

Exercises edge cases across:
    A. resolve_pipeline_state — clearml fallback (6 arbitration paths)
    B. CompetitionController — deadline ISO 8601, time limits, queue
    C. Dummy Game — empty/oversized params, all failures, ceiling, reset
    D. RepairStage — signal codes, empty logs, L2 params, describe_params
    E. GoalOrchestrator — key rejection, corrupt file, dedup, 20-cap

Relies on zero external services (no clearml, no torch, no GPU).
"""

from __future__ import annotations

import json
import os

import pytest

from expflow_pde.competition_controller import CompetitionController, _parse_deadline
from expflow_pde.dummy.game import DummyExperimentGame, _FAILURE_TEMPLATES, _CEILING
from expflow_pde.goal_orchestrator import (
    add_learned_failure,
    clear as gs_clear,
    load as gs_load,
    resolve_pipeline_state,
    save as gs_save,
    _set_state_path,
    _verify_state,
)
from expflow_pde.repair import RepairStage, _describe_params

# ════════════════════════════════════════════
# A. resolve_pipeline_state — fallback paths
# ════════════════════════════════════════════


class TestResolvePipelineState:
    """All clearml SDK lookups return None in CI/test — tests the fallback branch."""

    def test_empty_ids_returns_submit_new(self):
        r = resolve_pipeline_state({"last_pipeline_id": "", "last_train_task_id": ""})
        assert r["action"] == "submit_new"

    def test_nonexistent_ids_returns_submit_new(self):
        r = resolve_pipeline_state({
            "last_pipeline_id": "nonexistent_pipe",
            "last_train_task_id": "nonexistent_train",
        })
        assert r["action"] == "submit_new"

    @pytest.mark.parametrize("state", [
        {"last_pipeline_id": "p", "last_train_task_id": ""},
        {"last_pipeline_id": "", "last_train_task_id": "t"},
        {"last_pipeline_id": "p", "last_train_task_id": "t"},
    ])
    def test_unreachable_clearml_always_submit_new(self, state):
        """When clearml SDK returns None, we must NOT stall."""
        r = resolve_pipeline_state(state)
        assert r["action"] == "submit_new"

    def test_result_structure(self):
        r = resolve_pipeline_state({"last_pipeline_id": "", "last_train_task_id": ""})
        assert "action" in r
        assert "pipeline_status" in r
        assert "train_task_status" in r
        assert "eval_task_status" in r
        assert "train_killed" in r


# ════════════════════════════════════════════
# B. CompetitionController deadline / per-task
# ════════════════════════════════════════════


class TestCompetitionControllerBoundaries:

    def test_parse_deadline_plus8(self):
        dt = _parse_deadline("2026-06-30T23:59:59+08:00")
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt).total_seconds() == 8 * 3600

    def test_parse_deadline_utc_z(self):
        dt = _parse_deadline("2026-06-30T23:59:59Z")
        assert dt.tzinfo.utcoffset(dt).total_seconds() == 0

    def test_parse_deadline_date_only(self):
        dt = _parse_deadline("2026-06-30")
        assert dt.hour == 0
        assert dt.minute == 0

    def test_parse_deadline_fractional_seconds(self):
        dt = _parse_deadline("2026-06-30T12:00:00.123456+00:00")
        assert dt is not None
        assert dt.tzinfo.utcoffset(dt).total_seconds() == 0

    def test_parse_deadline_negative_offset(self):
        dt = _parse_deadline("2026-06-30T23:59:59-05:00")
        assert dt.tzinfo.utcoffset(dt).total_seconds() == -18000

    def test_parse_deadline_half_hour_offset(self):
        dt = _parse_deadline("2026-06-30T23:59:59+05:30")
        assert dt.tzinfo.utcoffset(dt).total_seconds() == 19800

    def test_parse_deadline_no_tz(self):
        dt = _parse_deadline("2026-06-30T23:59:59")
        assert dt.tzinfo is not None
        assert dt.tzinfo.utcoffset(dt).total_seconds() == 0

    def test_per_task_limit_equal(self):
        ctrl = CompetitionController(session_id="t", mode="sprint", per_task_max_hours=12)
        ctrl.record_task_time("task1", 12)
        assert not ctrl.check_per_task_limit("task1")

    def test_per_task_limit_just_under(self):
        ctrl = CompetitionController(session_id="t", mode="sprint", per_task_max_hours=12)
        ctrl.record_task_time("task1", 11.99)
        assert ctrl.check_per_task_limit("task1")

    def test_per_task_limit_explore_mode_ignored(self):
        ctrl = CompetitionController(session_id="t", mode="explore", per_task_max_hours=12)
        ctrl.record_task_time("task1", 100)
        assert ctrl.check_per_task_limit("task1")

    def test_auto_advance_over_limit(self):
        ctrl = CompetitionController(
            session_id="t", mode="sprint",
            task_order=["task1", "task2"], per_task_max_hours=12,
        )
        ctrl.record_task_time("task1", 12)
        assert ctrl.get_current_task() == "task2"

    def test_queue_depth_unreachable(self):
        ctrl = CompetitionController(session_id="t")
        d = ctrl.check_queue_depth("nonexistent")
        assert d == {"running": 0, "pending": 0, "total": 0}

    def test_should_wait_unreachable(self):
        ctrl = CompetitionController(session_id="t")
        assert not ctrl.should_wait_for_queue("nonexistent")

    def test_complete_task_chain(self):
        ctrl = CompetitionController(session_id="t", task_order=["a", "b", "c"])
        ctrl.record_task_time("a", 1)
        assert ctrl.complete_task("a") == "b"
        ctrl.record_task_time("b", 1)
        assert ctrl.complete_task("b") == "c"
        ctrl.record_task_time("c", 1)
        assert ctrl.complete_task("c") is None

    def test_to_dict_serializable(self):
        ctrl = CompetitionController(session_id="s1", mode="sprint")
        d = ctrl.to_dict()
        assert d["session_id"] == "s1"
        assert d["mode"] == "sprint"
        assert "task_time" in d

    def test_remaining_days_positive(self):
        import datetime
        future = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)).isoformat()
        ctrl = CompetitionController(session_id="t", deadline=future)
        days = ctrl.remaining_days()
        assert 6.0 < days < 8.0

    def test_remaining_days_near_zero(self):
        import datetime
        exact_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ctrl = CompetitionController(session_id="t", deadline=exact_now)
        assert ctrl.remaining_days() < 0.01


# ════════════════════════════════════════════
# C. Dummy Game edge cases
# ════════════════════════════════════════════


class TestDummyGameBoundaries:

    def test_empty_params(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        r = g.step(suggested_params={})
        assert r["status"] in ("completed", "failed")

    def test_none_params(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        r = g.step(suggested_params=None)
        assert r["status"] in ("completed", "failed")

    def test_large_params_dict(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        r = g.step(suggested_params={f"p{i}": i for i in range(100)})
        assert r["status"] in ("completed", "failed")

    @pytest.mark.parametrize("pattern_name", list(_FAILURE_TEMPLATES.keys()))
    def test_all_failure_injection_patterns(self, tmp_path, pattern_name):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        r = g.step(inject_failure=pattern_name)
        assert r["status"] == "failed"
        assert r["inject_failure"]["failure"] == pattern_name

    def test_inject_none_forces_success(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        r = g.step(inject_failure="none")
        assert r["status"] == "completed"

    def test_inject_unknown_pattern(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        r = g.step(inject_failure="__no_such_pattern__")
        assert r["status"] in ("completed", "failed")

    def test_reset_clears_state(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        g.step()
        g.reset()
        assert g._step_count == 0
        assert g._last_exp_id is None

    def test_ceiling_caps_scores(self, tmp_path):
        g = DummyExperimentGame(task_id="task1", db_path=str(tmp_path / "g.db"), seed=1)
        g.start()
        for _ in range(20):
            g.step(suggested_params={"n_modes": 24, "width": 64, "num_sub_steps": 10}, inject_failure="none")
        c = _CEILING["task1"]
        seg = g._current_seg
        assert seg["seg1"] <= c["seg1"]
        assert seg["seg2"] <= c["seg2"]
        assert seg["seg3"] <= c["seg3"]
        assert seg["seg1"] >= 0
        assert seg["seg2"] >= 0
        assert seg["seg3"] >= 0

    def test_steps_to_ceiling_decreases(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        initial = g._steps_to_ceiling()
        for _ in range(3):
            g.step(suggested_params={"n_modes": 20}, inject_failure="none")
        assert g._steps_to_ceiling() <= initial

    def test_restart_creates_new_root(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        r1 = g.start()
        r2 = g.start()
        assert r2["experiment_id"] != r1["experiment_id"]
        assert g._step_count == 0

    def test_seg_noise_does_not_crash(self, tmp_path):
        g = DummyExperimentGame(db_path=str(tmp_path / "g.db"))
        g.start()
        for _ in range(5):
            r = g.step(inject_failure="none")
            assert r["seg"]["seg1"] >= 0


# ════════════════════════════════════════════
# D. RepairStage boundary cases
# ════════════════════════════════════════════


class TestRepairBoundaries:

    def test_exit_0_no_repair(self):
        r = RepairStage().run(task_log="", exit_code=0)
        assert r["level"] == "none"
        assert not r["fixed"]

    def test_empty_log_exit_1(self):
        r = RepairStage().run(task_log="", exit_code=1)
        assert r["level"] == "L1"

    def test_very_long_log_no_errors(self):
        long_log = "\n".join([f"Line {i}" for i in range(10_000)])
        r = RepairStage().run(task_log=long_log, exit_code=1)
        assert r["level"] == "L1"
        assert "error" in r.get("exit_code_category", "")

    @pytest.mark.parametrize("ec,name", [(137, "SIGKILL"), (139, "SIGSEGV"), (134, "SIGABRT"), (143, "SIGTERM")])
    def test_signal_exit_codes(self, ec, name):
        r = RepairStage().run(task_log=f"{name}\n", exit_code=ec)
        assert r["level"] == "L1"

    def test_git_not_found_L0_rule(self):
        log, code, _ = _FAILURE_TEMPLATES["git_not_found"]
        r = RepairStage().run(task_log=log, exit_code=code)
        assert r["level"] == "L0"
        # L0 matched with needs_user_action is still valid
        assert any(
            h.get("matched") and h.get("needs_user_action")
            for h in r.get("history", [])
            if h.get("level") == "L0"
        )

    def test_unknown_error_enable_reflection_with_params(self):
        log, code, _ = _FAILURE_TEMPLATES["unknown_error"]
        r = RepairStage(experiment_id="exp:test_123").run(
            task_log=log, exit_code=code,
            enable_reflection=True,
            experiment_params={"n_modes": 24, "batch_size": 4},
        )
        assert r["level"] == "L2"
        assert r.get("input_valid", True) is True
        assert r["subagent_schema"].get("suggested_params", {}) == {"n_modes": 24, "batch_size": 4}

    def test_empty_log_L2_input_invalid(self):
        log, code, _ = _FAILURE_TEMPLATES["empty_log"]
        r = RepairStage().run(task_log=log, exit_code=code, enable_reflection=True)
        assert r.get("input_valid", True) is False

    def test_signal_oom_L1(self):
        log, code, _ = _FAILURE_TEMPLATES["signal_oom"]
        r = RepairStage(experiment_id="exp:oom_test").run(task_log=log, exit_code=code)
        assert r["level"] == "L1"
        assert "signal" in r.get("exit_code_category", "")

    def test_disk_quota_rule(self):
        log, code, _ = _FAILURE_TEMPLATES["disk_quota"]
        r = RepairStage().run(task_log=log, exit_code=code)
        assert r["level"] in ("L0", "L1")

    def test_import_error_rule(self):
        """ImportError may be caught by L0 (ModuleNotFoundError) or fall through to L1."""
        log, code, _ = _FAILURE_TEMPLATES["import_error"]
        r = RepairStage().run(task_log=log, exit_code=code)
        assert r["level"] in ("L0", "L1")
        assert r["level"] != "none"


class TestDescribeParams:
    def test_none_returns_empty(self):
        assert _describe_params(None) == ""

    def test_empty_dict(self):
        assert _describe_params({}) == ""

    def test_single_key(self):
        assert _describe_params({"n_modes": 24}) == "n_modes=24"

    def test_mixed_types(self):
        result = _describe_params({"lr": 0.001, "flag": True})
        assert "lr=0.001" in result
        assert "flag=True" in result


# ════════════════════════════════════════════
# E. GoalOrchestrator save/load boundaries
# ════════════════════════════════════════════


class TestGoalOrchestratorBoundaries:

    def test_verify_state_rejects_chinese_keys(self):
        with pytest.raises(ValueError, match="unexpected state key"):
            _verify_state({"best_score": 100, "建议": {"x": 1}})

    def test_verify_state_accepts_english_keys(self):
        _verify_state({"best_score": 100, "current_phase": "completed"})

    def test_unknown_phase_resets_to_idle(self, tmp_path):
        _set_state_path(str(tmp_path / "state.json"))
        gs_save({"current_phase": "garbage_phase"})
        assert gs_load()["current_phase"] == "idle"

    def test_corrupt_file_returns_defaults(self, tmp_path):
        p = str(tmp_path / "state.json")
        _set_state_path(p)
        with open(p, "w") as f:
            f.write("{corrupt")
        state = gs_load()
        assert state["root_experiment_id"] == ""
        assert state["current_phase"] == "idle"

    def test_no_file_returns_defaults(self, tmp_path):
        _set_state_path(str(tmp_path / "nope.json"))
        state = gs_load()
        assert state["root_experiment_id"] == ""
        assert state["current_phase"] == "idle"

    def test_load_strips_unknown_keys(self, tmp_path):
        p = str(tmp_path / "state.json")
        _set_state_path(p)
        with open(p, "w") as f:
            json.dump({"best_score": 142, "ChineseKey": "bad", "current_phase": "waiting"}, f)
        state = gs_load()
        assert state["best_score"] == 142.0
        assert "ChineseKey" not in state

    def test_learned_failures_capped_at_20(self, tmp_path):
        p = str(tmp_path / "state.json")
        _set_state_path(p)
        for i in range(25):
            add_learned_failure({"type": f"f{i}", "params": {}, "reason": str(i), "experiment_id": None})
        assert len(gs_load()["learned_failures"]) == 20

    def test_learned_failures_dedup_same_param_keys(self, tmp_path):
        p = str(tmp_path / "state.json")
        _set_state_path(p)
        add_learned_failure({"type": "oom", "params": {"n_modes": 24}, "reason": "original", "experiment_id": "e1"})
        add_learned_failure({"type": "oom", "params": {"n_modes": 24}, "reason": "updated", "experiment_id": "e2"})
        state = gs_load()
        assert len(state["learned_failures"]) == 1
        assert state["learned_failures"][0]["reason"] == "updated"
        assert state["learned_failures"][0]["experiment_id"] == "e2"

    def test_learned_failures_no_dedup_different_param_keys(self, tmp_path):
        p = str(tmp_path / "state.json")
        _set_state_path(p)
        add_learned_failure({"type": "oom", "params": {"n_modes": 24}, "reason": "a", "experiment_id": "e1"})
        add_learned_failure({"type": "oom", "params": {"batch_size": 4}, "reason": "b", "experiment_id": "e2"})
        state = gs_load()
        # Different param keys -> separate entries
        assert len(state["learned_failures"]) == 2
        reasons = {e["reason"] for e in state["learned_failures"]}
        assert "a" in reasons
        assert "b" in reasons

    def test_clear_removes_file(self, tmp_path):
        p = str(tmp_path / "state.json")
        _set_state_path(p)
        gs_save({"root_experiment_id": "x"})
        assert os.path.isfile(p)
        gs_clear()
        assert not os.path.isfile(p)
