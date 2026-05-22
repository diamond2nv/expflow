#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for CompetitionController — competition scheduling logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from expflow_pde.competition_controller import CompetitionController


def test_competition_controller_create():
    ctrl = CompetitionController(
        session_id="test_sess",
        mode="sprint",
        deadline="2026-06-30T23:59:59+08:00",
        task_order=["task1", "task2", "task3"],
        per_task_max_hours=12,
    )
    assert ctrl._session_id == "test_sess"
    assert ctrl._mode == "sprint"
    assert ctrl.get_current_task() == "task1"


def test_deadline_not_passed():
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    ctrl = CompetitionController(session_id="t1", deadline=future)
    assert not ctrl.check_deadline()


def test_deadline_passed():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    ctrl = CompetitionController(session_id="t2", deadline=past)
    assert ctrl.check_deadline()


def test_no_deadline():
    ctrl = CompetitionController(session_id="t3")
    assert not ctrl.check_deadline()


def test_per_task_limit_sprint():
    ctrl = CompetitionController(
        session_id="t4", mode="sprint", per_task_max_hours=12
    )
    assert ctrl.check_per_task_limit("task1")  # no time recorded = OK
    ctrl.record_task_time("task1", 10)
    assert ctrl.check_per_task_limit("task1")  # 10 < 12
    ctrl.record_task_time("task1", 3)  # total 13
    assert not ctrl.check_per_task_limit("task1")  # 13 >= 12


def test_per_task_limit_explore():
    ctrl = CompetitionController(
        session_id="t5", mode="explore", per_task_max_hours=12
    )
    ctrl.record_task_time("task1", 100)
    assert ctrl.check_per_task_limit("task1")  # no limit in explore mode


def test_complete_task_advances():
    ctrl = CompetitionController(
        session_id="t6",
        task_order=["task1", "task2", "task3"],
    )
    assert ctrl.get_current_task() == "task1"
    nxt = ctrl.complete_task("task1")
    assert nxt == "task2"
    assert ctrl.get_current_task() == "task1"  # get_current_task always returns first uncompleted
    # Mark task1 completed by adding time to it
    ctrl.record_task_time("task1", 100)
    assert ctrl.get_current_task() == "task2"
    nxt = ctrl.complete_task("task2")
    assert nxt == "task3"
    ctrl.record_task_time("task2", 100)
    assert ctrl.get_current_task() == "task3"
    nxt = ctrl.complete_task("task3")
    assert nxt is None  # all done


def test_complete_task_out_of_order():
    ctrl = CompetitionController(
        session_id="t7",
        task_order=["task1", "task2", "task3"],
    )
    # Mark task2 done while on task1
    ctrl.record_task_time("task2", 100)
    assert ctrl.get_current_task() == "task1"  # still task1
    ctrl.record_task_time("task1", 100)
    assert ctrl.get_current_task() == "task3"  # skip task2 (already "done")


def test_to_dict_serializable():
    ctrl = CompetitionController(session_id="t8", mode="sprint")
    d = ctrl.to_dict()
    assert d["session_id"] == "t8"
    assert d["mode"] == "sprint"
    assert "task_time" in d


def test_remaining_days_positive():
    future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    ctrl = CompetitionController(session_id="t9", deadline=future)
    days = ctrl.remaining_days()
    assert 6.0 < days < 8.0


def test_check_pipeline_in_flight_no_pipeline():
    """When last_pipeline_id is None, returns 'none' status."""
    ctrl = CompetitionController(session_id="t10")
    result = ctrl.check_pipeline_in_flight(None)
    assert result["status"] == "none"
    assert result["pipeline_id"] is None


def test_check_pipeline_recovery_no_pipeline():
    """When all IDs are empty, action is 'submit_new'."""
    ctrl = CompetitionController(session_id="t11")
    result = ctrl.check_pipeline_recovery(None, None, None)
    assert result["action"] == "submit_new"


def test_check_queue_depth_no_clearml():
    """When clearml is unreachable, returns zeros (not crash)."""
    ctrl = CompetitionController(session_id="t12")
    depth = ctrl.check_queue_depth("default")
    assert isinstance(depth, dict)
    assert depth.get("running", -1) >= 0
    assert depth.get("pending", -1) >= 0
