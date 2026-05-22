#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for CompetitionController — competition scheduling logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from expflow_pde.competition_controller import CompetitionController, _parse_deadline


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
    assert ctrl.check_per_task_limit("task1")
    ctrl.record_task_time("task1", 10)
    assert ctrl.check_per_task_limit("task1")
    ctrl.record_task_time("task1", 3)
    assert not ctrl.check_per_task_limit("task1")


def test_per_task_limit_explore():
    ctrl = CompetitionController(
        session_id="t5", mode="explore", per_task_max_hours=12
    )
    ctrl.record_task_time("task1", 100)
    assert ctrl.check_per_task_limit("task1")


def test_complete_task_advances():
    ctrl = CompetitionController(
        session_id="t6",
        task_order=["task1", "task2", "task3"],
    )
    assert ctrl.get_current_task() == "task1"
    nxt = ctrl.complete_task("task1")
    assert nxt == "task2"
    ctrl.record_task_time("task1", 100)
    assert ctrl.get_current_task() == "task2"
    nxt = ctrl.complete_task("task2")
    assert nxt == "task3"
    ctrl.record_task_time("task2", 100)
    assert ctrl.get_current_task() == "task3"
    nxt = ctrl.complete_task("task3")
    assert nxt is None


def test_complete_task_out_of_order():
    ctrl = CompetitionController(
        session_id="t7",
        task_order=["task1", "task2", "task3"],
    )
    ctrl.record_task_time("task2", 100)
    assert ctrl.get_current_task() == "task1"
    ctrl.record_task_time("task1", 100)
    assert ctrl.get_current_task() == "task3"


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


def test_check_queue_depth_no_clearml():
    """When clearml is unreachable, returns zeros (not crash)."""
    ctrl = CompetitionController(session_id="t10")
    depth = ctrl.check_queue_depth("default")
    assert isinstance(depth, dict)
    assert depth.get("running", -1) >= 0
    assert depth.get("pending", -1) >= 0


def test_to_dict_includes_mode_and_deadline():
    ctrl = CompetitionController(session_id="t11", mode="sprint")
    d = ctrl.to_dict()
    assert d["mode"] == "sprint"
    assert "deadline" in d
    assert "task_order" in d


# ── Pure-stdlib deadline parser tests ──


def test_parse_deadline_tz_plus8():
    dt = _parse_deadline("2026-06-30T23:59:59+08:00")
    assert dt.tzinfo is not None
    offset = dt.tzinfo.utcoffset(dt)
    assert offset.total_seconds() == 8 * 3600


def test_parse_deadline_utc():
    dt = _parse_deadline("2026-06-30T23:59:59Z")
    offset = dt.tzinfo.utcoffset(dt)
    assert offset.total_seconds() == 0


def test_parse_deadline_date_only():
    dt = _parse_deadline("2026-06-30")
    assert dt.hour == 0
    assert dt.minute == 0


def test_parse_deadline_fractional_seconds():
    """Fractional seconds in isoformat output should not break parsing."""
    dt = _parse_deadline("2026-06-30T12:00:00.123456+00:00")
    assert dt is not None
    offset = dt.tzinfo.utcoffset(dt)
    assert offset.total_seconds() == 0
