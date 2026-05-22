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
    assert ctrl.session_id == "test_sess"
    assert ctrl.mode == "sprint"
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
    assert ctrl.get_current_task() == "task2"
    nxt = ctrl.complete_task("task2")
    assert nxt == "task3"
    assert ctrl.get_current_task() == "task3"
    nxt = ctrl.complete_task("task3")
    assert nxt is None  # all done


def test_complete_task_out_of_order():
    ctrl = CompetitionController(
        session_id="t7",
        task_order=["task1", "task2", "task3"],
    )
    # Mark task2 done while on task1
    ctrl.complete_task("task2")
    assert ctrl.get_current_task() == "task1"  # still task1
    ctrl.complete_task("task1")
    assert ctrl.get_current_task() == "task3"  # skip task2


def test_budget_ok():
    ctrl = CompetitionController(session_id="t8", budget=100.0)
    assert ctrl.check_budget()


def test_budget_explore_no_limit():
    ctrl = CompetitionController(session_id="t9", budget=float("inf"))
    assert ctrl.check_budget()


def test_keyword_matching():
    ctrl = CompetitionController(
        session_id="k1",
        mode="explore",
        task_order=["task1"],
    )
    assert ctrl.mode == "explore"
    assert ctrl.get_current_task() == "task1"


def test_unknown_mode_raises():
    try:
        CompetitionController(session_id="x1", mode="invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass
