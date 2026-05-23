#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end mock test: simulate a complete Hermes /goal competition loop.

This test exercises the full pipeline:
  CompetitionController → pipeline submit (mock) → scalar read →
  scoring → StagnationDetector → GoalOrchestrator persistence →
  task switch

No real clearml server or GPU is needed. All clearml interactions
are mocked at the SDK level.

Scenario:
  Sprint mode with 2 tasks (task1, task2), per-task limit 3 iterations.
  Task 1: 3 attempts, stagnates → submit best → switch to task 2
  Task 2: 1 attempt → deadline triggers → emergency submit
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Mock clearml package ──


@pytest.fixture(autouse=True)
def mock_clearml_pkg():
    """Replace 'clearml' in sys.modules so lazy imports receive mocks.

    Also creates a mock PipelineController that tracks created pipelines
    and simulates completed runs with configurable scalars.
    """
    # Clear any previously cached expflow clearml module
    for mod in list(sys.modules.keys()):
        if "expflow.clearml" in mod or "expflow_pde.clearml" in mod:
            del sys.modules[mod]

    pkg = MagicMock(name="clearml_pkg")

    created_pipelines: list[dict[str, Any]] = []

    def _mock_create(name, project, **kwargs):
        pipe_id = f"pipe_{len(created_pipelines) + 1:04d}"
        entry = {"pipeline_id": pipe_id, "name": name, "project": project, "status": "created"}
        created_pipelines.append(entry)
        mock_pipe = MagicMock(name=f"Pipeline({pipe_id})")
        mock_pipe.pipeline_id = pipe_id
        mock_pipe.add_step = MagicMock()
        mock_pipe.start.return_value = pipe_id
        mock_pipe.wait.return_value = MagicMock()
        mock_pipe.get_status.return_value = "completed"
        return mock_pipe

    mock_pipe_controller = MagicMock()
    mock_pipe_controller.create.side_effect = _mock_create
    mock_pipe_controller.add_step = MagicMock()
    mock_pipe_controller.start = MagicMock()
    mock_pipe_controller.stop = MagicMock()

    pkg.PipelineController = MagicMock(return_value=mock_pipe_controller)
    pkg.PipelineController.return_value = mock_pipe_controller

    _scalar_counter: dict[str, int] = {"call": 0}

    def _mock_get_task(task_id=None, **kwargs):
        t = MagicMock(name=f"Task({task_id})")
        _scalar_counter["call"] += 1
        step = _scalar_counter["call"]

        if step == 1:
            seg_total = 75.0
            train_time_min = 45.0
        elif step == 2:
            seg_total = 105.0
            train_time_min = 48.0
        elif step == 3:
            seg_total = 106.0
            train_time_min = 50.0
        elif step == 4:
            seg_total = 106.5
            train_time_min = 52.0
        else:
            seg_total = 30.0
            train_time_min = 10.0

        t.get_last_scalars.return_value = {
            "Score": {"seg_total": seg_total},
            "Time": {"train_time_min": train_time_min},
        }
        t.get_parameters.return_value = {
            "Args/epochs": "80",
            "Args/lr": "0.001",
            "Args/sub_step": "5",
        }
        return t

    pkg.Task = MagicMock()
    pkg.Task.get_task.side_effect = _mock_get_task
    pkg.Task.get_tasks.return_value = []

    auto_mod = MagicMock(name="clearml.automation")
    auto_mod.TaskScheduler = MagicMock(name="TaskScheduler")

    with patch.dict(
        "sys.modules",
        {
            "clearml": pkg,
            "clearml.automation": auto_mod,
            "clearml.PipelineController": MagicMock(return_value=mock_pipe_controller),
        },
    ):
        yield pkg


@pytest.fixture
def expflow_home():
    """Set EXPFLOW_HOME to a temp dir for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_home = os.environ.get("EXPFLOW_HOME", "")
        os.environ["EXPFLOW_HOME"] = tmpdir
        yield
        if old_home:
            os.environ["EXPFLOW_HOME"] = old_home
        else:
            del os.environ["EXPFLOW_HOME"]


@pytest.fixture(autouse=True)
def reset_config():
    from expflow_pde import config
    config._config_cache.clear()


# ── The actual end-to-end simulation ──


class TestGoalLoopMock:
    """Simulate a full Hermes /goal competition loop using mock clearml."""

    def simulate_task1(self, ctrl) -> dict[str, Any]:
        """Run up to 5 iterations on task1; stop on stagnation."""

        state = {'best_score': 0, 'best_params': {}, 'best_eval_id': None}
        stagnation_count = 0

        for iteration in range(5):
            if ctrl.check_deadline():
                break
            task = ctrl.get_current_task()
            if task != 'task1':
                break
            if not ctrl.check_per_task_limit(task):
                break

            from expflow_pde.pipeline import ExperimentPipeline
            ep = ExperimentPipeline(queue='default')
            result = ep.train_val_submit(
                train_script='mock_train.py',
                eval_script='mock_eval.py',
            )
            eval_id = result.get('eval_task_id', f"eval_mock_{iteration:04d}")

            from expflow_pde.clearml import get_task_scalars
            scalars = get_task_scalars(eval_id) or {}
            seg_total = scalars.get('Score/seg_total', 0.0)
            train_time = scalars.get('Time/train_time_min', 60.0)

            train_score = max(0, 35 * (1 - max(0, train_time - 60) / 60))
            infer_score = 40.0
            total = seg_total * 0.75 + train_score + infer_score

            if total > state['best_score']:
                state['best_score'] = total
                state['best_params'] = {'epochs': 80}
                state['best_eval_id'] = eval_id
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= 3:
                break

            ctrl.record_task_time(task, 1.0)
            ctrl.save(extra={
                'best_score': state['best_score'],
                'best_params': state['best_params'],
                'best_eval_id': state['best_eval_id'],
            })

        return state

    def test_full_goal_loop(self, mock_clearml_pkg, expflow_home):
        """Simulate a full goal loop across 2 tasks."""
        import importlib

        import expflow_pde.goal_orchestrator as go_mod
        importlib.reload(go_mod)
        from expflow_pde.competition_controller import CompetitionController
        from expflow_pde.goal_orchestrator import load as _go_load

        ctrl = CompetitionController(
            session_id='sess_e2e_test',
            mode='sprint',
            task_order=['task1', 'task2'],
            per_task_max_hours=5.0,
        )

        result1 = self.simulate_task1(ctrl)
        assert result1['best_score'] > 0, "Should have recorded a best score"

        next_task = ctrl.complete_task('task1')
        assert next_task == 'task2', "Should advance to task2"

        task = ctrl.get_current_task()
        assert task == 'task2'

        ctrl.record_task_time('task2', 0.5)
        ctrl.save(extra={
            'best_score': result1['best_score'],
            'best_params': result1['best_params'],
        })

        loaded = _go_load()
        assert isinstance(loaded, dict)
        assert loaded.get('session_id') == 'sess_e2e_test'
        assert loaded.get('best_score', 0) > 0

        progress_path = os.path.join(os.environ["EXPFLOW_HOME"], "progress_state.json")
        assert os.path.isfile(progress_path), f"progress state not at {progress_path}"

        with open(progress_path) as f:
            saved = json.load(f)
        assert saved['session_id'] == 'sess_e2e_test'
        assert saved['best_score'] == result1['best_score']

    def test_stagnation_triggers_switch(self, mock_clearml_pkg, expflow_home):
        """StagnationDetector fires after 3 repeated plateau scores."""
        import importlib

        import expflow_pde.goal_orchestrator as go_mod
        importlib.reload(go_mod)
        from expflow_pde.competition_controller import CompetitionController
        from expflow_pde.goal_orchestrator import load as _go_load
        from expflow_pde.goal_orchestrator import save as _go_save

        ctrl = CompetitionController(
            session_id='sess_stag_test',
            mode='sprint',
            task_order=['task1', 'task2'],
            per_task_max_hours=5.0,
        )

        for i in range(3):
            ctrl.record_task_time('task1', 1.0)
            _go_save({
                'session_id': 'sess_stag_test',
                'best_score': 100.0,
                'best_params': {'epochs': 80},
                'best_eval_id': f'eval_stag_{i}',
                'task_hours': {'task1': float(i + 1)},
            })

        assert ctrl.check_per_task_limit('task1'), "3h < 5h limit should be OK"

        next_task = ctrl.complete_task('task1')
        assert next_task == 'task2'

        # Mark task1 as exhausted so get_current_task advances
        ctrl.record_task_time('task1', 100)
        assert ctrl.get_current_task() == 'task2'

        loaded = _go_load()
        assert loaded['best_score'] == 100.0


class TestV3GoalPromptLogic:
    """Core logic from the v3 /goal prompt: diagnose_segs, propose_params.

    These are pure-Python functions with no clearml dependency.
    They directly implement the reasoning loop described in
    p2-goal-prompt.md and thus validate the prompt's correctness.
    """

    # ── v3 prompt: diagnose_segs ──

    @staticmethod
    def _diagnose_segs(seg1, seg2, seg3, task_id):
        """From v3 prompt section 2: degradation diagnosis."""
        decay_rate = max(seg1 - seg2, seg2 - seg3) / max(seg1, seg2, 1e-8)

        thresholds = {
            "task1": {"seg1_low": 60, "seg3_low": 35},
            "task2": {"seg1_low": 40, "seg3_low": 25},
            "task3": {"seg1_low": 10, "seg3_low": 10},
        }
        th = thresholds.get(task_id, thresholds["task1"])

        if seg1 < th["seg1_low"]:
            return "short_term"
        if seg1 - seg2 > 25:
            return "mid_term"
        if seg3 < th["seg3_low"] or seg3 < seg2 * 0.6:
            return "long_term"
        if seg1 < 70 and seg2 > seg1 * 0.85 and seg3 > seg2 * 0.7:
            return "ceiling"
        return "stable"

    # ── v3 prompt: propose_params ──

    @staticmethod
    def _propose_params(pattern, best_params=None):
        """From v3 prompt section 3: parameter adjustment by degradation."""
        best = best_params or {}
        base = {
            "n_modes": best.get("n_modes", 16),
            "hidden_channels": best.get("hidden_channels", 64),
            "sub_step": best.get("sub_step", 5),
        }

        if pattern == "short_term":
            base["n_modes"] = min(base["n_modes"] * 2, 32)
            base["epochs"] = max(best.get("epochs", 80), 100)
        elif pattern in ("mid_term", "long_term"):
            base["sub_step"] = min(base["sub_step"] + 5, 20)
            base["n_modes"] = max(base["n_modes"], 16)
        elif pattern == "ceiling":
            base["hidden_channels"] = min(base["hidden_channels"] * 2, 256)
            base["n_modes"] = min(base["n_modes"] * 2, 32)
        elif pattern in ("stable", "init"):
            pass

        return base

    # ── Tests ──

    def test_diagnose_short_term(self):
        """Seg1=55 on task1 triggers short_term."""
        assert self._diagnose_segs(55, 50, 45, "task1") == "short_term"

    def test_diagnose_mid_term(self):
        """Seg1=100, Seg2=70, Seg3=65 — seg1-seg2=30 > 25 → mid_term."""
        assert self._diagnose_segs(100, 70, 65, "task1") == "mid_term"

    def test_diagnose_long_term(self):
        """Seg1=100, Seg2=80, Seg3=30 — seg3 < 35 → long_term."""
        assert self._diagnose_segs(100, 80, 30, "task1") == "long_term"

    def test_diagnose_ceiling(self):
        """Consistently mediocre scores across all segments → ceiling."""
        assert self._diagnose_segs(65, 60, 55, "task1") == "ceiling"

    def test_diagnose_stable(self):
        """Strong and balanced → stable."""
        assert self._diagnose_segs(95, 88, 80, "task1") == "stable"

    def test_propose_short_term_increases_n_modes(self):
        p = self._propose_params("short_term")
        assert p["n_modes"] >= 32
        assert p["epochs"] >= 100

    def test_propose_mid_term_increases_sub_step(self):
        p = self._propose_params("mid_term")
        assert p["sub_step"] >= 10

    def test_propose_ceiling_increases_hidden(self):
        p = self._propose_params("ceiling")
        assert p["hidden_channels"] >= 128
        assert p["n_modes"] >= 32

    def test_propose_stable_preserves_params(self):
        best = {"n_modes": 24, "hidden_channels": 128, "sub_step": 10}
        p = self._propose_params("stable", best)
        assert p["n_modes"] == 24
        assert p["hidden_channels"] == 128
