"""Tests for GoalOrchestrator progress persistence."""

from __future__ import annotations

import os

from expflow_pde.goal_orchestrator import GoalOrchestrator


class TestGoalOrchestrator:
    def test_empty_state_on_no_file(self, monkeypatch):
        tmp = "/tmp/no_such_dir_xyz"
        monkeypatch.setattr(
            "expflow_pde.goal_orchestrator._PROGRESS_PATH",
            os.path.join(tmp, "progress_state.json"),
        )
        state = GoalOrchestrator.load()
        assert state["root_experiment_id"] is None
        assert state["consecutive_failures"] == 0
        assert state["current_phase"] == "init"

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        state = {
            "root_experiment_id": "exp:snow_123",
            "best_score": 142.5,
            "consecutive_failures": 0,
            "last_pipeline_id": "pipe_abc",
            "current_phase": "wait",
            "suggestion_params": {"n_modes": 24},
            "stagnation_status": None,
        }
        saved_path = GoalOrchestrator.save(state)
        assert os.path.isfile(saved_path)
        loaded = GoalOrchestrator.load()
        assert loaded["root_experiment_id"] == "exp:snow_123"
        assert loaded["best_score"] == 142.5
        assert loaded["current_phase"] == "wait"

    def test_clear_removes_file(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        GoalOrchestrator.save({"test": True})
        assert os.path.isfile(p)
        GoalOrchestrator.clear()
        assert not os.path.isfile(p)

    def test_corrupt_file_returns_empty(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        with open(p, "w") as f:
            f.write("{invalid json")
        state = GoalOrchestrator.load()
        assert state["root_experiment_id"] is None

    def test_default_keys_all_present(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        GoalOrchestrator.save({"root_experiment_id": "abc"})
        loaded = GoalOrchestrator.load()
        # All default keys should still be present
        assert loaded["consecutive_failures"] == 0
        assert loaded["current_phase"] == "init"
        assert loaded["best_score"] is None
        assert loaded["root_experiment_id"] == "abc"
