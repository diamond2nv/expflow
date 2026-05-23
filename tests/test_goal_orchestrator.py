"""Tests for GoalOrchestrator progress persistence and two-phase recovery."""

from __future__ import annotations

import os

from expflow_pde.goal_orchestrator import (
    add_learned_failure,
    clear,
    load,
    resolve_pipeline_state,
    save,
)


class TestGoalOrchestrator:
    def test_empty_state_on_no_file(self, monkeypatch):
        tmp = "/tmp/no_such_dir_xyz"
        monkeypatch.setattr(
            "expflow_pde.goal_orchestrator._PROGRESS_PATH",
            os.path.join(tmp, "progress_state.json"),
        )
        state = load()
        assert state["root_experiment_id"] == ""
        assert state["consecutive_failures"] == 0
        assert state["current_phase"] == "idle"

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        state = {
            "root_experiment_id": "exp:snow_123",
            "best_score": 142.5,
            "consecutive_failures": 0,
            "last_pipeline_id": "pipe_abc",
            "current_phase": "completed",
        }
        saved = save(state)
        assert os.path.isfile(p)
        loaded = load()
        assert loaded["root_experiment_id"] == "exp:snow_123"
        assert loaded["best_score"] == 142.5
        assert loaded["current_phase"] == "completed"

    def test_clear_removes_file(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        save({"root_experiment_id": "test", "current_phase": "idle"})
        assert os.path.isfile(p)
        clear()
        assert not os.path.isfile(p)

    def test_corrupt_file_returns_empty(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        with open(p, "w") as f:
            f.write("{invalid json")
        state = load()
        assert state["root_experiment_id"] == ""

    def test_default_keys_all_present(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        save({"root_experiment_id": "abc", "current_phase": "completed"})
        loaded = load()
        assert loaded["consecutive_failures"] == 0
        assert loaded["current_phase"] == "completed"
        assert loaded["best_score"] == 0.0
        assert loaded["root_experiment_id"] == "abc"

    def test_add_learned_failure(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        # Add a failure
        add_learned_failure({
            "type": "oom",
            "params": {"n_modes": 24, "batch_size": 4},
            "reason": "OOM on 5090 with 24 modes",
            "experiment_id": "exp123",
        })
        state = load()
        assert len(state["learned_failures"]) == 1
        assert state["learned_failures"][0]["type"] == "oom"

    def test_add_learned_failure_dedup(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        add_learned_failure({
            "type": "oom",
            "params": {"n_modes": 24, "batch_size": 4},
            "reason": "original",
            "experiment_id": "exp123",
        })
        add_learned_failure({
            "type": "oom",
            "params": {"n_modes": 24, "batch_size": 4},  # identical keys
            "reason": "updated",
            "experiment_id": "exp456",
        })
        state = load()
        assert len(state["learned_failures"]) == 1  # dedup by (type, param_keys)
        assert state["learned_failures"][0]["reason"] == "updated"

    def test_add_learned_failure_max_20(self, tmp_path, monkeypatch):
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        for i in range(25):
            add_learned_failure({
                "type": f"type_{i}",
                "params": {"val": i},
                "reason": f"failure {i}",
                "experiment_id": None,
            })
        state = load()
        assert len(state["learned_failures"]) <= 20

    def test_resolve_pipeline_state_no_task_ids(self):
        """No last_pipeline_id -> submit_new."""
        state = load()
        state["last_pipeline_id"] = ""
        state["last_train_task_id"] = ""
        result = resolve_pipeline_state(state)
        assert result["action"] == "submit_new"

    def test_resolve_pipeline_state_clearml_unreachable(self, monkeypatch):
        """When clearml cannot be contacted, falls through to submit_new."""
        monkeypatch.setattr(
            "expflow_pde.goal_orchestrator._get_task_status",
            lambda tid: None,
        )
        state = {
            "last_pipeline_id": "pipe_xyz",
            "last_train_task_id": "task_xyz",
            "last_eval_task_id": "task_eval",
            "current_phase": "waiting",
        }
        result = resolve_pipeline_state(state)
        # clearml unreachable, no status -> submit_new to avoid stall
        assert result["action"] == "submit_new"

    def test_verify_state_rejects_bad_keys(self):
        """Non-English keys should raise ValueError."""
        import pytest

        from expflow_pde.goal_orchestrator import _verify_state
        with pytest.raises(ValueError, match="unexpected state key"):
            _verify_state({"建议": {"n_modes": 24}})

    def test_verify_state_accepts_english_keys(self):
        """English snake_case keys should pass without error."""
        from expflow_pde.goal_orchestrator import _verify_state
        _verify_state({"best_score": 142, "current_phase": "completed"})
        # no exception = pass

    def test_load_strips_unknown_keys(self, tmp_path, monkeypatch):
        """Persisted unknown keys should be stripped by load()."""
        import json
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)
        # Write a state with a non-compliant key
        with open(p, "w") as f:
            json.dump({"best_score": 132, "建议": {"n_modes": 24}, "current_phase": "completed"}, f)
        loaded = load()
        assert loaded["best_score"] == 132.0
        assert "建议" not in loaded  # stripped
        assert loaded["current_phase"] == "completed"

    def test_session_recovery_flows_through_goal_orchestrator(self, monkeypatch, tmp_path):
        """End-to-end: save pipeline state -> simulate crash -> recover."""
        p = str(tmp_path / "progress.json")
        monkeypatch.setattr("expflow_pde.goal_orchestrator._PROGRESS_PATH", p)

        # Simulate Hermes submitting a pipeline
        save({
            "best_score": 0.0,
            "last_pipeline_id": "pipe_001",
            "last_train_task_id": "train_001",
            "last_eval_task_id": "eval_001",
            "current_phase": "waiting",
            "consecutive_failures": 0,
        })

        # Simulate crash: clearml unreachable
        monkeypatch.setattr(
            "expflow_pde.goal_orchestrator._get_task_status",
            lambda tid: None,
        )
        state = load()
        assert state["current_phase"] == "waiting"
        result = resolve_pipeline_state(state)
        # clearml unreachable -> submit_new (not stalled)
        assert result["action"] in ("submit_new",)
