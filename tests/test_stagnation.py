"""Tests for StagnationDetector."""

from __future__ import annotations

import json

from expflow_pde.dispatch_db import DispatchDB
from expflow_pde.stagnation import StagnationDetector


def _mkfailed(db, parent_id) -> str:
    child = db.register_experiment(script="test.py", parent_id=parent_id)
    db.update_status(
        child["experiment_id"],
        "failed",
        result_summary=json.dumps({"status": "failed"}),
    )
    return child["experiment_id"]


def _mkscore(db, parent_id, score: float) -> str:
    child = db.register_experiment(script="test.py", parent_id=parent_id)
    db.update_status(
        child["experiment_id"],
        "completed",
        result_summary=json.dumps({"status": "completed", "score": score}),
    )
    return child["experiment_id"]


class TestStagnationDetectorBasics:
    def test_no_experiment_returns_not_stagnant(self, tmp_path):
        detector = StagnationDetector(str(tmp_path / "empty.db"))
        result = detector.check_iteration("nonexistent")
        assert result["stagnant"] is False
        assert result["recommendation"] == "continue"

    def test_single_completed_not_stagnant(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = DispatchDB(db_path)
        root = db.register_experiment(script="test.py")
        _mkscore(db, root["experiment_id"], 56.0)
        detector = StagnationDetector(db_path)
        result = detector.check_iteration(root["experiment_id"])
        assert result["stagnant"] is False
        assert result["details"]["consecutive_fail_count"] == 0


class TestStagnationConsecutiveFail:
    def test_three_fails_triggers(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = DispatchDB(db_path)
        root = db.register_experiment(script="test.py")
        for _ in range(3):
            _mkfailed(db, root["experiment_id"])
        detector = StagnationDetector(db_path)
        result = detector.check_iteration(root["experiment_id"])
        assert result["stagnant"] is True
        assert StagnationDetector.MODE_CONCLUSIVE_FAIL in result["patterns"]

    def test_two_fails_not_enough(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = DispatchDB(db_path)
        root = db.register_experiment(script="test.py")
        _mkfailed(db, root["experiment_id"])
        _mkfailed(db, root["experiment_id"])
        _mkscore(db, root["experiment_id"], 55.0)
        # score is last → trailing fail count = 0
        detector = StagnationDetector(db_path)
        result = detector.check_iteration(root["experiment_id"])
        assert result["stagnant"] is False

    def test_one_fail_then_success_resets(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = DispatchDB(db_path)
        root = db.register_experiment(script="test.py")
        _mkfailed(db, root["experiment_id"])
        _mkscore(db, root["experiment_id"], 55.0)
        _mkfailed(db, root["experiment_id"])
        _mkfailed(db, root["experiment_id"])
        # Only 2 trailing fails since a success sits between them
        detector = StagnationDetector(db_path)
        result = detector.check_iteration(root["experiment_id"])
        assert result["stagnant"] is False
        assert result["details"]["consecutive_fail_count"] == 2


class TestStagnationScorePlateau:
    def test_flat_scores_triggers(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = DispatchDB(db_path)
        root = db.register_experiment(script="test.py")
        for _ in range(4):
            _mkscore(db, root["experiment_id"], 142.0)
        detector = StagnationDetector(db_path)
        result = detector.check_iteration(
            root["experiment_id"],
            score_plateau_epsilon=1.0,
            score_plateau_window=3,
        )
        assert result["stagnant"] is True
        assert StagnationDetector.MODE_SCORE_PLATEAU in result["patterns"]

    def test_rising_scores_no_plateau(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = DispatchDB(db_path)
        root = db.register_experiment(script="test.py")
        for s in [132.0, 138.0, 142.0, 145.0]:
            _mkscore(db, root["experiment_id"], s)
        detector = StagnationDetector(db_path)
        result = detector.check_iteration(
            root["experiment_id"],
            score_plateau_epsilon=1.0,
            score_plateau_window=3,
        )
        assert result["stagnant"] is False


class TestStagnationHypothesisLock:
    def test_no_hypothesis_file_returns_not_locked(self, tmp_path):
        detector = StagnationDetector(str(tmp_path / "empty.db"))
        assert detector._hypothesis_self_lock_count(k=3) == 0


class TestStagnationRecommendation:
    def test_consecutive_fail_is_pause(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = DispatchDB(db_path)
        root = db.register_experiment(script="test.py")
        for _ in range(4):
            _mkfailed(db, root["experiment_id"])
        detector = StagnationDetector(db_path)
        result = detector.check_iteration(root["experiment_id"])
        assert result["recommendation"] == "pause"

    def test_explain_works(self):
        text = StagnationDetector._explain(
            ["consecutive_fail"],
            {"consecutive_fail_count": 3},
        )
        assert "3" in text
