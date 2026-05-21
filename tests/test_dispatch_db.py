#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.dispatch_db — SQLite-backed experiment dispatch database."""

import json
import os
import time

import pytest

from expflow_pde.dispatch_db import DispatchDB


@pytest.fixture
def db(tmp_path: str) -> DispatchDB:
    """Create a DispatchDB with a temp path (in-memory DB would work but
    we use tmp_path to verify file creation and WAL mode)."""
    db_path = os.path.join(str(tmp_path), "test_dispatch.db")
    return DispatchDB(db_path)


# ── Phase 1.1: Connection management ──


class TestConnectionManagement:
    """_conn factory, _write_tx, _read_tx patterns."""

    def test_creates_db_file(self, tmp_path):
        db_path = os.path.join(str(tmp_path), "fresh.db")
        assert not os.path.isfile(db_path)
        db = DispatchDB(db_path)
        assert os.path.isfile(db_path)

    def test_wal_mode_enabled(self, db):
        with db._read_tx() as conn:
            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal == "wal"

    def test_foreign_keys_enabled(self, db):
        with db._read_tx() as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_write_tx_commit(self, db):
        with db._write_tx() as conn:
            conn.execute(
                "INSERT INTO experiments (id, root_id, script) "
                "VALUES ('test:1', 'test:1', 'train.py')"
            )
        # Should persist after context exits
        with db._read_tx() as conn:
            row = conn.execute("SELECT id FROM experiments WHERE id='test:1'").fetchone()
        assert row is not None

    def test_write_tx_rollback_on_error(self, db):
        try:
            with db._write_tx() as conn:
                conn.execute(
                    "INSERT INTO experiments (id, root_id, script) "
                    "VALUES ('test:2', 'test:2', 'eval.py')"
                )
                raise ValueError("something went wrong")
        except ValueError:
            pass
        # Should NOT persist
        with db._read_tx() as conn:
            row = conn.execute("SELECT id FROM experiments WHERE id='test:2'").fetchone()
        assert row is None

    def test_read_tx_no_lock(self, db):
        """Read transactions should not acquire the write lock."""
        with db._read_tx() as conn:
            row = conn.execute("SELECT 1 as val").fetchone()
        assert row["val"] == 1

    def test_multiple_dbs_isolated(self, tmp_path):
        db1 = DispatchDB(os.path.join(str(tmp_path), "a.db"))
        db2 = DispatchDB(os.path.join(str(tmp_path), "b.db"))
        e1 = db1.register_experiment(script="train.py")
        e2 = db2.register_experiment(script="eval.py")
        assert db1.query_recent(limit=1)[0]["id"] == e1["experiment_id"]
        assert db2.query_recent(limit=1)[0]["id"] == e2["experiment_id"]


# ── Phase 1.2: Schema initialization + migration ──


class TestSchema:
    """Tables are created and migration is idempotent."""

    def test_tables_exist(self, db):
        with db._read_tx() as conn:
            tables = set(
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            )
        assert "experiments" in tables
        assert "branches" in tables
        assert "artifacts" in tables
        assert "metrics" in tables
        assert "audit_log" in tables

    def test_indexes_exist(self, db):
        with db._read_tx() as conn:
            indexes = set(
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            )
        expected = {
            "idx_experiments_status", "idx_experiments_parent",
            "idx_experiments_root", "idx_experiments_created",
            "idx_experiments_ctask", "idx_branches_parent",
            "idx_branches_child", "idx_artifacts_exp",
            "idx_artifacts_type", "idx_metrics_exp",
            "idx_metrics_name", "idx_audit_exp",
        }
        for idx in expected:
            assert idx in indexes, f"Missing index: {idx}"

    def test_migration_idempotent(self, db):
        """Calling _ensure_migration twice should not error."""
        db._ensure_migration()
        db._ensure_migration()  # second call

    def test_schema_reinitialization(self, tmp_path):
        """Multiple DispatchDB() calls on same path should not error."""
        db_path = os.path.join(str(tmp_path), "shared.db")
        d1 = DispatchDB(db_path)
        d1.register_experiment(script="a.py")
        d2 = DispatchDB(db_path)
        d2.register_experiment(script="b.py")
        assert len(d2.query_recent()) == 2


# ── Phase 1.3: Experiment CRUD ──


class TestExperimentCRUD:
    """register, update_status, get, query_recent."""

    def test_register_experiment(self, db):
        exp = db.register_experiment(
            script="train_task1.py",
            args={"lr": 0.001, "epochs": 80},
            queue="gpu_queue",
            project="PDEBench",
            tags=["task1", "baseline"],
        )
        assert exp["status"] == "pending"
        assert exp["script"] == "train_task1.py"
        assert exp["queue"] == "gpu_queue"
        assert exp["project"] == "PDEBench"
        assert exp["experiment_id"].startswith("exp:snow_")
        assert exp["parent_id"] is None
        assert exp["root_id"] == exp["experiment_id"]

    def test_register_with_parent(self, db):
        parent = db.register_experiment(script="hpo.py")
        child = db.register_experiment(
            script="train.py", parent_id=parent["experiment_id"],
        )
        assert child["parent_id"] == parent["experiment_id"]
        assert child["root_id"] == parent["root_id"]
        assert child["root_id"] != child["experiment_id"]

    def test_update_status_pending_to_queued(self, db):
        exp = db.register_experiment(script="train.py")
        updated = db.update_status(exp["experiment_id"], "queued")
        assert updated is not None
        assert updated["status"] == "queued"

    def test_update_status_running_sets_started_at(self, db):
        exp = db.register_experiment(script="train.py")
        updated = db.update_status(exp["experiment_id"], "running")
        assert updated["status"] == "running"
        assert updated["started_at"] is not None

    def test_update_status_completed_sets_completed_at(self, db):
        exp = db.register_experiment(script="train.py")
        db.update_status(exp["experiment_id"], "running")
        updated = db.update_status(
            exp["experiment_id"], "completed",
            best_value=57.09,
            result_summary="HPO completed, best seg=57.09",
        )
        assert updated["status"] == "completed"
        assert updated["completed_at"] is not None
        assert updated["best_value"] == 57.09

    def test_update_status_with_clearml_task_id(self, db):
        exp = db.register_experiment(script="train.py")
        updated = db.update_status(
            exp["experiment_id"], "queued",
            clearml_task_id="cm_abc123",
        )
        assert updated["clearml_task_id"] == "cm_abc123"

    def test_get_experiment_found(self, db):
        exp = db.register_experiment(script="train.py")
        fetched = db.get_experiment(exp["experiment_id"])
        assert fetched is not None
        assert fetched["id"] == exp["experiment_id"]

    def test_get_experiment_not_found(self, db):
        fetched = db.get_experiment("nonexistent")
        assert fetched is None

    def test_query_recent_empty(self, db):
        results = db.query_recent()
        assert results == []

    def test_query_recent_ordered(self, db):
        e1 = db.register_experiment(script="a.py")
        e2 = db.register_experiment(script="b.py")
        results = db.query_recent()
        assert len(results) == 2
        assert results[0]["id"] == e2["experiment_id"]  # newest first

    def test_query_recent_filter_by_status(self, db):
        db.register_experiment(script="a.py")
        e2 = db.register_experiment(script="b.py")
        db.update_status(e2["experiment_id"], "queued")
        results = db.query_recent(status="pending")
        assert len(results) == 1
        assert results[0]["script"] == "a.py"

    def test_query_recent_limit(self, db):
        for i in range(5):
            db.register_experiment(script=f"e{i}.py")
        results = db.query_recent(limit=3)
        assert len(results) == 3


# ── Phase 1.4: Branch tracking ──


class TestBranchTracking:
    """Tree-based experiment hierarchy via parent_id + branches table."""

    def test_register_creates_branch_record(self, db):
        parent = db.register_experiment(script="hpo.py")
        child = db.register_experiment(
            script="train.py", parent_id=parent["experiment_id"],
        )
        children = db.get_children(parent["experiment_id"])
        assert len(children) == 1
        assert children[0]["id"] == child["experiment_id"]

    def test_get_children_empty(self, db):
        exp = db.register_experiment(script="alone.py")
        children = db.get_children(exp["experiment_id"])
        assert children == []

    def test_multiple_children(self, db):
        parent = db.register_experiment(script="research.py")
        c1 = db.register_experiment(script="a.py", parent_id=parent["experiment_id"])
        c2 = db.register_experiment(script="b.py", parent_id=parent["experiment_id"])
        children = db.get_children(parent["experiment_id"])
        assert len(children) == 2
        child_ids = {c["id"] for c in children}
        assert child_ids == {c1["experiment_id"], c2["experiment_id"]}

    def test_query_by_root_returns_tree(self, db):
        root = db.register_experiment(script="root.py")
        child = db.register_experiment(script="child.py", parent_id=root["experiment_id"])
        grandchild = db.register_experiment(
            script="grand.py", parent_id=child["experiment_id"],
        )
        tree = db.query_by_root(root["experiment_id"])
        assert len(tree) == 3, f"Expected 3, got {len(tree)}: {[e['id'] for e in tree]}"
        all_ids = {e["id"] for e in tree}
        assert root["experiment_id"] in all_ids
        assert child["experiment_id"] in all_ids
        assert grandchild["experiment_id"] in all_ids

    def test_get_experiment_tree_nested(self, db):
        root = db.register_experiment(script="root.py")
        child = db.register_experiment(script="child.py", parent_id=root["experiment_id"])
        db.register_experiment(script="grand.py", parent_id=child["experiment_id"])

        tree = db.get_experiment_tree(root["experiment_id"])
        assert tree["id"] == root["experiment_id"]
        assert len(tree["children"]) == 1
        assert tree["children"][0]["id"] == child["experiment_id"]
        assert len(tree["children"][0]["children"]) == 1


# ── Phase 1.5: Metrics + Artifacts + AuditLog ──


class TestMetrics:
    """Metric recording and querying."""

    def test_record_and_get_metric(self, db):
        exp = db.register_experiment(script="train.py")
        db.record_metric(exp["experiment_id"], "seg_total", 57.09, iteration=80)
        metrics = db.get_metrics(exp["experiment_id"])
        assert len(metrics) == 1
        assert metrics[0]["name"] == "seg_total"
        assert metrics[0]["value"] == 57.09
        assert metrics[0]["iteration"] == 80

    def test_record_multiple_metrics(self, db):
        exp = db.register_experiment(script="train.py")
        for epoch in range(5):
            db.record_metric(exp["experiment_id"], "loss", 0.5 / (epoch + 1),
                             iteration=epoch * 10)
        metrics = db.get_metrics(exp["experiment_id"])
        assert len(metrics) == 5

    def test_record_metric_with_group(self, db):
        exp = db.register_experiment(script="train.py")
        db.record_metric(exp["experiment_id"], "seg_total", 57.09,
                         group_name="Score")
        metrics = db.get_metrics(exp["experiment_id"])
        assert metrics[0]["group_name"] == "Score"

    def test_compare_scores(self, db):
        e1 = db.register_experiment(script="a.py", project="PDEBench")
        e2 = db.register_experiment(script="b.py", project="PDEBench")
        db.record_metric(e1["experiment_id"], "seg_total", 57.09)
        db.record_metric(e2["experiment_id"], "seg_total", 42.0)
        scores = db.compare_scores("seg_total", project="PDEBench")
        assert len(scores) == 2
        assert scores[0]["id"] == e1["experiment_id"]  # higher first (desc)
        assert scores[0]["value"] == 57.09

    def test_compare_scores_ascending(self, db):
        e1 = db.register_experiment(script="a.py")
        e2 = db.register_experiment(script="b.py")
        db.record_metric(e1["experiment_id"], "pde_mean", 18.09)
        db.record_metric(e2["experiment_id"], "pde_mean", 12.5)
        scores = db.compare_scores("pde_mean", direction="asc")
        assert scores[0]["value"] == 12.5  # lower first (asc)


class TestArtifacts:
    """Artifact recording."""

    def test_add_and_get_artifact(self, db):
        exp = db.register_experiment(script="train.py")
        art = db.add_artifact(
            exp["experiment_id"], "checkpoint", "best_model.pt",
            "/checkpoints/best.pt", checksum="abc123", size_bytes=42 * 1024 * 1024,
        )
        assert art["artifact_id"].startswith("art:snow_")

        artifacts = db.get_artifacts(exp["experiment_id"])
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "best_model.pt"
        assert artifacts[0]["checksum"] == "abc123"

    def test_get_artifacts_filter_by_type(self, db):
        exp = db.register_experiment(script="train.py")
        db.add_artifact(exp["experiment_id"], "checkpoint", "m.pt", "/m.pt")
        db.add_artifact(exp["experiment_id"], "plot", "loss.png", "/l.png")
        plots = db.get_artifacts(exp["experiment_id"], artifact_type="plot")
        assert len(plots) == 1
        assert plots[0]["name"] == "loss.png"


class TestAuditLog:
    """Audit trail is automatically created."""

    def test_submit_creates_audit_entry(self, db):
        exp = db.register_experiment(script="train.py")
        log = db.get_audit_log(experiment_id=exp["experiment_id"])
        assert len(log) >= 1
        assert log[0]["event_type"] == "submit"

    def test_status_change_creates_audit_entry(self, db):
        exp = db.register_experiment(script="train.py")
        db.update_status(exp["experiment_id"], "running")
        log = db.get_audit_log(experiment_id=exp["experiment_id"],
                               event_type="status_change")
        assert len(log) >= 1

    def test_audit_log_without_experiment(self, db):
        """event_type can be recorded without an experiment_id."""
        with db._write_tx() as conn:
            db._write_audit(conn, None, "system_event",
                            {"msg": "clearml server health check"})
        log = db.get_audit_log(event_type="system_event")
        assert len(log) == 1
        assert log[0]["experiment_id"] is None


class TestStats:
    """Database statistics."""

    def test_stats_empty_db(self, db):
        stats = db.stats()
        assert stats["total_experiments"] == 0
        assert stats["total_metrics"] == 0
        assert stats["total_audit_entries"] == 0
        assert stats["db_size_bytes"] > 0

    def test_stats_with_data(self, db):
        exp = db.register_experiment(script="train.py")
        db.record_metric(exp["experiment_id"], "loss", 0.5)
        stats = db.stats()
        assert stats["total_experiments"] == 1
        assert stats["by_status"]["pending"] == 1
        assert stats["total_metrics"] == 1


class TestArchive:
    """Archiving old experiments."""

    def test_archive_no_data(self, db):
        result = db.archive("2020-01-01")
        assert result["moved_count"] == 0

    def test_archive_completed_experiments(self, db, tmp_path):
        exp = db.register_experiment(script="old.py")
        db.update_status(exp["experiment_id"], "completed")
        db.record_metric(exp["experiment_id"], "seg", 50.0)

        archive_path = os.path.join(str(tmp_path), "archive.db")
        result = db.archive("2099-01-01", archive_path=archive_path)
        assert result["moved_count"] == 1

        # Original should be empty
        remaining = db.query_recent()
        assert len(remaining) == 0

        # Archive should have it
        archived = DispatchDB(archive_path)
        archived_exp = archived.get_experiment(exp["experiment_id"])
        assert archived_exp is not None

    def test_archive_only_completed(self, db, tmp_path):
        running = db.register_experiment(script="running.py")
        db.update_status(running["experiment_id"], "running")
        done = db.register_experiment(script="done.py")
        db.update_status(done["experiment_id"], "completed")

        archive_path = os.path.join(str(tmp_path), "archive.db")
        result = db.archive("2099-01-01", archive_path=archive_path)
        assert result["moved_count"] == 1  # only the completed one
