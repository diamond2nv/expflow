#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow dispatch DB — SQLite-backed experiment dispatch database.

Architecture (based on hfpapers-crawler PaperStore pattern):
- Each CRUD operation creates a new connection (SQLite overhead ~0.1ms)
- Write operations: threading.Lock() serialization + BEGIN IMMEDIATE
- Read operations: no lock (WAL mode supports concurrent reads)
- row_factory = sqlite3.Row → dict(row) for JSON-serializable output
- Schema migrations: pragma_table_info check + idempotent ALTER TABLE

Usage:
    from expflow_pde.dispatch_db import DispatchDB

    db = DispatchDB()
    exp = db.register_experiment(script="train.py", args={"lr": 0.001})
    db.update_status(exp["id"], "queued")
    recent = db.query_recent(limit=10)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


# ─── Snowflake ID generator (yitter drift algorithm, port from hfpapers) ───

_SNOWFLAKE_LOCK = threading.Lock()
_SNOWFLAKE_GEN: _SnowflakeM1 | None = None
_SNOWFLAKE_WORKER_ID: int = 1  # worker_id=1 reserved for expflow
_SNOWFLAKE_BASE_TIME: int = 1728000000000  # 2024-10-04, aligned with hfpapers


class _IdGeneratorOptions:
    def __init__(self, worker_id: int = 0):
        self.method: int = 1
        self.base_time: int = _SNOWFLAKE_BASE_TIME
        self.worker_id: int = worker_id
        self.worker_id_bit_length: int = 6
        self.seq_bit_length: int = 6
        self.max_seq_number: int = 0
        self.min_seq_number: int = 5
        self.top_over_cost_count: int = 2000


class _SnowflakeM1:
    """Snowflake drift algorithm M1 — thread-safe, time-rollback-tolerant."""

    def __init__(self, options: _IdGeneratorOptions):
        self.base_time = int(options.base_time)
        self.worker_id_bit_length = int(options.worker_id_bit_length)
        self.worker_id = int(options.worker_id)
        self.seq_bit_length = int(options.seq_bit_length)
        self.max_seq_number = int(options.max_seq_number)
        if options.max_seq_number <= 0:
            self.max_seq_number = (1 << self.seq_bit_length) - 1
        self.min_seq_number = int(options.min_seq_number)
        self.top_over_cost_count = int(options.top_over_cost_count)

        self._timestamp_shift = self.worker_id_bit_length + self.seq_bit_length
        self._worker_id_shift = self.seq_bit_length

        self._last_gen_time = 0
        self._sequence = 0

    def _current_time(self) -> int:
        return int(time.time() * 1000)

    def _to_twepoch(self, ts: int) -> int:
        return ts - self.base_time

    def _calc_drift_back(self, ts: int) -> int:
        return ts - self._current_time()

    def _till_next_millis(self, last_gen_time: int) -> int:
        ts = self._current_time()
        while ts <= last_gen_time:
            ts = self._current_time()
        return ts

    def next_id(self) -> int:
        with _SNOWFLAKE_LOCK:
            current_time = self._current_time()
            if current_time < self._last_gen_time:
                self._sequence = self.min_seq_number
                self._last_gen_time = current_time
                twepoch = self._to_twepoch(current_time)
                return (
                    twepoch << self._timestamp_shift
                    | self.worker_id << self._worker_id_shift
                    | self._sequence
                )

            if current_time == self._last_gen_time:
                self._sequence += 1
                if self._sequence > self.max_seq_number:
                    self._sequence = self.min_seq_number
                    current_time = self._till_next_millis(self._last_gen_time)
            else:
                self._sequence = self.min_seq_number

            self._last_gen_time = current_time
            twepoch = self._to_twepoch(current_time)
            return (
                twepoch << self._timestamp_shift
                | self.worker_id << self._worker_id_shift
                | self._sequence
            )


def _get_snowflake() -> _SnowflakeM1:
    global _SNOWFLAKE_GEN
    if _SNOWFLAKE_GEN is None:
        opts = _IdGeneratorOptions(worker_id=_SNOWFLAKE_WORKER_ID)
        _SNOWFLAKE_GEN = _SnowflakeM1(opts)
    return _SNOWFLAKE_GEN


def _snowflake_id() -> int:
    """Generate a 64-bit snowflake ID."""
    return _get_snowflake().next_id()


def _exp_id() -> str:
    """Generate a human-readable experiment ID: exp:snow_<id>"""
    return f"exp:snow_{_snowflake_id()}"


# ─── Default DB path ───

_DEFAULT_DB_PATH: str | None = None


def _get_db_path() -> str:
    global _DEFAULT_DB_PATH
    if _DEFAULT_DB_PATH is None:
        expflow_dir = os.path.expanduser("~/.expflow")
        os.makedirs(expflow_dir, exist_ok=True)
        _DEFAULT_DB_PATH = os.path.join(expflow_dir, "dispatch.db")
    return _DEFAULT_DB_PATH


# ─── Validity bits for status/type CHECK constraints ───

_VALID_STATUSES = frozenset({
    "pending", "queued", "running", "completed", "failed", "cancelled", "pruned",
})

_VALID_FSM_STATES = frozenset({
    "ideation", "hpo_tuning", "training", "evaluation",
    "submission", "review", "archived",
})

_VALID_ARTIFACT_TYPES = frozenset({
    "checkpoint", "plot", "dataset", "log", "submission", "report",
})

_VALID_EVENT_TYPES = frozenset({
    "submit", "status_change", "hpo_trial", "repair",
    "deploy", "error", "cancel", "callback",
})


# ─── DispatchDB ───


class DispatchDB:
    """SQLite-backed experiment dispatch database.

    Thread-safe (WAL mode + serialized writes). Zero external dependencies.
    All public methods return plain dicts — JSON-serializable.
    """

    def __init__(self, db_path: str | None = None):
        self.path = db_path or _get_db_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()
        self._ensure_migration()

    # ── Connection management ──

    def _conn(self) -> sqlite3.Connection:
        """Create a new connection (paper_store factory pattern)."""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _write_tx(self):
        """Write transaction: lock + BEGIN IMMEDIATE + auto commit/rollback."""
        with self._lock, self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @contextmanager
    def _read_tx(self):
        """Read transaction: no lock, read-only."""
        with self._conn() as conn:
            yield conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        """Convert sqlite3.Row to plain dict (or return None)."""
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def _rows_to_list(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        """Convert list of sqlite3.Row to list of dicts."""
        return [dict(r) for r in rows]

    # ── Schema ──

    def _init_schema(self):
        """Initialize tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id              TEXT PRIMARY KEY,
                    parent_id       TEXT REFERENCES experiments(id),
                    root_id         TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    fsm_state       TEXT DEFAULT 'ideation',
                    script          TEXT NOT NULL,
                    args_json       TEXT NOT NULL DEFAULT '{}',
                    search_space_json TEXT,
                    tags_json       TEXT DEFAULT '[]',
                    queue           TEXT DEFAULT 'default',
                    project         TEXT DEFAULT 'expflow',
                    branch          TEXT,
                    commit_hash     TEXT,
                    clearml_task_id TEXT,
                    best_value      REAL,
                    best_params_json TEXT,
                    result_summary  TEXT,
                    error_message   TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    started_at      TEXT,
                    completed_at    TEXT,
                    source          TEXT DEFAULT 'hermes',
                    version         TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_experiments_status
                    ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_experiments_parent
                    ON experiments(parent_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_root
                    ON experiments(root_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_created
                    ON experiments(created_at);
                CREATE INDEX IF NOT EXISTS idx_experiments_ctask
                    ON experiments(clearml_task_id);

                CREATE TABLE IF NOT EXISTS branches (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_exp_id   TEXT NOT NULL REFERENCES experiments(id),
                    child_exp_id    TEXT NOT NULL REFERENCES experiments(id),
                    strategy        TEXT,
                    condition_json  TEXT,
                    depth           INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(child_exp_id)
                );

                CREATE INDEX IF NOT EXISTS idx_branches_parent
                    ON branches(parent_exp_id);
                CREATE INDEX IF NOT EXISTS idx_branches_child
                    ON branches(child_exp_id);

                CREATE TABLE IF NOT EXISTS artifacts (
                    id              TEXT PRIMARY KEY,
                    experiment_id   TEXT NOT NULL REFERENCES experiments(id),
                    type            TEXT NOT NULL,
                    name            TEXT NOT NULL,
                    path            TEXT NOT NULL,
                    checksum        TEXT,
                    size_bytes      INTEGER,
                    metadata_json   TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_exp
                    ON artifacts(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_type
                    ON artifacts(type);

                CREATE TABLE IF NOT EXISTS metrics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT NOT NULL REFERENCES experiments(id),
                    name            TEXT NOT NULL,
                    value           REAL NOT NULL,
                    iteration       INTEGER,
                    group_name      TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(experiment_id, name, iteration)
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_exp
                    ON metrics(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_name
                    ON metrics(name);

                CREATE TABLE IF NOT EXISTS audit_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id   TEXT,
                    event_type      TEXT NOT NULL,
                    detail_json     TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_audit_exp
                    ON audit_log(experiment_id);
            """)

    def _ensure_migration(self):
        """Idempotent schema migration (hfpapers pattern)."""
        with self._conn() as conn:
            has_version_col = conn.execute(
                "SELECT COUNT(*) FROM pragma_table_info('experiments') "
                "WHERE name = 'version'"
            ).fetchone()[0]
            if has_version_col:
                return

            migrations = [
                "ALTER TABLE experiments ADD COLUMN version TEXT",
                "ALTER TABLE experiments ADD COLUMN source TEXT DEFAULT 'hermes'",
            ]
            for stmt in migrations:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            conn.commit()

    # ── Experiment CRUD ──

    def register_experiment(
        self,
        script: str,
        args: dict[str, Any] | None = None,
        parent_id: str | None = None,
        queue: str = "default",
        project: str = "expflow",
        tags: list[str] | None = None,
        source: str = "hermes",
    ) -> dict[str, Any]:
        """Register a new experiment. Returns the experiment record."""
        exp_id = _exp_id()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        root_id = exp_id
        if parent_id:
            root_id = self._resolve_root_id(parent_id) or parent_id
        args_json = json.dumps(args or {})
        tags_json = json.dumps(tags or [])

        with self._write_tx() as conn:
            conn.execute(
                """INSERT INTO experiments
                   (id, parent_id, root_id, script, args_json, tags_json,
                    queue, project, source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (exp_id, parent_id, root_id, script, args_json, tags_json,
                 queue, project, source, now, now),
            )

            if parent_id:
                depth_row = conn.execute(
                    "SELECT depth FROM branches WHERE child_exp_id=?",
                    (parent_id,),
                ).fetchone()
                depth = (depth_row["depth"] + 1) if depth_row else 1
                conn.execute(
                    "INSERT INTO branches (parent_exp_id, child_exp_id, depth) "
                    "VALUES (?, ?, ?)",
                    (parent_id, exp_id, depth),
                )

            self._write_audit(conn, exp_id, "submit", {
                "script": script,
                "queue": queue,
                "project": project,
            })

        return {
            "experiment_id": exp_id,
            "root_id": root_id,
            "parent_id": parent_id,
            "status": "pending",
            "script": script,
            "queue": queue,
            "project": project,
            "created_at": now,
        }

    def update_status(
        self,
        experiment_id: str,
        status: str,
        **extra: Any,
    ) -> dict[str, Any] | None:
        """Update experiment status and optional fields.

        Args:
            experiment_id: The experiment ID.
            status: New status (pending/queued/running/completed/failed/cancelled/pruned).
            **extra: Additional fields to update (e.g. clearml_task_id=..., best_value=...).

        Returns:
            Updated record, or None if not found.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        set_clauses = ["status=?", "updated_at=?"]
        params: list[Any] = [status, now]

        if status == "running":
            set_clauses.append("started_at=?")
            params.append(now)
        elif status in ("completed", "failed", "cancelled", "pruned"):
            set_clauses.append("completed_at=?")
            params.append(now)

        for key, value in extra.items():
            if key in ("clearml_task_id", "error_message", "branch",
                       "commit_hash", "result_summary", "best_params_json"):
                set_clauses.append(f"{key}=?")
                params.append(str(value) if value is not None else None)
            elif key in ("best_value",):
                set_clauses.append(f"{key}=?")
                params.append(float(value) if value is not None else None)

        params.append(experiment_id)

        with self._write_tx() as conn:
            conn.execute(
                f"UPDATE experiments SET {', '.join(set_clauses)} WHERE id=?",
                params,
            )
            self._write_audit(conn, experiment_id, "status_change", {
                "new_status": status,
                "extra": {k: v for k, v in extra.items() if v is not None},
            })

        return self.get_experiment(experiment_id)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get a single experiment by ID."""
        with self._read_tx() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            return self._row_to_dict(row)

    def query_recent(
        self,
        limit: int = 20,
        status: str | None = None,
        project: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query recent experiments, most recent first."""
        conditions: list[str] = []
        params: list[Any] = []

        if status:
            conditions.append("status=?")
            params.append(status)
        if project:
            conditions.append("project=?")
            params.append(project)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._read_tx() as conn:
            rows = conn.execute(
                f"SELECT * FROM experiments {where} "
                "ORDER BY created_at DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            return self._rows_to_list(rows)

    def query_by_root(self, root_id: str) -> list[dict[str, Any]]:
        """Get all experiments in a tree (by root_id)."""
        with self._read_tx() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE root_id=? "
                "ORDER BY created_at ASC",
                (root_id,),
            ).fetchall()
            return self._rows_to_list(rows)

    # ── Branches (tree tracking) ──

    def get_children(
        self, experiment_id: str, include_subtree: bool = False,
    ) -> list[dict[str, Any]]:
        """Get child experiments. Optionally include full subtree."""
        with self._read_tx() as conn:
            if include_subtree:
                rows = conn.execute(
                    "SELECT e.* FROM experiments e "
                    "JOIN branches b ON e.id = b.child_exp_id "
                    "WHERE b.parent_exp_id=? "
                    "ORDER BY b.depth, e.created_at",
                    (experiment_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT e.* FROM experiments e "
                    "JOIN branches b ON e.id = b.child_exp_id "
                    "WHERE b.parent_exp_id=? AND b.depth=1",
                    (experiment_id,),
                ).fetchall()
            return self._rows_to_list(rows)

    def get_experiment_tree(self, root_id: str) -> dict[str, Any]:
        """Build a nested tree structure from a root experiment."""
        experiments = self.query_by_root(root_id)
        if not experiments:
            return {}

        # Build adjacency map
        children_map: dict[str, list[dict]] = {}
        exp_map: dict[str, dict] = {}
        for exp in experiments:
            exp_id = exp["id"]
            exp_map[exp_id] = dict(exp)
            exp_map[exp_id]["children"] = []
            pid = exp.get("parent_id")
            if pid:
                children_map.setdefault(pid, []).append(exp_id)

        # Attach children
        for parent_id, child_ids in children_map.items():
            if parent_id in exp_map:
                for cid in child_ids:
                    exp_map[parent_id]["children"].append(exp_map[cid])

        root = exp_map.get(root_id)
        return root or {}

    def _resolve_root_id(self, experiment_id: str) -> str | None:
        """Find the root experiment ID by traversing parent chain."""
        with self._read_tx() as conn:
            row = conn.execute(
                "SELECT root_id FROM experiments WHERE id=?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                return None
            return row["root_id"] or experiment_id

    # ── Metrics ──

    def record_metric(
        self,
        experiment_id: str,
        name: str,
        value: float,
        iteration: int | None = None,
        group_name: str | None = None,
    ) -> dict[str, Any]:
        """Record a metric value. Returns the record."""
        with self._write_tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metrics "
                "(experiment_id, name, value, iteration, group_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (experiment_id, name, value, iteration, group_name),
            )
        return {"experiment_id": experiment_id, "name": name, "value": value}

    def get_metrics(
        self,
        experiment_id: str,
        name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get metrics for an experiment."""
        if name:
            rows = self._execute_read(
                "SELECT * FROM metrics WHERE experiment_id=? AND name=? "
                "ORDER BY iteration ASC NULLS LAST LIMIT ?",
                (experiment_id, name, limit),
            )
        else:
            rows = self._execute_read(
                "SELECT * FROM metrics WHERE experiment_id=? "
                "ORDER BY name, iteration ASC NULLS LAST LIMIT ?",
                (experiment_id, limit),
            )
        return rows

    def compare_scores(
        self,
        metric_name: str,
        project: str | None = None,
        sort_by: str = "value",
        direction: str = "desc",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Rank experiments by a metric value."""
        conditions = ["m.name=?"]
        params: list[Any] = [metric_name]

        if project:
            conditions.append("e.project=?")
            params.append(project)

        order = "DESC" if direction == "desc" else "ASC"
        with self._read_tx() as conn:
            rows = conn.execute(
                f"SELECT e.id, e.status, e.created_at, "
                f"m.value, m.iteration, m.group_name "
                f"FROM metrics m "
                f"JOIN experiments e ON e.id = m.experiment_id "
                f"WHERE {' AND '.join(conditions)} "
                f"ORDER BY m.{sort_by} {order} LIMIT ?",
                [*params, limit],
            ).fetchall()
            return self._rows_to_list(rows)

    # ── Artifacts ──

    def add_artifact(
        self,
        experiment_id: str,
        artifact_type: str,
        name: str,
        path: str,
        checksum: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an experiment artifact."""
        artifact_id = f"art:snow_{_snowflake_id()}"
        metadata_json = json.dumps(metadata or {})
        with self._write_tx() as conn:
            conn.execute(
                """INSERT INTO artifacts
                   (id, experiment_id, type, name, path, checksum,
                    size_bytes, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (artifact_id, experiment_id, artifact_type, name, path,
                 checksum, size_bytes, metadata_json),
            )
        return {
            "artifact_id": artifact_id,
            "experiment_id": experiment_id,
            "type": artifact_type,
            "name": name,
        }

    def get_artifacts(
        self, experiment_id: str, artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get artifacts for an experiment."""
        with self._read_tx() as conn:
            if artifact_type:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE experiment_id=? AND type=? "
                    "ORDER BY created_at DESC",
                    (experiment_id, artifact_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE experiment_id=? "
                    "ORDER BY created_at DESC",
                    (experiment_id,),
                ).fetchall()
            return self._rows_to_list(rows)

    # ── Audit log ──

    def _write_audit(
        self,
        conn: sqlite3.Connection,
        experiment_id: str | None,
        event_type: str,
        detail: dict[str, Any] | None = None,
    ):
        """Write an audit log entry (internal, called within write_tx)."""
        conn.execute(
            "INSERT INTO audit_log (experiment_id, event_type, detail_json, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (experiment_id, event_type, json.dumps(detail or {})),
        )

    def get_audit_log(
        self,
        experiment_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit log."""
        conditions: list[str] = []
        params: list[Any] = []

        if experiment_id:
            conditions.append("experiment_id=?")
            params.append(experiment_id)
        if event_type:
            conditions.append("event_type=?")
            params.append(event_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._read_tx() as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_log {where} "
                "ORDER BY created_at DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            return self._rows_to_list(rows)

    # ── Internal helpers ──

    def _execute_read(
        self, sql: str, params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        """Execute a read query and return results."""
        with self._read_tx() as conn:
            return self._rows_to_list(conn.execute(sql, params).fetchall())

    # ─── Stats ───

    def stats(self) -> dict[str, Any]:
        """Return database statistics."""
        with self._read_tx() as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            by_status = {
                r[0]: r[1] for r in conn.execute(
                    "SELECT status, COUNT(*) FROM experiments GROUP BY status"
                ).fetchall()
            }
            metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
            audit_count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

            # DB file size
            try:
                size_bytes = os.path.getsize(self.path)
            except OSError:
                size_bytes = 0

        return {
            "total_experiments": total,
            "by_status": by_status,
            "total_metrics": metric_count,
            "total_audit_entries": audit_count,
            "db_size_bytes": size_bytes,
            "db_path": self.path,
        }

    # ─── Archive ───

    def archive(
        self,
        before_date: str,
        archive_path: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Atomically move completed/failed experiments before a date to an archive DB.

        Uses two-phase commit:
          1. INSERT INTO archive (within archive DB write_tx)
          2. DELETE FROM source (within source DB write_tx)

        If step 1 fails, step 2 never runs.
        Archive path includes a unix timestamp to prevent overwrites.

        A checkpoint entry is written to audit_log before and after the move.

        Args:
            before_date: ISO date string (e.g. '2025-06-01').
            archive_path: Optional path to archive DB. Auto-generated if None.
            dry_run: Count experiments to archive without moving.

        Returns:
            Dict with archive_path, moved_count, status.
        """
        if archive_path is None:
            archive_dir = os.path.join(os.path.dirname(self.path), "archive")
            os.makedirs(archive_dir, exist_ok=True)
            ts = int(time.time())
            archive_path = os.path.join(
                archive_dir, f"pre-{before_date}-{ts}.db"
            )

        # Find experiments to archive
        with self._read_tx() as conn:
            rows = conn.execute(
                "SELECT id FROM experiments "
                "WHERE status IN ('completed', 'failed', 'cancelled', 'pruned') "
                "AND completed_at < ?",
                (before_date,),
            ).fetchall()
            exp_ids = [r[0] for r in rows]

        if not exp_ids:
            return {"archive_path": archive_path, "moved_count": 0, "status": "no_op"}

        if dry_run:
            return {
                "archive_path": archive_path,
                "moved_count": len(exp_ids),
                "status": "dry_run",
                "experiment_ids": exp_ids,
            }

        placeholders = ",".join("?" for _ in exp_ids)

        # Write audit checkpoint BEFORE move
        self._write_audit_direct(
            None,
            "archive",
            {
                "action": "started",
                "archive_path": archive_path,
                "count": len(exp_ids),
                "before_date": before_date,
            },
        )

        # Two-phase commit: copy to archive (phase 1) then delete from source (phase 2)
        try:
            # Phase 1: insert into archive DB
            archive_db = DispatchDB(archive_path)
            with archive_db._write_tx() as dst_conn:
                for table in ("experiments", "branches", "artifacts", "metrics", "audit_log"):
                    # Determine the relevant ID column for this table
                    if table == "experiments":
                        id_col = "id"
                    elif table == "branches":
                        id_col = "parent_exp_id"
                    elif table in ("artifacts", "metrics", "audit_log"):
                        id_col = "experiment_id"
                    else:
                        continue

                    with self._read_tx() as src_conn:
                        rows = src_conn.execute(
                            f"SELECT * FROM {table} "
                            f"WHERE {id_col} IN ({placeholders}) "
                            f"ORDER BY created_at",
                            exp_ids,
                        ).fetchall()

                    if not rows:
                        continue

                    # Get column names from source
                    cur = src_conn.execute(f"SELECT * FROM {table} LIMIT 0")
                    columns = [d[0] for d in cur.description]
                    col_list = ", ".join(columns)
                    place = ", ".join("?" for _ in columns)

                    for row in rows:
                        dst_conn.execute(
                            f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({place})",
                            [row[c] for c in columns],
                        )
        except Exception as exc:
            self._write_audit_direct(
                None, "archive",
                {"action": "failed_phase1", "error": str(exc)},
            )
            raise

        try:
            # Phase 2: delete from source
            with self._write_tx() as src_conn:
                for table in ("metrics", "artifacts", "branches", "audit_log"):
                    if table in ("metrics", "artifacts"):
                        id_col = "experiment_id"
                    elif table == "branches":
                        id_col = "parent_exp_id"
                    else:
                        id_col = "experiment_id"
                    src_conn.execute(
                        f"DELETE FROM {table} WHERE {id_col} IN ({placeholders})",
                        exp_ids,
                    )
                src_conn.execute(
                    "DELETE FROM experiments WHERE id IN ({})".format(placeholders),
                    exp_ids,
                )
        except Exception as exc:
            # Phase 2 failed but phase 1 succeeded = broken state.
            # Log the error — manual recovery needed.
            self._write_audit_direct(
                None, "archive",
                {
                    "action": "failed_phase2",
                    "error": str(exc),
                    "archive_path": archive_path,
                    "recovery": (
                        f"Data is safe in archive ({archive_path}) but source DB "
                        "may have duplicate or dangling references."
                    ),
                },
            )
            raise

        # Write audit checkpoint AFTER successful move
        self._write_audit_direct(
            None, "archive",
            {
                "action": "committed",
                "archive_path": archive_path,
                "count": len(exp_ids),
                "before_date": before_date,
            },
        )

        return {
            "archive_path": archive_path,
            "moved_count": len(exp_ids),
            "status": "committed",
        }

    def _write_audit_direct(
        self,
        experiment_id: str | None,
        event_type: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Write audit log entry WITHOUT an existing write_tx (uses own connection).

        Used by archive() for checkpoints outside a write_tx context.
        """
        with self._write_tx() as conn:
            conn.execute(
                "INSERT INTO audit_log (experiment_id, event_type, detail_json, created_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (experiment_id, event_type, json.dumps(detail or {})),
            )
