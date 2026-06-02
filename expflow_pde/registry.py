#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow dead-end registry — Cross-session failure knowledge base.

Inspired by AutoScientists' dead-end registry (arXiv:2605.28655),
this module records failed experimental approaches so they are not
repeated across sessions. Each entry stores the approach signature,
the axis of failure, and a human-readable reason.

Storage is an append-only JSONL file at ~/.expflow/dead_ends.jsonl.
This is intentionally separate from dispatch.db to allow independent
lifecycle (dead ends persist even if dispatch is reset).

Design: A dead end is identified by an approach_hash computed from
(script_name, args_signature, axis). Lookup before experiment launch
can warn "this approach has N previous failures".

Reference:
    AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific
    Experimentation. Gao, Fang, Zitnik. arXiv:2605.28655, 2026.
    Dead-end registry concept: Shared State component D_k, cross-team readable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("expflow")

DEFAULT_DB_PATH = os.path.expanduser("~/.expflow/dead_ends.jsonl")


def _approach_hash(
    script: str,
    args: dict[str, Any] | None,
    axis: str,
) -> str:
    """Compute a deterministic hash for an experimental approach.

    Args:
        script: Script name (e.g. 'train_task1.py').
        args: Dict of hyperparameters.
        axis: Search axis (e.g. 'learning_rate', 'architecture').

    Returns:
        SHA256 hex digest (first 16 chars).
    """
    raw = json.dumps({"script": script, "args": args or {}, "axis": axis}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DeadEndRegistry:
    """SQLite-backed dead-end registry for cross-session failure tracking.

    Stores experiment approaches that have been tried and failed, so that
    the same direction is not explored twice. Each entry is identified by
    an approach_hash (script + args + axis) and records the failure reason,
    code version, and timestamp.

    The registry is queried before launching any new experiment. If the
    approach_hash matches a recorded dead end, the pipeline warns and
    suggests alternative axes.

    Args:
        db_path: Path to JSONL database file.
            Default: ~/.expflow/dead_ends.jsonl
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def register(
        self,
        script: str,
        axis: str,
        reason: str,
        args: dict[str, Any] | None = None,
        code_hash: str | None = None,
        metric_value: float | None = None,
    ) -> dict[str, Any]:
        """Register a failed experimental approach.

        Args:
            script: Script that was run.
            axis: Search axis that failed (e.g. 'learning_rate', 'architecture').
            reason: Human-readable failure reason.
            args: Hyperparameters used.
            code_hash: Git commit hash of code version.
            metric_value: Final metric value (if any).

        Returns:
            Dict with entry_id, approach_hash, timestamp.
        """
        entry_id = _approach_hash(script, args, axis)
        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": script,
            "axis": axis,
            "reason": reason,
            "args": args or {},
            "code_hash": code_hash,
            "metric_value": metric_value,
        }

        with open(self.db_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info("Registered dead end: %s — %s (%s)", script, reason, axis)
        return {
            "entry_id": entry_id,
            "approach_hash": entry_id,
            "timestamp": entry["timestamp"],
        }

    def lookup(
        self,
        script: str,
        axis: str,
        args: dict[str, Any] | None = None,
        exact: bool = True,
    ) -> list[dict[str, Any]]:
        """Look up dead-end entries matching an approach.

        Args:
            script: Script name.
            axis: Search axis.
            args: Hyperparameters (for exact hash match).
            exact: If True, requires perfect hash match (script+args+axis).
                   If False, matches on any combination of script/axis.

        Returns:
            List of matching dead-end entries (most recent first).
        """
        if not os.path.exists(self.db_path):
            return []

        # For exact match, compute hash
        if exact:
            target_hash = _approach_hash(script, args, axis)
            return self._query(lambda e: e.get("entry_id") == target_hash)

        # For fuzzy match, check script + axis separately
        return self._query(
            lambda e: e.get("script") == script and e.get("axis") == axis,
        )

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """List most recent dead-end entries.

        Args:
            limit: Max entries to return.

        Returns:
            List of dead-end dicts.
        """
        entries = self._load_all()
        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return entries[:limit]

    def list_axes(self) -> dict[str, int]:
        """Count dead ends per search axis.

        Returns:
            Dict of {axis_name: failure_count}.
        """
        counts: dict[str, int] = {}
        if not os.path.exists(self.db_path):
            return counts
        for entry in self._load_all():
            axis = entry.get("axis", "unknown")
            counts[axis] = counts.get(axis, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def clear(self) -> int:
        """Clear all dead-end entries (for testing).

        Returns:
            Number of entries removed.
        """
        entries = self._load_all()
        count = len(entries)
        # Rewrite empty
        with open(self.db_path, "w") as f:
            f.write("")
        return count

    # ── Internal ──

    def _load_all(self) -> list[dict[str, Any]]:
        """Load all entries from JSONL."""
        if not os.path.exists(self.db_path):
            return []
        entries: list[dict[str, Any]] = []
        with open(self.db_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _query(self, predicate) -> list[dict[str, Any]]:
        """Filter entries by predicate, sorted most recent first."""
        entries = self._load_all()
        matched = [e for e in entries if predicate(e)]
        matched.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return matched
