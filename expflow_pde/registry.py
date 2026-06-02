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
(script_name, args_signature, axis, bucket). For parameterized axes
like ``architecture:n_modes`` or ``learning_rate:value``, the
``bucket`` field groups related values into a range so that
``n_modes=8`` and ``n_modes=24`` are recognized as the same axis
family (``architecture:n_modes``) — even though they have different
exact hashes. The ``bucket_range`` field allows interval overlap
matching for quantitative axes.

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
    bucket: str | None = None,
) -> str:
    """Compute a deterministic hash for an experimental approach.

    Args:
        script: Script name (e.g. 'train_task1.py').
        args: Dict of hyperparameters.
        axis: Search axis (e.g. 'learning_rate', 'architecture').
        bucket: Optional sub-axis bucket (e.g. 'n_modes', 'lr_value').
            When provided, the hash includes bucket but NOT the args
            value — so all experiments on the same bucket match.

    Returns:
        SHA256 hex digest (first 16 chars).
    """
    key: dict[str, Any] = {"script": script, "axis": axis}
    if bucket:
        key["bucket"] = bucket
    else:
        key["args"] = args or {}
    raw = json.dumps(key, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _value_to_metric_bucket(value: float) -> str:
    """Convert a numeric value to a human-readable bucket name.

    This is a heuristic for common PDE/ML hyperparameter axes.
    Falls back to a logarithmic bucket for arbitrary numbers.

    Examples:
        1e-5  -> '1e-5'
        0.001 -> '0.001'
        0.01  -> '0.01'
        0.1   -> '0.1'
        8     -> '8'
        16    -> '16'
        32    -> '32'
    """
    # Keep common values as-is (readable)
    common = {1e-5, 1e-4, 1e-3, 0.001, 0.01, 0.1, 1.0, 8, 12, 16, 20, 24, 32, 64, 128, 256}
    # Use repr to avoid floating-point artifacts like 0.0010000000001
    s = repr(value)
    for c in sorted(common, key=lambda x: -abs(x)):
        if abs(value - c) / max(abs(c), 1e-10) < 0.01:
            s = repr(c)
            break
    return s


class DeadEndRegistry:
    """JSONL-backed dead-end registry for cross-session failure tracking.

    Stores experiment approaches that have been tried and failed, so that
    the same direction is not explored twice. Each entry is identified by
    an approach_hash (script + args + axis + bucket) and records the
    failure reason, code version, and timestamp.

    Two-tier lookup:
    - **Exact hash match**: requires identical script+args+axis(+bucket).
    - **Axis + bucket fuzzy**: matches on (script, axis, bucket) across
      all entries in that bucket. This is the primary mode for PDE
      experiments where the same axis family (e.g. ``architecture:n_modes``)
      spans multiple numeric values.
    - **Wildcard**: matches on (script, axis) only, ignoring bucket,
      for top-level axis exhaustion checks.

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
        bucket: str | None = None,
        bucket_low: float | None = None,
        bucket_high: float | None = None,
    ) -> dict[str, Any]:
        """Register a failed experimental approach.

        Args:
            script: Script that was run.
            axis: Search axis that failed (e.g. 'learning_rate', 'architecture').
            reason: Human-readable failure reason.
            args: Hyperparameters used.
            code_hash: Git commit hash of code version.
            metric_value: Final metric value (if any).
            bucket: Sub-axis bucket (e.g. 'n_modes', 'lr_value').
                When provided, dead-end matching will group all entries
                with the same (script, axis, bucket) regardless of
                exact args values.
            bucket_low: Low end of the numeric range for this entry
                (for interval-overlap matching).
            bucket_high: High end of the numeric range for this entry.

        Returns:
            Dict with entry_id, approach_hash, timestamp.
        """
        entry_id = _approach_hash(script, args, axis, bucket)
        entry: dict[str, Any] = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": script,
            "axis": axis,
            "reason": reason,
            "args": args or {},
            "code_hash": code_hash,
            "metric_value": metric_value,
            "bucket": bucket,
        }
        if bucket_low is not None:
            entry["bucket_low"] = bucket_low
        if bucket_high is not None:
            entry["bucket_high"] = bucket_high

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
        bucket: str | None = None,
        bucket_value: float | None = None,
    ) -> list[dict[str, Any]]:
        """Look up dead-end entries matching an approach.

        Three matching modes:
        1. **Exact** (exact=True): requires perfect match of
           script+args+axis(+bucket if provided).
        2. **Bucket fuzzy** (exact=False, bucket=...): matches on
           (script, axis, bucket) across all entries in that bucket.
           Ignores exact args values — so ``n_modes=8`` and
           ``n_modes=24`` both match the same bucket dead-ends.
        3. **Wildcard** (exact=False, bucket=None): matches on
           (script, axis) only — for top-level axis exhaustion checks.

        Interval overlap: when entries have ``bucket_low``/``bucket_high``
        stored, and ``bucket_value`` is provided, only entries whose
        range overlaps [bucket_value, bucket_value] are returned.
        This allows ``n_modes=16`` to match a dead-end registered for
        ``n_modes∈[0,12]`` but NOT one registered for ``n_modes∈[24,32]``.

        Args:
            script: Script name.
            axis: Search axis.
            args: Hyperparameters (for exact hash match only).
            exact: If True, requires perfect hash match.
            bucket: Sub-axis bucket for fuzzy matching.
            bucket_value: Numeric value for interval-overlap filtering.

        Returns:
            List of matching dead-end entries (most recent first).
        """
        if not os.path.exists(self.db_path):
            return []

        if exact:
            target_hash = _approach_hash(script, args, axis, bucket)
            entries = self._query(lambda e: e.get("entry_id") == target_hash)
        elif bucket is not None:
            # Bucket-level fuzzy: matches same axis+bucket, then
            # optionally filters by range overlap
            entries = self._query(
                lambda e: (
                    e.get("script") == script
                    and e.get("axis") == axis
                    and e.get("bucket") == bucket
                ),
            )
            # Apply interval-overlap filter if bucket_value is provided
            if bucket_value is not None:
                entries = [
                    e
                    for e in entries
                    if self._range_overlaps(
                        e,
                        bucket_value,
                    )
                ]
        else:
            # Wildcard: script + axis only
            entries = self._query(
                lambda e: e.get("script") == script and e.get("axis") == axis,
            )

        return entries

    def _range_overlaps(
        self,
        entry: dict[str, Any],
        value: float,
    ) -> bool:
        """Check if a numeric value falls within an entry's range.

        If the entry has no range bounds, it matches everything
        (backward compat: older entries without bucket_low/high).
        """
        low = entry.get("bucket_low")
        high = entry.get("bucket_high")
        if low is None and high is None:
            return True  # No range constraint → match
        low = low if low is not None else float("-inf")
        high = high if high is not None else float("inf")
        return low <= value <= high

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

    def list_axes(self, detail: bool = False) -> dict[str, int]:
        """Count dead ends per search axis.

        Args:
            detail: If True, returns {'axis:bucket': count}
                instead of just {'axis': count}.

        Returns:
            Dict of {axis_name: failure_count}.
        """
        counts: dict[str, int] = {}
        if not os.path.exists(self.db_path):
            return counts
        for entry in self._load_all():
            if detail and entry.get("bucket"):
                key = f"{entry['axis']}:{entry['bucket']}"
            else:
                key = entry.get("axis", "unknown")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def list_buckets(self, axis: str) -> list[dict[str, Any]]:
        """List all buckets with failure counts for a given axis.

        Args:
            axis: Search axis to filter by.

        Returns:
            List of {bucket, count, low, high} dicts.
        """
        buckets: dict[str, dict[str, Any]] = {}
        for entry in self._load_all():
            if entry.get("axis") != axis:
                continue
            bucket = entry.get("bucket") or "default"
            if bucket not in buckets:
                buckets[bucket] = {
                    "bucket": bucket,
                    "count": 0,
                    "low": entry.get("bucket_low"),
                    "high": entry.get("bucket_high"),
                }
            buckets[bucket]["count"] += 1
            # Range union
            bl = entry.get("bucket_low")
            bh = entry.get("bucket_high")
            if bl is not None:
                curr = buckets[bucket]["low"]
                buckets[bucket]["low"] = min(curr, bl) if curr is not None else bl
            if bh is not None:
                curr = buckets[bucket]["high"]
                buckets[bucket]["high"] = max(curr, bh) if curr is not None else bh
        return sorted(buckets.values(), key=lambda x: -x["count"])

    def clear(self) -> int:
        """Clear all dead-end entries (for testing).

        Returns:
            Number of entries removed.
        """
        entries = self._load_all()
        count = len(entries)
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
