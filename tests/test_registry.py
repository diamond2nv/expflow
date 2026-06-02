#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.registry — DeadEndRegistry.

Tests cover:
- Register dead-end entries
- Exact hash lookup
- Fuzzy (script + axis) lookup
- List recent entries
- Count axes
- Clear entries
- Non-existent DB file handling
"""

from __future__ import annotations

import tempfile

import pytest

from expflow_pde.registry import DeadEndRegistry


@pytest.fixture
def registry():
    """Create a fresh DeadEndRegistry with temp file."""
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False,
    ) as f:
        db_path = f.name
    return DeadEndRegistry(db_path=db_path)


class TestDeadEndRegistry:
    def test_register_returns_entry_id(self, registry):
        """Register returns entry_id + approach_hash."""
        result = registry.register(
            script="train_task1.py",
            axis="learning_rate",
            reason="Validation loss diverged.",
            args={"lr": 1e-2},
            code_hash="abc123",
            metric_value=0.95,
        )
        assert "entry_id" in result
        assert "approach_hash" in result
        assert "timestamp" in result

    def test_exact_lookup_finds_entry(self, registry):
        """Exact hash lookup returns matching entries."""
        registry.register(
            script="train.py",
            axis="architecture",
            reason="Model capacity insufficient.",
        )
        result = registry.lookup(
            script="train.py",
            axis="architecture",
            exact=True,
        )
        assert len(result) == 1
        assert result[0]["script"] == "train.py"
        assert result[0]["axis"] == "architecture"

    def test_exact_lookup_diff_hash_no_match(self, registry):
        """Different args → different hash → no exact match."""
        registry.register(
            script="train.py",
            axis="lr",
            args={"lr": 1e-2},
            reason="lr too high.",
        )
        result = registry.lookup(
            script="train.py",
            axis="lr",
            args={"lr": 1e-3},  # different params
            exact=True,
        )
        assert len(result) == 0

    def test_fuzzy_lookup_finds_by_script_axis(self, registry):
        """Fuzzy mode matches any entry with same script+axis."""
        registry.register(script="train.py", axis="lr", reason="A")
        registry.register(script="train.py", axis="lr", reason="B")
        registry.register(script="eval.py", axis="lr", reason="C")

        result = registry.lookup(
            script="train.py", axis="lr", exact=False,
        )
        assert len(result) == 2
        # Most recent first
        assert result[0]["reason"] == "B"

    def test_list_recent(self, registry):
        """list_recent returns most recent entries limited."""
        for i in range(5):
            registry.register(
                script=f"train_{i}.py",
                axis="lr",
                reason=f"Failure #{i}",
            )
        recent = registry.list_recent(limit=3)
        assert len(recent) == 3
        assert recent[0]["reason"] == "Failure #4"  # most recent first

    def test_list_axes(self, registry):
        """list_axes counts per axis, sorted by count desc."""
        registry.register(script="a.py", axis="lr", reason="a")
        registry.register(script="b.py", axis="lr", reason="b")
        registry.register(script="c.py", axis="arch", reason="c")

        axes = registry.list_axes()
        assert axes["lr"] == 2
        assert axes["arch"] == 1
        # lr first (higher count)
        assert list(axes.keys())[0] == "lr"

    def test_clear_removes_all_entries(self, registry):
        """Clear removes all entries and returns count."""
        registry.register(script="a.py", axis="lr", reason="test")
        registry.register(script="b.py", axis="arch", reason="test")

        count = registry.clear()
        assert count == 2
        assert registry.list_recent() == []

    def test_lookup_non_existent_file(self):
        """Non-existent DB → empty list (no crash)."""
        r = DeadEndRegistry(db_path="/tmp/nonexistent_def.jsonl")
        assert r.lookup("a.py", "lr") == []
        assert r.list_recent() == []
        assert r.list_axes() == {}

    def test_register_duplicate_hash(self, registry):
        """Same approach hash → both entries stored (append-only)."""
        r1 = registry.register(
            script="a.py", axis="lr", args={"lr": 0.1},
            reason="First attempt",
        )
        r2 = registry.register(
            script="a.py", axis="lr", args={"lr": 0.1},
            reason="Second attempt (same params)",
        )
        assert r1["entry_id"] == r2["entry_id"]
        result = registry.lookup("a.py", "lr", args={"lr": 0.1}, exact=True)
        assert len(result) == 2
