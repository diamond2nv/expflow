#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.hypothesis — HypothesisRegistry CRUD.

All tests use temp EXPFLOW_HOME to avoid polluting user data.
"""
from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def expflow_home():
    """Isolate hypothesis YAML to a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.environ.get("EXPFLOW_HOME", "")
        os.environ["EXPFLOW_HOME"] = tmpdir
        yield
        if old:
            os.environ["EXPFLOW_HOME"] = old
        else:
            del os.environ["EXPFLOW_HOME"]


class TestHypothesisRegistry:
    """HypothesisRegistry — record, list, close, show."""

    def test_record_returns_valid_id(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        h = HypothesisRegistry.record("Test hypothesis", "Rationale")
        assert h["id"].startswith("hyp_")
        assert h["status"] == "proposed"

    def test_list_all(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        HypothesisRegistry.record("H1", "r1")
        HypothesisRegistry.record("H2", "r2")
        hyps = HypothesisRegistry.list()
        assert len(hyps) == 2

    def test_list_filter_status(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        h = HypothesisRegistry.record("Test", "Rationale")
        HypothesisRegistry.close(h["id"], "rejected", "No improvement")
        proposed = HypothesisRegistry.list(status="proposed")
        rejected = HypothesisRegistry.list(status="rejected")
        assert len(proposed) == 0
        assert len(rejected) == 1

    def test_close_accepted(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        h = HypothesisRegistry.record("Test", "Rationale", suggested_params={"lr": 0.01})
        result = HypothesisRegistry.close(h["id"], "accepted", "Score +5%", evidence_task_id="tid_1")
        assert result is not None
        assert result["status"] == "accepted"
        assert result["evidence_task_id"] == "tid_1"

    def test_invalid_status_raises(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        h = HypothesisRegistry.record("Test", "Rationale")
        with pytest.raises(ValueError):
            HypothesisRegistry.close(h["id"], "invalid", "whatever")

    def test_show_returns_detail(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        h = HypothesisRegistry.record("Detail test", "Because", origin_task_id="tid_abc")
        shown = HypothesisRegistry.show(h["id"])
        assert shown is not None
        assert shown["hypothesis"] == "Detail test"
        assert shown["origin_task_id"] == "tid_abc"

    def test_show_unknown_returns_none(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        assert HypothesisRegistry.show("nonexistent") is None

    def test_get_rejected_directions(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        h1 = HypothesisRegistry.record("Try A", "r1")
        h2 = HypothesisRegistry.record("Try B", "r2")
        HypothesisRegistry.close(h1["id"], "rejected", "Bad")
        HypothesisRegistry.close(h2["id"], "accepted", "Good")
        rejected = HypothesisRegistry.get_rejected_directions()
        assert len(rejected) == 1
        assert rejected[0]["id"] == h1["id"]

    def test_get_open_hypotheses(self):
        from expflow_pde.hypothesis import HypothesisRegistry

        h = HypothesisRegistry.record("Open", "Rationale")
        opened = HypothesisRegistry.get_open_hypotheses()
        assert len(opened) == 1
        HypothesisRegistry.close(h["id"], "rejected", "done")
        assert len(HypothesisRegistry.get_open_hypotheses()) == 0

    def test_yaml_persistence(self):
        """Hypotheses survive across loads (YAML file written to EXPFLOW_HOME)."""
        from expflow_pde.hypothesis import HypothesisRegistry

        HypothesisRegistry.record("Persist test", "Rationale")
        # Force a new load — should read from YAML
        hyps = HypothesisRegistry.list()
        assert len(hyps) == 1
