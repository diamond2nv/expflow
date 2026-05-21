#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.repair — RepairStage three-level auto-repair."""

import pytest

from expflow_pde.repair import RepairStage


class TestRepairStageL0:
    """L0 rule engine matches and returns suggestions."""

    def test_l0_git_clone_failure(self):
        stage = RepairStage()
        log = "fatal: Could not read from remote repository."
        result = stage.run(log, 128)
        assert result["level"] == "L0"
        assert not result["fixed"]  # needs user action on remote node
        assert "git" in result["action"].lower()

    def test_l0_module_not_found(self):
        stage = RepairStage()
        log = "ModuleNotFoundError: No module named 'torch'"
        result = stage.run(log, 1)
        assert result["level"] == "L0"
        assert "system_site_packages" in result["action"].lower()

    def test_l0_pip_conflict(self):
        stage = RepairStage()
        log = "pip install impossible: package conflict"
        result = stage.run(log, 1)
        assert result["level"] == "L0"
        assert "packages=[]" in result["action"]

    def test_l0_no_match_returns_none(self):
        stage = RepairStage()
        log = "Something that doesn't match any pattern"
        result = stage.run(log, 0)  # exit 0 = no repair needed
        assert result["level"] == "none"
        assert not result["fixed"]

    def test_l0_success_exit_no_match(self):
        stage = RepairStage()
        log = "Training complete. Seg: 57.09"
        result = stage.run(log, 0)
        assert result["level"] == "none"
        assert not result["fixed"]


class TestRepairStageL1:
    """L1 traceback extraction and error localization."""

    def test_l1_extracts_traceback_info(self):
        stage = RepairStage()
        log = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/train.py", line 42, in train_step\n'
            "    loss = loss_fn(output, target)\n"
            '  File "/home/user/train.py", line 80, in loss_fn\n'
            "    diff_norm = torch.norm(x_flat - y_flat, p=2, dim=1)\n"
            "RuntimeError: The size of tensor a (32) must match the size of tensor b (16)\n"
        )
        result = stage.run(log, 1, enable_reflection=False)
        assert result["level"] == "L1"
        assert result["action"] is not None
        # Check history entries were recorded
        assert len(result["history"]) >= 1

    def test_l1_no_traceback(self):
        stage = RepairStage()
        log = "Killed\n"
        result = stage.run(log, 137)
        assert result["level"] == "L1"  # Falls to L1 even without Traceback
        assert result["fixed"] is False

    def test_l1_empty_log(self):
        stage = RepairStage()
        result = stage.run("", 1)
        assert result["level"] == "L1"
        assert not result["fixed"]


class TestRepairStageL2:
    """L2 reflection context preparation."""

    def test_l2_prepares_context(self):
        stage = RepairStage(max_l1_attempts=1)
        log = (
            "Some non-standard error: model failed after epoch 50\n"
            "loss became NaN starting from epoch 40\n"
            "gradient explosion detected\n"
        )
        result = stage.run(log, 1, enable_reflection=True)
        assert result["level"] == "L2"
        assert result["fixed"] is False
        assert result["action"] is not None

    def test_l2_disabled_by_default(self):
        stage = RepairStage()
        log = "Unknown error with no traceback"
        result = stage.run(log, 1, enable_reflection=False)
        # Without reflection, L1 runs but returns fixed=False,
        # and since L2 is disabled, level is "L1"
        assert result["level"] in ("L1", "none")

    def test_l2_subagent_prompt_in_history(self):
        stage = RepairStage(max_l1_attempts=1)
        log = "CUDA out of memory. Tried to allocate 2.00 GiB"
        result = stage.run(log, 1, enable_reflection=True)
        assert result["level"] in ("L2", "L1")
        # History should contain the attempt
        assert len(result["history"]) >= 1


class TestRepairStageProperties:
    """RepairStage state tracking."""

    def test_history_tracks_attempts(self):
        stage = RepairStage()
        log = "ModuleNotFoundError: No module named 'h5py'"
        result = stage.run(log, 1)
        assert len(stage.history) >= 1

    def test_fixed_property(self):
        stage = RepairStage()
        # A non-matching case
        stage.run("ok output", 0)
        assert not stage.fixed

    def test_to_json_serializable(self):
        stage = RepairStage(experiment_id="exp_abc123")
        stage.run("ModuleNotFoundError: No module named 'x'", 1)
        json_str = stage.to_json()
        import json
        parsed = json.loads(json_str)
        assert parsed["experiment_id"] == "exp_abc123"
        assert "repair_history" in parsed


class TestRepairStageEdgeCases:
    """Edge cases and resilience."""

    def test_long_error_log(self):
        """Very long logs should not break L1 extraction."""
        log = "\n".join([f"Line {i}" for i in range(1000)])
        log += "\nTraceback (most recent call last):\n  File \"test.py\", line 1, in <module>\n    raise ValueError(\"test\")\nValueError: test"
        stage = RepairStage(max_l1_attempts=1)
        result = stage.run(log, 1)
        assert result["level"] == "L1"

    def test_unicode_in_log(self):
        """Non-ASCII characters should not break parsing."""
        log = "Error: can't decode byte 0xff in position 0\nTraceback:\n  ValueError: 测试中文信息"
        stage = RepairStage()
        result = stage.run(log, 1)
        assert result is not None

    def test_no_newlines(self):
        """Single-line log should not crash."""
        stage = RepairStage()
        result = stage.run("Single line error: kaboom", 1)
        assert result is not None

    def test_multiple_repair_stages(self):
        """Multiple sequential repair calls should not leak state."""
        stage = RepairStage()
        r1 = stage.run("ModuleNotFoundError: No module named 'a'", 1)
        r2 = stage.run("OK", 0)
        r3 = stage.run("fatal: Could not read", 128)
        assert r1["level"] == "L0"
        assert r2["level"] == "none"
        assert r3["level"] == "L0"
