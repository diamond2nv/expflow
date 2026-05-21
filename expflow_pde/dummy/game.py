#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow dummy game — Zero-dependency simulated experiment loop.

A fully self-contained, no-torch, no-clearml "dummy" experiment game that
exercises the entire diagnose → suggest → submit → (fail) → repair → iterate
loop using synthetic seg scores.

Each "step" creates a real experiment record in dispatch_db, applies the
suggested fix, generates new seg scores (with noise + ceiling), and
optionally injects a failure pattern (git_not_found, module_not_found,
cuda_oom, etc.) for repair testing.

Usage:
    from expflow_pde.dummy.game import DummyExperimentGame

    game = DummyExperimentGame(seed=42)
    game.start()           # Create root experiment in dispatch_db
    game.step(params={})   # Run one diagnose → suggest → submit cycle
    game.status()          # Show current state
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any

from expflow_pde.dispatch_db import DispatchDB


# ── Baseline seg scores per task ──

_BASELINE: dict[str, dict[str, float]] = {
    "task1": {"seg1": 55.0, "seg2": 30.0, "seg3": 20.0},
    "task2": {"seg1": 40.0, "seg2": 20.0, "seg3": 10.0},
    "task3": {"seg1": 15.0, "seg2": 8.0, "seg3": 4.0},
}

# ── Ceiling per task ──

_CEILING: dict[str, dict[str, float]] = {
    "task1": {"seg1": 70.0, "seg2": 60.0, "seg3": 45.0},
    "task2": {"seg1": 55.0, "seg2": 45.0, "seg3": 30.0},
    "task3": {"seg1": 25.0, "seg2": 18.0, "seg3": 12.0},
}

# ── Effect of each hyperparam change on seg scores (additive) ──

_FIX_EFFECTS: dict[str, dict[str, float]] = {
    "n_modes":       {"seg1": 0.0, "seg2": 3.0, "seg3": 8.0},   # +modes helps high freq
    "width":         {"seg1": 0.0, "seg2": 2.0, "seg3": 4.0},
    "num_sub_steps": {"seg1": 2.0, "seg2": 3.0, "seg3": 5.0},
    "lr":            {"seg1": 5.0, "seg2": -2.0, "seg3": -1.0},  # high LR helps short, hurts long
    "stability_lambda": {"seg1": -1.0, "seg2": 6.0, "seg3": 0.0},  # stability helps mid
    "weight_decay":  {"seg1": 0.0, "seg2": 1.0, "seg3": 3.0},
    "epochs":        {"seg1": 1.0, "seg2": 1.0, "seg3": 2.0},
}

# ── Failure injection patterns ──

_FAILURE_TEMPLATES: dict[str, tuple[str, int, str]] = {
    "git_not_found": (
        "Cloning into 'PDEBench'...\n"
        "ERROR: Repository not found.\n"
        "fatal: Could not read from remote repository.\n",
        128,
        "L0",
    ),
    "module_not_found": (
        "Traceback (most recent call last):\n"
        '  File "/opt/train.py", line 3, in <module>\n'
        "    import torch\n"
        "ModuleNotFoundError: No module named 'torch'\n",
        1,
        "L0",
    ),
    "cuda_oom": (
        "Traceback (most recent call last):\n"
        '  File "/opt/train.py", line 42, in forward\n'
        "    x = self.conv(x)\n"
        "torch.cuda.OutOfMemoryError: CUDA out of memory. "
        "Tried to allocate 2.45 GiB. GPU 0 has 1.95 GiB total capacity.\n",
        1,
        "L1",
    ),
    "data_not_found": (
        "Traceback (most recent call last):\n"
        '  File "/opt/eval.py", line 15, in load_data\n'
        "    data = h5py.File('dataset.hdf5', 'r')\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'dataset.hdf5'\n",
        1,
        "L1",
    ),
    "unknown_error": (
        "Some random system error occurred during model initialization.\n"
        "Internal error code: 0xDEADBEEF\n"
        "Contact support with error ID: EXP-2026-XYZ-999\n",
        1,
        "L2",
    ),
}


class DummyExperimentGame:
    """Simulated experiment iteration game.

    Each step runs the full loop:
      1. Read previous experiment's results
      2. Optionally apply a failure (inject pattern)
      3. Generate new seg scores from the suggested params
      4. Record in dispatch_db
      5. Return result for Hermes to diagnose/repair/iterate
    """

    def __init__(
        self,
        task_id: str = "task1",
        seed: int = 42,
        db_path: str | None = None,
    ):
        self.task_id = task_id
        self._rng = random.Random(seed)
        self._db = DispatchDB(db_path)
        self._current_seg = dict(_BASELINE[task_id])
        self._root_id: str | None = None
        self._step_count = 0
        self._last_exp_id: str | None = None

    # ── Game lifecycle ──

    def start(self) -> dict[str, Any]:
        """Start a new game: create root experiment, return its ID."""
        self._step_count = 0
        self._current_seg = dict(_BASELINE[self.task_id])
        self._root_id = None

        exp = self._db.register_experiment(
            script="dummy_game",
            args={"task_id": self.task_id, "game": "start"},
            queue="dummy",
            project="DummyGame",
        )
        self._root_id = exp["experiment_id"]
        self._last_exp_id = self._root_id

        self._db.update_status(self._root_id, "completed", result_summary=json.dumps({
            "seg1": self._current_seg["seg1"],
            "seg2": self._current_seg["seg2"],
            "seg3": self._current_seg["seg3"],
            "total": self._total(),
            "task_id": self.task_id,
        }))

        return {
            "game": "started",
            "experiment_id": self._root_id,
            "seg": dict(self._current_seg),
            "task_id": self.task_id,
        }

    def step(
        self,
        suggested_params: dict[str, Any] | None = None,
        strategy: str | None = None,
        inject_failure: str | None = None,
        inject_task_log: str | None = None,
    ) -> dict[str, Any]:
        """Run one game step: apply fix, generate segs, record in DB.

        Args:
            suggested_params: Dict of param -> value from Hermes suggest.
            strategy: Strategy name (e.g. 'sub_step', 'ceiling_fix').
            inject_failure: Failure pattern name (git_not_found, etc.)
            inject_task_log: Override task log (for repair testing).

        Returns:
            Dict with experiment_id, seg, total, status, task_log, inject_info.
        """
        self._step_count += 1

        # Apply suggested params to seg scores
        if suggested_params:
            for key, new_val in suggested_params.items():
                # Detect delta from the effect lookup
                for effect_key, effect in _FIX_EFFECTS.items():
                    if effect_key in key:
                        self._current_seg["seg1"] += effect["seg1"]
                        self._current_seg["seg2"] += effect["seg2"]
                        self._current_seg["seg3"] += effect["seg3"]

        # Add noise (±2, gaussian)
        noise = self._rng.gauss(0, 2.0)
        self._current_seg["seg1"] = round(self._current_seg["seg1"] + noise, 1)
        self._current_seg["seg2"] = round(self._current_seg["seg2"] + self._rng.gauss(0, 2.0), 1)
        self._current_seg["seg3"] = round(self._current_seg["seg3"] + self._rng.gauss(0, 2.0), 1)

        # Apply ceiling
        ceil = _CEILING[self.task_id]
        self._current_seg["seg1"] = min(self._current_seg["seg1"], ceil["seg1"])
        self._current_seg["seg2"] = min(self._current_seg["seg2"], ceil["seg2"])
        self._current_seg["seg3"] = min(self._current_seg["seg3"], ceil["seg3"])
        self._current_seg["seg1"] = max(self._current_seg["seg1"], 0.0)
        self._current_seg["seg2"] = max(self._current_seg["seg2"], 0.0)
        self._current_seg["seg3"] = max(self._current_seg["seg3"], 0.0)

        total = self._total()

        # Determine if we inject a failure
        failure = self._pick_failure(inject_failure)
        if failure:
            task_log, exit_code, expected_level = failure
            status = "failed"
            result_summary = None
        else:
            task_log = inject_task_log or self._build_log(total)
            exit_code = 0
            status = "completed"
            result_summary = json.dumps({
                "seg1": self._current_seg["seg1"],
                "seg2": self._current_seg["seg2"],
                "seg3": self._current_seg["seg3"],
                "total": total,
                "task_id": self.task_id,
            })

        # Create child experiment
        parent_id = self._last_exp_id
        exp = self._db.register_experiment(
            script="dummy_game",
            args={
                "suggested_params": suggested_params or {},
                "task_id": self.task_id,
                "step": self._step_count,
            },
            queue="dummy",
            project="DummyGame",
            parent_id=parent_id,
        )
        self._last_exp_id = exp["experiment_id"]

        # Link branch
        with self._db._write_tx() as conn:
            self._db._write_audit(
                conn,
                parent_id,
                "branch",
                {"child_id": exp["experiment_id"], "strategy": strategy or "auto"},
            )

        self._db.update_status(
            exp["experiment_id"],
            status,
            error_message=task_log if failure else None,
            result_summary=result_summary,
        )

        inject_info = None
        expected_level = "none"
        if failure:
            _, _, expected_level = failure
            inject_info = {
                "failure": inject_failure or self._rng.choice(list(_FAILURE_TEMPLATES.keys())),
                "exit_code": exit_code,
                "expected_repair_level": expected_level,
            }

        return {
            "experiment_id": exp["experiment_id"],
            "seg": dict(self._current_seg),
            "total": total,
            "status": status,
            "task_log": task_log,
            "step": self._step_count,
            "inject_failure": inject_info,
        }

    def status(self) -> dict[str, Any]:
        """Return current game state."""
        if self._last_exp_id:
            exp = self._db.get_experiment(self._last_exp_id)
        else:
            exp = None

        return {
            "root_id": self._root_id,
            "step": self._step_count,
            "current_seg": dict(self._current_seg),
            "total": self._total(),
            "task_id": self.task_id,
            "last_experiment": exp,
            "steps_left_to_ceiling": self._steps_to_ceiling(),
        }

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset game state to baseline."""
        if seed is not None:
            self._rng = random.Random(seed)
        self._current_seg = dict(_BASELINE[self.task_id])
        self._root_id = None
        self._step_count = 0
        self._last_exp_id = None
        return {"status": "reset", "task_id": self.task_id}

    # ── Internal helpers ──

    def _total(self) -> float:
        return round(self._current_seg["seg1"] +
                     self._current_seg["seg2"] +
                     self._current_seg["seg3"], 1)

    def _steps_to_ceiling(self) -> int:
        """Estimate remaining steps before all segs hit ceiling."""
        ceil = _CEILING[self.task_id]
        remaining = max(
            ceil["seg1"] - self._current_seg["seg1"],
            ceil["seg2"] - self._current_seg["seg2"],
            ceil["seg3"] - self._current_seg["seg3"],
            0.0,
        )
        # Rough: each step gains ~5 points on average
        return int(math.ceil(remaining / 5.0))

    def _pick_failure(
        self, explicit: str | None = None
    ) -> tuple[str, int, str] | None:
        """Pick a failure pattern. Returns None = success.

        Explicit = force a specific failure. 'none' = force success.
        Otherwise ~30% probability of random failure.
        """
        if explicit == "none":
            return None
        if explicit:
            pattern = _FAILURE_TEMPLATES.get(explicit)
            if pattern:
                return pattern
            return None

        # ~30% random failure
        if self._rng.random() < 0.3:
            key = self._rng.choice(list(_FAILURE_TEMPLATES.keys()))
            return _FAILURE_TEMPLATES[key]
        return None

    def _build_log(self, total: float) -> str:
        return (
            f"Dummy experiment completed.\n"
            f"Seg total: {total:.1f}\n"
            f"Task: {self.task_id}\n"
        )
