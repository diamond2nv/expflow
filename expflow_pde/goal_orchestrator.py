#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal orchestrator — Hermes /goal-compatible progress persistence.

Saves and restores experiment iteration state so Hermes can resume
across sessions (e.g. overnight runs that exceed context window).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from typing import Any

logger = logging.getLogger("expflow_pde.goal_orchestrator")

_PROGRESS_PATH = os.path.expanduser("~/.expflow/progress_state.json")


def _write_with_flock(state: dict[str, Any], path: str) -> None:
    """Write JSON state with fcntl.flock for concurrency safety.

    Falls back to plain write on non-Linux platforms where fcntl
    is not available or applicable.
    """
    try:
        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (OSError, IOError):
        # Non-Linux platform or unsupported fs: fall back to plain write
        with open(path, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)


class GoalOrchestrator:
    """Structured progress persistence for Hermes /goal mode.

    GoalOrchestrator stores:
      - root_experiment_id (for DispatchDB tree tracking)
      - best_score seen so far
      - consecutive_failures counter
      - last_diagnosis (skip re-analyzing cost)
      - suggestion params for repeatability
      - current_phase: str — "diagnose" | "submit" | "wait" | "evaluate" | "done"
    """

    @staticmethod
    def save(state: dict[str, Any]) -> str:
        """Persist progress state to ~/.expflow/progress_state.json.

        Uses fcntl.flock for concurrency safety. On non-Linux platforms
        (where fcntl may not apply), the lock is silently skipped.
        Always overwrites (only one active goal session).
        """
        os.makedirs(os.path.dirname(_PROGRESS_PATH), exist_ok=True)
        _write_with_flock(state, _PROGRESS_PATH)
        logger.debug("Progress saved to %s", _PROGRESS_PATH)
        return _PROGRESS_PATH

    @staticmethod
    def load() -> dict[str, Any]:
        """Load saved progress, or return empty state dict.

        Returns:
            dict with keys:
                root_experiment_id: str|None — root of the iteration tree
                best_score: float|None — best score so far
                consecutive_failures: int
                last_pipeline_id: str|None
                current_phase: str
                suggestion_params: dict
                stagnation_status: dict|None — last check_iteration result
        """
        if not os.path.isfile(_PROGRESS_PATH):
            return GoalOrchestrator._empty()
        try:
            with open(_PROGRESS_PATH) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt progress state, starting fresh")
            return GoalOrchestrator._empty()
        # Ensure all expected keys exist
        state = GoalOrchestrator._empty()
        state.update(data)
        return state

    @staticmethod
    def clear() -> None:
        """Wipe saved progress."""
        if os.path.isfile(_PROGRESS_PATH):
            os.remove(_PROGRESS_PATH)

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "root_experiment_id": None,
            "best_score": None,
            "consecutive_failures": 0,
            "last_pipeline_id": None,
            "current_phase": "init",
            "suggestion_params": {},
            "stagnation_status": None,
        }
