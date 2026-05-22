#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""goal_orchestrator.py — Persist and recover /goal loop state across Hermes sessions.

Key capabilities:
1. Save/load progress state (best_score, current_phase, etc.)
2. fcntl.flock concurrency safety
3. Phase-aware TWO-PHASE session recovery:
   - Phase 1: Check clearml task real status (running/completed/failed)
   - Phase 2: Re-attach to running task, read scalars, or trigger repair

Phase state machine:
  idle            → session start
  submitted       → pipeline submit returned pipeline_id
  waiting         → waiting for pipeline completion (--wait in progress)
  completed       → scalars read, score computed
  recovered       → session recovery: previous wait was interrupted
  stalled         → session recovery: pipeline lost, no recovery path
  task_done       → task completed (best_score confirmed)
  deadline_pass   → deadline exceeded, emergency submit
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("expflow_pde.goal_orchestrator")

# The progress state file path — resolves via _get_state_path()
_PROGRESS_PATH: str | None = None


# ── Path resolution ──


def _get_expflow_home() -> str:
    return os.environ.get("EXPFLOW_HOME", os.path.expanduser("~/.expflow"))


def _get_state_path() -> str:
    if _PROGRESS_PATH is not None:
        return _PROGRESS_PATH
    return os.path.join(_get_expflow_home(), "progress_state.json")


def _set_state_path(path: str) -> None:
    """Override for testing."""
    global _PROGRESS_PATH
    _PROGRESS_PATH = path


# ── File I/O with lock ──


def _try_flock(f, lock_type: int) -> None:
    """Try fcntl.flock, silently skip on non-Linux."""
    try:
        import fcntl  # noqa: PLC0415

        fcntl.flock(f.fileno(), lock_type)
    except Exception:
        pass


# ── Public API ──

# State keys that define the well-known interface between Hermes and expflow.
# All keys are english lower_snake_case — do NOT use Chinese.
_INTERFACE_KEYS = frozenset({
    "root_experiment_id",
    "best_score",
    "best_params",
    "best_eval_id",
    "last_pipeline_id",
    "last_train_task_id",
    "last_eval_task_id",
    "current_phase",
    "consecutive_failures",
    "consecutive_no_improvement",
    "current_task",
    "submission_id",
    "pde_mean_best",
    "learned_failures",
    "iteration_count",
    "session_id",
    "mode",
    "deadline",
    "task_order",
    "per_task_max_hours",
    "task_time",
    "completed_tasks",
    "task_hours",
})

# Phase state machine — valid transitions (Hermes enforces, we just persist)
_VALID_PHASES = frozenset({
    "idle", "submitted", "waiting", "completed", "recovered",
    "stalled", "task_done", "deadline_pass",
})


def _verify_state(state: dict[str, Any]) -> None:
    """Verify state contains only english keys and valid phase."""
    for key in state:
        if key in _INTERFACE_KEYS:
            continue
        # Allow underscore-prefixed internal metadata (e.g., _timestamp)
        if key.startswith("_") and all(c.isascii() for c in key):
            continue
        logger.warning(
            "GoalOrchestrator: unexpected key '%s' in state — "
            "all keys must be English snake_case",
            key,
        )
    phase = state.get("current_phase", "idle")
    if phase not in _VALID_PHASES:
        logger.warning(
            "GoalOrchestrator: unknown phase '%s' (valid: %s)",
            phase, sorted(_VALID_PHASES),
        )


def _write_with_flock(state: dict[str, Any], path: str) -> None:
    """Atomically write state dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            _try_flock(f, 2)  # LOCK_EX
            try:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                _try_flock(f, 8)  # LOCK_UN
        os.replace(tmp, path)
    except Exception:
        # Fallback: direct write without lock
        with open(path, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def _read_with_flock(path: str) -> dict[str, Any] | None:
    """Read state dict from JSON file with shared lock."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            _try_flock(f, 1)  # LOCK_SH
            try:
                data = json.load(f)
            finally:
                _try_flock(f, 8)  # LOCK_UN
        if isinstance(data, dict):
            return data
        logger.warning("GoalOrchestrator: invalid state file (not a dict)")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("GoalOrchestrator: failed to read state: %s", e)
    return None


# ── Core save/load ──


def save(state: dict[str, Any]) -> dict[str, Any]:
    """Persist the current /goal loop state.

    Args:
        state: State dict with english snake_case keys. Acceptable keys:
            root_experiment_id, best_score, best_params, last_pipeline_id,
            last_train_task_id, last_eval_task_id, current_phase,
            consecutive_failures, consecutive_no_improvement, current_task,
            submission_id, pde_mean_best, learned_failures, iteration_count.
            Phase must be one of: idle, submitted, waiting, completed,
            recovered, stalled, task_done, deadline_pass.

    Returns:
        The saved state dict (after verification).
    """
    state.setdefault("current_phase", "idle")
    state.setdefault("consecutive_failures", 0)
    state.setdefault("consecutive_no_improvement", 0)
    state.setdefault("iteration_count", 0)
    state.setdefault("learned_failures", [])
    _verify_state(state)
    path = _get_state_path()
    _write_with_flock(state, path)
    logger.debug("GoalOrchestrator saved: phase=%s", state.get("current_phase", "?"))
    return state


def load() -> dict[str, Any]:
    """Load persisted /goal loop state.

    Returns:
        State dict with safe defaults for all missing keys.
    """
    path = _get_state_path()
    raw = _read_with_flock(path)
    if raw is None:
        return {
            "root_experiment_id": "",
            "best_score": 0.0,
            "best_params": {},
            "last_pipeline_id": "",
            "last_train_task_id": "",
            "last_eval_task_id": "",
            "current_phase": "idle",
            "consecutive_failures": 0,
            "consecutive_no_improvement": 0,
            "current_task": "task1",
            "submission_id": "",
            "pde_mean_best": 999.0,
            "learned_failures": [],
            "iteration_count": 0,
        }

    # Safe merge: ensure all required keys exist
    defaults = {
        "root_experiment_id": "",
        "best_score": 0.0,
        "best_params": {},
        "last_pipeline_id": "",
        "last_train_task_id": "",
        "last_eval_task_id": "",
        "current_phase": "idle",
        "consecutive_failures": 0,
        "consecutive_no_improvement": 0,
        "current_task": "task1",
        "submission_id": "",
        "pde_mean_best": 999.0,
        "learned_failures": [],
        "iteration_count": 0,
    }
    defaults.update(raw)
    phase = defaults.get("current_phase", "idle")
    if phase not in _VALID_PHASES:
        logger.warning(
            "GoalOrchestrator loaded unknown phase '%s' — resetting to 'idle'",
            phase,
        )
        defaults["current_phase"] = "idle"
    return defaults


def recover_pipeline(
    last_train_task_id: str,
    last_eval_task_id: str,
    current_phase: str,
    pipeline_id: str | None = None,
) -> dict[str, Any]:
    """Two-phase session recovery.

    Phase 1: Determine clearml task status.
    Phase 2: Produce recovery action.

    Returns:
        dict with keys:
            - recovered_phase: str — "recovered" | "stalled" | "completed"
            - action: str — instruction for Hermes
            - train_task_status: str | None — clearml task status
            - eval_task_status: str | None
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    result: dict[str, Any] = {
        "recovered_phase": "stalled",
        "action": "No recovery path — re-initialize from scratch",
        "train_task_status": None,
        "eval_task_status": None,
    }

    if not last_train_task_id and not last_eval_task_id:
        result["recovered_phase"] = "recovered"
        result["action"] = "No pipeline in flight — start fresh iteration"
        result["train_task_status"] = "none"
        return result

    # Phase 1: Try to get real status from clearml
    train_status = _get_task_status(last_train_task_id) if last_train_task_id else None
    eval_status = _get_task_status(last_eval_task_id) if last_eval_task_id else None

    result["train_task_status"] = train_status or "unknown"
    result["eval_task_status"] = eval_status or "unknown"

    # Phase 2: Recovery decision
    if train_status == "in_progress" or train_status == "queued":
        # Pipeline is still running — re-attach wait
        result["recovered_phase"] = "recovered"
        result["action"] = (
            f"Previous pipeline (train={last_train_task_id}) is still "
            f"{train_status}. Re-attach with --wait --json or increase "
            "wait timeout."
        )
    elif train_status == "completed":
        if eval_status == "completed":
            result["recovered_phase"] = "completed"
            result["action"] = (
                f"Pipeline completed (train={last_train_task_id}, "
                f"eval={last_eval_task_id}). Read scalars directly via "
                "Task.get_task(last_eval_task_id)."
            )
        elif eval_status in ("failed", "stopped"):
            result["recovered_phase"] = "recovered"
            result["action"] = (
                f"Train completed but eval failed/stopped "
                f"(status={eval_status}). "
                "Consider repairing eval step or skipping eval result."
            )
        else:
            result["recovered_phase"] = "recovered"
            result["action"] = (
                f"Train completed, eval status unknown ({eval_status}). "
                "Read train scalars as fallback."
            )
    elif train_status in ("failed", "stopped"):
        result["recovered_phase"] = "recovered"
        result["action"] = (
            f"Previous pipeline failed (train={last_train_task_id}, "
            f"status={train_status}). Trigger repair or resubmit."
        )
    else:
        result["recovered_phase"] = "stalled"
        result["action"] = (
            f"Cannot determine pipeline status (train={train_status}, "
            f"eval={eval_status}). Check clearml web UI manually."
        )

    return result


def _get_task_status(task_id: str) -> str | None:
    """Get clearml task status from server.

    Returns: one of "unknown", "queued", "in_progress", "completed",
    "failed", "stopped", "pending", or None if unreachable.

    Note: This function is intentionally tolerant of clearml SDK absence
    and network failures — returns None on any error.
    """
    try:
        from clearml import Task  # noqa: PLC0415

        task = Task.get_task(task_id=task_id)
        if task is None:
            return None
        return task.get_status()
    except ImportError:
        logger.debug("clearml SDK not available — cannot check task status")
        return None
    except Exception as exc:
        logger.debug("Failed to get status for task %s: %s", task_id, exc)
        return None


def add_learned_failure(failure: dict[str, Any]) -> None:
    """Add a learned failure to the persistent state.

    Args:
        failure: dict with keys:
            - type: str — e.g. "oom", "timeout", "pde_gate"
            - params: dict — the hyperparams that caused the failure
            - reason: str — human-readable explanation
            - experiment_id: str | None
    """
    state = load()
    failures = state.get("learned_failures", [])
    # Deduplicate: same type + same param keys → replace rather than append
    new_params = failure.get("params", {})
    for existing in failures:
        if (
            existing.get("type") == failure.get("type")
            and existing.get("params", {}).keys() == new_params.keys()
        ):
            existing.update(failure)
            save(state)
            return
    failures.append(failure)
    # Keep only last 20 to avoid unbounded growth
    if len(failures) > 20:
        failures[:] = failures[-20:]
    state["learned_failures"] = failures
    save(state)


def clear() -> None:
    """Delete the persisted state file."""
    path = _get_state_path()
    if os.path.exists(path):
        try:
            os.remove(path)
            logger.info("GoalOrchestrator cleared")
        except OSError as e:
            logger.warning("GoalOrchestrator failed to clear: %s", e)
