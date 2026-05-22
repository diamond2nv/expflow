#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""goal_orchestrator.py — Persist and recover /goal loop state across Hermes sessions.

Key capabilities:
1. Save/load progress state (best_score, current_phase, etc.)
2. fcntl.flock concurrency safety
3. resolve_pipeline_state: UNIFIED single source of truth for pipeline status
   - Replaces the old recover_pipeline() + competition_controller.check_pipeline_in_flight()
   - Single arbitration: one input state -> one action output
   - No ambiguity between two independent callers
  
Phase state machine:
  idle            -> session start
  submitted       -> pipeline submit returned pipeline_id
  waiting         -> waiting for pipeline completion (--wait in progress)
  completed       -> scalars read, score computed
  recovered       -> session recovery: previous wait was interrupted
  stalled         -> session recovery: pipeline lost, no recovery path
  task_done       -> task completed (best_score confirmed)
  deadline_pass   -> deadline exceeded, emergency submit

Interface:
  - All state keys are English lower_snake_case (enforced by _verify_state)
  - _verify_state REJECTS unknown keys (ValueError) — no silent proliferation
  - load() STRIPS unknown keys — won't propagate stale/noncompliant state
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

# State keys — all English lower_snake_case.
# Anything not in this frozenset is REJECTED by save() and STRIPPED by load().
_INTERFACE_KEYS: frozenset = frozenset({
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

# Action strings from resolve_pipeline_state() — unique semantics
_PIPELINE_ACTIONS = frozenset({
    "re_wait", "read_scalars", "repair", "submit_new", "skip_tick",
})

# Phase state machine — valid transitions (Hermes enforces, we just persist)
_VALID_PHASES = frozenset({
    "idle", "submitted", "waiting", "completed", "recovered",
    "stalled", "task_done", "deadline_pass",
})


def _verify_state(state: dict[str, Any]) -> None:
    """Verify state contains only allowed keys and valid phase.

    Raises ValueError on non-compliant keys — unknown keys cannot be
    persisted silently. This prevents Chinese/arbitrary keys from
    entering the state file.
    """
    unexpected: list[str] = []
    for key in state:
        if key in _INTERFACE_KEYS:
            continue
        # Allow underscore-prefixed internal metadata (e.g., _timestamp)
        if key.startswith("_") and all(c.isascii() for c in key):
            continue
        unexpected.append(key)
    if unexpected:
        msg = (
            f"GoalOrchestrator: unexpected state key(s): {unexpected}. "
            "All keys must be English lower_snake_case. "
            f"Allowed: {sorted(_INTERFACE_KEYS)}"
        )
        logger.error(msg)
        raise ValueError(msg)
    phase = state.get("current_phase", "idle")
    if phase not in _VALID_PHASES:
        logger.warning(
            "GoalOrchestrator: unknown phase '%s' (valid: %s) — resetting to 'idle'",
            phase, sorted(_VALID_PHASES),
        )
        state["current_phase"] = "idle"


_DEFAULTS: dict[str, Any] = {
    "root_experiment_id": "",
    "best_score": 0.0,
    "best_params": {},
    "best_eval_id": "",
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
    "session_id": "",
    "mode": "explore",
    "deadline": "2026-06-30T23:59:59+08:00",
    "task_order": ["task1", "task2", "task3"],
    "per_task_max_hours": 12.0,
    "task_time": {},
    "completed_tasks": [],
    "task_hours": {},
}


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
        state: State dict with english snake_case keys.

    Returns:
        The saved state dict (after verification).

    Raises:
        ValueError: if state contains keys outside _INTERFACE_KEYS.
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

    Strips any unknown keys from the persisted state — this prevents
    stale/noncompliant keys from propagating.

    Returns:
        State dict with safe defaults for all missing keys.
    """
    path = _get_state_path()
    raw = _read_with_flock(path)
    if raw is None:
        return dict(_DEFAULTS)

    # Strip unknown keys from persisted state
    raw = {k: v for k, v in raw.items() if k in _INTERFACE_KEYS or (k.startswith("_") and all(c.isascii() for c in k))}

    # Safe merge
    result = dict(_DEFAULTS)
    result.update(raw)

    phase = result.get("current_phase", "idle")
    if phase not in _VALID_PHASES:
        logger.warning(
            "GoalOrchestrator loaded unknown phase '%s' — resetting to 'idle'",
            phase,
        )
        result["current_phase"] = "idle"
    return result


# ── Unified pipeline state arbitration ──


def resolve_pipeline_state(state: dict[str, Any]) -> dict[str, Any]:
    """UNIFIED arbitration: determine pipeline state and produce single action.

    This REPLACES both the old recover_pipeline() (in goal_orchestrator.py)
    and check_pipeline_in_flight() (in competition_controller.py). There is
    ONE code path for determining what to do with a pipeline.

    Input: GoalOrchestrator.load() result dict.
    Output: dict with keys:
        action: str — one of PIPELINE_ACTIONS
        pipeline_status: str | None — clearml task status
        train_task_status: str | None
        eval_task_status: str | None
        train_killed: bool — whether train_step was killed by time_limit

    Action semantics:
        re_wait:     pipeline still running, re-attach --wait
        read_scalars: pipeline completed, read scalars and score
        repair:      pipeline failed, trigger repair / resubmit
        submit_new:  no pipeline in flight, start fresh
        skip_tick:   pipeline in queue, don't touch anything this tick
    """
    result: dict[str, Any] = {
        "action": "submit_new",
        "pipeline_status": None,
        "train_task_status": None,
        "eval_task_status": None,
        "train_killed": False,
    }

    last_pipeline_id = state.get("last_pipeline_id", "")
    last_train_task_id = state.get("last_train_task_id", "")
    last_eval_task_id = state.get("last_eval_task_id", "")

    if not last_pipeline_id and not last_train_task_id:
        result["action"] = "submit_new"
        return result

    # Get real status from clearml
    pipeline_status = _get_task_status(last_pipeline_id) if last_pipeline_id else None
    train_status = _get_task_status(last_train_task_id) if last_train_task_id else None
    eval_status = _get_task_status(last_eval_task_id) if last_eval_task_id else None

    result["pipeline_status"] = pipeline_status
    result["train_task_status"] = train_status
    result["eval_task_status"] = eval_status

    # ── Arbitration ──

    # 1. Pipeline is still in flight
    if pipeline_status in ("queued", "in_progress", "created", "pending"):
        result["action"] = "re_wait"
        return result

    # 2. Pipeline completed
    if pipeline_status == "completed":
        result["action"] = "read_scalars"
        # Check train killed
        if train_status == "stopped":
            result["train_killed"] = True
        return result

    # 3. Train step killed by time_limit
    if train_status == "stopped":
        result["action"] = "repair"
        result["train_killed"] = True
        return result

    # 4. Pipeline/step failed
    if pipeline_status in ("failed", "stopped") or train_status in ("failed",):
        result["action"] = "repair"
        return result

    # 5. Pipeline in queue but not yet created (edge case: no pipeline ID)
    if pipeline_status is None and train_status in ("queued", "pending"):
        result["action"] = "skip_tick"
        return result

    # 6. Pipeline unknown (clearml unreachable or task never existed)
    result["action"] = "submit_new"
    return result


def _get_task_status(task_id: str) -> str | None:
    """Get clearml task status from server.

    Returns: one of "queued", "in_progress", "completed", "failed",
    "stopped", "pending", "created", or None if unreachable.
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


# ── Learned failures ──


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
    # Deduplicate: same type + same param keys -> replace rather than append
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
    # Keep only last 20
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
