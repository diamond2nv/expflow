#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""competition_controller.py — Lightweight inline controller for /goal loops.

Used by Hermes within the /goal loop (NOT a daemon or separate process).
Provides constraint checks and state management for unattended competition.

Key additions in this version:
  - check_pipeline_in_flight(): idempotent cron guard — detects if a
    pipeline from a previous cron tick is still running and prevents
    duplicate submission.
  - check_pipeline_recovery(): tells Hermes what to do when resuming
    from a cron tick (re-attach wait vs. skip vs. repair).
  - Deadline, budget, per-task limits, queue depth checks.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

logger = logging.getLogger("expflow_pde.competition_controller")


# ── Constants ──

_DEFAULT_DEADLINE = "2026-06-30T23:59:59+08:00"
_DEFAULT_PER_TASK_MAX_HOURS = 12.0


# ── CompetitionController ──


class CompetitionController:
    """Inline controller for a /goal loop session.

    All checks are 0-token (pure Python). State persists via
    GoalOrchestrator.save().

    Args:
        session_id: Unique identifier for this goal session.
        mode: 'explore' (loose) or 'sprint' (hard constraints).
        deadline: ISO-8601 deadline string.
        task_order: List of task names in priority order.
        per_task_max_hours: Max cumulative hours per task (sprint only).
    """

    def __init__(
        self,
        session_id: str = "default",
        mode: str = "explore",
        deadline: str | None = None,
        task_order: list[str] | None = None,
        per_task_max_hours: float = _DEFAULT_PER_TASK_MAX_HOURS,
    ) -> None:
        self._session_id = session_id
        self._mode = mode
        self._deadline = deadline or _DEFAULT_DEADLINE
        self._task_order = task_order or ["task1", "task2", "task3"]
        self._per_task_max_hours = per_task_max_hours
        self._task_time: dict[str, float] = {}

    # ── Deadline checks ──

    def check_deadline(self) -> bool:
        """Check if the competition deadline has passed.

        Returns True if current time > deadline (emergency mode).
        """
        parsed = self._parse_deadline(self._deadline)
        return datetime.datetime.now(parsed.tzinfo) > parsed

    def remaining_days(self) -> float:
        """Days remaining before deadline (float, may be negative)."""
        parsed = self._parse_deadline(self._deadline)
        now = datetime.datetime.now(parsed.tzinfo)
        delta = parsed - now
        return max(0.0, delta.total_seconds() / 86400.0)

    def check_per_task_limit(self, task_name: str) -> bool:
        """Check if a task has exceeded its time budget.

        In sprint mode, returns False if cumulative hours > limit.
        In explore mode, returns True (no limit).
        """
        if self._mode != "sprint":
            return True
        cumulative = self._task_time.get(task_name, 0.0)
        if cumulative >= self._per_task_max_hours:
            logger.warning(
                "Task %s exceeded per-task limit: %.1f/%.1f hours",
                task_name, cumulative, self._per_task_max_hours,
            )
            return False
        return True

    def record_task_time(self, task_name: str, hours: float) -> None:
        """Record cumulative time spent on a task."""
        self._task_time[task_name] = self._task_time.get(task_name, 0.0) + hours

    # ── Cron idempotent guard ──

    def check_pipeline_in_flight(
        self, last_pipeline_id: str | None = None,
    ) -> dict[str, Any]:
        """Check if a previously submitted pipeline is still in flight.

        This is the cron idempotent guard. Call this at the START of each
        /goal tick (before any new submit). If the previous tick's pipeline
        is still running, returns status='running' — Herman should NOT submit.

        Returns:
            dict with keys:
            - status: 'running' | 'completed' | 'failed' | 'none'
            - action: instruction for Hermes
            - pipeline_id: str or None
        """
        if not last_pipeline_id:
            return {"status": "none", "action": "No previous pipeline", "pipeline_id": None}

        try:
            from expflow_pde.clearml import get_task  # noqa: PLC0415

            task = get_task(last_pipeline_id)
            if task is None:
                return {
                    "status": "unknown",
                    "action": f"Cannot find pipeline {last_pipeline_id} — "
                    "may have been deleted or clearml is unreachable",
                    "pipeline_id": last_pipeline_id,
                }
            task_status = task.get("status", "") or ""

            if task_status in ("queued", "in_progress", "created"):
                return {
                    "status": "running",
                    "action": f"Pipeline {last_pipeline_id} is still {task_status}. "
                    "Do NOT submit a new one yet. Re-attach --wait instead.",
                    "pipeline_id": last_pipeline_id,
                }
            if task_status == "completed":
                return {
                    "status": "completed",
                    "action": f"Pipeline {last_pipeline_id} completed. "
                    "Read scalars and continue.",
                    "pipeline_id": last_pipeline_id,
                }
            if task_status in ("failed", "stopped"):
                return {
                    "status": "failed",
                    "action": f"Pipeline {last_pipeline_id} {task_status}. "
                    "Trigger repair.",
                    "pipeline_id": last_pipeline_id,
                }
            return {
                "status": "unknown",
                "action": f"Pipeline {last_pipeline_id} has unexpected status: {task_status}",
                "pipeline_id": last_pipeline_id,
            }
        except Exception as exc:
            return {
                "status": "error",
                "action": f"Cannot check pipeline status: {exc}",
                "pipeline_id": last_pipeline_id,
            }

    def check_pipeline_recovery(
        self,
        last_pipeline_id: str | None,
        last_train_task_id: str | None,
        last_eval_task_id: str | None,
    ) -> dict[str, Any]:
        """Determine what Hermes should do when resuming from a cron tick.

        Three outcomes:
        - action='re_wait': pipeline still running, re-attach --wait
        - action='read_scalars': pipeline completed, read results directly
        - action='repair': pipeline failed, trigger repair

        Returns:
            dict with keys: action, pipeline_status, train_status, eval_status
        """
        result: dict[str, Any] = {
            "action": "submit_new",
            "pipeline_status": None,
            "train_status": None,
            "eval_status": None,
        }

        # Check pipeline status first
        in_flight = self.check_pipeline_in_flight(last_pipeline_id)
        pipeline_status = in_flight.get("status", "none")
        result["pipeline_status"] = pipeline_status

        if pipeline_status == "running":
            result["action"] = "re_wait"
            result["train_status"] = "in_progress"
        elif pipeline_status == "completed":
            result["action"] = "read_scalars"
            try:
                from expflow_pde.clearml import get_task  # noqa: PLC0415

                if last_eval_task_id:
                    eval_task = get_task(last_eval_task_id)
                    result["eval_status"] = eval_task.get("status", "") if eval_task else None
                if last_train_task_id:
                    train_task = get_task(last_train_task_id)
                    result["train_status"] = train_task.get("status", "") if train_task else None
            except Exception:
                pass
        elif pipeline_status in ("failed", "stopped"):
            result["action"] = "repair"
        else:
            # no previous or unknown — start fresh
            result["action"] = "submit_new"

        return result

    # ── Queue depth ──

    def check_queue_depth(self, queue_name: str = "default") -> dict[str, Any]:
        """Check clearml queue depth.

        Returns dict with running, pending, total counts.
        Returns all zeros on clearml connection error (non-fatal).
        """
        try:
            from expflow_pde.clearml import get_queue_depth  # noqa: PLC0415

            return get_queue_depth(queue_name)
        except Exception:
            return {"running": 0, "pending": 0, "total": 0}

    def should_wait_for_queue(
        self, queue_name: str = "default", max_pending: int = 1
    ) -> bool:
        """Check if queue backlog would inflate train_time.

        Returns True if backlog may cause false train_time inflation.
        """
        depth = self.check_queue_depth(queue_name)
        pending = depth.get("pending", 0)
        if pending > max_pending:
            logger.info(
                "Queue %s has %d pending tasks — waiting",
                queue_name, pending,
            )
            return True
        return False

    # ── Task scheduling ──

    def complete_task(self, task_name: str) -> str | None:
        """Mark a task as completed and return next task name, or None."""
        if task_name in self._task_order:
            idx = self._task_order.index(task_name)
            if idx + 1 < len(self._task_order):
                return self._task_order[idx + 1]
        return None

    def get_current_task(self) -> str | None:
        """Return the first uncompleted task."""
        for task in self._task_order:
            if self._task_time.get(task, 0.0) < self._per_task_max_hours:
                return task
        return self._task_order[-1] if self._task_order else None

    # ── Persistence ──

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for GoalOrchestrator persistence."""
        return {
            "session_id": self._session_id,
            "mode": self._mode,
            "deadline": self._deadline,
            "task_order": list(self._task_order),
            "per_task_max_hours": self._per_task_max_hours,
            "task_time": dict(self._task_time),
        }

    def save(self, extra: dict[str, Any] | None = None) -> None:
        """Persist state via GoalOrchestrator.save()."""
        from expflow_pde.goal_orchestrator import save as _g_save  # noqa: PLC0415

        state = self.to_dict()
        if extra:
            state.update(extra)
        _g_save(state)

    # ── Static helpers ──

    @staticmethod
    def _parse_deadline(deadline_str: str) -> datetime.datetime:
        """Parse ISO-8601 deadline string with timezone awareness."""
        # Remove trailing Z and normalize
        s = deadline_str.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if s.endswith("+08:00"):
            fmt = "%Y-%m-%dT%H:%M:%S%z"
            import re  # noqa: PLC0415

            m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-]\d{2}:\d{2})", s)
            if m:
                return datetime.datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)), int(m.group(6)),
                    tzinfo=datetime.timezone(
                        datetime.timedelta(
                            hours=int(m.group(7)[:3]),
                            minutes=int(m.group(7)[4:]) if len(m.group(7)) > 5 else 0,
                        )
                    ),
                )
        # Fallback to dateutil if available, else basic parsing
        try:
            from dateutil import parser  # noqa: PLC0415

            return parser.parse(s)
        except ImportError:
            # Basic fallback: treat as UTC
            try:
                return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(
                    tzinfo=datetime.timezone.utc
                )
            except ValueError:
                return datetime.datetime.strptime(s, "%Y-%m-%d").replace(
                    tzinfo=datetime.timezone.utc
                )
