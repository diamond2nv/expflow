#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""competition_controller.py — Lightweight inline controller for /goal loops.

Used by Hermes within the /goal loop (NOT a daemon or separate process).
Provides constraint checks and state management for unattended competition.

Pipeline state arbitration has been UNIFIED into
goal_orchestrator.resolve_pipeline_state() — this module no longer has
its own check_pipeline_in_flight() to avoid action ambiguity.

All deadline parsing uses stdlib only (re + datetime) — no dateutil dependency.
"""

from __future__ import annotations

import datetime
import logging
import os
import re as _re
from typing import Any

logger = logging.getLogger("expflow_pde.competition_controller")

# ── Constants ──

_DEFAULT_DEADLINE = "2026-06-30T23:59:59+08:00"
_DEFAULT_PER_TASK_MAX_HOURS = 12.0

# Pure-stdlib timezone offset regex: +08:00, -05:30, etc.
_ZONE_OFFSET_RE = _re.compile(r"([+-])(\d{2}):(\d{2})$")


# ── CompetitionController ──


class CompetitionController:
    """Inline controller for a /goal loop session.

    All checks are 0-token (pure Python). State persists via
    GoalOrchestrator.save().

    Pipeline status checks delegate to
    goal_orchestrator.resolve_pipeline_state() — there is ONE action source.

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
        """Check if the competition deadline has passed."""
        parsed = _parse_deadline(self._deadline)
        return datetime.datetime.now(parsed.tzinfo) > parsed

    def remaining_days(self) -> float:
        """Days remaining before deadline (float, may be negative)."""
        parsed = _parse_deadline(self._deadline)
        now = datetime.datetime.now(parsed.tzinfo)
        delta = parsed - now
        return max(0.0, delta.total_seconds() / 86400.0)

    def check_per_task_limit(self, task_name: str) -> bool:
        """Check if a task has exceeded its time budget."""
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

    def should_wait_for_queue(self, queue_name: str = "default", max_pending: int = 1) -> bool:
        """Check if queue backlog would inflate train_time."""
        depth = self.check_queue_depth(queue_name)
        pending = depth.get("pending", 0)
        if pending > max_pending:
            logger.info("Queue %s has %d pending tasks — waiting", queue_name, pending)
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


# ── Pure-stdlib deadline parser ──


def _parse_deadline(deadline_str: str) -> datetime.datetime:
    """Parse ISO-8601 deadline string with timezone awareness.

    Pure stdlib — no dateutil dependency. Supports:
    - 2026-06-30T23:59:59+08:00
    - 2026-06-30T23:59:59Z
    - 2026-06-30 (date only, treated as UTC midnight)
    """
    s = deadline_str.strip()

    # Handle trailing Z (normalize to +00:00)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # Try with timezone offset
    m = _ZONE_OFFSET_RE.search(s)
    if m:
        body = s[:m.start()]
        sign = 1 if m.group(1) == "+" else -1
        oh = int(m.group(2))
        om = int(m.group(3))
        tz = datetime.timezone(datetime.timedelta(hours=sign * oh, minutes=sign * om))
        # Strip fractional seconds if present
        body = body.split(".")[0]
        dt = datetime.datetime.strptime(body, "%Y-%m-%dT%H:%M:%S")
        return dt.replace(tzinfo=tz)

    # Date only (no time component)
    if "T" not in s:
        return datetime.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)

    # Full datetime without timezone — treat as UTC
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
