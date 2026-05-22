#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition controller — Hermes /goal-compatible competition scheduling.

Provides deadline/budget gating, task-to-task scheduling, and mode-aware
constraint checking for the PDEBench competition.

Two modes:
  - EXPLORE (7x24h): Loose time budget per task, token/fee soft budget.
  - SPRINT: Hard per-task ≤12h limit, strict deadline enforcement.

Design principle: CompetitionController is a pure-Python object that Hermes
uses inline in the /goal loop — NOT a daemon or separate scheduler. It
makes decisions available; Hermes executes them.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from expflow_pde.goal_orchestrator import GoalOrchestrator

logger = logging.getLogger("expflow_pde.competition_controller")

_TASK_META_PATH = os.path.expanduser("~/.expflow/task_meta.yaml")
_PROGRESS_PATH = os.path.expanduser("~/.expflow/progress_state.json")


class CompetitionController:
    """Competition scheduling controller for Hermes /goal loops.

    Args:
        session_id: Unique goal session ID (used to tag clearml tasks).
        mode: 'explore' (7x24h) or 'sprint' (fast, per-task ≤12h).
        deadline: ISO-8601 deadline string (e.g. '2026-06-30T23:59:59+08:00').
        task_order: Ordered list of tasks, e.g. ['task1', 'task2', 'task3'].
        budget: Soft budget limit in dollars/tokens (explore mode only).
        per_task_max_hours: Per-task hard limit in hours (sprint mode, default 12).
    """

    MODE_EXPLORE = "explore"
    MODE_SPRINT = "sprint"

    def __init__(
        self,
        session_id: str,
        mode: str = MODE_SPRINT,
        deadline: str | None = None,
        task_order: list[str] | None = None,
        budget: float = float("inf"),
        per_task_max_hours: float = 12.0,
    ) -> None:
        if mode not in (self.MODE_EXPLORE, self.MODE_SPRINT):
            raise ValueError(f"Unknown mode: {mode}. Use 'explore' or 'sprint'.")

        self.session_id = session_id
        self.mode = mode
        self.task_order = task_order or ["task1", "task2", "task3"]

        self._deadline: datetime | None = None
        if deadline:
            try:
                self._deadline = datetime.fromisoformat(deadline)
            except ValueError:
                logger.warning("Invalid deadline '%s', ignoring", deadline)

        self._budget = budget
        self._per_task_max_hours = per_task_max_hours

        # Restore state from GoalOrchestrator if resuming
        self._state = self._load_or_init_state()
        self._task_hours: dict[str, float] = self._state.get("task_hours", {})

    def check_deadline(self) -> bool:
        """Check if the overall competition deadline has passed.

        Returns:
            True if deadline passed (must emergency-submit best results).
        """
        if self._deadline is None:
            return False
        return datetime.now(timezone.utc).astimezone() > self._deadline

    def check_per_task_limit(self, task_name: str) -> bool:
        """Check if the current task has exceeded its time limit.

        In sprint mode, each task has a hard max hours.
        In explore mode, there is no per-task limit.

        Returns:
            True if the task has NOT exceeded the limit (OK to continue).
            False if the limit is exceeded (must submit best and move on).
        """
        if self.mode == self.MODE_EXPLORE:
            return True  # no per-task limit in explore mode

        elapsed = self._task_hours.get(task_name, 0.0)
        if elapsed >= self._per_task_max_hours:
            logger.warning(
                "Task %s exceeded limit: %.1f / %.1f hours",
                task_name, elapsed, self._per_task_max_hours,
            )
            return False
        return True

    def check_budget(self) -> bool:
        """Check if remaining budget is positive.

        Returns:
            True if budget remaining (or unlimited).
        """
        remaining = self._state.get("budget_remaining", float("inf"))
        return remaining > 0

    def check_queue_depth(self, queue_name: str = "default") -> dict[str, int]:
        """Check clearml queue depth before submitting a new task.

        If the queue has many pending items, the new task will experience
        scheduling delay that inflates train_time_min. Call this before
        submit to decide whether to wait.

        Args:
            queue_name: Queue name to check.

        Returns:
            dict with keys: running, pending, total.
            Returns zeros on clearml connection failure.
        """
        try:
            from expflow_pde.clearml import get_queue_status

            return get_queue_status(queue_name)
        except Exception:
            logger.warning("Cannot check queue depth (clearml may be down)")
            return {"running": 0, "pending": 0, "total": 0}

    def should_wait_for_queue(self, queue_name: str = "default",
                               max_pending: int = 1) -> bool:
        """Decide whether Hermes should wait before submitting.

        Returns True if queue is too deep and submitting now would cause
        scheduling delay inflation.

        Args:
            queue_name: Queue to check.
            max_pending: Max acceptable pending tasks (default: 1).

        Returns:
            True = wait before submit, False = submit now.
        """
        status = self.check_queue_depth(queue_name)
        pending = status.get("pending", 0)
        if pending > max_pending:
            logger.info("Queue %s has %d pending tasks — consider waiting",
                        queue_name, pending)
            return True
        return False

    def record_task_time(self, task_name: str, additional_hours: float) -> None:
        """Record time spent on a task."""
        current = self._task_hours.get(task_name, 0.0)
        self._task_hours[task_name] = current + additional_hours

    def complete_task(self, task_name: str) -> str | None:
        """Mark a task as completed and return the next task to work on.

        Args:
            task_name: The task that just finished.

        Returns:
            Next task name, or None if all tasks are done.
        """
        completed = set(self._state.get("completed_tasks", []))
        completed.add(task_name)
        self._state["completed_tasks"] = list(completed)

        for t in self.task_order:
            if t not in completed:
                logger.info("Advancing to next task: %s", t)
                return t
        return None  # all done

    def get_current_task(self) -> str | None:
        """Return the first uncompleted task."""
        completed = set(self._state.get("completed_tasks", []))
        for t in self.task_order:
            if t not in completed:
                return t
        return None

    def save(self, extra: dict | None = None) -> None:
        """Persist all controller state via GoalOrchestrator (with flock)."""
        state = {
            "session_id": self.session_id,
            "mode": self.mode,
            "task_order": self.task_order,
            "task_hours": self._task_hours,
            "completed_tasks": self._state.get("completed_tasks", []),
            "budget_remaining": self._state.get("budget_remaining", float("inf")),
            "deadline": str(self._deadline) if self._deadline else None,
        }
        if extra:
            state.update(extra)
        GoalOrchestrator.save(state)

    def load_state(self) -> dict[str, Any]:
        """Reload persisted state (for Hermes recovery across sessions)."""
        self._state = self._load_or_init_state()
        self._task_hours = self._state.get("task_hours", {})
        return dict(self._state)

    def _load_or_init_state(self) -> dict[str, Any]:
        """Load from GoalOrchestrator or return empty defaults."""
        raw = GoalOrchestrator.load()
        # If the loaded state has our session_id, use it
        if raw.get("session_id") == self.session_id:
            return raw
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "task_order": self.task_order,
            "task_hours": {},
            "completed_tasks": [],
            "budget_remaining": self._budget,
            "deadline": str(self._deadline) if self._deadline else None,
        }
