#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment state machine — fysom-based finite state machine.

Manages the lifecycle of an experiment with strict transition rules.
Provides callbacks for state entry/leave actions and validation.

States:
    created    — Experiment record created, not yet dispatched
    dispatched — Submitted to clearml queue or worktree
    queued     — Enqueued in clearml, waiting for agent
    running    — Being executed by clearml agent
    completed  — Execution finished successfully
    failed     — Execution finished with errors
    cancelled  — Cancelled by user before completion

Events:
    dispatch  → created → dispatched
    queue     → dispatched → queued
    cancel    → {created, dispatched, queued, running} → cancelled
    start     → queued → running
    complete  → running → completed
    fail      → running → failed

Usage:
    from expflow_pde.fsm import ExperimentFSM

    fsm = ExperimentFSM(experiment_id="abc123")
    fsm.dispatch()
    fsm.queue()
    fsm.cancel()
    print(fsm.current)  # "cancelled"
    print(fsm.is_finished())  # True
"""

from typing import Any, Callable, Optional

from fysom import Fysom, FysomError

# ── State & event names as constants ──

STATE_CREATED = "created"
STATE_DISPATCHED = "dispatched"
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

EVENT_DISPATCH = "dispatch"
EVENT_QUEUE = "queue"
EVENT_CANCEL = "cancel"
EVENT_START = "start"
EVENT_COMPLETE = "complete"
EVENT_FAIL = "fail"

# ── Terminal states ──
TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED})

# ── Human-readable labels ──
STATE_LABELS: dict[str, str] = {
    STATE_CREATED: "Created",
    STATE_DISPATCHED: "Dispatched",
    STATE_QUEUED: "Queued",
    STATE_RUNNING: "Running",
    STATE_COMPLETED: "Completed",
    STATE_FAILED: "Failed",
    STATE_CANCELLED: "Cancelled",
}


def state_label(state: str) -> str:
    """Return a human-readable label for a state constant."""
    return STATE_LABELS.get(state, state)


# ── Experiment FSM ──


class ExperimentFSM:
    """Finite state machine for experiment lifecycle.

    Args:
        experiment_id: Unique experiment identifier (used in callbacks).
        callbacks: Optional dict of event callbacks:
            - on_enter_<state>(e): called when entering a state
            - on_leave_<state>(e): called when leaving a state
            - on_<event>(e): called after event completes
            - on_error(exc): called when a transition is rejected

    Example:
        >>> fsm = ExperimentFSM("exp-001")
        >>> fsm.current
        'created'
        >>> fsm.can("dispatch")
        True
        >>> fsm.dispatch()
        >>> fsm.current
        'dispatched'
        >>> fsm.is_finished()
        False
        >>> fsm.cancel()
        >>> fsm.is_finished()
        True
    """

    def __init__(
        self,
        experiment_id: str = "",
        callbacks: Optional[dict[str, Callable[..., Any]]] = None,
    ) -> None:
        self._experiment_id = experiment_id
        self._callbacks = callbacks or {}
        self._last_event: Optional[str] = None
        self._errors: list[str] = []

        # Build FSM with deferred startup
        self._fsm = Fysom(
            initial={"state": STATE_CREATED, "defer": True},
            events=[
                (EVENT_DISPATCH, STATE_CREATED, STATE_DISPATCHED),
                (EVENT_QUEUE, STATE_DISPATCHED, STATE_QUEUED),
                (EVENT_CANCEL, [STATE_CREATED, STATE_DISPATCHED, STATE_QUEUED, STATE_RUNNING], STATE_CANCELLED),
                (EVENT_START, STATE_QUEUED, STATE_RUNNING),
                (EVENT_COMPLETE, STATE_RUNNING, STATE_COMPLETED),
                (EVENT_FAIL, [STATE_QUEUED, STATE_RUNNING], STATE_FAILED),
            ],
            final=None,
        )
        # Monkey-patch fysom lifecycle methods to support generic callbacks.
        # fysom's built-in _enter_state etc. only look for `onenter<dst>` and `on<dst>`
        # (specific callbacks per state/event), not `onenterstate` / `onchangestate` etc.
        # We wrap each so generic callbacks dispatched through user_cbs fire first.
        user_cbs = self._callbacks
        _orig_enter = self._fsm._enter_state
        _orig_leave = self._fsm._leave_state
        _orig_after = self._fsm._after_event
        _orig_change = self._fsm._change_state

        def _patched_enter(e: Any) -> None:
            cb = user_cbs.get(f"on_enter_{e.dst}")
            if cb:
                cb(self, e)
            _orig_enter(e)

        def _patched_leave(e: Any) -> None:
            cb = user_cbs.get(f"on_leave_{e.src}")
            if cb:
                cb(self, e)
            _orig_leave(e)

        def _patched_after(e: Any) -> None:
            cb = user_cbs.get(f"on_{e.event}")
            if cb:
                cb(self, e)
            _orig_after(e)

        def _patched_change(e: Any) -> None:
            cb = user_cbs.get("on_change_state")
            if cb:
                cb(self, e, str(e.dst))
            _orig_change(e)

        self._fsm._enter_state = _patched_enter  # type: ignore[method-assign]
        self._fsm._leave_state = _patched_leave  # type: ignore[method-assign]
        self._fsm._after_event = _patched_after  # type: ignore[method-assign]
        self._fsm._change_state = _patched_change  # type: ignore[method-assign]

        # Fire the deferred startup transition
        self._fsm.startup()

    # ── Properties ──

    @property
    def current(self) -> str:
        """Current state of the experiment."""
        return str(self._fsm.current)

    @property
    def is_finished(self) -> bool:
        """True if experiment is in a terminal state (completed/failed/cancelled)."""
        return self.current in TERMINAL_STATES

    @property
    def errors(self) -> list[str]:
        """List of errors encountered during state transitions."""
        return list(self._errors)

    # ── Public API ──

    def can(self, event: str) -> bool:
        """Check if an event can be triggered in the current state."""
        return bool(self._fsm.can(event))

    def cannot(self, event: str) -> bool:
        """Check if an event cannot be triggered in the current state."""
        return bool(self._fsm.cannot(event))

    def trigger(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Trigger an event by name (dynamic dispatch).

        Raises FysomError if the event is invalid in the current state.
        """
        try:
            self._last_event = event
            self._fsm.trigger(event, *args, **kwargs)
        except FysomError as exc:
            self._errors.append(f"Cannot {event} from {self.current}: {exc}")
            on_error = self._callbacks.get("on_error")
            if on_error:
                on_error(self, exc)
            raise

    # ── Convenience event methods ──

    def dispatch(self) -> None:
        """Transition from created → dispatched."""
        self.trigger(EVENT_DISPATCH)

    def queue(self) -> None:
        """Transition from dispatched → queued."""
        self.trigger(EVENT_QUEUE)

    def cancel(self) -> None:
        """Transition from {created, dispatched, queued, running} → cancelled."""
        self.trigger(EVENT_CANCEL)

    def start(self) -> None:
        """Transition from queued → running (agent picked up the task)."""
        self.trigger(EVENT_START)

    def complete(self) -> None:
        """Transition from running → completed."""
        self.trigger(EVENT_COMPLETE)

    def fail(self) -> None:
        """Transition from {queued, running} → failed."""
        self.trigger(EVENT_FAIL)

    # ── State representation ──

    def __repr__(self) -> str:
        return (
            f"ExperimentFSM(id={self._experiment_id!r}, "
            f"state={self.current!r}, "
            f"finished={self.is_finished})"
        )

    def summary(self) -> dict[str, Any]:
        """Return a dict summary of the current FSM state."""
        return {
            "experiment_id": self._experiment_id,
            "state": self.current,
            "state_label": state_label(self.current),
            "is_finished": self.is_finished,
            "terminal_states": sorted(TERMINAL_STATES),
        }

    # ── Internal (methods below reserved for future use) ──
