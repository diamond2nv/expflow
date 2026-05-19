#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.fsm — ExperimentFSM state machine.

Covers:
- All valid state transitions (happy path)
- All invalid transitions (error paths)
- Callback invocation
- Terminal state detection
- Summary serialization
- Edge cases (cancel from all valid sources)
"""

import pytest
from fysom import FysomError

from expflow_pde.fsm import (
    ExperimentFSM,
    STATE_CREATED,
    STATE_DISPATCHED,
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_CANCELLED,
    TERMINAL_STATES,
    state_label,
)


class TestStateLabel:
    """state_label() helper."""

    def test_known_states(self):
        assert state_label("created") == "Created"
        assert state_label("dispatched") == "Dispatched"
        assert state_label("queued") == "Queued"
        assert state_label("running") == "Running"
        assert state_label("completed") == "Completed"
        assert state_label("failed") == "Failed"
        assert state_label("cancelled") == "Cancelled"

    def test_unknown_state_returns_raw(self):
        assert state_label("weird_state") == "weird_state"


class TestInitialState:
    """Initial state and basic properties."""

    def test_initial_state_is_created(self):
        fsm = ExperimentFSM("exp-001")
        assert fsm.current == STATE_CREATED
        assert not fsm.is_finished

    def test_default_experiment_id(self):
        fsm = ExperimentFSM()
        assert fsm.current == STATE_CREATED

    def test_can_cannot_at_start(self):
        fsm = ExperimentFSM("exp-002")
        assert fsm.can("dispatch")
        assert not fsm.can("queue")
        assert not fsm.can("complete")
        assert not fsm.cannot("dispatch")  # cannot is inverse of can
        assert fsm.cannot("queue")


class TestHappyPath:
    """All valid transitions from created to completion."""

    def test_full_lifecycle(self):
        fsm = ExperimentFSM("exp-010")
        assert fsm.current == STATE_CREATED

        fsm.dispatch()
        assert fsm.current == STATE_DISPATCHED

        fsm.queue()
        assert fsm.current == STATE_QUEUED

        fsm.start()
        assert fsm.current == STATE_RUNNING

        fsm.complete()
        assert fsm.current == STATE_COMPLETED
        assert fsm.is_finished

    def test_created_to_cancelled(self):
        fsm = ExperimentFSM("exp-011")
        fsm.cancel()
        assert fsm.current == STATE_CANCELLED
        assert fsm.is_finished

    def test_dispatched_to_cancelled(self):
        fsm = ExperimentFSM("exp-012")
        fsm.dispatch()
        fsm.cancel()
        assert fsm.current == STATE_CANCELLED

    def test_queued_to_cancelled(self):
        fsm = ExperimentFSM("exp-013")
        fsm.dispatch()
        fsm.queue()
        fsm.cancel()
        assert fsm.current == STATE_CANCELLED

    def test_running_to_cancelled(self):
        fsm = ExperimentFSM("exp-014")
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        fsm.cancel()
        assert fsm.current == STATE_CANCELLED

    def test_fail_from_running(self):
        fsm = ExperimentFSM("exp-015")
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        fsm.fail()
        assert fsm.current == STATE_FAILED
        assert fsm.is_finished

    def test_fail_from_queued(self):
        fsm = ExperimentFSM("exp-016")
        fsm.dispatch()
        fsm.queue()
        fsm.fail()
        assert fsm.current == STATE_FAILED
        assert fsm.is_finished

    def test_trigger_dynamic(self):
        fsm = ExperimentFSM("exp-017")
        fsm.trigger("dispatch")
        assert fsm.current == STATE_DISPATCHED
        fsm.trigger("queue")
        assert fsm.current == STATE_QUEUED


class TestInvalidTransitions:
    """Transition attempts that should raise FysomError."""

    def test_cannot_dispatch_twice(self):
        fsm = ExperimentFSM("exp-020")
        fsm.dispatch()
        with pytest.raises(FysomError):
            fsm.dispatch()

    def test_cannot_queue_from_created(self):
        fsm = ExperimentFSM("exp-021")
        with pytest.raises(FysomError):
            fsm.queue()

    def test_cannot_start_from_created(self):
        fsm = ExperimentFSM("exp-022")
        with pytest.raises(FysomError):
            fsm.start()

    def test_cannot_complete_from_created(self):
        fsm = ExperimentFSM("exp-023")
        with pytest.raises(FysomError):
            fsm.complete()

    def test_cannot_complete_from_queued(self):
        fsm = ExperimentFSM("exp-024")
        fsm.dispatch()
        fsm.queue()
        with pytest.raises(FysomError):
            fsm.complete()

    def test_cannot_restart_after_completion(self):
        fsm = ExperimentFSM("exp-025")
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        fsm.complete()
        # After terminal, no events allowed
        for evt in ("dispatch", "queue", "start", "complete", "fail"):
            assert fsm.cannot(evt), f"Should not allow {evt} from {fsm.current}"

    def test_cannot_restart_after_cancel(self):
        fsm = ExperimentFSM("exp-026")
        fsm.cancel()
        assert fsm.cannot("dispatch")
        assert fsm.cannot("fail")

    def test_cannot_cancel_after_completed(self):
        fsm = ExperimentFSM("exp-027")
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        fsm.complete()
        with pytest.raises(FysomError):
            fsm.cancel()

    def test_cannot_fail_from_created(self):
        fsm = ExperimentFSM("exp-028")
        with pytest.raises(FysomError):
            fsm.fail()

    def test_trigger_fails_for_invalid_event(self):
        fsm = ExperimentFSM("exp-029")
        with pytest.raises(FysomError):
            fsm.trigger("complete")


class TestTerminalStates:
    """Terminal state detection and constants."""

    def test_terminal_states_set(self):
        assert STATE_COMPLETED in TERMINAL_STATES
        assert STATE_FAILED in TERMINAL_STATES
        assert STATE_CANCELLED in TERMINAL_STATES
        assert STATE_CREATED not in TERMINAL_STATES
        assert STATE_DISPATCHED not in TERMINAL_STATES
        assert STATE_QUEUED not in TERMINAL_STATES
        assert STATE_RUNNING not in TERMINAL_STATES

    def test_is_finished_after_complete(self):
        fsm = ExperimentFSM("exp-030")
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        assert not fsm.is_finished
        fsm.complete()
        assert fsm.is_finished

    def test_is_finished_after_fail(self):
        fsm = ExperimentFSM("exp-031")
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        fsm.fail()
        assert fsm.is_finished

    def test_is_finished_after_cancel(self):
        fsm = ExperimentFSM("exp-032")
        fsm.dispatch()
        fsm.cancel()
        assert fsm.is_finished


class TestSummary:
    """Summary serialization."""

    def test_summary_initial(self):
        fsm = ExperimentFSM("exp-040")
        s = fsm.summary()
        assert s["experiment_id"] == "exp-040"
        assert s["state"] == STATE_CREATED
        assert s["state_label"] == "Created"
        assert not s["is_finished"]
        assert "cancelled" in s["terminal_states"]

    def test_summary_after_complete(self):
        fsm = ExperimentFSM("exp-041")
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        fsm.complete()
        s = fsm.summary()
        assert s["state"] == STATE_COMPLETED
        assert s["is_finished"]

    def test_repr(self):
        fsm = ExperimentFSM("exp-042")
        r = repr(fsm)
        assert "exp-042" in r
        assert "created" in r


class TestCallbacks:
    """Callback invocation during transitions."""

    def test_on_enter_callbacks(self):
        entered = []

        def on_enter_running(fsm_obj, event):
            entered.append(fsm_obj.current)

        fsm = ExperimentFSM(
            "exp-050",
            callbacks={"on_enter_running": on_enter_running},
        )
        fsm.dispatch()
        fsm.queue()
        fsm.start()
        assert len(entered) == 1
        assert entered[0] == STATE_RUNNING

    def test_on_event_callbacks(self):
        after_events = []

        def on_queue(fsm_obj, event):
            after_events.append(event.event)

        fsm = ExperimentFSM(
            "exp-051",
            callbacks={"on_queue": on_queue},
        )
        fsm.dispatch()
        fsm.queue()
        assert len(after_events) == 1
        assert after_events[0] == "queue"

    def test_on_change_state_callback(self):
        changes = []

        def on_change(fsm_obj, event, new_state):
            changes.append((event.event, new_state))

        fsm = ExperimentFSM(
            "exp-052",
            callbacks={"on_change_state": on_change},
        )
        fsm.dispatch()
        fsm.queue()
        assert len(changes) >= 3  # startup + dispatch + queue
        assert changes[0] == ("startup", STATE_CREATED)
        assert changes[1] == ("dispatch", STATE_DISPATCHED)
        assert changes[2] == ("queue", STATE_QUEUED)

    def test_on_error_callback(self):
        errors = []

        def on_error(fsm_obj, exc):
            errors.append(str(exc))

        fsm = ExperimentFSM("exp-053", callbacks={"on_error": on_error})
        with pytest.raises(FysomError):
            fsm.complete()  # invalid transition
        assert len(errors) == 1
        assert "complete" in errors[0]


class TestErrors:
    """Error tracking during invalid transitions."""

    def test_errors_list(self):
        fsm = ExperimentFSM("exp-060")
        assert fsm.errors == []

        with pytest.raises(FysomError):
            fsm.complete()

        assert len(fsm.errors) == 1
        assert "complete" in fsm.errors[0]

    def test_errors_accumulate(self):
        fsm = ExperimentFSM("exp-061")
        with pytest.raises(FysomError):
            fsm.complete()
        with pytest.raises(FysomError):
            fsm.fail()

        assert len(fsm.errors) == 2
