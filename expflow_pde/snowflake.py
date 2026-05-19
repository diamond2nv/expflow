#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snowflake ID generator for expflow — pure stdlib, no external deps.

Uses yitter snowflake drift algorithm (M1), same as hfpapers-crawler's
paper_store.py implementation. Thread-safe, time-rollback-tolerant,
~500K IDs/sec on single machine.

Configuration (for expflow's autonomous session_id generation):
  - worker_id = 1 (reserved for expflow, 0..63 range)
  - worker_id_bit_length = 6
  - seq_bit_length = 6
  - base_time = 2024-10-04 (same as hfpapers for cross-tooling consistency)

Changelog:
  - 2026-05-19: Ported from hfpapers-crawler for expflow session_id generation
"""

import threading
import time

# ── Types ──────────────────────────────────────────────────

__all__ = ["next_snowflake_id", "snowflake_session_id"]

# ── Constants ──────────────────────────────────────────────

_DRIFT_BASE_TIME = 1_728_000_000_000  # 2024-10-04 (aligned with hfpapers)
_WORKER_ID_BIT_LENGTH: int = 6
_SEQ_BIT_LENGTH: int = 6
_TIMESTAMP_SHIFT: int = _WORKER_ID_BIT_LENGTH + _SEQ_BIT_LENGTH  # 12
_MAX_SEQ_NUMBER: int = (1 << _SEQ_BIT_LENGTH) - 1  # 63
_MIN_SEQ_NUMBER: int = 5  # First 5 reserved for rollback tolerance
_TOP_OVER_COST_COUNT: int = 2000  # Max drift length
_WORKER_ID_MASK: int = (1 << _WORKER_ID_BIT_LENGTH) - 1  # 63


# ── Internal Generator (yitter drift algorithm) ────────────


class _SnowflakeM1:
    """Snowflake drift algorithm — time-rollback-tolerant, high-throughput."""

    def __init__(self, worker_id: int = 1) -> None:
        assert 0 <= worker_id <= _WORKER_ID_MASK, (
            f"worker_id {worker_id} out of range [0, {_WORKER_ID_MASK}]"
        )
        self._worker_id = worker_id
        self._last_time_tick: int = 0
        self._current_seq_number: int = _MIN_SEQ_NUMBER
        self._turn_back_time_tick: int = 0
        self._turn_back_index: int = 0
        self._is_over_cost = False
        self._over_cost_count = 0
        self._lock = threading.Lock()

    # ── Time helpers ──

    @staticmethod
    def _current_tick() -> int:
        return int((time.time_ns() / 1e6) - _DRIFT_BASE_TIME)

    def _next_tick(self) -> int:
        tick = self._current_tick()
        while tick <= self._last_time_tick:
            time.sleep(0.001)
            tick = self._current_tick()
        return tick

    # ── ID calculation ──

    def _calc_id(self, tick: int) -> int:
        self._current_seq_number += 1
        return (
            (tick << _TIMESTAMP_SHIFT)
            | (self._worker_id << _SEQ_BIT_LENGTH)
            | self._current_seq_number
        )

    def _calc_turn_back_id(self, tick: int) -> int:
        self._turn_back_time_tick -= 1
        return (
            (tick << _TIMESTAMP_SHIFT)
            | (self._worker_id << _SEQ_BIT_LENGTH)
            | self._turn_back_index
        )

    # ── Over-cost (high-throughput) path ──

    def _next_over_cost_id(self) -> int:
        current = self._current_tick()
        if current > self._last_time_tick:
            self._last_time_tick = current
            self._current_seq_number = _MIN_SEQ_NUMBER
            self._is_over_cost = False
            self._over_cost_count = 0
            return self._calc_id(self._last_time_tick)

        if self._over_cost_count >= _TOP_OVER_COST_COUNT:
            self._last_time_tick = self._next_tick()
            self._current_seq_number = _MIN_SEQ_NUMBER
            self._is_over_cost = False
            self._over_cost_count = 0
            return self._calc_id(self._last_time_tick)

        if self._current_seq_number > _MAX_SEQ_NUMBER:
            self._last_time_tick += 1
            self._current_seq_number = _MIN_SEQ_NUMBER
            self._is_over_cost = True
            self._over_cost_count += 1
            return self._calc_id(self._last_time_tick)

        return self._calc_id(self._last_time_tick)

    # ── Normal path ──

    def _next_normal_id(self) -> int:
        current = self._current_tick()

        # Time rollback
        if current < self._last_time_tick:
            if self._turn_back_time_tick < 1:
                self._turn_back_time_tick = self._last_time_tick - 1
                self._turn_back_index += 1
                if self._turn_back_index > 4:
                    self._turn_back_index = 1
            return self._calc_turn_back_id(self._turn_back_time_tick)

        self._turn_back_time_tick = min(self._turn_back_time_tick, 0)

        if current > self._last_time_tick:
            self._last_time_tick = current
            self._current_seq_number = _MIN_SEQ_NUMBER
            return self._calc_id(self._last_time_tick)

        if self._current_seq_number > _MAX_SEQ_NUMBER:
            self._last_time_tick += 1
            self._current_seq_number = _MIN_SEQ_NUMBER
            self._is_over_cost = True
            self._over_cost_count = 1
            return self._calc_id(self._last_time_tick)

        return self._calc_id(self._last_time_tick)

    # ── Public ──

    def next_id(self) -> int:
        with self._lock:
            if self._is_over_cost:
                return self._next_over_cost_id()
            return self._next_normal_id()


# ── Module-level singleton ────────────────────────────────

_GENERATOR = _SnowflakeM1(worker_id=1)


def next_snowflake_id() -> int:
    """Generate a unique Snowflake ID (int).

    Uses yitter drift algorithm with worker_id=1 (reserved for expflow).
    Thread-safe.
    """
    return _GENERATOR.next_id()


def snowflake_session_id() -> str:
    """Generate a Langfuse-compatible session_id string.

    Format: ``exp:snow_<int_id>``

    - ``exp`` prefix: generated by expflow (namespace)
    - ``snow_<id>``: snowflake ID, unique and sortable by time

    Langfuse accepts US-ASCII strings < 200 characters — this format
    is ~35 chars, well within limits.
    """
    return f"exp:snow_{next_snowflake_id()}"
