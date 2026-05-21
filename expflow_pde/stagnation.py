#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stagnation detector — detect when experiment iteration is stalled.

Three stagnation modes:
1. CONSECUTIVE_FAIL — N consecutive experiment failures (default: 3)
2. SCORE_PLATEAU — Last M experiments show no improvement (default: 3, eps=1.0)
3. HYPOTHESIS_SELF_LOCK — Last K hypotheses all rejected (default: 3)

Usage:
    detector = StagnationDetector()
    status = detector.check_iteration(last_experiment_id)
    # -> {"stagnant": False, "patterns": [], "recommendation": "continue"}
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("expflow_pde.stagnation")


class StagnationDetector:
    """Detect when experiment iteration is stalled.

    Uses DispatchDB for experiment status access and hypothesis registry
    for the self-lock pattern. Can operate with or without hypothesis data.
    """

    MODE_CONCLUSIVE_FAIL = "consecutive_fail"
    MODE_SCORE_PLATEAU = "score_plateau"
    MODE_HYPOTHESIS_SELF_LOCK = "hypothesis_self_lock"

    _RECOMMENDATIONS = {
        "continue": "continue",
        "pause": "pause",  # pause and re-evaluate
        "switch_direction": "switch_direction",  # try a different hypothesis
        "stop": "stop",  # stop the iteration entirely
    }

    def __init__(self, db_path: str | None = None):
        from expflow_pde.dispatch_db import DispatchDB

        self._db = DispatchDB(db_path)

    def check_iteration(
        self,
        last_experiment_id: str,
        max_consecutive_fail: int = 3,
        score_plateau_window: int = 3,
        score_plateau_epsilon: float = 1.0,
        hypothesis_self_lock_k: int = 3,
    ) -> dict[str, Any]:
        """Check iteration status against all stagnation patterns.

        Args:
            last_experiment_id: Root experiment ID to trace from.
            max_consecutive_fail: Threshold for MODE_CONCLUSIVE_FAIL.
            score_plateau_window: Window size for MODE_SCORE_PLATEAU.
            score_plateau_epsilon: Max absolute score change for "plateau".
            hypothesis_self_lock_k: Recent K rejected hypotheses threshold.

        Returns:
            dict with keys:
                stagnant: bool
                patterns: list[str] — which stagnation patterns triggered
                details: dict — per-pattern detail
                recommendation: str — continue | pause | switch_direction | stop
        """
        details: dict[str, Any] = {}
        patterns: list[str] = []
        recommendation = self._RECOMMENDATIONS["continue"]

        # 1. Consecutive fail check
        fail_count = self._consecutive_failures(last_experiment_id)
        details["consecutive_fail_count"] = fail_count
        if fail_count >= max_consecutive_fail:
            patterns.append(self.MODE_CONCLUSIVE_FAIL)
            recommendation = self._RECOMMENDATIONS["pause"]

        # 2. Score plateau check
        plateau_info = self._score_plateau_detail(
            last_experiment_id, score_plateau_window, score_plateau_epsilon
        )
        details["plateau"] = plateau_info
        if plateau_info["plateau"]:
            patterns.append(self.MODE_SCORE_PLATEAU)
            if recommendation == self._RECOMMENDATIONS["continue"]:
                recommendation = self._RECOMMENDATIONS["switch_direction"]

        # 3. Hypothesis self-lock check
        lock_count = self._hypothesis_self_lock_count(hypothesis_self_lock_k)
        details["hypothesis_self_lock_count"] = lock_count
        if lock_count >= hypothesis_self_lock_k:
            patterns.append(self.MODE_HYPOTHESIS_SELF_LOCK)
            recommendation = self._RECOMMENDATIONS["stop"]

        # Detailed recommendation explanation
        if patterns:
            details["explanation"] = self._explain(patterns, details)

        return {
            "stagnant": len(patterns) > 0,
            "patterns": patterns,
            "details": details,
            "recommendation": recommendation,
        }

    # ── Private helpers ──

    def _consecutive_failures(self, experiment_id: str) -> int:
        """Count consecutive failed experiments from the root.

        Walks forward from root_id through the experiment tree,
        counting trailing 'failed' statuses.
        """
        root = self._db.get_experiment(experiment_id)
        if not root:
            return 0

        # Walk to root
        root_id = root.get("root_id") or experiment_id

        # Get descendant chain
        children = self._db.get_children(root_id)
        if not children:
            return 0

        # Sort by creation time descending (fallback to id for stable sort)
        sorted_children = sorted(
            children, key=lambda c: (c.get("created_at", ""), c.get("id", "")), reverse=True
        )
        fail_count = 0
        for child in sorted_children:
            summary_raw = child.get("result_summary", "") or ""
            try:
                summary = json.loads(summary_raw) if summary_raw else {}
            except (json.JSONDecodeError, TypeError):
                summary = {}
            status = summary.get("status", "")
            if status == "failed":
                fail_count += 1
            else:
                break
        return fail_count

    def _score_plateau_detail(
        self, experiment_id: str, window: int, epsilon: float
    ) -> dict[str, Any]:
        """Check if the last `window` experiments have plateaued in score.

        Returns:
            dict with: plateau (bool), window_scores (list), deltas (list)
        """
        root = self._db.get_experiment(experiment_id)
        if not root:
            return {"plateau": False, "window_scores": [], "deltas": []}

        root_id = root.get("root_id") or experiment_id
        children = self._db.get_children(root_id)
        if not children:
            return {"plateau": False, "window_scores": [], "deltas": []}

        # Sort by creation time ascending (fallback to id for stable sort)
        sorted_children = sorted(children, key=lambda c: (c.get("created_at", ""), c.get("id", "")))
        last_n = sorted_children[-window:]

        scores: list[float] = []
        for child in last_n:
            summary_raw = child.get("result_summary", "") or ""
            try:
                summary = json.loads(summary_raw) if summary_raw else {}
            except (json.JSONDecodeError, TypeError):
                summary = {}
            # Try total score first, then seg_total
            if "score" in summary:
                scores.append(float(summary["score"]))
            elif "seg_total" in summary:
                scores.append(float(summary["seg_total"]))

        if len(scores) < 2:
            return {"plateau": False, "window_scores": scores, "deltas": []}

        deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        plateau = all(abs(d) < epsilon for d in deltas)

        return {
            "plateau": plateau,
            "window_scores": scores,
            "deltas": deltas,
        }

    def _hypothesis_self_lock_count(self, k: int = 3) -> int:
        """Count recent discarded hypotheses, return 0 if no hypothesis file."""
        from pathlib import Path

        hyp_path = Path.home() / ".expflow" / "hypotheses.yaml"
        if not hyp_path.exists():
            return 0

        try:
            import yaml  # optional
        except ImportError:
            return 0

        with open(hyp_path) as f:
            try:
                data = yaml.safe_load(f) or []
            except yaml.YAMLError:
                return 0

        if not isinstance(data, list):
            return 0

        # Take last K closed hypotheses that are rejected
        closed = [h for h in data if h.get("status") in ("rejected",)]
        return len(closed[-k:])

    @staticmethod
    def _explain(patterns: list[str], details: dict[str, Any]) -> str:
        """Generate human-readable explanation from patterns + details."""
        parts = []
        if StagnationDetector.MODE_CONCLUSIVE_FAIL in patterns:
            n = details.get("consecutive_fail_count", 0)
            parts.append(f"{n} consecutive experiment failures")
        if StagnationDetector.MODE_SCORE_PLATEAU in patterns:
            scores = details.get("plateau", {}).get("window_scores", [])
            parts.append(f"score plateau at {scores}")
        if StagnationDetector.MODE_HYPOTHESIS_SELF_LOCK in patterns:
            n = details.get("hypothesis_self_lock_count", 0)
            parts.append(f"{n} recent hypotheses all rejected")
        return "; ".join(parts) if parts else "no stagnation detected"
