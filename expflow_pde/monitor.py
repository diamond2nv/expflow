#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow monitor — Experiment stagnation detection and dead-end analysis.

This module implements the experiment-level monitoring layer inspired by
AutoScientists' stagnation detection (arXiv:2605.28655). It operates on
experiment trees from the dispatch database and detects when a search
has hit a plateau.

Two detection modes:
1. **KEEP-count stagnation**: When consecutive champion-update experiments
   produce no meaningful metric gain, the search is stagnant.
2. **Single-axis exhaustion**: When most failed experiments cluster on a
   small number of search axes, those axes are exhausted.

Reference:
    AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific
    Experimentation. Gao, Fang, Zitnik. arXiv:2605.28655, 2026.
    Stagnation detection: $\\ge 3$ consecutive KEEP with no real gain,
    $\\ge 8$ DISCARD on $\\le 3$ axes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("expflow")


def detect_stagnation(
    experiment_history: list[dict[str, Any]],
    keep_window: int = 5,
    max_keep_no_progress: int = 3,
    single_axis_threshold: int = 8,
    max_exhausted_axes: int = 3,
    significant_improvement_ratio: float = 0.005,
) -> dict[str, Any]:
    """Detect stagnation in an experiment tree.

    Analyses a list of experiments sorted chronologically and determines
    whether the search is stagnant. Two complementary modes detect
    different failure patterns (AutoScientists mechanisms 1 and 3).

    Args:
        experiment_history: List of experiment dicts, chronologically sorted.
            Each dict must have: status, metric_value (if completed),
            axis (search axis), action (promote/confirm/reject).
        keep_window: How many recent KEEP decisions to check (default: 5).
        max_keep_no_progress: Max KEEPs without significant gain before
            declaring stagnation (default: 3).
        single_axis_threshold: Min discards on a single axis to mark it
            as exhausted (default: 8).
        max_exhausted_axes: How many exhausted axes before triggering
            stagnation (default: 3).
        significant_improvement_ratio: Minimum relative gain to consider
            a KEEP meaningful (default: 0.005 = 0.5%).

    Returns:
        Dict with keys:
            - stagnant: Boolean.
            - reason: Description of stagnation mode.
            - suggested_action: 'regroup', 'explore_new_axis', 'terminate',
              or 'continue'.
            - axis_distribution: Dict of {axis_name: experiment_count}.
            - keep_count: Number of recent KEEPs without progress.
            - exhausted_axes: List of exhausted axis names.
    """
    result: dict[str, Any] = {
        "stagnant": False,
        "reason": "",
        "suggested_action": "continue",
        "axis_distribution": {},
        "keep_count": 0,
        "exhausted_axes": [],
    }

    if not experiment_history:
        return result

    # Build axis distribution
    axis_counts: dict[str, int] = {}
    discard_axes: dict[str, int] = {}
    for exp in experiment_history:
        axis = exp.get("axis", "unknown")
        axis_counts[axis] = axis_counts.get(axis, 0) + 1
        if exp.get("action") in ("reject", "confirm") and exp.get("status") in (
            "failed",
            "pruned",
        ):
            discard_axes[axis] = discard_axes.get(axis, 0) + 1

    result["axis_distribution"] = dict(
        sorted(axis_counts.items(), key=lambda x: -x[1]),
    )

    # Mode 1: KEEP-count stagnation
    recent_keeps: list[dict[str, Any]] = []
    for exp in reversed(experiment_history):
        if exp.get("action") == "promote":
            recent_keeps.append(exp)
        if len(recent_keeps) >= keep_window:
            break

    keep_without_progress = 0
    for i in range(len(recent_keeps) - 1):
        current = recent_keeps[i].get("metric_value", 0.0)
        previous = recent_keeps[i + 1].get("metric_value", 0.0)
        # Skip the first comparison (no previous champion)
        if i == len(recent_keeps) - 1:
            break
        scale = max(abs(previous), 1.0)
        rel_gain = abs(current - previous) / scale
        if rel_gain < significant_improvement_ratio:
            keep_without_progress += 1

    result["keep_count"] = keep_without_progress

    if keep_without_progress >= max_keep_no_progress:
        result["stagnant"] = True
        result["reason"] = (
            f"KEEP-count stagnation: {keep_without_progress} consecutive "
            f"champion updates without significant gain "
            f"(threshold: >{significant_improvement_ratio*100:.1f}% relative progress)."
        )
        result["suggested_action"] = "regroup"

    # Mode 2: Single-axis exhaustion
    exhausted = [
        axis
        for axis, count in discard_axes.items()
        if count >= single_axis_threshold
    ]
    result["exhausted_axes"] = exhausted

    if len(exhausted) >= max_exhausted_axes:
        if result["stagnant"]:
            result["reason"] += (
                f" Additionally, {len(exhausted)} axes are exhausted: {', '.join(exhausted)}."
            )
        else:
            result["stagnant"] = True
            result["reason"] = (
                f"Axis exhaustion: {len(exhausted)} axes have >= "
                f"{single_axis_threshold} discards each: {', '.join(exhausted)}."
            )
        result["suggested_action"] = "explore_new_axis"

    return result


def generate_monitor_report(
    experiment_history: list[dict[str, Any]],
    registry_axes: dict[str, int] | None = None,
) -> str:
    """Generate a human-readable stagnation report.

    Args:
        experiment_history: List of experiment dicts.
        registry_axes: Dead-end axis counts (optional).

    Returns:
        Markdown-formatted report string.
    """
    result = detect_stagnation(experiment_history)

    lines = [
        "## Experiment Monitor Report",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Status:** {'⚠️ STAGNANT' if result['stagnant'] else '✅ Active'}",
        f"**Reason:** {result['reason'] or 'No issues detected.'}",
        f"**Suggested action:** `{result['suggested_action']}`",
        "",
        "### Experiment Axis Distribution",
    ]

    for axis, count in result.get("axis_distribution", {}).items():
        exhausted_mark = " ⚠️" if axis in result.get("exhausted_axes", []) else ""
        lines.append(f"- **{axis}**: {count} runs{exhausted_mark}")

    if result.get("keep_count", 0) > 0:
        lines.extend(
            [
                "",
                f"**KEEPs without significant progress:** {result['keep_count']}",
            ],
        )

    if registry_axes:
        lines.extend(
            [
                "",
                "### Dead-End Registry (all sessions)",
            ],
        )
        for axis, count in sorted(
            registry_axes.items(), key=lambda x: -x[1],
        )[:10]:
            lines.append(f"- **{axis}**: {count} dead ends")

    return "\n".join(lines)
