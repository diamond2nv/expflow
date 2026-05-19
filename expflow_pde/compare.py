#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow compare — side-by-side experiment comparison and score ranking."""

from typing import Any


def _apply_gate(value: float, operator: str, threshold: float) -> bool:
    """Apply a gate filter to a value.

    Args:
        value: The metric value to check.
        operator: Comparison operator (lt, le, gt, ge).
        threshold: The threshold to compare against.

    Returns:
        True if the value passes the gate, False otherwise.
        Unknown operators return True (pass-through).
    """
    if operator == "lt":
        return value < threshold
    elif operator == "le":
        return value <= threshold
    elif operator == "gt":
        return value > threshold
    elif operator == "ge":
        return value >= threshold
    return True  # Unknown operator → pass through


def compare_tasks(task_id_a: str, task_id_b: str) -> dict[str, Any]:
    """Compare two clearml tasks side by side.

    Args:
        task_id_a: First task ID.
        task_id_b: Second task ID.

    Returns:
        Dict with 'a' and 'b' task summaries, or error dict.
    """
    from expflow_pde.clearml import get_task

    try:
        task_a = get_task(task_id_a)
        task_b = get_task(task_id_b)
    except Exception as e:
        return {"error": str(e)}

    return {
        "a": task_a,
        "b": task_b,
    }
