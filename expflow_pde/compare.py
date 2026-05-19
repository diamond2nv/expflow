#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow compare — side-by-side experiment comparison."""

from typing import Any


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
