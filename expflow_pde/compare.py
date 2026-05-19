#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow compare — experiment comparison and model selection.

Provides:
- compare_scores(): Multi-task ranking by metric with gating.
- compare_tasks(): Side-by-side comparison of two tasks.
- _apply_gate(): Gate filter for metric-based pass/fail.
"""

from typing import Any


def _apply_gate(value: float, op: str, threshold: float) -> bool:
    """Apply a comparison operator for gating.

    Args:
        value: Metric value.
        op: One of 'lt', 'le', 'gt', 'ge'.
        threshold: Threshold value.

    Returns:
        True if the value passes the gate, False otherwise.
        Unknown operator returns True (pass-through).
    """
    if op == "lt":
        return value < threshold
    elif op == "le":
        return value <= threshold
    elif op == "gt":
        return value > threshold
    elif op == "ge":
        return value >= threshold
    return True  # Unknown op = pass


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


def compare_scores(
    project: str = "PDEBench",
    tags: list[str] | None = None,
    sort_by: str = "seg_total",
    ascending: bool = False,
    gates: list[dict[str, Any]] | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Rank clearml tasks by metric score with optional gating.

    Fetches all tasks in a project (optionally filtered by tags),
    retrieves their latest scalar metrics, applies gates, and returns
    a sorted ranking.

    Args:
        project: clearml project name (default: PDEBench).
        tags: Filter by tags (tasks must have ALL specified tags).
        sort_by: Metric name to sort by (default: seg_total).
        ascending: Sort ascending (default: False = best first).
        gates: List of gate dicts, each with keys:
            - metric: Metric name to check.
            - op: 'lt', 'le', 'gt', 'ge' (comparison operator).
            - value: Threshold value.
            Example: [{"metric": "pde_mean", "op": "lt", "value": 18.09}]
        max_results: Max results (default: 20).

    Returns:
        List of task dicts, each with id, name, status, metrics dict, gates_passed.
    """
    from expflow_pde.clearml import list_tasks

    tasks = list_tasks(project_name=project, tags=tags)

    results: list[dict[str, Any]] = []
    for t in tasks:
        task_id = t["id"]
        metrics = _get_task_metrics(task_id)

        # Apply gates
        gates_passed = True
        gate_results: list[dict[str, Any]] = []
        if gates:
            for gate in gates:
                metric_name = gate.get("metric", "")
                op = gate.get("op", "lt")
                threshold = gate.get("value", 0)
                metric_val = metrics.get(metric_name)
                if metric_val is None:
                    gates_passed = False
                    gate_results.append(
                        {
                            "metric": metric_name,
                            "value": None,
                            "op": op,
                            "threshold": threshold,
                            "passed": False,
                            "reason": "Metric not found",
                        }
                    )
                    continue
                passed = _apply_gate(float(metric_val), op, float(threshold))
                if not passed:
                    gates_passed = False
                gate_results.append(
                    {
                        "metric": metric_name,
                        "value": metric_val,
                        "op": op,
                        "threshold": threshold,
                        "passed": passed,
                    }
                )

        results.append(
            {
                "id": task_id,
                "name": t.get("name", ""),
                "status": t.get("status", ""),
                "metrics": metrics,
                "gates_passed": gates_passed,
                "gate_results": gate_results,
            }
        )

    # Sort by sort_by metric
    def _sort_key(r: dict[str, Any]) -> float:
        val = r.get("metrics", {}).get(sort_by)
        if val is None:
            return float("inf") if not ascending else float("-inf")
        try:
            return float(val)
        except (ValueError, TypeError):
            return float("inf") if not ascending else float("-inf")

    results.sort(key=_sort_key, reverse=not ascending)

    return results[:max_results]


def _get_task_metrics(task_id: str) -> dict[str, float]:
    """Fetch the latest scalar metrics for a clearml task.

    Returns a dict of {metric_name: value} by reading the task's
    last scalar metrics via SDK.

    Args:
        task_id: clearml task ID.

    Returns:
        Dict of metric name -> latest value.
    """
    try:
        from clearml import Task
    except ImportError:
        return {}

    try:
        task = Task.get_task(task_id=task_id)
        scalar_metrics = getattr(task, "get_last_scalar_metrics", lambda: {})()
    except Exception:
        return {}

    # Flatten: group/series -> {series: value}
    flat: dict[str, float] = {}
    for group_name, series_dict in scalar_metrics.items():
        if not isinstance(series_dict, dict):
            continue
        for series_name, metric_info in series_dict.items():
            if isinstance(metric_info, dict):
                value = metric_info.get("last") or metric_info.get("value")
                if value is not None:
                    try:
                        flat[series_name] = float(value)
                    except (ValueError, TypeError):
                        pass
            elif isinstance(metric_info, (int, float)):
                flat[series_name] = float(metric_info)
    return flat
