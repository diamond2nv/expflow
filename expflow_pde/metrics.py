#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow metrics — standardized metric registry for PDEBench experiments.

Provides a central registry of known metrics with metadata (group, threshold,
higher_is_better), and a report_standard() helper to report them to clearml.

Usage:
    from expflow_pde.metrics import get_registered_metrics, report_standard

    # In a training script:
    report_standard(seg_total=57.09, pde_mean=20.29, train_time_min=50.5)
"""

from typing import Any

# ── Metric descriptor ──

STANDARD_METRICS: dict[str, dict[str, Any]] = {
    "seg_total": {
        "type": "scalar",
        "group": "Score",
        "higher_is_better": True,
        "description": "Total segment score (primary competition metric)",
    },
    "seg1": {
        "type": "scalar",
        "group": "Score",
        "higher_is_better": True,
        "description": "Segment 1 score",
    },
    "seg2": {
        "type": "scalar",
        "group": "Score",
        "higher_is_better": True,
        "description": "Segment 2 score",
    },
    "seg3": {
        "type": "scalar",
        "group": "Score",
        "higher_is_better": True,
        "description": "Segment 3 score",
    },
    "val_mse": {
        "type": "scalar",
        "group": "Loss",
        "higher_is_better": False,
        "description": "Validation MSE",
    },
    "val_relmse": {
        "type": "scalar",
        "group": "Loss",
        "higher_is_better": False,
        "description": "Validation relative MSE",
    },
    "pde_mean": {
        "type": "scalar",
        "group": "PDE",
        "higher_is_better": False,
        "threshold": 18.09,
        "description": "Mean PDE residual (competition gate: <18.09)",
    },
    "pde_seg1": {
        "type": "scalar",
        "group": "PDE",
        "higher_is_better": False,
        "description": "PDE residual for segment 1",
    },
    "pde_seg2": {
        "type": "scalar",
        "group": "PDE",
        "higher_is_better": False,
        "description": "PDE residual for segment 2",
    },
    "pde_seg3": {
        "type": "scalar",
        "group": "PDE",
        "higher_is_better": False,
        "description": "PDE residual for segment 3",
    },
    "train_time_min": {
        "type": "scalar",
        "group": "Time",
        "higher_is_better": False,
        "threshold": 60,
        "description": "Training time in minutes (competition limit: <60min)",
    },
    "arch_params": {
        "type": "scalar",
        "group": "Model",
        "higher_is_better": False,
        "description": "Model architecture parameter count",
    },
    "epochs": {
        "type": "scalar",
        "group": "Training",
        "higher_is_better": False,
        "description": "Number of training epochs",
    },
}

# ── Public API ──


def get_registered_metrics() -> dict[str, dict[str, Any]]:
    """Return a copy of the standard metrics registry."""
    return dict(STANDARD_METRICS)


def get_metric_info(name: str) -> dict[str, Any] | None:
    """Get metadata for a single metric by name, or None if unknown."""
    info = STANDARD_METRICS.get(name)
    return dict(info) if info else None


def report_standard(
    task: Any | None = None,
    **kwargs: float,
) -> dict[str, float]:
    """Report standard metrics to clearml (or return a dict if no task).

    Each keyword argument must match a metric name in STANDARD_METRICS.
    Reports are sent to clearml via task.report_scalar(group, name, value, iteration=0).

    Args:
        task: Optional clearml Task object. If None, metrics are returned as a dict.
        **kwargs: Metric name=value pairs (e.g. seg_total=57.09).

    Returns:
        Dict of {metric_name: value} that were processed.

    Raises:
        ValueError: If an unknown metric name is passed.
    """
    reported: dict[str, float] = {}
    for name, value in kwargs.items():
        info = STANDARD_METRICS.get(name)
        if info is None:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Registered metrics: {', '.join(sorted(STANDARD_METRICS.keys()))}"
            )
        reported[name] = float(value)

        if task is not None:
            group = info.get("group", "default")
            task.report_scalar(title=group, series=name, value=float(value), iteration=0)

    return reported


# ── Validation helpers ──


def validate_metric_threshold(
    name: str,
    value: float,
) -> dict[str, Any]:
    """Check if a metric value passes its threshold (if defined).

    Args:
        name: Metric name.
        value: Metric value to check.

    Returns:
        Dict with name, value, threshold, passed, detail.
    """
    info = STANDARD_METRICS.get(name)
    if info is None:
        return {
            "name": name,
            "value": value,
            "threshold": None,
            "passed": True,
            "detail": "Unknown metric",
        }

    threshold = info.get("threshold")
    if threshold is None:
        return {
            "name": name,
            "value": value,
            "threshold": None,
            "passed": True,
            "detail": "No threshold",
        }

    higher_is_better = info.get("higher_is_better", True)
    if higher_is_better:
        passed = float(value) >= threshold
    else:
        passed = float(value) <= threshold

    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": passed,
        "detail": f"{'PASS' if passed else 'FAIL'}: {name}={value} {'≥' if higher_is_better else '≤'} {threshold}",
    }
