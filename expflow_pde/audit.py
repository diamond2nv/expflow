#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow audit — experiment validation, compliance checking, report generation."""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_COMPLIANCE_VALUES = frozenset(["allowed", "forbidden"])


def validate_experiment(
    experiment_id: str,
    config_snapshot: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Run reproducibility checks on an experiment.

    Args:
        experiment_id: The experiment ID.
        config_snapshot: Training config dict.
        metrics: Result metrics dict.

    Returns:
        Dict with experiment_id, checks (list), timestamp.
    """
    now = datetime.now(timezone.utc).isoformat()
    checks = []

    # Config check
    config_has_keys = bool(config_snapshot)
    checks.append(
        {
            "name": "config_present",
            "passed": config_has_keys,
            "detail": f"Config has {len(config_snapshot)} keys"
            if config_has_keys
            else "Config is empty",
        }
    )

    # Metrics check
    metrics_has_keys = bool(metrics)
    checks.append(
        {
            "name": "metrics_present",
            "passed": metrics_has_keys,
            "detail": f"Metrics has {len(metrics)} keys"
            if metrics_has_keys
            else "Metrics is empty",
        }
    )

    return {
        "experiment_id": experiment_id,
        "checks": checks,
        "timestamp": now,
    }


def check_dataset_compliance(
    dataset_name: str,
    compliance: str,
) -> dict[str, Any]:
    """Check if a dataset complies with competition rules.

    Args:
        dataset_name: Dataset name.
        compliance: 'allowed' or 'forbidden'.

    Returns:
        Dict with dataset_name, compliance, compliant.

    Raises:
        ValueError: If compliance value is invalid.
    """
    if compliance not in _COMPLIANCE_VALUES:
        raise ValueError(
            f"compliance must be one of {sorted(_COMPLIANCE_VALUES)}, got '{compliance}'"
        )

    return {
        "dataset_name": dataset_name,
        "compliance": compliance,
        "compliant": compliance == "allowed",
    }


def validate_competition_rules(
    task_metrics: dict[str, float],
    task_params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate PDEBench competition rules against experiment metrics and params.

    Rules checked:
      - ``seg_total``: primary score (reported, no threshold gating here)
      - ``pde_mean``: PDE residual gate < 18.09 (from STANDARD_METRICS threshold)
      - ``train_time_min``: training time limit < 60 minutes
      - ``sub_step``: must be present and > 0 (enforces dt correction)

    Args:
        task_metrics: Flat dict of metric names to float values (e.g.
            ``{"seg_total": 57.09, "pde_mean": 18.29, "train_time_min": 50.5}``).
        task_params: Optional clearml task parameters dict for sub_step check.

    Returns:
        Dict with:
            ``all_pass``: bool
            ``checks``: list of per-rule check dicts (name, label, value, passed, detail)
            ``metrics``: input metrics dict
    """
    from expflow_pde.metrics import (
        get_metric_threshold,
        get_registered_metrics,
        validate_metric_threshold,
    )

    checks: list[dict[str, Any]] = []
    _ = get_registered_metrics()  # trigger lazy import for side effect

    rule_defs: list[tuple[str, str]] = [
        ("seg_total", "Primary score"),
        ("pde_mean", "PDE residual (gate: <18.09)"),
        ("train_time_min", "Training time (limit: <60min)"),
    ]

    for metric_name, label in rule_defs:
        value = task_metrics.get(metric_name)
        passed = True
        detail = ""

        if value is None:
            passed = False
            detail = f"metric '{metric_name}' not found"
            checks.append(
                {
                    "name": metric_name,
                    "label": label,
                    "value": None,
                    "passed": passed,
                    "detail": detail,
                }
            )
            continue

        # Use validate_metric_threshold which internally checks _THRESHOLDS
        passed = validate_metric_threshold(metric_name, float(value))
        thresh = get_metric_threshold(metric_name)
        if thresh is not None:
            fv = f"{value}"
            detail = f"{fv} (gate: <{thresh})"
        else:
            detail = f"{value} (no threshold set)"

        checks.append(
            {
                "name": metric_name,
                "label": label,
                "value": value,
                "passed": passed,
                "detail": detail,
            }
        )

    # sub_step check
    sub_step_found = False
    if task_params:
        for key in task_params or {}:
            if "sub_step" in key.lower() or "substep" in key.lower():
                val = task_params[key]
                try:
                    sub_step_found = int(val) > 0
                except (ValueError, TypeError):
                    sub_step_found = val is not None
                break

    checks.append(
        {
            "name": "sub_step",
            "label": "sub_step parameter",
            "value": sub_step_found,
            "passed": sub_step_found,
            "detail": "present and > 0" if sub_step_found else "missing or zero",
        }
    )

    all_pass = all(c["passed"] for c in checks)

    return {
        "all_pass": all_pass,
        "checks": checks,
        "metrics": task_metrics,
    }


def generate_report(
    experiment_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Generate a Markdown experiment report.

    Args:
        experiment_id: Experiment ID.
        config: Training config dict.
        metrics: Result metrics dict.

    Returns:
        Dict with experiment_id, markdown (string).
    """
    validation = validate_experiment(experiment_id, config, metrics)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# Experiment Report: {experiment_id}",
        "",
        f"**Generated:** {now}",
        "",
        "## Validation",
        "",
    ]

    for check in validation["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- [{status}] {check['name']}: {check['detail']}")

    if config:
        lines.extend(["", "## Configuration", ""])
        for k, v in config.items():
            lines.append(f"- `{k}`: `{v}`")

    if metrics:
        lines.extend(["", "## Metrics", "", "| Metric | Value |", "|--------|-------|"])
        for k, v in metrics.items():
            lines.append(f"| {k} | {v} |")

    lines.append("")

    return {
        "experiment_id": experiment_id,
        "markdown": "\n".join(lines),
    }
