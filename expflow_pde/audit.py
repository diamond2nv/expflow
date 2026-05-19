#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow audit — experiment validation, compliance checking, report generation."""

from datetime import datetime, timezone
from typing import Any

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
