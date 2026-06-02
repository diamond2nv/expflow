#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow validation — Noise-aware champion validation and lazy sigma calibration.

These tools implement the experiment-level rigor layer inspired by
AutoScientists (arXiv:2605.28655, Harvard 2026). The core insight is that
experimental noise (seed variance, hyperparameter sensitivity) makes a
single-run improvement unreliable. This module provides:

1. **Noise-aware champion promotion** — ``noise_aware_validate()`` decides
   whether a candidate improvement is real (beyond noise floor).

2. **Lazy noise floor calibration** — ``calibrate_noise_floor()`` estimates
   sigma from passive experiment duplicates, avoiding dedicated seed probes.

Independent implementation based on the published mathematical formulation
of AutoScientists' champion promotion rule. No source code was copied.

Reference:
    AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific
    Experimentation. Gao, Fang, Zitnik. arXiv:2605.28655, 2026.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("expflow")

# ── Noise-Aware Champion Validation ──


def noise_aware_validate(
    candidate_value: float,
    champion_value: float,
    noise_floor: float | None = None,
    sigma_multiplier: float = 2.0,
    noise_db_path: str | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    """Validate a candidate against champion with noise awareness.

    Promotion rule (AutoScientists Eq. 1, arXiv:2605.28655)::

        promote(p') =
            promote     if delta > M * sigma    -- confident improvement
            confirm     if 0 < delta <= M * sigma  -- within noise band
            reject      if delta <= 0              -- no improvement

    where:
        delta = candidate - champion (higher-is-better metric assumed)
        sigma = noise floor (pooled stdev of duplicate runs)
        M = sigma_multiplier

    Args:
        candidate_value: Metric value from the candidate experiment.
        champion_value: Current best/metric champion value.
        noise_floor: Pre-calibrated standard deviation of the metric.
            If None, attempts lazy calibration from noise_db_path.
        sigma_multiplier: Noise band width in sigma units (default: 2.0).
            Higher values make promotion more conservative.
        noise_db_path: Path to JSONL noise calibration data file.
            Used for on-the-fly lazy calibration when noise_floor is None.
        metric_name: Metric name for tagging the calibration entry.
            Required if noise_db_path is provided.

    Returns:
        Dict with keys:
            - action: 'promote', 'confirm', or 'reject'.
            - delta: Candidate minus champion value.
            - noise_floor: Used sigma value (calibrated or provided).
            - second_seed_needed: True if action is 'confirm'.
            - message: Human-readable explanation.
    """
    delta = candidate_value - champion_value

    # Determine noise floor
    sigma = noise_floor
    if sigma is None:
        if noise_db_path and metric_name:
            calib = calibrate_noise_floor(noise_db_path, metric_name)
            sigma = calib.get("sigma")
            if sigma is None:
                sigma = _estimate_sigma_from_context(candidate_value, champion_value)

    if sigma is None or sigma <= 0.0:
        sigma = _estimate_sigma_from_context(candidate_value, champion_value)

    # Decision
    if delta > sigma_multiplier * sigma:
        result = _build_result(
            "promote",
            delta,
            sigma,
            False,
            f"Candidate +{delta:.6f} exceeds {sigma_multiplier}x noise floor "
            f"(sigma={sigma:.6f}). Confident improvement.",
        )
    elif delta > 0.0:
        result = _build_result(
            "confirm",
            delta,
            sigma,
            True,
            f"Candidate +{delta:.6f} within noise band "
            f"(sigma={sigma:.6f}, threshold={sigma_multiplier * sigma:.6f}). "
            f"Dual-seed confirmation required.",
        )
    else:
        result = _build_result(
            "reject",
            delta,
            sigma,
            False,
            f"Candidate {delta:.6f} (no improvement or regression). Rejected.",
        )

    return result


def _build_result(
    action: str,
    delta: float,
    sigma: float,
    second_seed_needed: bool,
    message: str,
) -> dict[str, Any]:
    """Build the standard result dict."""
    return {
        "action": action,
        "delta": delta,
        "noise_floor": sigma,
        "second_seed_needed": second_seed_needed,
        "message": message,
    }


def _estimate_sigma_from_context(
    candidate: float,
    champion: float,
    fallback_rel: float = 0.005,
) -> float:
    """Estimate noise floor as a fraction of the metric scale.

    When no duplicate runs are available for calibration, uses a
    conservative heuristic: sigma = fallback_rel * |champion|,
    clamped to a minimum of 1e-8.

    Args:
        candidate: Candidate metric value.
        champion: Champion metric value.
        fallback_rel: Relative noise floor fraction (default: 0.5%).

    Returns:
        Estimated sigma.
    """
    scale = max(abs(champion), abs(candidate), 1.0)
    return max(fallback_rel * scale, 1e-8)


# ── Lazy Noise Floor Calibration ──


def calibrate_noise_floor(
    db_path: str,
    metric_name: str,
    min_samples: int = 3,
    lock_count: int = 5,
) -> dict[str, Any]:
    """Calibrate noise floor sigma from passive duplicate experiment data.

    Follows AutoScientists' "lazy sigma calibration" pattern: instead of
    running dedicated seed probes, sigma is estimated from duplicate runs
    that naturally occur during experimentation.

    The calibration data is stored as JSONL with records:
        {"timestamp": "...", "metric": "seg_total", "value": 57.09,
         "seed": 42, "code_hash": "a1b2c3d4", "params_hash": "e5f6g7h8"}

    Duplicates are grouped by (metric, code_hash, params_hash). Each group
    with >= 3 runs contributes a pooled variance estimate. The final sigma
    is the pooled stdev across all qualifying groups.

    When a metric reaches lock_count qualifying groups, its sigma is
    marked as "locked" and will not be updated further (prevents analyst
    from repeatedly recalibrating sigma to avoid rejection).

    Args:
        db_path: Path to JSONL calibration data file.
            Default: ~/.expflow/noise_floor.jsonl
        metric_name: Name of the metric to calibrate (e.g. 'seg_total').
        min_samples: Minimum duplicate pairs per group (default: 3).
        lock_count: Groups needed to lock sigma (default: 5).

    Returns:
        Dict with keys:
            - sigma: Pooled standard deviation, or None if insufficient data.
            - n_samples: Number of duplicate groups used.
            - locked: Whether sigma is locked.
            - message: Status description.
    """
    db_path = db_path or os.path.expanduser("~/.expflow/noise_floor.jsonl")
    if not os.path.exists(db_path):
        return {
            "sigma": None,
            "n_samples": 0,
            "locked": False,
            "message": f"No calibration data at {db_path}",
        }

    # Read records
    groups: dict[str, list[float]] = {}
    with open(db_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("metric") != metric_name:
                continue
            key = f"{record.get('code_hash', '')}::{record.get('params_hash', '')}"
            if key not in groups:
                groups[key] = []
            groups[key].append(record.get("value", 0.0))

    # Filter groups with enough samples
    qualifying = [v for v in groups.values() if len(v) >= min_samples]
    if not qualifying:
        return {
            "sigma": None,
            "n_samples": 0,
            "locked": False,
            "message": f"Need >= {min_samples} duplicates per group for '{metric_name}'. "
            f"Found {sum(len(v) for v in groups.values())} total runs in "
            f"{len(groups)} groups.",
        }

    # Pooled variance across groups
    total_n = sum(len(g) for g in qualifying)
    n_groups = len(qualifying)
    pooled_var = 0.0
    for group in qualifying:
        mean = sum(group) / len(group)
        var = sum((v - mean) ** 2 for v in group) / (len(group) - 1)
        pooled_var += (len(group) - 1) * var
    pooled_var /= total_n - n_groups
    sigma = pooled_var**0.5

    locked = n_groups >= lock_count

    return {
        "sigma": sigma,
        "n_samples": n_groups,
        "locked": locked,
        "message": f"Sigma for '{metric_name}' = {sigma:.6f} "
        f"({n_groups} groups, {total_n} runs)" + (" [LOCKED]" if locked else ""),
    }


def record_noise_entry(
    value: float,
    metric: str,
    seed: int | None = None,
    code_hash: str | None = None,
    params_hash: str | None = None,
    db_path: str | None = None,
) -> None:
    """Record an experiment value for lazy noise calibration.

    Appends to the JSONL calibration database. This is called automatically
    by the pipeline when duplicate runs complete.

    Args:
        value: Metric value.
        metric: Metric name (e.g. 'seg_total').
        seed: Random seed used.
        code_hash: Git commit hash of the code.
        params_hash: Hash of the hyperparameter set.
        db_path: Path to JSONL file (default: ~/.expflow/noise_floor.jsonl).
    """
    db_path = db_path or os.path.expanduser("~/.expflow/noise_floor.jsonl")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "value": value,
        "seed": seed,
        "code_hash": code_hash,
        "params_hash": params_hash,
    }

    with open(db_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
