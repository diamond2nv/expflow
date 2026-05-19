#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow metrics — Standard metric registry and PDEBench evaluation functions.

The central metric registry is ``STANDARD_METRICS`` (dict), which maps
metric names to their metadata (type, group, higher_is_better).

Phase 12 enhancements:
- Added PDEBench 6-metric suite + competition score metrics
- Added ``compute_pdebench_metrics()`` — wraps PDEBench's metric_func()
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("expflow")

# ── Standard Metrics Registry ──

STANDARD_METRICS: dict[str, dict[str, Any]] = {
    # ── Score group (higher is better) ──
    "seg_total": {"type": "scalar", "group": "Score", "higher_is_better": True},
    "seg1": {"type": "scalar", "group": "Score", "higher_is_better": True},
    "seg2": {"type": "scalar", "group": "Score", "higher_is_better": True},
    "seg3": {"type": "scalar", "group": "Score", "higher_is_better": True},
    # ── Loss group (lower is better) ──
    "val_mse": {"type": "scalar", "group": "Loss", "higher_is_better": False},
    "val_relmse": {"type": "scalar", "group": "Loss", "higher_is_better": False},
    # ── HyperNOs-style training loss (lower is better) ──
    "val_lprel": {"type": "scalar", "group": "Loss", "higher_is_better": False},
    "val_h1rel": {"type": "scalar", "group": "Loss", "higher_is_better": False},
    # ── PDEBench Error metrics (lower is better) ──
    "val_rmse": {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_nrmse": {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_max_err": {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_bd_err": {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_csv_err": {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_fourier_low": {"type": "scalar", "group": "Fourier", "higher_is_better": False},
    "val_fourier_mid": {"type": "scalar", "group": "Fourier", "higher_is_better": False},
    "val_fourier_high": {"type": "scalar", "group": "Fourier", "higher_is_better": False},
    # ── PDE group (lower is better) ──
    "pde_mean": {"type": "scalar", "group": "PDE", "higher_is_better": False},
    "pde_seg1": {"type": "scalar", "group": "PDE", "higher_is_better": False},
    "pde_seg2": {"type": "scalar", "group": "PDE", "higher_is_better": False},
    "pde_seg3": {"type": "scalar", "group": "PDE", "higher_is_better": False},
    # ── Other ──
    "train_time_min": {"type": "scalar", "group": "Time", "higher_is_better": False},
    "arch_params": {"type": "scalar", "group": "Model", "higher_is_better": False},
    "epochs": {"type": "scalar", "group": "Training", "higher_is_better": False},
}

# Default thresholds (used by validate_metric_threshold for pass/fail gating)
_THRESHOLDS: dict[str, float] = {
    "pde_mean": 18.09,
    "train_time_min": 60.0,
}


# ── Public API ──


def get_registered_metrics() -> dict[str, dict[str, Any]]:
    """Return a copy of the standard metrics registry."""
    return dict(STANDARD_METRICS)


def get_metric_info(name: str) -> dict[str, Any] | None:
    """Return metadata for a single metric, or None if not found."""
    return STANDARD_METRICS.get(name)


def report_standard(
    task: Any | None,
    **metrics: float,
) -> dict[str, float]:
    """Report metrics via clearml Task or return them as a dict.

    For each kwarg, validates the metric name is in STANDARD_METRICS,
    then either calls ``task.report_scalar()`` (if task is provided)
    or just returns a dict.

    Args:
        task: Optional clearml Task instance.
        **metrics: Metric name=value pairs.

    Returns:
        Dict of validated metric name -> value.
    """
    reported: dict[str, float] = {}
    for name, value in metrics.items():
        info = STANDARD_METRICS.get(name)
        if info is None:
            logger.warning("Unknown metric '%s', skipping", name)
            continue
        group = info["group"]
        reported[name] = value
        if task is not None:
            try:
                task.report_scalar(title=group, series=name, value=value, iteration=0)
            except Exception:
                pass
    return reported


def validate_metric_threshold(
    metric_name: str, value: float, threshold: float | None = None
) -> bool:
    """Check if a metric value passes its threshold (pass = below threshold).

    Args:
        metric_name: Key in STANDARD_METRICS.
        value: Actual metric value.
        threshold: Override threshold. Uses _THRESHOLDS[metric_name] if None.

    Returns:
        True if value <= threshold (pass), False otherwise.
    """
    threshold = threshold if threshold is not None else _THRESHOLDS.get(metric_name)
    if threshold is None:
        return True
    if value <= threshold:
        return True
    return False


def set_metric_threshold(metric_name: str, threshold: float) -> None:
    """Set a custom threshold for a metric.

    Args:
        metric_name: Key in STANDARD_METRICS.
        threshold: New threshold value.
    """
    _THRESHOLDS[metric_name] = threshold


def get_metric_threshold(metric_name: str) -> float | None:
    """Get the threshold for a metric, or None if not set.

    Args:
        metric_name: Key in STANDARD_METRICS or _THRESHOLDS.

    Returns:
        Threshold value, or None if no threshold is defined.
    """
    return _THRESHOLDS.get(metric_name)


# ── PDEBench Metric Suite ──


def compute_pdebench_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    initial_step: int = 10,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    iLow: int = 4,
    iHigh: int = 12,
) -> dict[str, float]:
    """Compute the full PDEBench 6-metric suite.

    Wraps ``pdebench.models.metrics.metric_func()`` with ``if_mean=True``
    and returns a ``dict`` keyed by the standard metric names for expflow.

    The 6 metrics:
    - ``val_rmse`` — Root Mean Squared Error
    - ``val_nrmse`` — Normalised RMSE
    - ``val_csv_err`` — Conserved variable (integral) error
    - ``val_max_err`` — Max absolute error
    - ``val_bd_err`` — Boundary error
    - ``val_fourier_{low,mid,high}`` — Fourier-domain error by frequency band

    Args:
        pred: Predicted tensor. Shape (N, *spatial, T, C).
        target: Target tensor. Same shape as pred.
        initial_step: Number of initial context steps to exclude from metric
                      computation (default 10 — standard FNO rollout).
        Lx, Ly, Lz: Domain lengths for Fourier normalisation.
        iLow, iHigh: Fourier band boundaries (low < iLow, mid iLow-iHigh, high > iHigh).

    Returns:
        Dict mapping expflow metric names to scalar float values.
    """
    from pdebench.models.metrics import metric_func as pdebench_metric_func

    result = pdebench_metric_func(
        pred,
        target,
        if_mean=True,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        iLow=iLow,
        iHigh=iHigh,
        initial_step=initial_step,
    )
    # result: (rmse, nrmse, csv, max_err, bd, fourier_3band) — each is
    # a scalar or (3,) tensor. Fourier_3band -> [low, mid, high]
    rmse_val, nrmse_val, csv_val, max_val, bd_val, fourier_val = result

    # Convert to Python floats
    def _to_scalar(t) -> float:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().item() if t.numel() == 1 else t.detach().cpu().mean().item()
        return float(t)

    # Fourier is a (3,) tensor -> unpack
    if isinstance(fourier_val, torch.Tensor):
        fourier_vals = fourier_val.detach().cpu()
    else:
        fourier_vals = torch.as_tensor(fourier_val)

    return {
        "val_rmse": _to_scalar(rmse_val),
        "val_nrmse": _to_scalar(nrmse_val),
        "val_csv_err": _to_scalar(csv_val),
        "val_max_err": _to_scalar(max_val),
        "val_bd_err": _to_scalar(bd_val),
        "val_fourier_low": float(fourier_vals[0]) if fourier_vals.numel() >= 1 else 0.0,
        "val_fourier_mid": float(fourier_vals[1]) if fourier_vals.numel() >= 2 else 0.0,
        "val_fourier_high": float(fourier_vals[2]) if fourier_vals.numel() >= 3 else 0.0,
    }


def compute_rel_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute relative MSE (Rel-MSE) as used in PDE competition scoring.

    ``rel_mse = MSE(pred, target) / (MSE(0, target) + eps)``
    where MSE is taken over all non-batch dimensions.

    This is the primary competition scoring metric.

    Args:
        pred: Predicted tensor, shape (N, *dims).
        target: Target tensor, same shape.
        eps: Small epsilon to avoid division by zero.

    Returns:
        Per-sample relative MSE: shape (N,) tensor.
    """
    N = pred.size(0)
    pred_flat = pred.reshape(N, -1)
    target_flat = target.reshape(N, -1)
    mse_diff = (pred_flat - target_flat).pow(2).mean(dim=1)
    mse_ref = target_flat.pow(2).mean(dim=1)
    return mse_diff / (mse_ref + eps)
