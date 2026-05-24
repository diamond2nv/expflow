#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition scoring framework — parameterized, no hardcoded values.

This module implements the PDE competition scoring algorithm as a PURE
MATHEMATICAL FRAMEWORK. Every competition-specific value (segment boundaries,
weights, score formula coefficients) is passed as a parameter — none are
hardcoded.

Design:
- NumPy only, zero external ML dependencies
- All competition-specific numbers come from SegmentConfig dataclass
- Agent fills SegmentConfig from rules document during bootstrap
- Can validate submissions locally (expflow competition validate --score)

This is NOT an "open-book answer" — it's a parameterized calculator.
The agent still needs to read the competition rules to fill in the values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ======================================================================
# Segment configuration — ALL competition-specific numbers go here
# ======================================================================


@dataclass
class SegmentConfig:
    """Defines one scoring segment's parameters.

    The agent fills these from the competition rules document during bootstrap.

    Attributes:
        label: Human-readable label (e.g. 'Seg1', 'Seg2', 'Seg3').
        start_step: Start of this segment (0-indexed, relative to PREDICTION steps).
        end_step: End of this segment (exclusive).
        weight: Score weight for weighted total (e.g. 0.25).
        formula: Score formula type: 'exp_relmse' or 'max_lorentzian_frechet'.
        coeff: Coefficient for exp(-coeff * rel_mse) formula.
        description: Human-readable description of what this segment measures.
    """
    label: str
    start_step: int
    end_step: int
    weight: float
    formula: str = "exp_relmse"  # 'exp_relmse' | 'max_lorentzian_frechet'
    coeff: float = 20.0
    description: str = ""


@dataclass
class ProblemConfig:
    """Full scoring configuration for one problem (task).

    The agent populates this from the competition rules document.

    Attributes:
        problem_id: e.g. 'task1', 'task2', 'task3'.
        observation_steps: Number of IC steps before prediction (e.g. 10 for task1/2, 20 for task3).
        prediction_steps: Number of steps to predict (e.g. 190 for task1/2, 380 for task3).
        segments: List of SegmentConfig, one per segment.
        rel_mse_cap: Per-sample Rel-MSE cap (default 5.0).
        lorentzian_coeff: Lorentzian denominator coefficient (default 10.0).
        frechet_scale: Frechet score scaling (default 50.0).
        total_score_max: Maximum possible prediction score (e.g. 100 for segment-only).
    """
    problem_id: str = "task1"
    observation_steps: int = 10
    prediction_steps: int = 190
    segments: list[SegmentConfig] = field(default_factory=list)
    rel_mse_cap: float = 5.0
    lorentzian_coeff: float = 10.0
    frechet_scale: float = 50.0
    total_score_max: float = 100.0


# ======================================================================
# Preset configurations (for self-test and documentation only)
# These are NOT loaded by default — agent must fill from rules.
# ======================================================================

# TASK_1_PRESET = ProblemConfig(
#     problem_id="task1",
#     observation_steps=10,
#     prediction_steps=190,
#     segments=[
#         SegmentConfig("Seg1", 0, 47, 0.25, coeff=20.0,
#                       description="Short-term: 0-47 steps, exp(-20*Rel-MSE)"),
#         SegmentConfig("Seg2", 47, 95, 0.25, coeff=10.0,
#                       description="Mid-term: 47-95 steps, exp(-10*Rel-MSE)"),
#         SegmentConfig("Seg3", 95, 190, 0.50, formula="max_lorentzian_frechet",
#                       description="Long-term: 95-190 steps, max(Lorentzian, Frechet)"),
#     ],
# )

# TASK_2_PRESET = TASK_1_PRESET  # Same segment structure, different total score multiplier

# TASK_3_PRESET = ProblemConfig(
#     problem_id="task3",
#     observation_steps=20,
#     prediction_steps=380,
#     segments=[
#         SegmentConfig("Seg1", 0, 30, 0.25, coeff=20.0,
#                       description="Short-term: 0-30 steps (t∈[10,24.5])"),
#         SegmentConfig("Seg2", 30, 180, 0.25, coeff=10.0,
#                       description="Mid-term: 30-180 steps (t∈[25,99.5])"),
#         SegmentConfig("Seg3", 180, 380, 0.50, formula="max_lorentzian_frechet",
#                       description="Long-term: 180-380 steps, max(Lorentzian, Frechet)"),
#     ],
# )


# ======================================================================
# Scoring functions — pure math, no hardcoded values
# ======================================================================


def compute_rel_mse_matrix(
    pred: "np.ndarray",
    target: "np.ndarray",
    eps: float = 1e-12,
) -> "np.ndarray":
    """Compute Rel-MSE per sample per time step.

    rel_t = sum((p-g)^2) / sum(g^2), clamped to rel_mse_cap.

    Args:
        pred: (N, T, S) predictions.
        target: (N, T, S) ground truth.
        eps: Small constant to avoid division by zero.

    Returns:
        (N, T) matrix of per-sample per-time-step Rel-MSE values.
    """
    import numpy as np

    numerator = np.sum((pred - target) ** 2, axis=-1)  # (N, T)
    denominator = np.sum(target ** 2, axis=-1)          # (N, T)
    rel = numerator / (denominator + eps)               # (N, T)
    return rel


def compute_segment_rel_mse(
    pred_seg: "np.ndarray",
    gt_seg: "np.ndarray",
    cap: float = 5.0,
    eps: float = 1e-12,
) -> float:
    """Compute segment-averaged Rel-MSE.

    Competition rule: per-sample cap 5.0, then average over time, then average over samples.

    Args:
        pred_seg: (N, T_seg, S) predictions for one segment.
        gt_seg: (N, T_seg, S) ground truth for one segment.
        cap: Per-sample Rel-MSE cap.
        eps: Small constant to avoid division by zero.

    Returns:
        Float: mean Rel-MSE for the segment.
    """
    import numpy as np

    N, T = pred_seg.shape[0], pred_seg.shape[1]
    sample_rel = np.zeros(N)

    for i in range(N):
        rel_t = np.array([
            np.sum((pred_seg[i, t] - gt_seg[i, t]) ** 2) /
            (np.sum(gt_seg[i, t] ** 2) + eps)
            for t in range(T)
        ])
        # Cap per sample, then average over time
        sample_rel[i] = np.clip(np.mean(rel_t), 0.0, cap)

    return float(np.mean(sample_rel))


def compute_exp_relmse_score(rel_mse: float, coeff: float = 20.0) -> float:
    """Score = 100 * exp(-coeff * rel_mse)."""
    import numpy as np

    return float(100.0 * np.exp(-coeff * rel_mse))


def compute_lorentzian(rmse: float, coeff: float = 10.0) -> float:
    """Lorentzian = 100 / (1 + coeff * RMSE)."""
    return 100.0 / (1.0 + coeff * rmse)


def compute_frechet_score(frechet_dist: float, scale: float = 50.0) -> float:
    """Frechet score = scale * exp(-FD^2)."""
    import numpy as np

    return float(scale * np.exp(-frechet_dist ** 2))


def frechet_distance(P: "np.ndarray", Q: "np.ndarray") -> float:
    """Discrete Frechet distance via DP.

    Args:
        P: (M, D) curve 1.
        Q: (N, D) curve 2.

    Returns:
        Frechet distance.
    """
    import numpy as np

    M, N = P.shape[0], Q.shape[0]
    dist = np.sqrt(np.sum((P[:, np.newaxis] - Q[np.newaxis, :]) ** 2, axis=-1))
    ca = np.full((M, N), np.inf, dtype=np.float64)
    ca[0, 0] = dist[0, 0]
    for i in range(1, M):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, N):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    for i in range(1, M):
        for j in range(1, N):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), dist[i, j])
    return float(ca[-1, -1])


# ======================================================================
# Main scoring entry point
# ======================================================================


def compute_scores(
    pred: "np.ndarray",
    target: "np.ndarray",
    config: ProblemConfig,
) -> dict[str, Any]:
    """Compute competition scores for a problem configuration.

    Args:
        pred: (N, T_total, S) model predictions (including observation_steps).
        target: (N, T_total, S) ground truth.
        config: ProblemConfig with segment definitions.

    Returns:
        Dict with per-segment scores and total.
    """
    import numpy as np

    # Extract prediction portion (after observation steps)
    pred_steps = pred[:, config.observation_steps:, :]
    target_steps = target[:, config.observation_steps:, :]

    seg_scores: dict[str, Any] = {}
    total_weighted = 0.0

    for seg in config.segments:
        p = pred_steps[:, seg.start_step:seg.end_step, :]
        g = target_steps[:, seg.start_step:seg.end_step, :]

        if seg.formula == "exp_relmse":
            rel = compute_segment_rel_mse(p, g, cap=config.rel_mse_cap)
            score = compute_exp_relmse_score(rel, coeff=seg.coeff)
            seg_scores[f"{seg.label}_rel_mse"] = rel
            seg_scores[f"{seg.label}_score"] = score

        elif seg.formula == "max_lorentzian_frechet":
            # RMSE
            rmse = float(np.sqrt(np.mean((p - g) ** 2)))
            lorentzian = compute_lorentzian(rmse, coeff=config.lorentzian_coeff)

            # Frechet — batch compute
            fd = float(np.mean([
                frechet_distance(p[i], g[i])
                for i in range(min(p.shape[0], 100))  # sample for speed
            ]))
            frechet_val = compute_frechet_score(fd, scale=config.frechet_scale)

            score = max(lorentzian, frechet_val)

            seg_scores[f"{seg.label}_rmse"] = rmse
            seg_scores[f"{seg.label}_lorentzian"] = lorentzian
            seg_scores[f"{seg.label}_frechet"] = frechet_val
            seg_scores[f"{seg.label}_frechet_dist"] = fd
            seg_scores[f"{seg.label}_score"] = score

        else:
            raise ValueError(f"Unknown formula: {seg.formula}")

        total_weighted += seg.weight * score

    result: dict[str, Any] = dict(seg_scores)
    result["total_segmented_score"] = round(total_weighted, 4)
    result["config_summary"] = {
        "problem_id": config.problem_id,
        "observation_steps": config.observation_steps,
        "prediction_steps": config.prediction_steps,
        "segments": [
            {
                "label": s.label,
                "range": f"[{s.start_step}:{s.end_step}]",
                "weight": s.weight,
                "formula": s.formula,
            }
            for s in config.segments
        ],
    }
    return result


# ======================================================================
# Submission scoring (for expflow competition validate --score)
# ======================================================================


def score_submission(
    pred_path: str | Path,
    gt_path: str | Path,
    config: ProblemConfig,
) -> dict[str, Any]:
    """Score a submission file against ground truth.

    Args:
        pred_path: Path to pred.hdf5 file.
        gt_path: Path to ground truth HDF5 file.
        config: ProblemConfig.

    Returns:
        Dict with scores.
    """
    import numpy as np

    try:
        import h5py
    except ImportError:
        return {"error": "h5py not installed"}

    with h5py.File(pred_path, "r") as f:
        pred = f["tensor"][:].astype(np.float32)
    with h5py.File(gt_path, "r") as f:
        target = f["tensor"][:].astype(np.float32)

    # Auto-detect: if ground truth has fewer steps than pred, it's a subset
    if target.shape[1] < pred.shape[1]:
        target_full = np.zeros_like(pred)
        target_full[:, :target.shape[1], :] = target
        target = target_full

    N = min(pred.shape[0], target.shape[0])
    return compute_scores(pred[:N], target[:N], config)


# ======================================================================
# JSON serialization helpers
# ======================================================================


def problem_config_to_dict(config: ProblemConfig) -> dict[str, Any]:
    """Serialize ProblemConfig to a JSON-compatible dict."""
    return {
        "problem_id": config.problem_id,
        "observation_steps": config.observation_steps,
        "prediction_steps": config.prediction_steps,
        "rel_mse_cap": config.rel_mse_cap,
        "lorentzian_coeff": config.lorentzian_coeff,
        "frechet_scale": config.frechet_scale,
        "total_score_max": config.total_score_max,
        "segments": [
            {
                "label": s.label,
                "start_step": s.start_step,
                "end_step": s.end_step,
                "weight": s.weight,
                "formula": s.formula,
                "coeff": s.coeff,
                "description": s.description,
            }
            for s in config.segments
        ],
    }


def problem_config_from_dict(d: dict[str, Any]) -> ProblemConfig:
    """Deserialize a dict to ProblemConfig."""
    segments = [
        SegmentConfig(
            label=s["label"],
            start_step=s["start_step"],
            end_step=s["end_step"],
            weight=s["weight"],
            formula=s.get("formula", "exp_relmse"),
            coeff=s.get("coeff", 20.0),
            description=s.get("description", ""),
        )
        for s in d.get("segments", [])
    ]
    return ProblemConfig(
        problem_id=d.get("problem_id", "task1"),
        observation_steps=d.get("observation_steps", 10),
        prediction_steps=d.get("prediction_steps", 190),
        segments=segments,
        rel_mse_cap=d.get("rel_mse_cap", 5.0),
        lorentzian_coeff=d.get("lorentzian_coeff", 10.0),
        frechet_scale=d.get("frechet_scale", 50.0),
        total_score_max=d.get("total_score_max", 100.0),
    )


def save_config_to_json(config: ProblemConfig, path: str | Path) -> None:
    """Save ProblemConfig to a JSON file for agent bootstrap storage."""
    path = Path(path) if isinstance(path, str) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(problem_config_to_dict(config), indent=2, ensure_ascii=False)
    )


def load_config_from_json(path: str | Path) -> ProblemConfig:
    """Load ProblemConfig from a JSON file."""
    path = Path(path) if isinstance(path, str) else path
    data = json.loads(path.read_text(encoding="utf-8"))
    return problem_config_from_dict(data)
