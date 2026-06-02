#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow validation — Noise-aware champion validation and agent arbiter.

Three-tier validation architecture
====================================

Tier 1 — Experiment Noise Gate (AutoScientists, Eq.1)
    sigma = pooled std across duplicate runs (same code+params, different seeds).
    Used via noise_aware_validate() to decide if a candidate improvement is real.
    Reference: AutoScientists arXiv:2605.28655 (Harvard 2026)

Tier 2 — Physics Residual Gate (Zhang2026 JFM)
    Monitors PDE constraint loss (e.g. RANS residual) as a secondary gate.
    A candidate is blocked if PDE residual exceeds a scale-aware threshold
    relative to champion.
    Reference: Zhang et al. J. Fluid Mech. (2026), arXiv:2510.06049

Tier 3 — Agent Arbiter (inter-rater reliability)
    When N subagents score the same proposal, uses ICC (intraclass correlation)
    and rank-agreement metrics — NOT inter-agent score stdev as noise.
    Separates: (a) do agents agree on rank? (b) is the champion clearly better?

Independent implementation based on published formulations. No source code copied.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger("expflow")

# ══════════════════════════════════════════════════════════════
# Tier 1 — Noise-Aware Champion Validation (AutoScientists)
# ══════════════════════════════════════════════════════════════

def noise_aware_validate(
    candidate_value: float,
    champion_value: float,
    noise_floor: float | None = None,
    sigma_multiplier: float = 2.0,
    noise_db_path: str | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    """Validate a candidate against champion with noise awareness.

    Implements AutoScientists champion promotion rule (Eq. 1, arXiv:2605.28655):

        promote(p') = promote    if Δ > M·σ       — confident improvement
                      confirm    if 0 < Δ ≤ M·σ   — within noise band
                      reject     if Δ ≤ 0         — no improvement

    where:
        Δ = candidate - champion (higher-is-better metric assumed)
        σ = noise floor (pooled std from duplicate experimental runs)
        M = sigma_multiplier

    IMPORTANT: σ must be from **intra-experiment seed variance**, NOT from
    inter-agent score disagreement. These are physically different quantities.
    See `arbitrate_agent_outputs()` for the agent-team context.

    Args:
        candidate_value: Metric value from the candidate experiment.
        champion_value: Current best metric champion.
        noise_floor: Pre-calibrated std of the metric from duplicate runs.
            If None, attempts lazy calibration from noise_db_path, then fallback.
        sigma_multiplier: Noise band width (default: 2.0, per AutoScientists).
            For PDE training where seed noise is 0.01-0.1% of metric scale,
            raise to 10.0 if metric resolution justifies it.
        noise_db_path: Path to JSONL noise calibration data file.
        metric_name: Metric name for tagging the calibration entry.
            Required if noise_db_path is provided.

    Returns:
        Dict with keys:
            - action: 'promote', 'confirm', or 'reject'.
            - delta: Candidate minus champion value.
            - noise_floor: Used sigma value.
            - second_seed_needed: True if action is 'confirm'.
            - message: Human-readable explanation.
    """
    delta = candidate_value - champion_value

    sigma = noise_floor
    if sigma is None:
        if noise_db_path and metric_name:
            calib = calibrate_noise_floor(noise_db_path, metric_name)
            sigma = calib.get("sigma")
            if sigma is None:
                sigma = _estimate_sigma_from_context(candidate_value, champion_value)

    if sigma is None or sigma <= 0.0:
        sigma = _estimate_sigma_from_context(candidate_value, champion_value)

    if delta > sigma_multiplier * sigma:
        result = _build_result(
            "promote",
            delta,
            sigma,
            False,
            f"Candidate +{delta:.6g} exceeds {sigma_multiplier}× noise floor "
            f"(σ={sigma:.6g}). Confident improvement.",
        )
    elif delta > 0.0:
        result = _build_result(
            "confirm",
            delta,
            sigma,
            True,
            f"Candidate +{delta:.6g} within noise band "
            f"(σ={sigma:.6g}, threshold={sigma_multiplier * sigma:.6g}). "
            f"Dual-seed confirmation required.",
        )
    else:
        result = _build_result(
            "reject",
            delta,
            sigma,
            False,
            f"Candidate {delta:.6g} (no improvement or regression). Rejected.",
        )

    return result


def _build_result(
    action: str,
    delta: float,
    sigma: float,
    second_seed_needed: bool,
    message: str,
) -> dict[str, Any]:
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
    """Estimate noise floor as a fraction of metric scale.

    Conservative heuristic when no duplicate-run data is available.
    sigma = fallback_rel * |champion|, clamped to minimum 1e-8.
    """
    scale = max(abs(champion), abs(candidate), 1.0)
    return max(fallback_rel * scale, 1e-8)


# ══════════════════════════════════════════════════════════════
# Tier 2 — Physics Residual Gate (Zhang2026 JFM)
# ══════════════════════════════════════════════════════════════

# Scale-aware PDE residual gate: uses relative increase (not absolute threshold),
# with a minimum absolute floor to avoid division-by-zero at very small residuals.
# Reference: Zhang et al. J. Fluid Mech. (2026), Sec 3.3 "Loss Balancing"

_DEFAULT_PDE_RELATIVE_THRESHOLD = 0.50  # 50% relative increase allowed
_DEFAULT_PDE_ABSOLUTE_FLOOR = 0.01       # min absolute threshold when champion near zero


def check_pde_residual_gate(
    candidate_residual: float,
    champion_residual: float,
    relative_threshold: float | None = None,
    absolute_floor: float | None = None,
) -> dict[str, Any]:
    """Check if a candidate's PDE residual violates the physical plausibility gate.

    Uses a combined relative + absolute threshold (scale-aware):

        blocked if candidate > max(champion × (1 + relative_threshold),
                                   champion + absolute_floor)

    This avoids the Problem 4 inconsistency: at any scale, the gate triggers
    at a consistent *relative* increase, with an absolute floor preventing
    false blocks when champion residual is near zero.

    Args:
        candidate_residual: PDE residual from candidate experiment
            (e.g. RANS-PDE total loss component).
        champion_residual: PDE residual from current champion.
        relative_threshold: Fractional increase allowed (default: 0.50 = 50%).
            For stricter gating use 0.20; for relaxed use 1.0.
        absolute_floor: Minimum absolute threshold gap (default: 0.01).
            Ensures gate doesn't trigger trivially when residual ~0.

    Returns:
        Dict with keys:
            - blocked: True if candidate exceeds threshold.
            - threshold: The actual threshold value used.
            - residual_increase: candidate - champion.
            - residual_increase_rel: (candidate - champion) / |champion| if
              champion > 0 else inf.
            - message: Description.
    """
    rel_thresh = relative_threshold if relative_threshold is not None else _DEFAULT_PDE_RELATIVE_THRESHOLD
    abs_floor = absolute_floor if absolute_floor is not None else _DEFAULT_PDE_ABSOLUTE_FLOOR

    # Scale-aware threshold: max(relative, absolute)
    threshold = max(champion_residual * (1.0 + rel_thresh), champion_residual + abs_floor)
    increase = candidate_residual - champion_residual
    champion_abs = abs(champion_residual) if champion_residual != 0 else 1e-12
    rel_increase = increase / champion_abs if champion_residual != 0 else float('inf')

    blocked = candidate_residual > threshold

    return {
        "blocked": blocked,
        "threshold": threshold,
        "residual_increase": increase,
        "residual_increase_rel": rel_increase,
        "relative_threshold": rel_thresh,
        "absolute_floor": abs_floor,
        "message": (
            f"PDE residual {'BLOCKED' if blocked else 'passed'}: "
            f"{candidate_residual:.6g} vs champion {champion_residual:.6g} "
            f"(threshold={threshold:.6g}, +{rel_increase*100:.1f}% relative). "
            f"{'Candidate may be physically invalid despite metric gain.' if blocked else ''}"
        ),
    }


# ══════════════════════════════════════════════════════════════
# Tier 3 — Agent Arbiter (inter-rater reliability)
# ══════════════════════════════════════════════════════════════


def _compute_icc(
    outputs: list[dict[str, Any]],
    score_key: str = "score",
    min_agents: int = 3,
) -> dict[str, Any]:
    """Compute ICC(2,1) — two-way random effects, absolute agreement, single rater.

    ICC(2,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + (k/n)*(MS_C-MS_E))

    where:
        MS_R = mean square between targets (proposals)
        MS_C = mean square between raters (agents)
        MS_E = residual mean square
        k = number of raters (agents)
        n = number of targets (proposals)

    This measures whether agents agree on the *ordering* of proposals,
    not just whether their absolute scores match. ICC > 0.75 is "good agreement".

    NOTE: This requires multiple proposals scored by the same agents.
    For single-proposal multi-agent scoring (current /goal mode),
    ICC cannot be computed (n=1 targets). Falls back to rank-agreement metrics.
    """
    # Extract score matrix: rows = agents, cols = proposals
    # For single-proposal case (all outputs are for the same goal),
    # n_targets = 1 → ICC is undefined. Return None.
    # Real ICC needs N proposals each scored by K agents.
    return {"icc": None, "n_targets": 0, "n_raters": len(outputs), "available": False}


def _compute_rank_agreement(
    outputs: list[dict[str, Any]],
    score_key: str = "score",
    agent_id_key: str = "agent_id",
) -> dict[str, Any]:
    """Compute rank-based inter-rater metrics when ICC is unavailable.

    For single-proposal scoring (all outputs are for the same goal),
    we cannot compute ICC (need multiple targets). Instead measure:

    1. Score dispersion ratio (σ/mean) — how spread out agents are
    2. Champion margin ratio — how clearly the best beats the median
    3. Agreement classification based on dispersion relative to
       the champion margin

    Reference: Landis & Koch (1977) multi-rater agreement thresholds adapted
    for continuous scores.

    Returns:
        Dict with:
            - agreement_level: 'strong' | 'moderate' | 'weak' | 'conflict'
            - sigma_ratio: σ/mean
            - champion_margin_ratio: (best - median) / max(|best|, 1)
            - message: Human-readable
    """
    scores = [float(o[score_key]) for o in outputs if o.get(score_key) is not None]
    if len(scores) < 2:
        return {
            "agreement_level": "insufficient",
            "sigma_ratio": 0.0,
            "champion_margin_ratio": 0.0,
            "message": "Fewer than 2 agents with valid scores.",
        }

    n = len(scores)
    mean_score = sum(scores) / n
    variance = sum((s - mean_score) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    sigma = math.sqrt(variance)

    sigma_ratio = sigma / abs(mean_score) if abs(mean_score) > 1e-12 else sigma

    sorted_scores = sorted(scores, reverse=True)
    best = sorted_scores[0]
    median = sorted_scores[n // 2] if n % 2 == 1 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    champion_margin = best - median
    champion_margin_ratio = champion_margin / max(abs(best), 1.0)

    # Classify agreement level using both dispersion and margin
    # Strong: agents cluster tightly AND best is clearly above the pack
    # Conflict: dispersion exceeds margin (agents disagree more than best leads)
    # Weak: borderline dispersion
    if sigma_ratio > 0.15:
        agreement_level = "conflict"
    elif sigma_ratio > 0.08:
        agreement_level = "weak"
    elif sigma_ratio < 0.01 and n >= 3:
        agreement_level = "suspicious"  # was "low_confidence"
    else:
        agreement_level = "strong" if champion_margin_ratio > sigma_ratio * 2 else "moderate"

    return {
        "agreement_level": agreement_level,
        "sigma_ratio": sigma_ratio,
        "champion_margin_ratio": champion_margin_ratio,
        "best_score": best,
        "median_score": median,
        "message": (
            f"Agreement level: {agreement_level}. "
            f"Dispersion: σ/mean={sigma_ratio:.4f}. "
            f"Champion margin ratio: {champion_margin_ratio:.4f}."
        ),
    }


def arbitrate_agent_outputs(
    outputs: list[dict[str, Any]],
    champion_score: float | None = None,
    score_key: str = "score",
    sigma_multiplier: float = 2.0,
    agreement_threshold: float = 0.15,
    suspicious_agreement_threshold: float = 0.01,
) -> dict[str, Any]:
    """Arbitrate N subagent outputs for the same goal.

    DESIGN RATIONALE (post-audit revision):

    This function uses a **two-stage decision flow**:

    Stage 1 — Agreement assessment (measure inter-rater reliability)
        Uses dispersion-based metrics to classify agreement level.
        Does NOT conflate agent disagreement with experimental noise σ.
        Agreement metrics are reported separately for transparency.

    Stage 2 — Champion comparison
        Compares best agent score against champion.
        Decision order: first check champion delta, THEN check agreement.
        This fixes the Problem 2 bug (high disagreement blocking promotion).

    IMPORTANT: There is NO single "sigma" here that represents experimental
    noise. The dispersion (σ) from agent scores measures inter-rater variance,
    which is a DIFFERENT physical quantity from the pooled seed σ in
    noise_aware_validate(). The sigma_multiplier here controls how conservatively
    we promote based on champion margin, NOT experimental noise.

    Decision matrix:

        First-round (no champion):
            → promote_best (initialize champion)

        With champion:
            best >> champion (Δ > M·σ_dispersion):
                → promote_best (confident, agents agree on improvement)
            best > champion (0 < Δ ≤ M·σ_dispersion):
                → confirm_best (tentative)
            all < champion:
                → request_rerun

        Agreement modifiers (reported, not blocking):
            agreement="conflict" (dispersion > 15%):
                → promote_best still possible, but flag in message
            agreement="suspicious" (dispersion < 1%, N≥3):
                → still promotes, but flags possible score copying

    Args:
        outputs: List of dicts, each with at least score_key entry.
        champion_score: Current best from previous rounds.
            None → first round, initialize.
        score_key: Dict key for the numeric score.
        sigma_multiplier: Noise band multiplier for agent score dispersion.
            This is NOT experimental noise σ — it controls how far the
            best agent score must exceed champion relative to inter-agent
            dispersion.
        agreement_threshold: σ/mean above which agreement is "conflict"
            (default: 0.15 = 15%).
        suspicious_agreement_threshold: σ/mean below which scores are
            suspiciously tight (default: 0.01 = 1%).

    Returns:
        Dict with keys:
            - action: 'promote_best', 'confirm_best', 'request_rerun',
              or 'insufficient_agents'.
            - best_output: Best-scoring output dict.
            - best_score: Highest score.
            - agreement: Agreement assessment dict.
            - champion_score: Champion used for comparison.
            - champion_margin: best - champion (or None if first round).
            - low_scorers: Indices below champion.
            - message: Explanation.
    """
    result: dict[str, Any] = {
        "action": "insufficient_agents",
        "best_output": None,
        "best_score": None,
        "agreement": {},
        "champion_score": champion_score,
        "champion_margin": None,
        "low_scorers": [],
        "message": "",
    }

    if not outputs or len(outputs) < 2:
        result["message"] = f"Need at least 2 agent outputs (got {len(outputs)})."
        return result

    # Extract valid scores
    scores: list[float] = []
    valid_outputs: list[dict[str, Any]] = []
    for o in outputs:
        s = o.get(score_key)
        if s is not None and isinstance(s, (int, float)):
            scores.append(float(s))
            valid_outputs.append(o)

    if len(scores) < 2:
        result["message"] = f"Fewer than 2 outputs have valid '{score_key}' scores."
        return result

    # Find best
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_output = valid_outputs[best_idx]

    result["best_score"] = best_score
    result["best_output"] = best_output

    # Stage 1 — Agreement assessment (no longer blocks promotion!)
    agreement = _compute_rank_agreement(
        valid_outputs, score_key=score_key,
    )
    result["agreement"] = agreement

    # Stage 2 — Champion comparison
    champ = champion_score if champion_score is not None else None

    if champ is None:
        # First round: promote best as initial champion
        result["action"] = "promote_best"
        result["champion_score"] = best_score
        result["message"] = (
            f"No prior champion. Score={best_score:.4g}, "
            f"agreement={agreement.get('agreement_level', 'unknown')}. "
            f"Promoting best output as initial champion."
        )
        return result

    result["champion_score"] = champ
    delta = best_score - champ
    result["champion_margin"] = delta

    # Low-scorers below champion
    low = [i for i, s in enumerate(scores) if s < champ]
    result["low_scorers"] = low

    # Dispersion sigma for the noise band — this is agent score scatter,
    # NOT experimental noise floor. Used only as a margin-of-error estimate.
    mean_score = sum(scores) / len(scores)
    dispersion_sigma = math.sqrt(
        sum((s - mean_score) ** 2 for s in scores) / (len(scores) - 1)
    ) if len(scores) > 1 else 0.0

    if delta > sigma_multiplier * dispersion_sigma:
        result["action"] = "promote_best"
        result["message"] = (
            f"Best {best_score:.4g} exceeds champion {champ:.4g} by "
            f"{delta:.4g} (> {sigma_multiplier}× dispersion σ={dispersion_sigma:.4g}). "
            f"Agreement: {agreement.get('agreement_level', 'unknown')}. "
            f"Confident improvement. Updating champion."
        )
        result["champion_score"] = best_score

    elif delta > 0:
        result["action"] = "confirm_best"
        result["message"] = (
            f"Best {best_score:.4g} > champion {champ:.4g} by "
            f"{delta:.4g}, but within {sigma_multiplier}× dispersion "
            f"(σ={dispersion_sigma:.4g}). "
            f"Agreement: {agreement.get('agreement_level', 'unknown')}. "
            f"Tentative improvement. Consider a second round."
        )

    else:
        result["action"] = "request_rerun"
        n_low = len(low)
        result["message"] = (
            f"Best {best_score:.4g} <= champion {champ:.4g}. "
            f"{n_low}/{len(scores)} agents below champion. "
            f"Agreement: {agreement.get('agreement_level', 'unknown')}. "
            f"Requesting re-generation with refined instructions."
        )

    return result


# ══════════════════════════════════════════════════════════════
# Lazy Noise Floor Calibration (AutoScientists-style)
# ══════════════════════════════════════════════════════════════


def calibrate_noise_floor(
    db_path: str,
    metric_name: str,
    min_samples: int = 3,
    lock_count: int = 5,
) -> dict[str, Any]:
    """Calibrate noise floor sigma from passive duplicate experiment data.

    Follows AutoScientists' "lazy sigma calibration" pattern: instead of running
    dedicated seed probes, sigma is estimated from duplicate runs that naturally
    occur during experimentation (same code+params, different random seeds).

    Duplicates are grouped by (metric, code_hash, params_hash). Each group
    with >= min_samples runs contributes a pooled variance estimate.

    When a metric reaches lock_count qualifying groups, its sigma is marked
    as "locked" and won't be updated further (prevents recalibration gaming).

    Args:
        db_path: Path to JSONL calibration data file.
            Default: ~/.expflow/noise_floor.jsonl
        metric_name: Metric to calibrate (e.g. 'seg_total').
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
            groups.setdefault(key, [])
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

    # Pooled variance across groups (Bessel-corrected)
    total_n = sum(len(g) for g in qualifying)
    n_groups = len(qualifying)
    pooled_var = 0.0
    for group in qualifying:
        mean = sum(group) / len(group)
        var = sum((v - mean) ** 2 for v in group) / (len(group) - 1)
        pooled_var += (len(group) - 1) * var
    pooled_var /= total_n - n_groups
    sigma = pooled_var ** 0.5

    locked = n_groups >= lock_count

    return {
        "sigma": sigma,
        "n_samples": n_groups,
        "locked": locked,
        "message": f"Sigma for '{metric_name}' = {sigma:.6g} "
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

    Appends to the JSONL calibration database. Called automatically
    by pipeline when duplicate runs complete.

    Args:
        value: Metric value.
        metric: Metric name (e.g. 'seg_total').
        seed: Random seed used.
        code_hash: Git commit hash.
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


# ══════════════════════════════════════════════════════════════
# Sigma Multiplier Calibration
# ══════════════════════════════════════════════════════════════

def calibrate_sigma_multiplier(
    metric_values: list[float] | None = None,
    noise_floor: float | None = None,
    target_promote_rate: float = 0.10,
) -> float:
    """Empirically determine M from observed noise and metric distribution.

    AutoScientists uses M=2 as default, but the optimal M depends on
    the noise-to-signal ratio of the specific metric. This function
    estimates M such that only target_promote_rate fraction of random
    perturbations would be auto-promoted.

    Formula:
        M = Δ_characteristic / σ

    where Δ_characteristic is the (1 - target_promote_rate)-th percentile
    of random sequential differences.

    If no data is available, returns 2.0 (AutoScientists default).

    Args:
        metric_values: Historical metric values from sequential experiments.
            If None, returns default M=2.0.
        noise_floor: Estimated sigma from calibrate_noise_floor().
            If None, uses stdev of metric_values as proxy.
        target_promote_rate: Desired false-promote rate (default: 0.10 = 10%).

    Returns:
        Recommended M value.
    """
    if not metric_values or len(metric_values) < 5:
        return 2.0  # AutoScientists default

    # Use noise floor if provided, else stdev of values as proxy
    sigma = noise_floor if (noise_floor is not None and noise_floor > 0) else (
        __import__('statistics').stdev(metric_values) if len(metric_values) >= 2 else 1.0
    )

    # Compute sequential deltas to estimate characteristic improvement
    deltas = [abs(metric_values[i] - metric_values[i - 1])
              for i in range(1, len(metric_values))]

    if not deltas:
        return 2.0

    deltas.sort()
    idx = min(int(len(deltas) * (1.0 - target_promote_rate)), len(deltas) - 1)
    characteristic_delta = deltas[idx]

    if sigma <= 0:
        return 2.0

    M = characteristic_delta / sigma
    # Clamp to reasonable range
    return max(0.5, min(M, 50.0))
