#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow analyze — Task intelligence and strategic advising for PDE competition.

Provides analysis of competition tasks, PDE equations, difficulty assessment,
score projections, and strategic recommendations for research focus.

Usage:
    from expflow_pde.analyze import (
        analyze_task,
        list_task_summaries,
        estimate_score_potential,
        get_strategic_recommendation,
    )
"""

import os
from typing import Any

from expflow_pde.equations import (
    get_equation,
    get_equations,
    list_equations_for_task,
)

# ── Default thresholds per task ──
# Loaded from task_meta.yaml, fallback to these defaults.
_DEFAULT_DIAGNOSE_THRESHOLDS: dict[str, dict[str, float]] = {
    "task1": {
        "ceiling_seg1_high": 70.0,
        "ceiling_seg1_seg2_gap": 10.0,
        "ceiling_seg3_ratio": 0.7,
        "mid_term_gap": 25.0,
        "long_term_seg3_low": 35.0,
        "long_term_seg3_seg2_ratio": 0.6,
        "short_term_seg1_low": 70.0,
        "dist_shift_max_seg": 40.0,
        "dist_shift_mse_high": 0.1,
    },
    "task2": {
        "ceiling_seg1_high": 50.0,
        "ceiling_seg1_seg2_gap": 8.0,
        "ceiling_seg3_ratio": 0.6,
        "mid_term_gap": 20.0,
        "long_term_seg3_low": 25.0,
        "long_term_seg3_seg2_ratio": 0.5,
        "short_term_seg1_low": 50.0,
        "dist_shift_max_seg": 30.0,
        "dist_shift_mse_high": 0.1,
    },
    "task3": {
        "ceiling_seg1_high": 20.0,
        "ceiling_seg1_seg2_gap": 5.0,
        "ceiling_seg3_ratio": 0.5,
        "mid_term_gap": 15.0,
        "long_term_seg3_low": 10.0,
        "long_term_seg3_seg2_ratio": 0.4,
        "short_term_seg1_low": 20.0,
        "dist_shift_max_seg": 15.0,
        "dist_shift_mse_high": 0.1,
    },
}


def _get_diagnose_thresholds(task_id: str = "task1") -> dict[str, float]:
    """Load diagnose thresholds from task_meta.yaml, fallback to defaults."""
    meta = _load_task_meta()
    task_meta = meta.get(task_id, {})
    task_ths = task_meta.get("diagnose_thresholds", {})
    defaults = _DEFAULT_DIAGNOSE_THRESHOLDS.get(task_id, _DEFAULT_DIAGNOSE_THRESHOLDS["task1"])
    merged = dict(defaults)
    merged.update(task_ths)
    return merged


_FALLBACK_SEG1_LOW: dict[str, float] = {
    "task1": 60.0,
    "task2": 40.0,
    "task3": 10.0,
}

_TASK_META_YAML: str | None = None


def _get_task_meta_path() -> str:
    if _TASK_META_YAML is not None:
        return _TASK_META_YAML
    home = os.environ.get("EXPFLOW_HOME", os.path.expanduser("~/.expflow"))
    return os.path.join(home, "task_meta.yaml")


def _load_task_meta() -> dict[str, dict[str, Any]]:
    path = _get_task_meta_path()
    if not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def update_task_meta(task_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = _get_task_meta_path()
    meta = _load_task_meta()
    if task_id not in meta:
        meta[task_id] = {}
    meta[task_id].update(updates)
    try:
        import yaml

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(meta, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Failed to persist task metadata: {e}") from e
    return dict(meta[task_id])


def get_task_meta(task_id: str | None = None) -> dict[str, Any] | dict[str, dict[str, Any]]:
    meta = _load_task_meta()
    if task_id is not None:
        return meta.get(task_id, {})
    return meta


# ── Public API ──


def list_task_summaries() -> list[dict[str, Any]]:
    """Return a summary of all competition tasks with current status.

    Returns:
        List of task summary dicts, one per competition task.
    """
    result: list[dict[str, Any]] = []
    for task_id, meta in sorted((get_task_meta() or {}).items()):
        eqs = list_equations_for_task(task_id)
        eq_names = [e.get("name", "?") for e in eqs]
        dims = [e.get("dim", "?") for e in eqs]

        summary = {
            "task_id": task_id,
            "label": meta["label"],
            "max_score": meta["max_score"],
            "difficulty": meta["difficulty"],
            "priority": meta["priority"],
            "status": meta["status"],
            "equations": eq_names,
            "dimensions": dims,
            "current_best": meta.get("current_best_seg"),
            "current_total": meta.get("current_best_total"),
            "estimated_ceiling": meta["estimated_ceiling"],
            "remaining_headroom": meta["remaining_headroom"],
            "key_bottlenecks": meta["key_bottlenecks"],
            "proven_strategies": meta["proven_strategies"],
            "next_steps": meta["next_steps"],
        }
        result.append(summary)
    return result


def analyze_task(task_id: str) -> dict[str, Any] | None:
    """Get detailed analysis for a specific task.

    Args:
        task_id: 'task1', 'task2', or 'task3'.

    Returns:
        Detailed analysis dict with equations, scoring, strategies, or None.
    """
    meta = get_task_meta(task_id)
    if not meta:
        return None

    eqs = list_equations_for_task(task_id)
    eq_details = []
    for e in eqs:
        eq_details.append(
            {
                "name": e.get("name"),
                "full_name": e.get("full_name"),
                "latex": e.get("latex_short"),
                "dim": e.get("dim"),
                "time_dependent": e.get("time_dependent"),
                "samples": e.get("data_samples"),
                "solver": e.get("solver"),
            }
        )

    return {
        "task_id": task_id,
        "label": meta["label"],
        "max_score": meta["max_score"],
        "difficulty": meta["difficulty"],
        "priority": meta["priority"],
        "status": meta["status"],
        "equations": eq_details,
        "current_best": meta.get("current_best_seg"),
        "current_total": meta.get("current_best_total"),
        "estimated_ceiling": meta["estimated_ceiling"],
        "remaining_headroom": meta["remaining_headroom"],
        "score_composition": _get_score_composition(task_id),
        "key_bottlenecks": meta["key_bottlenecks"],
        "proven_strategies": meta["proven_strategies"],
        "next_steps": meta["next_steps"],
        "score_estimate": estimate_score_potential(task_id),
    }


def _get_score_composition(task_id: str) -> dict[str, Any] | None:
    """Return score breakdown for a task."""
    if task_id == "task1":
        return {
            "prediction": {"max": 75, "current_estimate": 60},
            "train_time": {"max": 35, "current_estimate": 35},
            "inference": {"max": 40, "current_estimate": 40},
            "note": "Training time <60min is achievable; inference <2min is achievable",
        }
    elif task_id == "task2":
        return {
            "prediction": {"max": 150, "current_estimate": 0},
            "note": "Multi-nu score = segmented score × 1.5",
        }
    elif task_id == "task3":
        return {
            "plan_a": "Task1(150) + Task2(150) + T3seg×0.5",
            "plan_b": "Task1(150) + T3seg×2",
            "seg_max": 100,
            "max_via_plan_a": 350,
            "max_via_plan_b": 350,
        }
    return None


def _compute_convergence_estimate(
    seg_history: list[float],
) -> dict[str, Any]:
    """Given ordered Seg total history [s0, s1, ..., sn], estimate ceiling.

    Uses exponential decay of incremental gains to project asymptotic limit.
    Low-gain-decay (near 1.0 = diminishing returns) means near ceiling.

    Args:
        seg_history: Ordered list of Seg totals from most recent experiments.

    Returns:
        Dict with optimistic, expected, conservative, confidence, gain_decay.
    """
    if not seg_history:
        return {
            "optimistic": 0,
            "expected": 0,
            "conservative": 0,
            "confidence": "none",
            "gain_decay": 1.0,
            "seg_history": [],
        }

    if len(seg_history) < 3:
        return {
            "optimistic": round(seg_history[-1] + 5, 1),
            "expected": round(seg_history[-1] + 2, 1),
            "conservative": round(seg_history[-1], 1),
            "confidence": "low",
            "gain_decay": 0.5,
            "seg_history": seg_history,
        }

    # Compute incremental gains
    gains = [seg_history[i + 1] - seg_history[i] for i in range(len(seg_history) - 1)]

    # Estimate gain decay ratio from the last few steps
    gain_decay = 0.5  # default halving
    if len(gains) >= 2:
        decay_ratios = [gains[i + 1] / max(1e-8, gains[i]) for i in range(len(gains) - 1)]
        # Clamp decay to [0, 1) — negative or >1 values are unphysical for
        # the geometric series formula a / (1 - r)
        filtered_ratios = [max(0.0, min(r, 0.99)) for r in decay_ratios if r > 0]
        if filtered_ratios:
            gain_decay = min(filtered_ratios)
        else:
            gain_decay = 0.5

    last_gain = gains[-1] if gains else 0.0

    # If last gain is negative or near-zero, we're already at ceiling
    if last_gain <= 1e-6:
        projected = round(seg_history[-1], 1)
        conf = "high"
        return _convergence_result(projected, seg_history, conf, 0.0)

    # Asymptotic series: last_gain * (1 + r + r^2 + ...) = last_gain / (1 - r)
    denom = max(1.0 - gain_decay, 0.1)
    asymptotic_gain = last_gain / denom
    projected = round(seg_history[-1] + asymptotic_gain, 1)

    # Confidence based on stability of decline + prediction interval width
    if len(gains) >= 3:
        stable = all(
            0.3 <= gains[i + 1] / max(1e-8, gains[i]) <= 2.0 for i in range(len(gains) - 1)
        )
        # Prediction interval: wider = lower confidence
        pi_ratio = asymptotic_gain / max(abs(last_gain), 1.0)
        if stable and pi_ratio <= 5.0:
            conf = "high"
        elif stable:
            conf = "medium"
        else:
            conf = "low"
    else:
        conf = "medium"

    return _convergence_result(projected, seg_history, conf, gain_decay)


def _convergence_result(
    projected: float,
    seg_history: list[float],
    confidence: str,
    gain_decay: float,
) -> dict[str, Any]:
    """Build the convergence estimate result dict."""
    last = seg_history[-1] if seg_history else 0.0
    return {
        "optimistic": round(projected + max(1.0, max(last - projected, 0) * 0.2), 1),
        "expected": projected,
        "conservative": round(max(projected - 2, last), 1),
        "confidence": confidence,
        "gain_decay": round(gain_decay, 3),
        "seg_history": seg_history,
    }


def estimate_score_potential(
    task_id: str,
    seg_history: list[float] | None = None,
) -> dict[str, Any]:
    """Estimate best-case and expected score for a task given current knowledge.

    If seg_history is provided, uses data-driven convergence estimation.
    Otherwise falls back to hardcoded expert estimates.

    Args:
        task_id: 'task1', 'task2', or 'task3'.
        seg_history: Optional ordered list of Seg totals for data-driven estimate.

    Returns:
        Dict with optimistic, expected, conservative estimates.
    """
    if seg_history and len(seg_history) >= 1:
        return _compute_convergence_estimate(seg_history)

    # Fallback: hardcoded expert estimates (used when no history available)
    if task_id == "task1":
        return {
            "optimistic": 148,
            "expected": 145,
            "conservative": 140,
            "confidence": "high",
            "note": "Hardcoded estimate (no seg_history provided). "
            "Seg ~60-65 achievable with more epochs.",
        }
    elif task_id == "task2":
        return {
            "optimistic": 120,
            "expected": 90,
            "conservative": 60,
            "confidence": "low",
            "note": "Hardcoded estimate (no seg_history provided). Nu generalization gap unknown.",
        }
    elif task_id == "task3":
        return {
            "optimistic": 200,
            "expected": 150,
            "conservative": 100,
            "confidence": "low",
            "note": "Hardcoded estimate (no seg_history provided). "
            "Chaotic KS is fundamentally harder.",
        }
    return {"optimistic": 0, "expected": 0, "conservative": 0, "confidence": "none"}


def _get_competition_deadline():
    """Read competition deadline from config or fall back to a sensible default.

    Looks in ~/.expflow/config.yaml under ``competition.deadline``.
    Falls back to ``EXPFLOW_COMPETITION_DEADLINE`` env var or 2026-06-30.
    """
    from datetime import date

    deadline_str: str | None = None
    try:
        from expflow_pde.config import get

        deadline_str = get("competition.deadline")
    except Exception:
        pass
    if not deadline_str:
        deadline_str = os.environ.get("EXPFLOW_COMPETITION_DEADLINE", "")
    if not deadline_str:
        return date(2026, 6, 30)

    # Handle ISO-8601 with timezone
    import re as _re

    m = _re.match(r"(\d{4})-(\d{2})-(\d{2})", deadline_str)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return date(2026, 6, 30)


def _format_deadline_str() -> str:
    """Human-readable deadline string for the recommendation dict."""
    try:
        from expflow_pde.config import get

        raw = get("competition.deadline", "")
        if raw:
            return raw
    except Exception:
        pass
    raw = os.environ.get("EXPFLOW_COMPETITION_DEADLINE", "")
    if raw:
        return raw
    return "2026-06-30 23:59:59 UTC+8"


def get_strategic_recommendation() -> dict[str, Any]:
    """Get overall strategic recommendation across all tasks.

    Adjusts strategy based on remaining days before deadline.
    - T+<=2: Sprint mode -- focus on highest-scored task, no new starts.
    - T+<=5: Mid-range -- start secondary tasks only if primary is stable.
    - T+>5: Normal mode -- full exploration.

    Returns:
        Dict with recommended focus, schedule, and reasoning.
    """
    from datetime import date

    t1 = get_task_meta("task1")
    t2 = get_task_meta("task2")
    t3 = get_task_meta("task3")

    deadline = _get_competition_deadline()
    today = date.today()
    remaining_days = (deadline - today).days
    if remaining_days < 0:
        remaining_days = 0

    deadline_str = _format_deadline_str()

    t1_total = t1.get("current_best_total") or 0
    t1_max = t1.get("max_score") or 150
    t1_room = t1.get("remaining_headroom") or 0
    t3_max = t3.get("max_score") or 350
    t3_room = t3.get("remaining_headroom") or 0
    t2_room = t2.get("remaining_headroom") or 0

    # Sprint mode (<=2 days)
    if remaining_days <= 2:
        if t1_total >= 140:
            return {
                "primary_focus": "task3",
                "primary_rationale": (
                    f"Only {remaining_days} days left. Task 1 is at "
                    f"{t1_total}/{t1_max} near ceiling. "
                    "Invest remaining time in a Task 3 baseline run."
                ),
                "secondary_focus": "task1",
                "secondary_rationale": ("Fine-tune Task 1 submission with current best config."),
                "tertiary_focus": None,
                "tertiary_rationale": (
                    "Task 2 not viable: starting from scratch with "
                    f"{remaining_days} days has near-zero success probability."
                ),
                "suggested_schedule": {
                    "day_1": "Task 3: Run baseline FNO (data already downloaded)",
                    "day_2": "Task 1: Final submission with best config",
                },
                "remaining_days": remaining_days,
                "competition_deadline": deadline_str,
                "submissions_per_day": 1,
                "mode": "sprint",
            }
        else:
            return {
                "primary_focus": "task1",
                "primary_rationale": (
                    f"Only {remaining_days} days left. "
                    f"Task 1 at {t1_total}/{t1_max} with {t1_room} pts "
                    "remaining focus all remaining submissions here."
                ),
                "secondary_focus": None,
                "secondary_rationale": ("No time to start Task 2 or 3 from scratch."),
                "tertiary_focus": None,
                "tertiary_rationale": "",
                "suggested_schedule": {
                    "day_1": "Task 1: Run final HPO sweep",
                    "day_2": "Task 1: Best model submission",
                },
                "remaining_days": remaining_days,
                "competition_deadline": deadline_str,
                "submissions_per_day": 1,
                "mode": "sprint",
            }

    # Mid-range mode (3-5 days)
    if remaining_days <= 5:
        return {
            "primary_focus": "task1",
            "primary_rationale": (
                f"Task 1 at {t1_total}/{t1_max} with {t1_room} pts headroom. "
                f"{remaining_days} days remaining final push."
            ),
            "secondary_focus": "task3",
            "secondary_rationale": (
                f"Task 3 has potential ({t3_room} pts headroom) but limited time. "
                "Run baseline only no iterative optimization."
            ),
            "tertiary_focus": None,
            "tertiary_rationale": (
                f"Task 2 ({t2_room} pts headroom) effectively dead "
                "insufficient time to train multi-nu from scratch."
            ),
            "suggested_schedule": {
                "day_1_2": "Task 1: Final HPO on stability FT + longer epochs",
                "day_3_4": "Task 3: Download data + baseline FNO evaluation",
                "day_5": "Task 1: Final submission",
            },
            "remaining_days": remaining_days,
            "competition_deadline": deadline_str,
            "submissions_per_day": 1,
            "mode": "mid_range",
        }

    # Normal mode (>5 days)
    return {
        "primary_focus": "task1",
        "primary_rationale": (
            f"Task 1 is at ~{t1_total}/{t1_max} "
            f"with {t1_room} points headroom. "
            "Seg improvement from current best is achievable."
        ),
        "secondary_focus": "task3",
        "secondary_rationale": (
            f"Task 3 has {t3_room} points potential at {t3_max} max. "
            "KS equation shares AR stability challenges with Task 1 "
            "stability FT knowledge transfers directly."
        ),
        "tertiary_focus": "task2",
        "tertiary_rationale": (
            f"Task 2 has high remaining headroom ({t2_room}) "
            "but requires generalizing across nu from scratch "
            "highest risk with lowest confidence."
        ),
        "suggested_schedule": {
            "day_1_2": "Task 1: HPO on lambda_stab + longer epochs",
            "day_3_4": "Task 3: Download data + baseline FNO + lambda_2 inference",
            "day_5_6": "Task 1: Submit final optimized version",
            "day_7_8": "Task 3: Apply stability FT + optimize submission",
        },
        "remaining_days": remaining_days,
        "competition_deadline": deadline_str,
        "submissions_per_day": 1,
        "mode": "normal",
    }


def get_equation_analysis(equation_name: str) -> dict[str, Any] | None:
    """Get detailed analysis for a specific PDE equation.

    Args:
        equation_name: Equation key (e.g. 'burgers', 'kuramoto_sivashinsky').

    Returns:
        Dict with equation metadata and strategic context, or None if not found.
    """
    eq = get_equation(equation_name)
    if eq is None:
        return None

    # Find which tasks use this equation
    assigned_tasks: list[str] = []
    for task_id, eq_names in get_task_meta().items():
        task_eqs = list_equations_for_task(task_id)
        if any(e.get("name") == equation_name for e in task_eqs):
            assigned_tasks.append(task_id)

    comp_info = eq.get("competition_info")
    if comp_info:
        info = dict(comp_info)
    else:
        info = {}

    return {
        "name": equation_name,
        "full_name": eq.get("full_name"),
        "latex": eq.get("latex_short"),
        "dim": eq.get("dim"),
        "time_dependent": eq.get("time_dependent"),
        "viscosity_params": eq.get("viscosity_params"),
        "data_samples": eq.get("data_samples"),
        "solver": eq.get("solver"),
        "assigned_tasks": assigned_tasks,
        "competition_info": info,
        "metrics": eq.get("metrics", []),
        "description": eq.get("description"),
    }


def list_all_equations_summary() -> list[dict[str, Any]]:
    """Return a compact summary of all PDE equations in the registry.

    Returns:
        List of equation summary dicts with name, dim, task, difficulty.
    """
    result: list[dict[str, Any]] = []
    for name, eq in sorted(get_equations().items()):
        task = eq.get("competition_task", "none")
        dim = eq.get("dim", "?")
        is_time = eq.get("time_dependent", True)

        # Assign difficulty based on task
        difficulty = "info"
        if task == "task1":
            difficulty = "medium"
        elif task == "task2":
            difficulty = "hard"
        elif task == "task3":
            difficulty = "very_hard"

        result.append(
            {
                "name": name,
                "full_name": eq.get("full_name", ""),
                "latex": eq.get("latex_short", ""),
                "dim": f"{dim}D",
                "time_dependent": is_time,
                "competition_task": task or "-",
                "difficulty": difficulty,
                "viscosity_params": eq.get("viscosity_params", ""),
            }
        )
    return result


# ── Diagnosis Engine ──


def _load_experiment_metrics(
    task_id: str | None = None,
    json_path: str | None = None,
) -> dict | None:
    """Load experiment metrics from clearml task or local JSON.

    Supports two input sources:
    - json_path: local eval JSON file (for unit tests / offline use)
    - task_id: clearml task ID (fetches metrics from clearml server)

    Returns a dict with metrics on success, a dict with _error key on
    clearml connection failure, or None if no input source provided.
    """
    import json
    import os

    if json_path:
        if not os.path.exists(json_path):
            return None
        with open(json_path) as f:
            return json.load(f)

    if task_id:
        try:
            from expflow_pde.clearml import get_task_scalars

            result = get_task_scalars(task_id)
            if result is None:
                return {
                    "_error": (
                        f"clearml task {task_id} returned no scalars: "
                        "task may not exist or have no reported data"
                    )
                }
            return result
        except ImportError:
            return {"_error": "clearml SDK not installed — cannot fetch task scalars"}
        except ConnectionError as e:
            return {"_error": f"clearml server connection failed: {e}"}
        except Exception as e:
            return {"_error": f"clearml error for task {task_id}: {e}"}

    return None


def diagnose_experiment(
    task_id: str | None = None,
    json_path: str | None = None,
) -> dict | None:
    """Analyze experiment results and identify degradation patterns.

    Reads metrics from a clearml task or local eval JSON, then applies
    rule-based diagnosis to classify degradation mode.

    Args:
        task_id: ClearML task ID (alternative to json_path).
        json_path: Path to local eval JSON file (alternative to task_id).

    Returns:
        Dict with seg scores, diagnosis list, and degradation pattern string.
        Returns None if metrics cannot be loaded.
    """
    metrics = _load_experiment_metrics(task_id, json_path)
    if metrics is None:
        return None

    # Check for clearml connection error
    if "_error" in metrics:
        return {
            "seg1": 0.0,
            "seg2": 0.0,
            "seg3": 0.0,
            "total": 0.0,
            "total_mse": 0.0,
            "diagnosis": [f"CLEARML_ERROR: {metrics['_error']}"],
            "degradation_pattern": "error",
            "_connection_error": metrics["_error"],
        }

    # Extract seg scores — handles both flat and nested JSON shapes
    seg = metrics.get("segmented_scores", metrics)
    if isinstance(seg, dict):
        seg1 = seg.get("seg1_score", seg.get("seg1", seg.get("Seg1", 0)))
        seg2 = seg.get("seg2_score", seg.get("seg2", seg.get("Seg2", 0)))
        seg3 = seg.get("seg3_score", seg.get("seg3", seg.get("Seg3", 0)))
        total = seg.get("total_segmented_score", seg.get("total", seg.get("Total", 0)))
    else:
        seg1 = seg2 = seg3 = total = 0

    # Extract total_mse from possibly nested results dict
    results = metrics.get("results", {})
    total_mse = results.get("total_mse", metrics.get("total_mse", metrics.get("Total MSE", 0)))
    if not isinstance(total_mse, (int, float)):
        total_mse = 0.0

    # ── Diagnosis rules (v2: composite + ceiling-aware, task-aware) ──
    diagnosis: list[str] = []
    degradation_pattern = "stable"

    # Normalize types
    s1 = float(seg1) if isinstance(seg1, (int, float)) else 0.0
    s2 = float(seg2) if isinstance(seg2, (int, float)) else 0.0
    s3 = float(seg3) if isinstance(seg3, (int, float)) else 0.0
    mse = float(total_mse) if isinstance(total_mse, (int, float)) else 0.0

    # Load task-aware thresholds
    task = task_id or "task1"
    th = _get_diagnose_thresholds(task)

    ceiling_seg1_high = th["ceiling_seg1_high"]
    ceiling_gap = th["ceiling_seg1_seg2_gap"]
    ceiling_seg3_ratio = th["ceiling_seg3_ratio"]
    mid_gap = th["mid_term_gap"]
    long_seg3_low = th["long_term_seg3_low"]
    long_seg3_seg2_ratio = th["long_term_seg3_seg2_ratio"]
    short_seg1_low = th["short_term_seg1_low"]
    dist_max_seg = th["dist_shift_max_seg"]
    dist_mse_high = th["dist_shift_mse_high"]

    # Detect ceiling: Seg1 below threshold but Seg2 close and Seg3 not collapsing
    is_ceiling = (
        s1 < ceiling_seg1_high
        and s2 > 0
        and s3 > 0
        and (s1 - s2) < ceiling_gap
        and s3 > s2 * ceiling_seg3_ratio
    )
    is_short_term = s1 < short_seg1_low and not is_ceiling

    if is_ceiling:
        diagnosis.append(
            "Score ceiling — Seg uniformly low but stable (model capacity or data limit)"
        )
        degradation_pattern = "ceiling"

    elif is_short_term:
        diagnosis.append("Short-term prediction is weak (Seg1 low)")
        degradation_pattern = "short_term"

    # Medium-term — detect independently (not mutually exclusive)
    mid_term_detected = s1 > 0 and s2 > 0 and (s1 - s2) > mid_gap
    if mid_term_detected:
        diagnosis.append("Medium-term stability degraded (Seg2 drops >%d from Seg1)" % int(mid_gap))

    # Long-term — detect independently
    long_term_detected = s3 < long_seg3_low or (s2 > 0 and s3 < s2 * long_seg3_seg2_ratio)
    if long_term_detected:
        diagnosis.append("Long-term autoregressive collapse (Seg3 collapse)")

    # Resolve composite pattern
    if is_ceiling:
        # Ceiling takes priority — mid/long_term are artifacts of plateau
        pass
    elif is_short_term:
        # Short-term takes priority — mid/long_term are secondary effects
        pass
    elif mid_term_detected and long_term_detected:
        degradation_pattern = "compound_mid_long"
    elif long_term_detected:
        degradation_pattern = "long_term"
    elif mid_term_detected:
        if degradation_pattern == "stable":
            degradation_pattern = "mid_term"

    # Distribution shift (independent check)
    if s1 > 0 and s2 > 0 and s3 > 0 and max(s1, s2, s3) < dist_max_seg and mse < dist_mse_high:
        diagnosis.append("Consistent underperformance — possible IC distribution mismatch")
        if degradation_pattern == "stable":
            degradation_pattern = "distribution_shift"

    if not diagnosis:
        diagnosis.append("No critical degradation detected")

    return {
        "seg1": round(float(seg1), 2) if isinstance(seg1, (int, float)) else 0,
        "seg2": round(float(seg2), 2) if isinstance(seg2, (int, float)) else 0,
        "seg3": round(float(seg3), 2) if isinstance(seg3, (int, float)) else 0,
        "total": round(float(total), 2) if isinstance(total, (int, float)) else 0,
        "total_mse": (round(float(total_mse), 6) if isinstance(total_mse, (int, float)) else 0),
        "diagnosis": diagnosis,
        "degradation_pattern": degradation_pattern,
    }


def suggest_next_params(
    diagnosis: dict,
    current_hparams: dict | None = None,
    task_id: str = "task1",
) -> dict:
    """Suggest next experiment parameters based on diagnosis.

    Uses rule-based suggestions derived from proven strategies
    (sub_step, stability FT, HyperNOs best practices).
    Zero token cost — deterministic rules only.

    Args:
        diagnosis: Output from diagnose_experiment().
        current_hparams: Current experiment's hyperparameters.
        task_id: Competition task ID (for context).

    Returns:
        Dict with suggested_params, rationale list.
    """
    pattern = diagnosis.get("degradation_pattern", "stable")
    seg1 = diagnosis.get("seg1", 0)
    seg3 = diagnosis.get("seg3", 0)

    hp = dict(current_hparams) if current_hparams else {}
    suggestions: dict = {}
    rationale: list[str] = []

    if (
        pattern == "long_term"
        or pattern == "compound_mid_long"
        or (
            isinstance(seg3, (int, float))
            and seg3 < 30
            and isinstance(seg1, (int, float))
            and seg1 > _FALLBACK_SEG1_LOW.get(task_id, 60)
        )
    ):
        current_modes = int(hp.get("n_modes", 12))
        suggestions["n_modes"] = min(current_modes + 4, 24)
        suggestions["num_sub_steps"] = 5
        suggestions["tag"] = "auto_seg3_fix"
        rationale.append(
            f"Seg3 collapse ({seg3:.1f}): Increase n_modes "
            f"{current_modes}->{suggestions['n_modes']} "
            "to capture more spatial frequencies"
        )
        rationale.append(
            "Add sub_step=5 to fix dt mismatch between training (0.01) and inference (0.05)"
        )
        if not hp.get("weight_decay"):
            suggestions["weight_decay"] = 1e-4
            rationale.append("Add weight_decay=1e-4 (HyperNOs Burgers best practice)")

    elif pattern == "mid_term":
        suggestions["tag"] = "auto_mid_fix"
        if "stability_lambda" not in hp or not hp.get("stability_lambda"):
            suggestions["stability_lambda"] = 0.001
            rationale.append("Seg2 drop: add step-wise stability penalty (stability_lambda=0.001)")

    elif pattern == "ceiling":
        current_modes = int(hp.get("n_modes", 12))
        current_width = int(hp.get("width", 32))
        suggestions["n_modes"] = min(current_modes + 4, 24)
        suggestions["width"] = min(current_width + 16, 64)
        suggestions["epochs"] = max(int(hp.get("epochs", 80)), 120)
        suggestions["tag"] = "auto_ceiling_fix"
        rationale.append(
            f"Score ceiling detected: increase model capacity "
            f"(n_modes {current_modes}->{suggestions['n_modes']}, "
            f"width {current_width}->{suggestions['width']}) "
            "to break through plateau"
        )
        rationale.append(
            "Ceiling may also indicate data-limited — consider "
            "data augmentation if capacity increase doesn't help"
        )

    elif pattern == "short_term":
        current_lr = float(hp.get("lr", 0.001))
        suggestions["lr"] = min(current_lr * 2, 0.005)
        suggestions["epochs"] = max(int(hp.get("epochs", 80)), 100)
        suggestions["tag"] = "auto_short_fix"
        rationale.append(
            f"Seg1 low ({seg1:.1f}): increase LR "
            f"{current_lr}->{suggestions['lr']} and extend training"
        )

    else:
        suggestions["tag"] = "auto_hpo_round"
        rationale.append("Experiment stable. Run targeted HPO on remaining strategies.")

    return {
        "suggested_params": suggestions,
        "rationale": rationale,
        "task_id": task_id,
        "degradation_pattern": pattern,
    }


# ── Experiment sync from clearml ──


def sync_task_meta_from_clearml(
    project_name: str = "PDEBench",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch latest completed task metrics from clearml, update task_meta.yaml.

    Groups completed tasks by tag matching (task1/task2/task3), finds the
    max seg_total per group, and updates the YAML current_best values.

    Args:
        project_name: clearml project name to search (default: PDEBench).
        tags: Optional filter tags (e.g. ['task1']).

    Returns:
        Dict with updated tasks and their new best scores.
    """
    from expflow_pde.clearml import list_tasks

    tasks = list_tasks(
        project_name=project_name,
        tags=tags,
        status=["completed"],
    )

    if not tasks:
        return {"updated": [], "message": "No completed tasks found"}

    # Group by inferred task_id from tags
    task_groups: dict[str, list[dict[str, Any]]] = {
        "task1": [],
        "task2": [],
        "task3": [],
    }
    for t in tasks:
        t_tags = t.get("tags", [])
        for task_id in ("task1", "task2", "task3"):
            if task_id in t_tags:
                task_groups[task_id].append(t)

    updated: list[dict[str, Any]] = []
    for task_id, group in task_groups.items():
        if not group:
            continue

        meta = get_task_meta(task_id)

        current_best = meta.get("current_best_total") or 0

        best_task = max(
            group,
            key=lambda x: x.get("last_iteration", 0),
        )
        new_total = best_task.get("last_iteration", 0) or 0

        # Update if better
        if new_total > current_best:
            update_task_meta(
                task_id,
                {
                    "current_best_seg": round(float(new_total), 2),
                    "current_best_total": round(float(new_total), 2),
                    "remaining_headroom": max(
                        0,
                        (meta.get("max_score") or 150) - new_total,
                    ),
                },
            )
            updated.append(
                {
                    "task_id": task_id,
                    "previous_best": current_best,
                    "new_best": new_total,
                    "best_task_id": best_task.get("id"),
                }
            )

    return {"updated": updated, "message": f"Synced {len(updated)} tasks"}
