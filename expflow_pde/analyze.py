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

from typing import Any

from expflow_pde.equations import (
    get_equation,
    get_equations,
    list_equations_for_task,
)

# ── Task metadata ──

_TASK_META: dict[str, dict[str, Any]] = {
    "task1": {
        "label": "Task 1 — Burgers (fixed nu=0.001)",
        "max_score": 150,
        "difficulty": "medium",
        "priority": "high",
        "status": "in_progress",
        "current_best_seg": 57.09,
        "current_best_total": 142,
        "estimated_ceiling": 150,
        "remaining_headroom": 8,
        "key_bottlenecks": [
            "Seg3 long-horizon stability (190-step AR rollout)",
            "IC distribution mismatch between train/val",
            "Training time <60min to get full 35/35 time score",
        ],
        "proven_strategies": [
            "sub_step=5: +11.37 Seg (dt mismatch fix)",
            "Stability FT (rollout_stability_penalty): +23.45 Seg",
            "P2 architecture (16/32, 50K params): optimal size",
            "FT lr≈1e-7: preserves pretrained features",
        ],
        "next_steps": [
            "HPO on lambda_stab (0.0001-0.01) for stability penalty",
            "Stability FT more epochs (20-30) for higher Seg",
            "P3 (24/32) baseline + sub_step=5 + stability FT",
        ],
    },
    "task2": {
        "label": "Task 2 — Burgers (multi-nu generalization)",
        "max_score": 150,
        "difficulty": "hard",
        "priority": "low",
        "status": "not_started",
        "current_best_seg": None,
        "current_best_total": None,
        "estimated_ceiling": 130,
        "remaining_headroom": 130,
        "key_bottlenecks": [
            "Multi-nu generalization: nu not provided at inference",
            "Must train from scratch (no pretrained checkpoints)",
            "Different IC distributions across nu values",
        ],
        "proven_strategies": [
            "nu-conditional FNO: embed nu as additional input channel",
            "Multi-decoder or adaptive normalization per nu",
            "Meta-learning across nu: MAML or Reptile",
        ],
        "next_steps": [
            "Investigate nu-conditional input encoding",
            "Run baseline FNO on multi-nu training set",
            "Evaluate generalization gap across nu range",
        ],
    },
    "task3": {
        "label": "Task 3 — Kuramoto-Sivashinsky (bonus, chaotic)",
        "max_score": 350,
        "difficulty": "very_hard",
        "priority": "medium",
        "status": "not_started",
        "current_best_seg": None,
        "current_best_total": None,
        "estimated_ceiling": 250,
        "remaining_headroom": 250,
        "key_bottlenecks": [
            "Chaotic dynamics: exponential error growth",
            "400-step total trajectory, 380-step prediction",
            "lambda_2 not provided at inference",
            "Must train from scratch (no pretrained checkpoints)",
            "Only 2100 total samples (2000 train)",
        ],
        "proven_strategies": [
            "Pseudo-spectral / ETD numerical solver as reference",
            "FNO with spectral normalization for chaotic stability",
            "lambda_2 inference head: estimate lambda from 20-step window",
            "Stability FT directly applicable (步间方差惩罚)",
        ],
        "next_steps": [
            "Download task3 data (800MB)",
            "Run baseline FNO on KS training set",
            "Evaluate lambda_2 inference accuracy",
            "Benchmark: 20-step observation → parameter guess",
        ],
    },
}


# ── Public API ──


def list_task_summaries() -> list[dict[str, Any]]:
    """Return a summary of all competition tasks with current status.

    Returns:
        List of task summary dicts, one per competition task.
    """
    result: list[dict[str, Any]] = []
    for task_id, meta in sorted(_TASK_META.items()):
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
    meta = _TASK_META.get(task_id)
    if meta is None:
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


def estimate_score_potential(task_id: str) -> dict[str, Any]:
    """Estimate best-case and expected score for a task given current knowledge.

    Args:
        task_id: 'task1', 'task2', or 'task3'.

    Returns:
        Dict with optimistic, expected, conservative estimates.
    """
    if task_id == "task1":
        return {
            "optimistic": 148,
            "expected": 145,
            "conservative": 140,
            "confidence": "high",
            "note": "Seg ~60-65 achievable with more epochs; main risk is training time",
        }
    elif task_id == "task2":
        return {
            "optimistic": 120,
            "expected": 90,
            "conservative": 60,
            "confidence": "low",
            "note": "Difficulty depends on nu generalization gap — needs baseline evaluation",
        }
    elif task_id == "task3":
        return {
            "optimistic": 200,
            "expected": 150,
            "conservative": 100,
            "confidence": "low",
            "note": "Chaotic KS is fundamentally harder; stability FT helps but unknown scaling",
        }
    return {"optimistic": 0, "expected": 0, "conservative": 0, "confidence": "none"}


def get_strategic_recommendation() -> dict[str, Any]:
    """Get overall strategic recommendation across all tasks.

    Returns:
        Dict with recommended focus, schedule, and reasoning.
    """
    # Determine focus
    # Task 1 is near ceiling (~142/150), Task 2 and 3 have high headroom
    t1 = _TASK_META["task1"]
    t2 = _TASK_META["task2"]
    t3 = _TASK_META["task3"]

    remaining_days = 8  # competition ends 2026-05-27, today is 2026-05-19

    return {
        "primary_focus": "task1",
        "primary_rationale": (
            f"Task 1 is at ~{t1['current_best_total']}/{t1['max_score']} "
            f"with {t1['remaining_headroom']} points headroom. "
            "Seg improvement from 57→65 is achievable within 8 days."
        ),
        "secondary_focus": "task3",
        "secondary_rationale": (
            f"Task 3 has {t3['remaining_headroom']} points potential at {t3['max_score']} max. "
            "KS equation shares AR stability challenges with Task 1 — "
            "stability FT knowledge transfers directly."
        ),
        "tertiary_focus": "task2",
        "tertiary_rationale": (
            f"Task 2 has high remaining headroom ({t2['remaining_headroom']}) "
            "but requires generalizing across nu from scratch — "
            "highest risk with lowest confidence."
        ),
        "suggested_schedule": {
            "day_1_2": "Task 1: HPO on lambda_stab + longer epochs",
            "day_3_4": "Task 3: Download data + baseline FNO + lambda_2 inference",
            "day_5_6": "Task 1: Submit final optimized version",
            "day_7_8": "Task 3: Apply stability FT + optimize submission",
        },
        "remaining_days": remaining_days,
        "competition_deadline": "2026-05-27 14:00 UTC+8",
        "submissions_per_day": 1,
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
    for task_id, eq_names in _TASK_META.items():
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
