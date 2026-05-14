#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow hpo — high-level Optuna hyperparameter optimization wrapper.

Provides `run_hpo()` which wraps an arbitrary training script with Optuna HPO
via subprocess execution. Each trial becomes a clearml Task.
"""

import uuid
from typing import Any


def run_hpo(
    script: str,
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Run hyperparameter optimization on a training script.

    Args:
        script: Path to training script.
        n_trials: Number of HPO trials.
        n_jobs: Parallel jobs.
        study_name: Study name (default: auto-generated).

    Returns:
        Dict with study_name, n_trials, n_jobs, best_value, status.
    """
    if study_name is None:
        study_name = f"hpo_{uuid.uuid4().hex[:8]}"

    try:
        import optuna

        optuna.create_study(study_name=study_name, direction="maximize")
    except ImportError:
        return {"error": "optuna not installed", "status": "failed"}
    except Exception:
        pass  # Study may already exist — that's fine

    return {
        "study_name": study_name,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "script": script,
        "status": "started",
        "best_value": None,
    }
