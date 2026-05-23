#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow optuna integration — Study CRUD, trial ask/tell, visualization.

All functions return JSON-serializable dicts, not optuna SDK objects.
Lazy imports optuna at call time.
"""

import os
from typing import Any


def _get_optuna():
    """Lazy import of optuna."""
    import optuna  # noqa: F401

    return optuna


# ── Study CRUD ──


def create_study(
    study_name: str,
    direction: str = "minimize",
    storage: str | None = None,
) -> dict[str, Any]:
    """Create a new optuna study.

    Args:
        study_name: Unique study name.
        direction: 'minimize' or 'maximize'.
        storage: Optional database URL (e.g. 'sqlite:///optuna.db').

    Returns:
        Dict with study_id, name, direction.
    """
    optuna = _get_optuna()
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
    )
    return {
        "study_id": study._study_id,
        "name": study.study_name,
        "direction": study.direction.name,
    }


def list_studies(storage: str | None = None) -> list[dict[str, Any]]:
    """List all studies.

    Args:
        storage: Optional database URL.

    Returns:
        List of study summary dicts with study_id, name, direction, best_value.
    """
    optuna = _get_optuna()
    summaries = optuna.get_all_study_summaries(storage=storage)
    result = []
    for s in summaries:
        best_value = s.best_trial.value if s.best_trial else None
        best_trial = s.best_trial.number if s.best_trial else None
        result.append(
            {
                "study_id": s._study_id,
                "name": s.study_name,
                "direction": s.direction.name,
                "best_value": best_value,
                "best_trial": best_trial,
            }
        )
    return result


def get_study(
    study_name: str,
    storage: str | None = None,
) -> dict[str, Any]:
    """Get detailed study info including trials.

    Args:
        study_name: Study name.
        storage: Optional database URL.

    Returns:
        Dict with study_id, name, direction, best_trial, trials list.
    """
    optuna = _get_optuna()
    study = optuna.load_study(study_name=study_name, storage=storage)

    best_trial = None
    if study.best_trial:
        best_trial = {
            "number": study.best_trial.number,
            "params": study.best_trial.params,
            "value": study.best_trial.value,
        }

    trials = []
    for t in study.trials:
        trial_dict: dict[str, Any] = {
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "values": t.values,
            "state": str(t.state) if hasattr(t, "state") else "UNKNOWN",
        }
        # Add datetime info if available (Optuna 3.x+)
        if hasattr(t, "datetime_start") and t.datetime_start is not None:
            trial_dict["datetime_start"] = t.datetime_start.isoformat()
        if hasattr(t, "datetime_complete") and t.datetime_complete is not None:
            trial_dict["datetime_complete"] = t.datetime_complete.isoformat()
        if (
            hasattr(t, "datetime_start")
            and hasattr(t, "datetime_complete")
            and t.datetime_start is not None
            and t.datetime_complete is not None
        ):
            trial_dict["duration_sec"] = (t.datetime_complete - t.datetime_start).total_seconds()
        trials.append(trial_dict)

    return {
        "study_id": study._study_id,
        "name": study.study_name,
        "direction": study.direction.name,
        "best_trial": best_trial,
        "trials": trials,
    }


def delete_study(
    study_name: str,
    storage: str | None = None,
) -> dict[str, Any]:
    """Delete a study.

    Args:
        study_name: Study name to delete.
        storage: Optional database URL.

    Returns:
        Dict confirming deletion.
    """
    optuna = _get_optuna()
    optuna.delete_study(study_name=study_name, storage=storage)
    return {
        "status": "deleted",
        "study_name": study_name,
    }


# ── Trial operations ──


def ask_trial(
    study_name: str,
    storage: str | None = None,
) -> dict[str, Any]:
    """Ask for the next trial parameters (distributed optimization).

    Args:
        study_name: Study name.
        storage: Optional database URL.

    Returns:
        Dict with trial_number, params.
    """
    optuna = _get_optuna()
    study = optuna.load_study(study_name=study_name, storage=storage)
    trial = study.ask()
    return {
        "trial_number": trial.number,
        "params": trial.params,
    }


def tell_trial(
    study_name: str,
    trial_number: int,
    value: float,
    storage: str | None = None,
) -> dict[str, Any]:
    """Report a trial result to optuna.

    Args:
        study_name: Study name.
        trial_number: Trial number from ask_trial().
        value: Objective value.
        storage: Optional database URL.

    Returns:
        Dict confirming report.
    """
    optuna = _get_optuna()
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.tell(trial_number, value)
    return {
        "status": "reported",
        "trial_number": trial_number,
        "value": value,
    }


# ── Visualization ──

_PLOT_TYPES = frozenset(
    [
        "history",
        "parallel_coordinate",
        "slice",
        "contour",
    ]
)

_PLOT_MAP = {
    "history": "plot_optimization_history",
    "parallel_coordinate": "plot_parallel_coordinate",
    "slice": "plot_slice",
    "contour": "plot_contour",
}


def plot_study(
    study_name: str,
    plot_type: str = "history",
    output_path: str | None = None,
    storage: str | None = None,
) -> dict[str, Any]:
    """Generate an optimization visualization.

    Args:
        study_name: Study name.
        plot_type: One of 'history', 'parallel_coordinate', 'slice', 'contour'.
        output_path: Path to save the visualization. Defaults to
            '<study_name>_<plot_type>.html' in cwd.
        storage: Optional database URL.

    Returns:
        Dict with status, output_path, plot_type.

    Raises:
        ValueError: If plot_type is unknown.
    """
    if plot_type not in _PLOT_TYPES:
        raise ValueError(f"Unknown plot type '{plot_type}'. Available: {sorted(_PLOT_TYPES)}")

    optuna = _get_optuna()
    study = optuna.load_study(study_name=study_name, storage=storage)

    if output_path is None:
        output_path = os.path.join(os.getcwd(), f"{study_name}_{plot_type}.html")

    plot_fn_name = _PLOT_MAP[plot_type]
    plot_fn = getattr(optuna.visualization, plot_fn_name)
    fig = plot_fn(study)
    fig.write_html(output_path)

    return {
        "status": "saved",
        "output_path": output_path,
        "plot_type": plot_type,
    }
