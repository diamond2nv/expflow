#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow hpo — Hyperparameter optimization runner.

Supports three modes:
1. **Local serial** (default): runs trials sequentially on this machine.
2. **Distributed via clearml ask/tell**: each trial as independent clearml Task.
3. **HyperParameterOptimizer** (recommended for production): uses ClearML's
   native Optuna integration — auto-creates/clones/enqueues tasks.

The training script must:
- Accept hyperparameters as CLI arguments: `--lr=0.001 --epochs=80`
- Report metrics via clearml `Task.report_scalar()` for objective collection
- Or output METRIC:<name>=<value> to stdout for local mode

Phase 11 enhancements:
- Optuna HyperbandPruner for early stopping of poor trials
- ClearML HyperParameterOptimizer native integration (mode 3)
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any

# ── Default search space ──

_DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "lr": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "batch_size": {"type": "int", "low": 16, "high": 256, "step": 16},
    "epochs": {"type": "int", "low": 20, "high": 150, "step": 10},
    "weight_decay": {"type": "float", "low": 1e-8, "high": 1e-3, "log": True},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
    "sub_step": {"type": "int", "low": 1, "high": 10, "step": 1},
    "width": {"type": "int", "low": 16, "high": 128, "step": 16},
    "n_layers": {"type": "int", "low": 2, "high": 8, "step": 1},
    "modes": {"type": "int", "low": 8, "high": 32, "step": 4},
}

# Supported pruner types
_PRUNER_TYPES: dict[str, Any] = {}


def _get_pruner(pruner_name: str | None = None) -> Any:
    """Get an Optuna pruner instance.

    Args:
        pruner_name: One of 'hyperband', 'median', 'percentile', or None (no pruner).

    Returns:
        Optuna pruner instance or None.
    """
    if pruner_name is None:
        return None

    import optuna.pruners

    pruner_map = {
        "hyperband": lambda: optuna.pruners.HyperbandPruner(
            min_epochs=10,
            reduction_factor=3,
            max_epochs=200,
        ),
        "median": lambda: optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
        "percentile": lambda: optuna.pruners.PercentilePruner(
            percentile=25.0,
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    }
    if pruner_name in pruner_map:
        return pruner_map[pruner_name]()
    raise ValueError(f"Unknown pruner '{pruner_name}'. Options: {list(pruner_map.keys())}")


# ── Public API ──


def get_search_space() -> dict[str, dict[str, Any]]:
    """Return the default hyperparameter search space definition."""
    return dict(_DEFAULT_SEARCH_SPACE)


def get_available_pruners() -> list[str]:
    """Return list of available pruner names."""
    return ["hyperband", "median", "percentile"]


def run_hpo(
    script: str,
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str | None = None,
    search_space: dict[str, dict[str, Any]] | None = None,
    storage: str | None = None,
    direction: str = "maximize",
    objective_metric: str = "seg_total",
    timeout_minutes: float | None = None,
    distributed: bool = False,
    queue: str | None = None,
    project: str = "PDEBench",
    pruner: str | None = "hyperband",
    use_hpo_optimizer: bool = False,
) -> dict[str, Any]:
    """Run hyperparameter optimization.

    Three modes:
    1. **local** (default): subprocess-based, no clearml.
    2. **distributed** (distributed=True): ask/tell + clearml Task clone/enqueue.
    3. **hpo_optimizer** (use_hpo_optimizer=True): ClearML HyperParameterOptimizer.

    Args:
        script: Path to training script.
        n_trials: Number of trials.
        n_jobs: Parallel/concurrent jobs.
        study_name: Optuna study name (auto-generated if None).
        search_space: Hyperparameter search space (default if None).
        storage: Optuna storage URL.
        direction: 'maximize' or 'minimize'.
        objective_metric: Name of the objective metric.
        timeout_minutes: Max runtime.
        distributed: If True, use clearml ask/tell distribution.
        queue: Target queue (required if distributed or use_hpo_optimizer).
        project: ClearML project name.
        pruner: Pruner type ('hyperband', 'median', 'percentile', or None).
        use_hpo_optimizer: If True, use clearml HyperParameterOptimizer.

    Returns:
        Dict with study metadata and best results.
    """
    ss = search_space or _DEFAULT_SEARCH_SPACE

    if use_hpo_optimizer:
        return _run_hpo_optimizer(
            script=script,
            n_trials=n_trials,
            parallel=n_jobs,
            study_name=study_name,
            search_space=ss,
            direction=direction,
            objective_metric=objective_metric,
            timeout_minutes=timeout_minutes,
            queue=queue or "default",
            project=project,
        )

    if distributed:
        return _run_hpo_distributed(
            script=script,
            n_trials=n_trials,
            parallel=n_jobs,
            study_name=study_name,
            search_space=ss,
            direction=direction,
            objective_metric=objective_metric,
            timeout_minutes=timeout_minutes,
            queue=queue or "default",
            project=project,
            pruner=pruner,
        )

    return _run_hpo_local(
        script=script,
        n_trials=n_trials,
        n_jobs=n_jobs,
        study_name=study_name,
        search_space=ss,
        storage=storage,
        direction=direction,
        objective_metric=objective_metric,
        timeout_minutes=timeout_minutes,
        pruner=pruner,
    )


# ── Local HPO (mode 1) ──


def _run_hpo_local(
    script: str,
    n_trials: int,
    n_jobs: int,
    study_name: str | None,
    search_space: dict[str, dict[str, Any]],
    storage: str | None,
    direction: str,
    objective_metric: str,
    timeout_minutes: float | None,
    pruner: str | None = "hyperband",
) -> dict[str, Any]:
    """Run HPO locally on this machine."""
    optuna = _import_optuna()

    if study_name is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        study_name = f"hpo_{ts}"

    pruner_instance = _get_pruner(pruner)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            pruner=pruner_instance,
        )

    start_time = datetime.datetime.now(datetime.timezone.utc)
    completed = 0
    failed = 0

    for trial_idx in range(n_trials):
        if timeout_minutes is not None:
            elapsed = (
                datetime.datetime.now(datetime.timezone.utc) - start_time
            ).total_seconds() / 60
            if elapsed >= timeout_minutes:
                break

        trial = study.ask()
        params = _suggest_params(trial, search_space)
        try:
            value = _run_trial_local(script, params, objective_metric)
            if value is not None:
                study.tell(trial=trial, values=value)
                completed += 1
            else:
                _tell_failed(study, trial, direction)
                failed += 1
        except Exception:
            _tell_failed(study, trial, direction)
            failed += 1

    duration = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
    return _build_result(
        study, study_name, n_trials, completed, failed, direction, duration, timeout_minutes
    )


# ── Distributed HPO (mode 2: ask/tell + clearml Task clone) ──


def _run_hpo_distributed(
    script: str,
    n_trials: int,
    parallel: int,
    study_name: str | None,
    search_space: dict[str, dict[str, Any]],
    direction: str,
    objective_metric: str,
    timeout_minutes: float | None,
    queue: str,
    project: str,
    pruner: str | None = "hyperband",
) -> dict[str, Any]:
    """Run HPO via clearml queue distribution (ask/tell mode)."""
    from clearml import Task

    optuna = _import_optuna()

    if study_name is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        study_name = f"hpo_dist_{ts}"

    # SQLite storage for persistence
    storage_path = os.path.expanduser(f"~/.expflow/optuna_{study_name}.db")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    pruner_instance = _get_pruner(pruner)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            pruner=pruner_instance,
        )

    script_abspath = os.path.abspath(script)
    source_task = Task.create(
        task_name=f"hpo_source_{study_name}",
        project_name=project,
        script=script_abspath,
        add_task_init_call=True,
    )

    start_time = datetime.datetime.now(datetime.timezone.utc)
    completed = 0
    failed = 0
    pending: list[tuple[Any, dict[str, Any], Any]] = []

    for trial_idx in range(n_trials):
        if timeout_minutes is not None:
            elapsed = (
                datetime.datetime.now(datetime.timezone.utc) - start_time
            ).total_seconds() / 60
            if elapsed >= timeout_minutes:
                break

        trial = study.ask()
        params = _suggest_params(trial, search_space)

        trial_task = Task.clone(
            source_task=source_task,
            name=f"trial_{trial.number}",
            parent=source_task,
        )
        for k, v in params.items():
            trial_task.set_parameter(f"Args/--{k}", str(v))
        trial_task.enqueue(queue)
        pending.append((trial, params, trial_task))

        while len(pending) >= parallel:
            collected = _collect_one_trial(study, pending, objective_metric, direction, optuna)
            if collected is not None:
                c, f = collected
                completed += c
                failed += f

    while pending:
        collected = _collect_one_trial(study, pending, objective_metric, direction, optuna)
        if collected is not None:
            c, f = collected
            completed += c
            failed += f

    source_task.delete(force=True)

    duration = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
    return _build_result(
        study, study_name, n_trials, completed, failed, direction, duration, timeout_minutes
    )


def _collect_one_trial(
    study: Any,
    pending: list,
    objective_metric: str,
    direction: str,
    optuna: Any,
    poll_interval: float = 5.0,
    timeout_minutes: float = 60.0,
) -> tuple[int, int] | None:
    """Wait for one pending trial to complete and report its result."""
    if not pending:
        return None

    start = time.time()
    timeout_sec = timeout_minutes * 60

    while pending and (time.time() - start) < timeout_sec:
        for i, (trial, params, task) in enumerate(pending):
            try:
                status = task.status
            except Exception:
                status = "unknown"

            if status in ("completed", "failed", "stopped"):
                trial_obj, param, tsk = pending.pop(i)
                if status == "completed":
                    value = _extract_metric_from_task(tsk, objective_metric)
                    if value is not None:
                        study.tell(trial=trial_obj, values=value)
                        return (1, 0)
                    else:
                        study.tell(trial=trial_obj, values=_failed_value(direction))
                        return (0, 1)
                else:
                    study.tell(trial=trial_obj, values=_failed_value(direction))
                    return (0, 1)
        time.sleep(poll_interval)
    return None


def _extract_metric_from_task(task: Any, metric_name: str) -> float | None:
    """Extract a metric value from a completed clearml task."""
    try:
        scalars = task.get_last_scalar_metrics()
        for group, metrics in scalars.items():
            if metric_name in metrics:
                return float(metrics[metric_name]["last"])
    except Exception:
        pass
    return None


# ── HyperParameterOptimizer (mode 3: ClearML native) ──


def _run_hpo_optimizer(
    script: str,
    n_trials: int,
    parallel: int,
    study_name: str | None,
    search_space: dict[str, dict[str, Any]],
    direction: str,
    objective_metric: str,
    timeout_minutes: float | None,
    queue: str,
    project: str,
) -> dict[str, Any]:
    """Run HPO via ClearML HyperParameterOptimizer.

    This uses ClearML's native Optuna integration — no manual ask/tell/collect.
    ClearML automatically creates Task clones, manages concurrency, and retries.

    The script must report the objective via clearml Task.report_scalar()
    during training so HyperParameterOptimizer can read it.
    """
    from clearml import Task
    from clearml.automation import (
        HyperParameterOptimizer,
        LogUniformParameterRange,
        UniformIntegerParameterRange,
        UniformParameterRange,
    )

    if study_name is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        study_name = f"hpo_opt_{ts}"

    # Determine metric sign for HyperParameterOptimizer
    metric_sign = "max" if direction == "maximize" else "min"

    # Determine metric title/series from the dot-notation metric name
    # e.g. "Score/seg_total" or just "seg_total"
    parts = objective_metric.split("/", 1)
    if len(parts) == 2:
        metric_title, metric_series = parts
    else:
        metric_title = objective_metric
        metric_series = objective_metric

    # Convert our search_space dict to ClearML parameter ranges
    hpo_params: list[Any] = []
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "float":
            if spec.get("log", False):
                hpo_params.append(
                    LogUniformParameterRange(
                        f"Args/--{name}",
                        min_value=spec["low"],
                        max_value=spec["high"],
                    )
                )
            else:
                hpo_params.append(
                    UniformParameterRange(
                        f"Args/--{name}",
                        min_value=spec["low"],
                        max_value=spec["high"],
                        step_size=spec.get("step"),
                    )
                )
        elif ptype == "int":
            hpo_params.append(
                UniformIntegerParameterRange(
                    f"Args/--{name}",
                    min_value=spec["low"],
                    max_value=spec["high"],
                    step_size=spec.get("step", 1),
                )
            )
        elif ptype == "categorical":
            hpo_params.append(
                UniformParameterRange(
                    f"Args/--{name}",
                    min_value=0,
                    max_value=len(spec["choices"]) - 1,
                )
            )

    script_abspath = os.path.abspath(script)
    source_task = Task.create(
        task_name=f"hpo_{study_name}_template",
        project_name=project,
        script=script_abspath,
        add_task_init_call=True,
    )

    start_time = datetime.datetime.now(datetime.timezone.utc)

    optimizer = HyperParameterOptimizer(
        base_task_id=source_task.id,
        hyper_parameters=hpo_params,
        objective_metric_title=metric_title,
        objective_metric_series=metric_series,
        objective_metric_sign=metric_sign,
        max_number_of_experiments=n_trials,
        execution_queue=queue,
        pool_period_min=1.0,
        concurrent_jobs=parallel,
        total_jobs=parallel,
    )

    optimizer.start()

    # Wait with optional timeout
    wait_kwargs: dict[str, Any] = {}
    if timeout_minutes is not None:
        wait_kwargs["timeout"] = timeout_minutes
    optimizer.wait(**wait_kwargs)

    top_jobs = optimizer.get_top_experiments(top_k=3)

    duration = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

    # Build result from top job
    completed = n_trials  # approximate — ClearML handles exact tracking
    failed = 0

    best_value = None
    best_params = None
    if top_jobs:
        best_job = top_jobs[0]
        best_value = getattr(best_job, "value", None)
        best_params = best_job.get_parameters() if hasattr(best_job, "get_parameters") else None

    # Cleanup source template
    source_task.delete(force=True)

    return {
        "study_name": study_name,
        "n_trials": n_trials,
        "completed": completed,
        "failed": failed,
        "best_value": best_value,
        "best_params": best_params,
        "direction": direction,
        "duration_sec": duration,
        "timeout_minutes": timeout_minutes,
        "method": "hyperparameter_optimizer",
    }


# ── Best params lookup (shared by Phase 12) ──


def get_study_best_params(study_name: str, storage: str | None = None) -> dict[str, Any] | None:
    """Get the best parameters from a completed Optuna study.

    Args:
        study_name: Optuna study name.
        storage: Optional storage URL (default: ~/.expflow/optuna_<name>.db).

    Returns:
        Dict of best parameters, or None if no completed trials.
    """
    optuna = _import_optuna()

    if storage is None:
        storage_path = os.path.expanduser(f"~/.expflow/optuna_{study_name}.db")
        if os.path.exists(storage_path):
            storage = f"sqlite:///{storage_path}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return None

    if study.best_trial and study.best_trial.params:
        return dict(study.best_trial.params)
    return None


# ── Helpers ──


def _import_optuna():
    """Lazy import of optuna."""
    try:
        import optuna  # noqa: F401

        return sys.modules["optuna"]
    except ImportError:
        raise ImportError("optuna is required. pip install optuna")


def _suggest_params(trial: Any, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample hyperparameters from a search space."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step"),
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{name}'")
    return params


def _run_trial_local(script: str, params: dict[str, Any], objective_metric: str) -> float | None:
    """Run a single trial locally."""
    cmd = [script]
    for k, v in params.items():
        cmd.append(f"--{k}={v}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    prefix = f"METRIC:{objective_metric}="
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            try:
                return float(line[len(prefix) :])
            except (ValueError, TypeError):
                continue

    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        try:
            data = json.loads(line)
            if objective_metric in data:
                return float(data[objective_metric])
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _failed_value(direction: str) -> float:
    """Return the worst possible value for a given direction."""
    return float("-inf") if direction == "maximize" else float("inf")


def _tell_failed(study: Any, trial: Any, direction: str) -> None:
    """Tell optuna that a trial failed."""
    try:
        study.tell(trial=trial, values=_failed_value(direction))
    except Exception:
        pass


def _build_result(
    study: Any,
    study_name: str,
    n_trials: int,
    completed: int,
    failed: int,
    direction: str,
    duration_sec: float,
    timeout_minutes: float | None,
) -> dict[str, Any]:
    """Build a standardized result dict from study data."""
    best_trial = study.best_trial if hasattr(study, "best_trial") and study.best_trial else None
    return {
        "study_name": study_name,
        "n_trials": n_trials,
        "completed": completed,
        "failed": failed,
        "best_value": best_trial.value if best_trial else None,
        "best_params": best_trial.params if best_trial else None,
        "direction": direction,
        "duration_sec": duration_sec,
        "timeout_minutes": timeout_minutes,
    }
