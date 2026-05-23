#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow hpo — Hyperparameter optimization runner.

Supports three modes:
1. **Local serial** (default): runs trials sequentially on this machine.
2. **Distributed via clearml ask/tell**: each trial as independent clearml Task.
3. **HyperParameterOptimizer** (recommended for production): uses ClearML's
   native Optuna integration — auto-creates/clones/enqueues tasks.

v0.9.0 enhancements:
- combined_score(): competition-aware objective (seg_total × 0.75 + time_score)
- cond_search_space(): narrow HPO ranges using diagnosis bias + OOM history
- best_params clean: strip Args/ prefix, auto-cast strings to float

The training script must:
- Accept hyperparameters as CLI arguments: `--lr=0.001 --epochs=80`
- Report metrics via clearml `Task.report_scalar()` for objective collection
  (REQUIRED for mode 3 — HyperParameterOptimizer reads scalar metrics)
- Report training wall time as `train_time_minutes` scalar for combined_score
- Or output METRIC:<name>=<value> to stdout for local mode
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any

# ── Combined score: seg_total + train_time ──

COMBINED_SEG_WEIGHT: float = 0.75
COMBINED_TIME_MAX: float = 60.0  # minutes
COMBINED_TIME_FULL_SCORE: float = 35.0
COMBINED_TIME_DECAY_WINDOW: float = 120.0  # minutes beyond max → 0


def combined_score(seg_total: float, train_minutes: float) -> float:
    """Compute competition-style combined score.

    Total = seg_total * 0.75 + train_time_score + inference_time_score
    At HPO time we only have train_time; infer_time is assumed 0 for ranking.

    Args:
        seg_total: seg_total from eval (0-150ish).
        train_minutes: training wall time in minutes.

    Returns:
        Combined score (higher is better).
    """
    time_score = COMBINED_TIME_FULL_SCORE
    if train_minutes > COMBINED_TIME_MAX:
        excess = train_minutes - COMBINED_TIME_MAX
        decay = max(0.0, 1.0 - excess / COMBINED_TIME_DECAY_WINDOW)
        time_score = COMBINED_TIME_FULL_SCORE * decay
    return seg_total * COMBINED_SEG_WEIGHT + time_score


# ── Conditional search space ──


def cond_search_space(
    base_space: dict[str, dict[str, Any]] | None = None,
    bias: dict[str, dict[str, float]] | None = None,
    constraints: dict[str, Any] | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Conditional search space: narrow ranges based on diagnosis and history.

    Args:
        base_space: Default search space (uses _DEFAULT_SEARCH_SPACE if None).
        bias: From suggest_next_params — dict of {param: {"low": ..., "high": ...}}
              indicating where to focus sampling.
        constraints: Time budget etc. {"max_train_minutes": float}.
        failures: From GoalOrchestrator.load()["learned_failures"].
                  Type "oom" suppresses capacity-increasing params.

    Returns:
        Modified search space with narrowed ranges and suppressed params.
    """
    space = dict(base_space or _DEFAULT_SEARCH_SPACE)

    # Step 1: Apply diagnosis bias (narrow ranges)
    if bias:
        for param_name, bounds in bias.items():
            if param_name not in space:
                continue
            low = bounds.get("low")
            high = bounds.get("high")
            spec = dict(space[param_name])
            if low is not None:
                spec["low"] = float(low) if isinstance(spec["low"], (int, float)) else low
            if high is not None:
                spec["high"] = float(high) if isinstance(spec["high"], (int, float)) else high
            space[param_name] = spec

    # Step 2: Suppress capacity-increasing params after OOM failures
    if failures:
        capacity_keys = {"n_modes", "width", "batch_size", "n_layers"}
        oom_types = {f.get("type", "") for f in failures if f.get("type") in ("oom", "signal")}
        if oom_types:
            for key in capacity_keys:
                if key in space:
                    spec = dict(space[key])
                    cur_high = spec.get("high", 999)
                    # Cap at (low + high) / 2
                    half = (spec.get("low", 0) + cur_high) / 2.0
                    spec["high"] = half
                    space[key] = spec

    # Step 3: Apply time constraints — limit epochs if tight on time
    if constraints and constraints.get("max_train_minutes"):
        max_min = constraints["max_train_minutes"]
        if max_min < 60 and "epochs" in space:
            spec = dict(space["epochs"])
            # Cap epochs to stay under time budget
            spec["high"] = min(spec.get("high", 150), int(max_min * 2))
            space["epochs"] = spec

    return space


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
            min_resource=10,
            reduction_factor=3,
            max_resource=200,
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
    direction: str | list[str] = "maximize",
    objective_metric: str | list[str] = "seg_total",
    timeout_minutes: float | None = None,
    distributed: bool = False,
    queue: str | None = None,
    project: str = "PDEBench",
    pruner: str | None = "hyperband",
    use_hpo_optimizer: bool = False,
    loss: str | None = None,
    search_bias: dict[str, dict[str, float]] | None = None,
    constraints: dict[str, Any] | None = None,
    failures: list[dict[str, Any]] | None = None,
    use_combined_score: bool = False,
    param_prefix: str = "Args/",
) -> dict[str, Any]:
    """Run hyperparameter optimization.

    Three modes:
    1. **local** (default): subprocess-based, no clearml.
    2. **distributed** (distributed=True): ask/tell + clearml Task clone/enqueue.
    3. **hpo_optimizer** (use_hpo_optimizer=True): ClearML HyperParameterOptimizer.

    Multi-objective support: pass direction as a list of strings and
    objective_metric as a list of metric names. Example:
        run_hpo(..., direction=["maximize", "minimize"],
                objective_metric=["seg_total", "train_time_minutes"])

    Args:
        ...
        direction: 'maximize', 'minimize', or a list for multi-objective.
        objective_metric: Metric name or list of metric names for multi-objective.
        ...
    """
    ss = cond_search_space(
        base_space=search_space or _DEFAULT_SEARCH_SPACE,
        bias=search_bias,
        constraints=constraints,
        failures=failures,
    )

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
            loss=loss,
            use_combined_score=use_combined_score,
            param_prefix=param_prefix,
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
            loss=loss,
            use_combined_score=use_combined_score,
            param_prefix=param_prefix,
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
        loss=loss,
        use_combined_score=use_combined_score,
    )


# ── Local HPO (mode 1) ──


def _run_hpo_local(
    script: str,
    n_trials: int,
    n_jobs: int,
    study_name: str | None,
    search_space: dict[str, dict[str, Any]],
    storage: str | None,
    direction: str | list[str],
    objective_metric: str | list[str],
    timeout_minutes: float | None,
    pruner: str | None = "hyperband",
    loss: str | None = None,
    use_combined_score: bool = False,
) -> dict[str, Any]:
    """Run HPO locally on this machine.

    Supports multi-objective: pass direction as a list, objective_metric as a list.
    """
    optuna = _import_optuna()

    if study_name is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        study_name = f"hpo_{ts}"

    pruner_instance = _get_pruner(pruner)

    # Normalize to multi-objective-friendly format
    multi = isinstance(direction, list)
    dirs = direction if multi else [direction]
    metrics = objective_metric if multi else [objective_metric]

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        study = optuna.create_study(
            study_name=study_name,
            directions=dirs if multi else direction,
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
            values = _run_trial_local_multi(script, params, metrics, loss=loss)
            if values is not None and all(v is not None for v in values):
                if multi:
                    study.tell(trial=trial, values=values)
                else:
                    study.tell(trial=trial, values=values[0])
                completed += 1
            else:
                _tell_failed(study, trial, dirs[0] if not multi else dirs[0])
                failed += 1
        except Exception:
            _tell_failed(study, trial, dirs[0] if not multi else dirs[0])
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
    direction: str | list[str],
    objective_metric: str | list[str],
    timeout_minutes: float | None,
    queue: str,
    project: str,
    pruner: str | None = "hyperband",
    loss: str | None = None,
    use_combined_score: bool = False,
    param_prefix: str = "Args/",
) -> dict[str, Any]:
    """Run HPO via clearml queue distribution (ask/tell mode).

    Supports multi-objective: pass direction as a list, objective_metric as a list.
    """
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
        multi = isinstance(direction, list)
        study = optuna.create_study(
            study_name=study_name,
            directions=direction if multi else direction,
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
            trial_task.set_parameter(f"{param_prefix}{k}", str(v))
        if loss is not None:
            trial_task.set_parameter(f"{param_prefix}loss", loss)
        Task.enqueue(task=trial_task, queue_name=queue)
        pending.append((trial, params, trial_task))

        while len(pending) >= parallel:
            collected = _collect_one_trial(
                study,
                pending,
                objective_metric,
                direction,
                optuna,
                use_combined_score=use_combined_score,
                timeout_minutes=timeout_minutes,
            )
            if collected is not None:
                c, f = collected
                completed += c
                failed += f

    while pending:
        collected = _collect_one_trial(
            study,
            pending,
            objective_metric,
            direction,
            optuna,
            use_combined_score=use_combined_score,
            timeout_minutes=timeout_minutes,
        )
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
    pending: list[tuple[Any, dict[str, Any], Any]],
    objective_metric: str | list[str],
    direction: str | list[str],
    optuna: Any,
    poll_interval: float = 5.0,
    timeout_minutes: float | None = 60.0,
    use_combined_score: bool = False,
) -> tuple[int, int] | None:
    """Wait for one pending trial to complete and report its result.

    Supports both single and multi-objective. For multi-objective, returns
    a list of values extracted from each metric name in ``objective_metric``.
    """
    if not pending:
        return None

    multi = isinstance(direction, list)
    metrics = objective_metric if multi else [objective_metric]

    start = time.time()
    timeout_sec = (timeout_minutes or 60.0) * 60

    while pending and (time.time() - start) < timeout_sec:
        for i, (trial, params, task) in enumerate(pending):
            try:
                status = task.status
            except Exception:
                status = "unknown"

            if status in ("completed", "failed", "stopped"):
                trial_obj, param, tsk = pending.pop(i)
                if status == "completed":
                    values = _extract_metrics_from_task(tsk, metrics)
                    if values is not None and all(v is not None for v in values):
                        if multi:
                            study.tell(trial=trial_obj, values=values)
                        else:
                            study.tell(trial=trial_obj, values=values[0])
                        return (1, 0)
                    else:
                        study.tell(
                            trial=trial_obj,
                            values=_failed_value(direction[0] if multi else direction),
                        )
                        return (0, 1)
                else:
                    study.tell(
                        trial=trial_obj, values=_failed_value(direction[0] if multi else direction)
                    )
                    return (0, 1)
        time.sleep(poll_interval)
    return None


def _extract_metrics_from_task(task: Any, metrics: list[str]) -> list[float | None] | None:
    """Extract multiple metric values from a completed clearml task.

    Returns a list of values in the same order as ``metrics``, or None if none found.
    """
    if not metrics:
        return None
    try:
        scalars = task.get_last_scalar_metrics()
    except Exception:
        return None

    values: list[float | None] = []
    for m in metrics:
        val = _extract_single_metric_from_scalars(scalars, m)
        values.append(val)

    if all(v is None for v in values):
        return None
    return values


def _extract_single_metric_from_scalars(scalars: dict, metric_name: str) -> float | None:
    """Extract a single metric value from clearml scalars dict using fuzzy matching."""
    norm_target = _normalize_metric_name(metric_name)
    for group, metrics in scalars.items():
        for key, val in metrics.items():
            if _normalize_metric_name(key) == norm_target:
                return float(val["last"])
    return None


def _normalize_metric_name(name: str) -> str:
    """Normalize metric name for fuzzy matching.

    Strips case, spaces, underscores, and hyphens so that
    'seg_total', 'Seg Total', 'seg-total', and 'SEGTOTAL' all match.
    """
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def _extract_metric_from_task(task: Any, metric_name: str) -> float | None:
    """Extract a metric value from a completed clearml task.

    Uses fuzzy matching on metric keys (case/space/underscore insensitive)
    so that e.g. 'seg_total' matches 'Seg Total' reported by the training script.
    """
    try:
        scalars = task.get_last_scalar_metrics()
        norm_target = _normalize_metric_name(metric_name)
        for group, metrics in scalars.items():
            for key, val in metrics.items():
                if _normalize_metric_name(key) == norm_target:
                    return float(val["last"])
    except Exception:
        pass
    return None


def _extract_combined_score(task: Any) -> float | None:
    """Extract combined_score(seg_total, train_minutes) from a clearml task.

    Uses fuzzy metric name matching (case/space/underscore insensitive)
    so 'seg_total' matches 'Seg Total' reported by the training script.

    Requires the training script to report both:
        Task.report_scalar("Score", "seg_total", value)
        Task.report_scalar("Time", "train_time_minutes", value)
    """
    try:
        scalars = task.get_last_scalar_metrics()
        seg = None
        time_min = None
        for group, metrics in scalars.items():
            for key, val in metrics.items():
                norm = _normalize_metric_name(key)
                if norm == "segtotal" or norm == "segmenttotal":
                    seg = float(val["last"])
                if norm == "traintimeminutes":
                    time_min = float(val["last"])
        if seg is not None and time_min is not None:
            return combined_score(seg, time_min)
        return seg  # fallback to seg_total only
    except Exception:
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
    loss: str | None = None,
    use_combined_score: bool = False,
    param_prefix: str = "Args/",
    pruner: str | None = "hyperband",
) -> dict[str, Any]:
    """Run HPO via ClearML HyperParameterOptimizer.

    Uses OptimizerOptuna (default) for TPE-based sampling with pruner support.

    Args:
        param_prefix: Prefix for parameter keys. The base task uses one of:
            "Args/" (clearml Args section, standard) or "Args/--" (with -- prefix).
            Default "Args/".
        pruner: Optuna pruner type ('hyperband', 'median', 'percentile', or None).
    """
    from clearml import Task
    from clearml.automation import (
        HyperParameterOptimizer,
        LogUniformParameterRange,
        UniformIntegerParameterRange,
        UniformParameterRange,
    )
    from clearml.automation.optuna import OptimizerOptuna

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
                        f"{param_prefix}{name}",
                        min_value=spec["low"],
                        max_value=spec["high"],
                    )
                )
            else:
                hpo_params.append(
                    UniformParameterRange(
                        f"{param_prefix}{name}",
                        min_value=spec["low"],
                        max_value=spec["high"],
                        step_size=spec.get("step"),
                    )
                )
        elif ptype == "int":
            hpo_params.append(
                UniformIntegerParameterRange(
                    f"{param_prefix}{name}",
                    min_value=spec["low"],
                    max_value=spec["high"],
                    step_size=spec.get("step", 1),
                )
            )
        elif ptype == "categorical":
            from clearml.automation import DiscreteParameterRange

            hpo_params.append(
                DiscreteParameterRange(
                    f"{param_prefix}{name}",
                    values=spec["choices"],
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

    # Build optimizer_kwargs for SearchStrategy (OptimizerOptuna)
    optimizer_kwargs: dict[str, Any] = {
        "total_max_jobs": n_trials,
    }

    # Convert pruner name to optuna pruner object for OptimizerOptuna
    pruner_instance = _get_pruner(pruner)
    if pruner_instance is not None:
        optimizer_kwargs["optuna_pruner"] = pruner_instance

    optimizer = HyperParameterOptimizer(
        base_task_id=source_task.id,
        hyper_parameters=hpo_params,
        objective_metric_title=metric_title,
        objective_metric_series=metric_series,
        objective_metric_sign=metric_sign,
        optimizer_class=OptimizerOptuna,
        max_number_of_concurrent_tasks=parallel,
        execution_queue=queue,
        pool_period_min=1.0,
        **optimizer_kwargs,
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


def _run_trial_local_multi(
    script: str,
    params: dict[str, Any],
    metrics: list[str],
    loss: str | None = None,
) -> list[float | None] | None:
    """Run a single trial locally and extract multiple metrics.

    Returns a list of values in the same order as ``metrics``,
    or None if none of the metrics could be extracted.
    """
    cmd = [script]
    for k, v in params.items():
        cmd.append(f"--{k}={v}")
    if loss is not None:
        cmd.append(f"--loss={loss}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None

    values: list[float | None] = []
    for m in metrics:
        val = _extract_stdout_metric(result.stdout, m)
        values.append(val)

    if all(v is None for v in values):
        return None
    return values


def _extract_stdout_metric(output: str, metric_name: str) -> float | None:
    """Extract a single metric value from a subprocess stdout."""
    prefix = f"METRIC:{metric_name}="
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            try:
                return float(line[len(prefix) :])
            except (ValueError, TypeError):
                continue

    for line in reversed(output.splitlines()):
        line = line.strip()
        try:
            data = json.loads(line)
            if metric_name in data:
                return float(data[metric_name])
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _run_trial_local(
    script: str, params: dict[str, Any], objective_metric: str, loss: str | None = None
) -> float | None:
    """Run a single trial locally."""
    cmd = [script]
    for k, v in params.items():
        cmd.append(f"--{k}={v}")
    if loss is not None:
        cmd.append(f"--loss={loss}")

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
    direction: str | list[str],
    duration_sec: float,
    timeout_minutes: float | None,
) -> dict[str, Any]:
    """Build a standardized result dict from study data.

    For multi-objective studies, ``best_value`` will be a list of values.
    """
    best_trial = study.best_trial if hasattr(study, "best_trial") and study.best_trial else None
    return {
        "study_name": study_name,
        "n_trials": n_trials,
        "completed": completed,
        "failed": failed,
        "best_value": best_trial.values
        if best_trial and isinstance(direction, list)
        else (best_trial.value if best_trial else None),
        "best_params": best_trial.params if best_trial else None,
        "direction": direction,
        "duration_sec": duration_sec,
        "timeout_minutes": timeout_minutes,
    }
