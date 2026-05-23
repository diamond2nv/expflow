#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow hpo - Hyperparameter optimization runner.

Supports three modes:
1. **Local serial** (default): runs trials sequentially on this machine.
2. **Distributed via clearml ask/tell**: each trial as independent clearml Task.
3. **HyperParameterOptimizer** (recommended for production): uses ClearML's
   native Optuna integration - auto-creates/clones/enqueues tasks.

== For third-party users ==

All tunable defaults are module-level constants at the top of this file.
Search for CAPACITY_KEYS, PRUNER_*, NARROW_*, HIERARCHICAL_*, etc.
Override before calling run_hpo():

    from expflow_pde import hpo
    hpo.PRUNER_N_STARTUP_TRIALS = 10
    hpo.NARROW_SHRINK_FACTOR = 0.3
    hpo.HIERARCHICAL_STAGE1_FRAC = 0.3

Key constants to adjust for your competition / dataset:

  SEG_WEIGHT (0.75)            - Different competition metric mix
  TIME_FULL_SCORE (35.0)       - Different time bonus scale
  TIME_MAX_MINUTES (60.0)      - Different training time budget
  PRUNER_HYPERBAND_MIN_RESOURCE (10) - Tutorial script uses more epochs
  PRUNER_HYPERBAND_MAX_RESOURCE (200) - Max epochs across all trials
  EARLY_STOP_MIN_REPORTS_DEFAULT (10) - Your training reports less often
  HIERARCHICAL_STAGE1_FRAC (0.2) - More/less random exploration
  _DEFAULT_SEARCH_SPACE values (various) - Your arch/equation differs

The training script must:
- Accept hyperparameters as CLI arguments: --lr=0.001 --epochs=80
- Report metrics via clearml Task.report_scalar() for objective collection
  (REQUIRED for mode 3 - HyperParameterOptimizer reads scalar metrics)
- Report training wall time as train_time_minutes scalar for combined_score
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
# These constants define how the competition's combined score is computed.
# `combined_score = seg_total * SEG_WEIGHT + time_score`
# time_score decays linearly from TIME_FULL_SCORE to 0 when train time
# exceeds TIME_MAX_MINUTES, with full decay over TIME_DECAY_WINDOW_MINUTES.
SEG_WEIGHT: float = 0.75  # Weight of seg_total in combined score
TIME_FULL_SCORE: float = 35.0  # Maximum time bonus (achieved at TIME_MAX_MINUTES or under)
TIME_MAX_MINUTES: float = 60.0  # Training time for full time_score
TIME_DECAY_WINDOW_MINUTES: float = 120.0  # Minutes beyond TIME_MAX to decay to 0


# ── Default hyperparameter search space ──
# Each param: {"type": "float"|"int"|"categorical", "low": ..., "high": ..., ...}
# Used by all three HPO modes. Users can override via `run_hpo(search_space=...)`.
# Note: ranges assume PDEBench FNO/DeepONet on 1D Burgers; adjust for
# your equation / architecture / GPU memory budget.
_DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # PDEBench Task 1 (Burgers FNO) verified ranges — May 2026 experiments
    "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    # w64+cosine verified lr=1e-3 optimal, AdamW+wd diverges at 1e-3
    "batch_size": {"type": "int", "low": 64, "high": 256, "step": 32},
    # Sweep: 128 sweet spot, 256 works, 512 degrades (fewer gradient updates)
    "epochs": {"type": "int", "low": 40, "high": 120, "step": 10},
    # w64 peaks at epoch 60; >80 exceeds 60min time budget for n=500
    "weight_decay": {"type": "float", "low": 0.0, "high": 1e-5, "log": True},
    # 0 optimal for fresh train; AdamW+wd=1e-5 diverges under 60min constraint
    "dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
    "sub_step": {"type": "int", "low": 1, "high": 6, "step": 1},
    # 5 best (exactly matches dt ratio 0.05/0.01); >6 degrades
    "width": {"type": "int", "low": 16, "high": 128, "step": 16},
    # 64 far superior to 32 (+118% Seg); 16 underfits, 128 TBD
    "n_layers": {"type": "int", "low": 2, "high": 6, "step": 1},
    # 4 optimal; 8+ overparameterized at n=1000
    "modes": {"type": "int", "low": 8, "high": 32, "step": 4},
    # 16 optimal; 24+ diminishing returns, 12 baseline
}


# ── Conditional search space defaults ──
# When OOM/signal failures accumulate, these params are capped at (low + high) / 2.
CAPACITY_KEYS: set[str] = {"n_modes", "width", "batch_size", "n_layers"}
# Epoch cap formula: min(original_high, max_train_minutes * EPOCHS_PER_MINUTE)
# Used when constraints["max_train_minutes"] < TIME_LIMIT_FOR_EPOCH_CAP_MINUTES.
EPOCHS_PER_MINUTE: int = 2
TIME_LIMIT_FOR_EPOCH_CAP_MINUTES: float = 60.0


# ── Pruner defaults ──
# These values are passed to Optuna pruner constructors when the user selects
# a pruner type via `run_hpo(pruner="hyperband")`. Override by calling
# `get_pruner(pruner_name)` directly with custom values.

# HyperbandPruner: min_resource=max_resource/reduction_factor^max_n_brackets
# Typical: 10 * 3^3 = 10 to 270 epoch range
PRUNER_HYPERBAND_MIN_RESOURCE: int = 10
PRUNER_HYPERBAND_MAX_RESOURCE: int = 200
PRUNER_HYPERBAND_REDUCTION_FACTOR: int = 3

# MedianPruner / PercentilePruner: common warmup parameters
PRUNER_N_STARTUP_TRIALS: int = 5  # Trials before pruning kicks in
PRUNER_N_WARMUP_STEPS: int = 10  # Epochs per trial before considering pruning
PRUNER_PERCENTILE: float = 25.0  # Percentile threshold for PercentilePruner


# ── Distributed HPO (ask/tell) polling defaults ──
# _collect_one_trial waits for clearml tasks to complete.
DEFAULT_POLL_INTERVAL_SEC: float = 5.0  # Seconds between poll iterations
DEFAULT_TRIAL_TIMEOUT_MINUTES: float = 60.0  # Max wait per individual trial
EARLY_STOP_MIN_REPORTS_DEFAULT: int = 10  # Min scalar reports before early-stop check
EARLY_STOP_RECENT_N: int = 5  # Number of recent values to compare against threshold


# ── _narrow_space defaults ──
# Used by run_hpo_hierarchical to shrink search space around best trials.
NARROW_TOP_FRAC: float = 0.2  # Fraction of best trials to analyze
NARROW_SHRINK_FACTOR: float = 0.5  # New range = spread * factor / 2
# If top-performer spread is below this fraction of original range,
# fall back to MIN_SPREAD_FRAC * original_range
NARROW_MIN_SPREAD_FRAC: float = 0.1
NARROW_IDENTICAL_SPREAD_FRAC: float = 0.01


# ── run_hpo_hierarchical defaults ──
HIERARCHICAL_STAGE1_FRAC: float = 0.2  # Phase 1 trials = max(5, n_trials * STAGE1_FRAC)
HIERARCHICAL_STAGE1_MIN: int = 5  # Minimum Phase 1 trials
HIERARCHICAL_STAGE2_PRUNER: str = "hyperband"  # Optuna pruner for Phase 2


# ── Combined score deprecation ──
# These are kept for backward compatibility and will be removed in v0.12.
# Use SEG_WEIGHT / TIME_FULL_SCORE / TIME_MAX_MINUTES / TIME_DECAY_WINDOW_MINUTES instead.
COMBINED_SEG_WEIGHT: float = SEG_WEIGHT
COMBINED_TIME_MAX: float = TIME_MAX_MINUTES
COMBINED_TIME_FULL_SCORE: float = TIME_FULL_SCORE
COMBINED_TIME_DECAY_WINDOW: float = TIME_DECAY_WINDOW_MINUTES


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
        oom_types = {f.get("type", "") for f in failures if f.get("type") in ("oom", "signal")}
        if oom_types:
            for key in CAPACITY_KEYS:
                if key in space:
                    spec = dict(space[key])
                    cur_high = spec.get("high", 999)
                    # Cap at (low + high) / 2 — conservative to avoid repeat OOM
                    half = (spec.get("low", 0) + cur_high) / 2.0
                    spec["high"] = half
                    space[key] = spec

    # Step 3: Apply time constraints — limit epochs if tight on time
    if constraints and constraints.get("max_train_minutes"):
        max_min = constraints["max_train_minutes"]
        if max_min < TIME_LIMIT_FOR_EPOCH_CAP_MINUTES and "epochs" in space:
            spec = dict(space["epochs"])
            # Cap epochs to stay under time budget
            spec["high"] = min(spec.get("high", 150), int(max_min * EPOCHS_PER_MINUTE))
            space["epochs"] = spec

    return space


# ── Pruner factory ──


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
            min_resource=PRUNER_HYPERBAND_MIN_RESOURCE,
            reduction_factor=PRUNER_HYPERBAND_REDUCTION_FACTOR,
            max_resource=PRUNER_HYPERBAND_MAX_RESOURCE,
        ),
        "median": lambda: optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER_N_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_N_WARMUP_STEPS,
        ),
        "percentile": lambda: optuna.pruners.PercentilePruner(
            percentile=PRUNER_PERCENTILE,
            n_startup_trials=PRUNER_N_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_N_WARMUP_STEPS,
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
    early_stop_threshold: float | None = None,
    early_stop_min_reports: int = 10,
) -> dict[str, Any]:
    """Run hyperparameter optimization.

    Three modes:
    1. **local** (default): subprocess-based, no clearml.
    2. **distributed** (distributed=True): ask/tell + clearml Task clone/enqueue.
    3. **hpo_optimizer** (use_hpo_optimizer=True): ClearML HyperParameterOptimizer.

    Multi-objective support: pass direction as a list of strings and
    objective_metric as a list of metric names. Example::

        run_hpo(..., direction=["maximize", "minimize"],
                objective_metric=["seg_total", "train_time_minutes"])

    When ``early_stop_threshold`` is set (distributed mode), trials whose
    intermediate scalar values fall below the threshold are stopped early.

    Args:
        direction: 'maximize', 'minimize', or a list for multi-objective.
        objective_metric: Metric name or list of metric names for multi-objective.
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
            early_stop_threshold=early_stop_threshold,
            early_stop_min_reports=early_stop_min_reports,
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
    early_stop_threshold: float | None = None,
    early_stop_min_reports: int = 10,
) -> dict[str, Any]:
    """Run HPO via clearml queue distribution (ask/tell mode).

    Supports multi-objective: pass direction as a list, objective_metric as a list.
    When ``early_stop_threshold`` is set, underperforming trials are stopped early
    by checking intermediate scalars during polling.
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
                early_stop_threshold=early_stop_threshold,
                early_stop_min_reports=early_stop_min_reports,
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
            early_stop_threshold=early_stop_threshold,
            early_stop_min_reports=early_stop_min_reports,
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


def _should_early_stop(
    task: Any,
    metric_name: str,
    threshold: float | None = None,
    min_reports: int = EARLY_STOP_MIN_REPORTS_DEFAULT,
    recent_n: int = EARLY_STOP_RECENT_N,
) -> bool:
    """Check if a running clearml task should be stopped early.

    Reads the task's reported scalars for ``metric_name`` (fuzzy match).
    If at least ``min_reports`` values have been reported and the max of
    the most recent ``recent_n`` is below ``threshold``, returns True.

    Args:
        task: clearml Task object.
        metric_name: Metric name to check (e.g. ``seg_total``).
        threshold: Early-stop threshold. If the recent max is below this,
            the task should be stopped. None means no early stopping.
        min_reports: Minimum number of scalar reports required before
            checking (default 10). Prevents premature stopping.
        recent_n: Number of most recent values to consider (default 5).

    Returns:
        True if the task should be stopped, False otherwise.
    """
    if threshold is None:
        return False

    try:
        # get_reported_scalars returns Iterable[tuple[float, float]]
        # where each tuple is (iteration, value)
        full_name = _resolve_metric_title_series(metric_name)
        reported = task.get_reported_scalars(*full_name)
    except Exception:
        return False

    values: list[float] = []
    for v in list(reported or []):
        try:
            pair = (v[0], v[1]) if isinstance(v, (list, tuple)) else (0, float(v))
            values.append(pair[1])
        except (TypeError, ValueError):
            continue

    if len(values) < min_reports:
        return False

    recent = values[-recent_n:]
    return max(recent) < threshold


def _resolve_metric_title_series(metric_name: str) -> tuple[str, str]:
    """Resolve a dot-notation metric name into (title, series).

    E.g. ``Score/seg_total`` -> (``Score``, ``seg_total``).
          ``seg_total`` -> (``seg_total``, ``seg_total``).
    """
    parts = metric_name.split("/", 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return (metric_name, metric_name)


def _collect_one_trial(
    study: Any,
    pending: list[tuple[Any, dict[str, Any], Any]],
    objective_metric: str | list[str],
    direction: str | list[str],
    optuna: Any,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SEC,
    timeout_minutes: float | None = DEFAULT_TRIAL_TIMEOUT_MINUTES,
    use_combined_score: bool = False,
    early_stop_threshold: float | None = None,
    early_stop_min_reports: int = EARLY_STOP_MIN_REPORTS_DEFAULT,
) -> tuple[int, int] | None:
    """Wait for one pending trial to complete and report its result.

    Supports both single and multi-objective. For multi-objective, returns
    a list of values extracted from each metric name in ``objective_metric``.

    When ``early_stop_threshold`` is set, periodically checks intermediate
    scalars and stops underperforming trials early.
    """
    if not pending:
        return None

    multi = isinstance(direction, list)
    metrics = objective_metric if multi else [objective_metric]
    # Primary metric for early stop (use first metric)
    primary_metric: str = metrics[0] if metrics else objective_metric  # type: ignore[assignment]

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
            elif (
                early_stop_threshold is not None
                and status == "running"
                and _should_early_stop(
                    task,
                    primary_metric,
                    threshold=early_stop_threshold,
                    min_reports=early_stop_min_reports,
                )
            ):
                # Early stop this underperforming trial
                task.stop()
                # It will be picked up in a future poll as "stopped"
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


# ── Hierarchical HPO (Sprint 2) ──


def _load_trials_from_storage(
    study_name: str,
    storage: str | None = None,
    direction: str = "maximize",
) -> list[dict[str, Any]]:
    """Load all completed trials from an Optuna study storage.

    Args:
        study_name: Optuna study name.
        storage: Storage URL (default: ~/.expflow/optuna_<name>.db).
        direction: Optimization direction ('maximize' or 'minimize').

    Returns:
        List of trial dicts with keys: value, params, number.
        Empty list if the study cannot be loaded or has no trials.
    """
    try:
        optuna = _import_optuna()
    except ImportError:
        return []

    resolved_storage: str | None = storage
    if resolved_storage is None:
        import glob

        db_pattern = os.path.expanduser(f"~/.expflow/optuna_{study_name}.db")
        matching = glob.glob(db_pattern)
        if matching:
            resolved_storage = f"sqlite:///{matching[0]}"

    if not resolved_storage:
        return []

    try:
        study = optuna.load_study(study_name=study_name, storage=resolved_storage)
    except Exception:
        return []

    is_maximize = direction == "maximize"
    trials: list[dict[str, Any]] = []
    for t in study.trials:
        val = t.value
        if val is None:
            continue
        # Optuna internally minimizes everything.
        # For maximize: it stores -value, so we negate back.
        # For minimize: it stores value directly.
        restored = val if not is_maximize else (-val if val is not None else None)
        trials.append(
            {
                "number": t.number,
                "value": restored,
                "params": dict(t.params) if t.params else {},
            }
        )
    # Sort by value descending (best first, regardless of direction)
    trials.sort(key=lambda x: x["value"], reverse=True)
    return trials


def _narrow_space(
    trials: list[dict[str, Any]],
    search_space: dict[str, dict[str, Any]],
    top_frac: float = NARROW_TOP_FRAC,
    shrink_factor: float = NARROW_SHRINK_FACTOR,
    direction: str = "maximize",
) -> dict[str, dict[str, Any]]:
    """Narrow search space based on top-performing trial params.

    Analyzes the top ``top_frac`` fraction of trials by value (descending
    for maximize, ascending for minimize) and shrinks float/int ranges
    to a tighter interval around the top performers.

    Args:
        trials: List of trial dicts with keys: value, params (dict of param_name=val).
            Values should already be positively-ordered (higher = better).
        search_space: Original search space definition.
        top_frac: Fraction of best trials to use for narrowing (default 0.2).
        shrink_factor: Multiplier for new range width relative to best-value
            spread. 0.5 means the new range is 50% of the top-performer spread.
            Must be in (0, 1].
        direction: Optimization direction ('maximize' or 'minimize').

    Returns:
        A new search space dict with narrowed float/int ranges.
        Categorical and boolean params are returned unchanged.
    """
    if len(trials) < 3:
        return dict(search_space)

    top_n = max(1, int(len(trials) * top_frac))
    if direction == "maximize":
        sorted_trials = sorted(trials, key=lambda t: t.get("value", 0) or 0, reverse=True)
    else:
        sorted_trials = sorted(trials, key=lambda t: t.get("value", 0) or 0, reverse=False)
    top_trials = sorted_trials[:top_n]

    narrowed: dict[str, dict[str, Any]] = {}
    for name, spec in search_space.items():
        ptype = spec.get("type", "")
        if ptype not in ("float", "int"):
            narrowed[name] = dict(spec)
            continue

        top_vals = [
            t["params"].get(name)
            for t in top_trials
            if t.get("params") and t["params"].get(name) is not None
        ]
        top_vals = [v for v in top_vals if isinstance(v, (int, float))]
        if len(top_vals) < 2:
            narrowed[name] = dict(spec)
            continue

        orig_low = spec["low"]
        orig_high = spec["high"]
        orig_range = orig_high - orig_low

        best_min = min(top_vals)
        best_max = max(top_vals)
        best_mid = (best_min + best_max) / 2.0
        best_spread = best_max - best_min
        # If all top values are identical, use the original range as reference
        if best_spread < orig_range * NARROW_IDENTICAL_SPREAD_FRAC:
            best_spread = orig_range * NARROW_MIN_SPREAD_FRAC

        half_range = best_spread * shrink_factor * 0.5
        new_low = max(orig_low, best_mid - half_range)
        new_high = min(orig_high, best_mid + half_range)

        # Round for int params
        if ptype == "int":
            new_low = int(round(new_low))
            new_high = int(round(new_high))
            if new_high - new_low < 1:
                new_low = max(orig_low, new_low - 1)
                new_high = min(orig_high, new_high + 1)

        # Ensure at least some range
        if new_high <= new_low:
            new_low = orig_low
            new_high = orig_high

        narrowed[name] = dict(spec)
        narrowed[name]["low"] = new_low
        narrowed[name]["high"] = new_high

    return narrowed


def run_hpo_hierarchical(
    script: str,
    n_trials: int = 50,
    n_stage1: int | None = None,
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
    loss: str | None = None,
    top_frac: float = 0.2,
    shrink_factor: float = 0.5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Two-stage hierarchical hyperparameter optimization.

    Phase 1 -- uniform random sampling (wide exploration, ~20% budget).
    Phase 2 -- Optuna TPE with Hyperband pruner on narrowed space (~80% budget).

    The combined result reflects the best trial across both phases.

    Args:
        script: Training script path.
        n_trials: Total number of trials across both phases.
        n_stage1: Number of trials for Phase 1 (default: max(5, n_trials // 5)).
        n_jobs: Parallel execution count (local or distributed).
        study_name: Optional study name override.
        search_space: Hyperparameter search space (default: _DEFAULT_SEARCH_SPACE).
        storage: Optuna storage URL (local mode only).
        direction: Optimization direction ('maximize' or 'minimize').
        objective_metric: Metric name to optimize.
        timeout_minutes: Max runtime in minutes.
        distributed: Use clearml ask/tell distribution.
        queue: Queue for distributed trials.
        project: ClearML project name.
        loss: Loss function name passed to training script.
        top_frac: Fraction of best trials used for space narrowing (default 0.2).
        shrink_factor: Shrink multiplier for narrowed space (default 0.5).
        **kwargs: Additional keyword arguments passed to both Phase run_hpo calls.

    Returns:
        Dict with merged results from both phases, including 'phase_1' and
        'phase_2' sub-results.
    """
    ss = search_space or dict(_DEFAULT_SEARCH_SPACE)
    if n_stage1 is None:
        n_stage1 = max(HIERARCHICAL_STAGE1_MIN, int(n_trials * HIERARCHICAL_STAGE1_FRAC))
    n_stage2 = n_trials - n_stage1

    base_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    s1_name = study_name or f"hpo_s1_{base_ts}"
    s2_name = study_name or f"hpo_s2_{base_ts}"

    # Phase 1: uniform random, no pruner
    phase1 = run_hpo(
        script=script,
        n_trials=n_stage1,
        n_jobs=n_jobs,
        study_name=s1_name,
        search_space=ss,
        storage=storage,
        direction=direction,
        objective_metric=objective_metric,
        timeout_minutes=timeout_minutes,
        distributed=distributed,
        queue=queue,
        project=project,
        loss=loss,
        pruner=None,
        **kwargs,
    )

    # Build trial history for narrowing from phase1 study storage
    phase1_trials: list[dict[str, Any]] = _load_trials_from_storage(
        study_name=s1_name,
        storage=storage,
        direction=direction,
    )
    if not phase1_trials:
        # Fallback: use best trial only
        bv = phase1.get("best_value")
        bp = phase1.get("best_params")
        if bv is not None and bp is not None:
            phase1_trials.append({"value": bv, "params": bp})

    # Narrow search space
    if len(phase1_trials) >= 3:
        narrowed_ss = _narrow_space(
            phase1_trials, ss, top_frac=top_frac, shrink_factor=shrink_factor, direction=direction
        )
    else:
        narrowed_ss = dict(ss)

    # Phase 2: TPE + Hyperband pruner on narrowed space
    phase2 = run_hpo(
        script=script,
        n_trials=n_stage2,
        n_jobs=n_jobs,
        study_name=s2_name,
        search_space=narrowed_ss,
        storage=storage,
        direction=direction,
        objective_metric=objective_metric,
        timeout_minutes=timeout_minutes,
        distributed=distributed,
        queue=queue,
        project=project,
        loss=loss,
        pruner=HIERARCHICAL_STAGE2_PRUNER,
        **kwargs,
    )

    # Merge results -- pick the best across both phases
    p1_best = phase1.get("best_value")
    p2_best = phase2.get("best_value")

    is_maximize = direction == "maximize"

    if p1_best is not None and p2_best is not None:
        if is_maximize:
            use_phase1 = p1_best >= p2_best
        else:
            use_phase1 = p1_best <= p2_best
    elif p1_best is not None:
        use_phase1 = True
    else:
        use_phase1 = False

    merged: dict[str, Any] = {
        "study_name": study_name or f"hpo_merged_{base_ts}",
        "n_trials": n_trials,
        "completed": (phase1.get("completed", 0) or 0) + (phase2.get("completed", 0) or 0),
        "failed": (phase1.get("failed", 0) or 0) + (phase2.get("failed", 0) or 0),
        "best_value": (phase1 if use_phase1 else phase2).get("best_value"),
        "best_params": (phase1 if use_phase1 else phase2).get("best_params"),
        "direction": direction,
        "duration_sec": (phase1.get("duration_sec", 0) or 0) + (phase2.get("duration_sec", 0) or 0),
        "timeout_minutes": timeout_minutes,
        "method": "hierarchical",
        "phase_1": phase1,
        "phase_2": phase2,
    }
    return merged


# ── Training curve analysis (Paradigm 5) ──


TRAIN_CURVE_CLASSES = frozenset({"sigmoid", "linear", "plateau", "oscillating"})
"""Four recognised training curve shapes."""


def _classify_training_curve(
    scalar_history: list[float],
    curve_early_frac: float = 0.2,
    curve_late_frac: float = 0.5,
    oscillation_threshold: float = 0.05,
) -> str:
    """Classify a training curve into one of four shapes.

    Args:
        scalar_history: Ordered list of scalar values (e.g. per-epoch val Seg).
        curve_early_frac: Fraction of total steps considered "early" (default 0.2).
        curve_late_frac: Fraction of total steps considered "late" (default 0.5).
        oscillation_threshold: CV threshold for oscillating classification.

    Returns:
        One of ``"sigmoid"``, ``"linear"``, ``"plateau"``, ``"oscillating"``.

    Shape definitions:
        sigmoid: early_20% > 80% of max — model capacity too high.
        linear:  near-constant per-step improvement throughout.
        plateau: last 50% of steps show negligible gain (<5% relative).
        oscillating: coefficient of variation > oscillation_threshold.
    """
    if len(scalar_history) < 5:
        return "linear"  # Too few data points, assume linear

    n = len(scalar_history)
    early = scalar_history[:max(1, int(n * curve_early_frac))]
    late = scalar_history[max(0, n - int(n * curve_late_frac)):]
    max_val = max(scalar_history)

    if max_val <= 0:
        return "linear"

    # Plateau: last 50% of values have very small max-min relative to max
    # Check BEFORE oscillating because a climb-then-plateau curve may have
    # high overall CV but is clearly not oscillating.
    if n >= 10:
        late_increase = max(late) - min(late)
        if late_increase < 0.02 * max_val:
            return "plateau"

    # Oscillating check: high coefficient of variation (std/mean)
    mean_val = sum(scalar_history) / n
    variance = sum((v - mean_val) ** 2 for v in scalar_history) / n
    cv = (variance ** 0.5) / max(mean_val, 1e-10)
    if cv > oscillation_threshold and n >= 10:
        return "oscillating"

    # Sigmoid: early values already near max
    early_max = max(early)
    if early_max >= 0.8 * max_val:
        return "sigmoid"

    # Plateau: last 50% of values have very small max-min relative to max
    late_increase = max(late) - min(late)
    if late_increase < 0.05 * max_val and n >= 10:
        return "plateau"

    return "linear"


def train_curve_feedback(
    scalar_history: list[float],
    current_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate HPO feedback based on training curve shape.

    Args:
        scalar_history: Per-epoch metric values from a completed trial.
        current_params: The hyperparameters used for this trial (optional, for
                        contextualising the feedback).

    Returns:
        Dict with keys::

            {"curve": str,           # shape classification
             "severity": str,        # "info" | "warn" | "critical"
             "message": str,         # human-readable description
             "adjustments": dict}    # HPO space narrowing suggestions
    """
    curve = _classify_training_curve(scalar_history)
    max_val = max(scalar_history) if scalar_history else 0.0
    final_val = scalar_history[-1] if scalar_history else 0.0
    n = len(scalar_history)

    if curve == "sigmoid":
        return {
            "curve": "sigmoid",
            "severity": "warn",
            "message": (
                f"Converged too fast: early {n}% ({scalar_history[:max(1, n//5)]}) "
                f"already reached {max_val:.2f}. Model capacity may be excessive."
            ),
            "adjustments": {
                "reduce_capacity": True,
                "suggest_decrease": ["width", "n_layers", "modes"],
            },
        }

    if curve == "oscillating":
        return {
            "curve": "oscillating",
            "severity": "warn",
            "message": (
                f"Loss oscillating (CV > 0.05). "
                f"Final value {final_val:.2f}. Try lower LR or gradient clipping."
            ),
            "adjustments": {
                "reduce_lr": True,
                "suggest_decrease": ["lr"],
                "suggest_clip_grad": True,
            },
        }

    if curve == "plateau":
        return {
            "curve": "plateau",
            "severity": "critical",
            "message": (
                f"Plateaued at {max_val:.2f} for the last {int(n * 0.5)} steps. "
                f"Model may be stuck in a local minimum. Try architecture change or LR warmup."
            ),
            "adjustments": {
                "change_architecture": True,
                "suggest_increase": ["lr", "width"],
            },
        }

    # linear
    return {
        "curve": "linear",
        "severity": "info",
        "message": (
            f"Steady linear improvement to {final_val:.2f} over {n} steps. "
            f"More epochs may still help."
        ),
        "adjustments": {
            "increase_epochs": True,
            "suggest_increase": ["epochs"],
        },
    }


# ── SearchGraph: NetworkX-based trial history tracking ──


class SearchGraph:
    """Directed graph tracking trial-to-trial parameter transitions.

    Uses ``networkx.DiGraph`` where each node is a trial (identified by trial
    number) and edges represent consecutive trial progression.  Node attributes
    store the full parameter dict and objective value.

    This is purely a **recording** / **visualisation** tool — it does not
    influence sampling.  Use it to understand HPO dynamics after a run.

    Example::

        g = SearchGraph()
        for trial in study.trials:
            g.add_trial(trial.number, trial.params, trial.value, "maximize")
        g.summary(top_k=3)
        # -> {"node_count": 50, "edge_count": 49, "top_trials": [...]}
    """

    def __init__(self) -> None:
        import networkx as nx

        self._nx = nx
        self.graph: nx.DiGraph = nx.DiGraph()
        self.trials: list[dict[str, Any]] = []

    def add_trial(
        self,
        trial_number: int,
        params: dict[str, Any],
        value: float | list[float] | None,
        direction: str | list[str],
    ) -> None:
        """Record a completed trial.

        Args:
            trial_number: Integer trial identifier (typically ``trial.number``).
            params: Sampled hyperparameters.
            value: Objective value(s).  Single float or list for multi-objective.
            direction: 'maximize', 'minimize', or a list for multi-objective.
        """
        norm_val: float
        if isinstance(value, (int, float)):
            norm_val = float(value)
        elif isinstance(value, (list, tuple)):
            # For multi-objective, take the first objective as primary
            norm_val = float(value[0]) if value else 0.0
        else:
            norm_val = 0.0

        self.graph.add_node(
            trial_number,
            params=params,
            value=norm_val,
            direction=direction,
        )
        self.trials.append({
            "number": trial_number,
            "params": params,
            "value": norm_val,
        })

        # Create a transition edge from the previous trial (if any)
        if len(self.trials) >= 2:
            prev = self.trials[-2]["number"]
            # Determine if value improved
            if isinstance(direction, str) and direction == "maximize":
                improved = norm_val > self.trials[-2]["value"]
            else:
                improved = norm_val < self.trials[-2]["value"]
            self.graph.add_edge(prev, trial_number, improved=improved)

    def summary(self, top_k: int = 5) -> dict[str, Any]:
        """Return a structured summary of the search history.

        Args:
            top_k: Number of top trials to include.

        Returns:
            Dict with node_count, edge_count, param_transitions, top_trials.
        """
        if not self.trials:
            return {"node_count": 0, "edge_count": 0, "top_trials": []}

        is_best = (
            (lambda v: v)
            if isinstance(self.trials[0].get("direction"), str)
            and self.trials[0]["direction"] == "maximize"
            else (lambda v: v)
        )
        # Sort by value
        sorted_trials = sorted(
            self.trials, key=lambda t: t["value"], reverse=True
        )

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "param_transitions": [
                (u, v) for u, v in self.graph.edges()
            ][:10],
            "top_trials": sorted_trials[:top_k],
        }

    def to_json(self) -> dict[str, Any]:
        """Export graph as a JSON-serializable dict for CLI / visualisation.

        Returns::

            {"nodes": [{"id": ..., "params": ..., "value": ...}],
             "edges": [{"source": ..., "target": ..., "improved": ...}],
             "top_trials": [...]}
        """
        nodes = [
            {
                "id": n,
                "params": self.graph.nodes[n].get("params", {}),
                "value": self.graph.nodes[n].get("value"),
            }
            for n in self.graph.nodes()
        ]
        edges = [
            {
                "source": u,
                "target": v,
                "improved": self.graph.edges[u, v].get("improved", False),
            }
            for u, v in self.graph.edges()
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            **self.summary(top_k=5),
        }


# ── pymoo integration ──


try:
    import pymoo  # noqa: F401

    _has_pymoo = True
except ImportError:
    _has_pymoo = False


def run_hpo_pymoo(
    eval_fn: Any,
    search_space: dict[str, dict[str, Any]],
    n_trials: int = 50,
    pop_size: int = 20,
    direction: str | list[str] = "maximize",
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run hyperparameter optimisation using pymoo's NSGA-II.

    pymoo provides a rich family of evolutionary multi-objective algorithms.
    This function wraps the ``pymoo.algorithms.moo.nsga2.NSGA2`` for both
    single- and multi-objective problems.  Multi-objective results include
    a Pareto-front approximation.

    Args:
        eval_fn: Callable ``f(params: dict[str, float]) -> float | list[float]``.
                 Must accept a flat dict of hyperparameter names → values and
                 return a scalar (single-objective) or list of scalars.
        search_space: Standard expflow search space dict (same format as
                      ``_DEFAULT_SEARCH_SPACE``).
        n_trials: Total number of function evaluations (population × generations).
        pop_size: NSGA-II population size per generation.
        direction: 'maximize', 'minimize', or a list for multi-objective.
        seed: Random seed for reproducibility.
        verbose: If True, print progress.

    Returns:
        Dict with keys ``best_params``, ``best_value``, ``n_trials``,
        ``direction``, and optionally ``pareto_front``.
    """
    if not _has_pymoo:
        raise ImportError(
            "pymoo is required for run_hpo_pymoo. Install with: pip install pymoo"
        )

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    import numpy as np  # noqa: N812

    # Build variable bounds and types from search space
    xl: list[float] = []
    xu: list[float] = []
    param_names: list[str] = []
    param_log: list[bool] = []

    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype not in ("float", "int"):
            continue  # skip categorical for now
        param_names.append(name)
        xl.append(float(spec["low"]))
        xu.append(float(spec["high"]))
        param_log.append(spec.get("log", False))

    n_var = len(param_names)

    # Determine number of objectives
    multi = isinstance(direction, list)
    n_obj = len(direction) if multi else 1

    # Build pymoo problem
    class _HpoProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                xl=np.array(xl),
                xu=np.array(xu),
            )

        def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> None:
            # Convert to flat param dict, handling log scale
            params: dict[str, float] = {}
            for i, name in enumerate(param_names):
                val = float(x[i])
                if param_log[i]:
                    # pymoo operates in linear space; convert from log
                    val = np.exp(val)
                params[name] = val

            result = eval_fn(params)

            if isinstance(result, (int, float)):
                out["F"] = [float(result)]
            else:
                out["F"] = [float(v) for v in result]

    # Normalise direction: pymoo always minimises → flip sign for maximise
    # We do the flip inside by negating the objective
    class _HpoProblemWithDirection(_HpoProblem):
        def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> None:
            params = {}
            for i, name in enumerate(param_names):
                val = float(x[i])
                if param_log[i]:
                    val = np.exp(val)
                params[name] = val

            result = eval_fn(params)

            if isinstance(result, (int, float)):
                vals = [float(result)]
            else:
                vals = [float(v) for v in result]

            # Flip sign for maximise objectives
            if multi:
                vals = [
                    -v if d == "maximize" else v
                    for v, d in zip(vals, direction)
                ]
            elif direction == "maximize":
                vals = [-v for v in vals]

            out["F"] = vals

    n_generations = max(1, n_trials // pop_size)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
    )

    if verbose:
        display = None
    else:
        display = None

    problem = _HpoProblemWithDirection()

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        seed=seed,
        display=display,
        verbose=verbose,
    )

    # Convert best solution back to param dict
    best_params: dict[str, Any] = {}
    x_flat = res.X.flatten() if hasattr(res.X, "flatten") else [res.X]
    for i, name in enumerate(param_names):
        val = float(x_flat[i]) if i < len(x_flat) else float(res.X[0])
        if param_log[i]:
            val = np.exp(val)
        best_params[name] = val

    # Undo the direction flip for the best value
    best_f = res.F.flatten() if hasattr(res.F, "flatten") else [res.F]
    best_value_raw = float(best_f[0])
    if multi:
        best_value = -best_value_raw if direction[0] == "maximize" else best_value_raw
    elif direction == "maximize":
        best_value = -best_value_raw
    else:
        best_value = best_value_raw

    result_dict: dict[str, Any] = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "completed": n_trials,
        "failed": 0,
        "direction": direction,
        "method": "pymoo_nsga2",
    }

    if multi:
        # Collect Pareto front
        pareto_points = []
        all_x = res.pop.get("X")
        for i, x_i in enumerate(all_x):
            f = res.pop.get("F")[i]
            params_i = {}
            x_flat_i = x_i.flatten() if hasattr(x_i, "flatten") else [x_i]
            for j, name in enumerate(param_names):
                val = float(x_flat_i[j]) if j < len(x_flat_i) else float(x_i[0])
                if param_log[j]:
                    val = np.exp(val)
                params_i[name] = val
            # Undo flip
            f_values = [
                -fv if direction[k] == "maximize" else fv
                for k, fv in enumerate(f)
            ]
            pareto_points.append({
                "params": params_i,
                "values": [float(v) for v in f_values],
            })
        result_dict["pareto_front"] = pareto_points

    return result_dict
