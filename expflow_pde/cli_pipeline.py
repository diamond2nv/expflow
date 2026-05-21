#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow pipeline CLI — pipeline sub-commands."""

from __future__ import annotations

import logging
from typing import Any, Optional

import typer

logger = logging.getLogger("expflow_pde.cli_pipeline")

pipeline_app = typer.Typer(
    name="pipeline",
    help="High-level PDEBench experiment pipeline (train -> eval -> submit)",
    no_args_is_help=True,
)


def get_pipeline_app() -> typer.Typer:
    """Return the pipeline sub-command group."""
    return pipeline_app


# ── Mode B (fast): Train → Eval ──


@pipeline_app.command("submit")
def submit_cmd(
    train_script: str = typer.Argument(..., help="Training script path (e.g. train_task1.py)"),
    eval_script: Optional[str] = typer.Option(
        None, "--eval-script", "-e", help="Evaluation script path (e.g. eval_task1.py)"
    ),
    queue: str = typer.Option("default", "--queue", "-q", help="Execution queue"),
    project: str = typer.Option("PDEBench", "--project", "-p", help="ClearML project name"),
    pipeline_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Pipeline name (auto-generated if empty)"
    ),
    version: str = typer.Option("0.1.0", "--version", help="Pipeline version"),
    train_param: Optional[list[str]] = typer.Option(
        None,
        "--train-param",
        "-t",
        help="Training parameter e.g. --train-param epochs=80",
    ),
    eval_param: Optional[list[str]] = typer.Option(
        None,
        "--eval-param",
        help="Evaluation parameter e.g. --eval-param sub_step=5",
    ),
    docker: Optional[str] = typer.Option(
        None, "--docker", help="Docker image for remote execution"
    ),
    abort_on_failure: bool = typer.Option(
        False, "--abort-on-failure", help="Stop all steps on any failure"
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for pipeline completion"),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", help="Max wait time in minutes (None = no limit)"
    ),
    skip: Optional[list[str]] = typer.Option(
        None,
        "--skip",
        help="Steps to skip: e.g. --skip eval --skip train",
    ),
    repair: bool = typer.Option(False, "--repair", "-r",
        help="Auto-repair on failure (must also use --wait)"),
    repair_reflection: bool = typer.Option(False, "--repair-reflection",
        help="Enable L2 reflection subagent for repair"),
    repair_output: Optional[str] = typer.Option(
        None, "--repair-output",
        help="Write repair result to JSON file (for L2 subagent watcher)"),
) -> None:
    """Submit a fast train -> eval pipeline (Mode B).

    Skips HPO, runs directly with given parameters. Best for competition
    sprint where you already know the best hyperparameters.

    Examples:

        # Train only
        expflow pipeline submit train_task1.py --queue default

        # Train + eval
        expflow pipeline submit train_task1.py --eval-script eval_task1.py \\
            --train-param epochs=80 --train-param lr=0.001 \\
            --eval-param sub_step=5

        # Skip eval (train only)
        expflow pipeline submit train_task1.py --skip eval

        # Repair with L2 reflection output to file (for Hermes subagent)
        expflow pipeline submit train_task1.py --queue default \\
            --wait --repair --repair-reflection \\
            --repair-output /tmp/l2_repair.json
    """
    from expflow_pde.pipeline import ExperimentPipeline

    train_params: dict[str, str] = {}
    if train_param:
        for p in train_param:
            if "=" in p:
                k, v = p.split("=", 1)
                train_params[k] = v

    eval_params: dict[str, str] = {}
    if eval_param:
        for p in eval_param:
            if "=" in p:
                k, v = p.split("=", 1)
                eval_params[k] = v

    ep = ExperimentPipeline(
        project=project,
        queue=queue,
        docker=docker,
        abort_on_failure=abort_on_failure,
    )

    result = ep.train_val_submit(
        train_script=train_script,
        train_params=train_params if train_params else None,
        eval_script=eval_script,
        eval_params=eval_params if eval_params else None,
        pipeline_name=pipeline_name,
        version=version or None,
        execution_queue=queue,
        timeout=timeout if wait else None,
        skip_steps=skip,
    )

    _print_result(result, wait, timeout)

    if repair and wait:
        max_retries = 2
        for retry_attempt in range(max_retries):
            repair_result = _maybe_repair_pipeline(
                result, repair_reflection, queue, project, repair_output
            )
            if not repair_result:
                break
            if repair_result.get("fixed") and repair_result.get("level") != "L2":
                if retry_attempt < max_retries - 1:
                    print(f"  Repair:   re-submitting (attempt {retry_attempt + 1}/{max_retries})...")
                    ep = ExperimentPipeline(
                        project=project,
                        queue=queue,
                        docker=docker,
                        abort_on_failure=abort_on_failure,
                    )
                    result = ep.train_val_submit(
                        train_script=train_script,
                        train_params=train_params if train_params else None,
                        eval_script=eval_script,
                        eval_params=eval_params if eval_params else None,
                        pipeline_name=pipeline_name,
                        version=version or None,
                        execution_queue=queue,
                        timeout=timeout if wait else None,
                        skip_steps=skip,
                    )
                    _print_result(result, wait, timeout)
                else:
                    print(f"  Repair:   max retries reached, escalating to manual inspection")
            else:
                break


# ── Mode A (full): HPO → Train → Eval ──


@pipeline_app.command("submit-full")
def submit_full_cmd(
    train_script: str = typer.Argument(..., help="Training script path (e.g. train_task1.py)"),
    eval_script: Optional[str] = typer.Option(
        None, "--eval-script", "-e", help="Evaluation script path"
    ),
    queue: str = typer.Option("default", "--queue", "-q", help="Execution queue"),
    project: str = typer.Option("PDEBench", "--project", "-p", help="ClearML project name"),
    trials: int = typer.Option(50, "--trials", "-t", help="Number of HPO trials"),
    parallel: int = typer.Option(4, "--parallel", "-j", help="Max concurrent HPO trials"),
    study_name: Optional[str] = typer.Option(
        None, "--study-name", "-n", help="Optuna study name (auto if empty)"
    ),
    metric: str = typer.Option("seg_total", "--metric", "-m", help="Objective metric to optimize"),
    direction: str = typer.Option("maximize", "--direction", help="Optimization direction"),
    pruner: str = typer.Option("hyperband", "--pruner", help="Optuna pruner type"),
    eval_param: Optional[list[str]] = typer.Option(
        None, "--eval-param", help="Eval parameter e.g. --eval-param sub_step=5"
    ),
    pipeline_name: Optional[str] = typer.Option(
        None, "--name", help="Pipeline name (auto if empty)"
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for pipeline completion"),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="Max wait in minutes"),
    skip: Optional[list[str]] = typer.Option(
        None,
        "--skip",
        help="Steps to skip: e.g. --skip hpo --skip eval",
    ),
    repair: bool = typer.Option(False, "--repair", "-r",
        help="Auto-repair on failure (must also use --wait)"),
    repair_reflection: bool = typer.Option(False, "--repair-reflection",
        help="Enable L2 reflection subagent for repair"),
    repair_output: Optional[str] = typer.Option(
        None, "--repair-output",
        help="Write repair result to JSON file (for L2 subagent watcher)"),
) -> None:
    """Submit a full HPO -> train -> eval pipeline (Mode A).

    Fully automated competition pipeline:
    1. HPO: runs N trials via clearml queue to find best hyperparams
    2. Train: trains with best params from HPO
    3. Eval: evaluates the best checkpoint

    Examples:

        # Full HPO pipeline
        expflow pipeline submit-full train_task1.py --queue default \\
            --trials 50 --parallel 4 --eval-script eval_task1.py

        # HPO only (skip train and eval)
        expflow pipeline submit-full train_task1.py --trials 50 \\
            --skip train --skip eval
    """
    from expflow_pde.pipeline import ExperimentPipeline

    eval_params: dict[str, str] = {}
    if eval_param:
        for p in eval_param:
            if "=" in p:
                k, v = p.split("=", 1)
                eval_params[k] = v

    ep = ExperimentPipeline(
        project=project,
        queue=queue,
    )

    result = ep.train_hpo_val_submit(
        train_script=train_script,
        eval_script=eval_script,
        eval_params=eval_params if eval_params else None,
        n_trials=trials,
        parallel=parallel,
        hpo_study_name=study_name,
        objective_metric=metric,
        direction=direction,
        pruner=pruner,
        pipeline_name=pipeline_name,
        execution_queue=queue,
        timeout=timeout if wait else None,
        skip_steps=skip,
    )

    _print_result(result, wait, timeout)

    if repair and wait:
        max_retries = 2
        for retry_attempt in range(max_retries):
            repair_result = _maybe_repair_pipeline(
                result, repair_reflection, queue, project, repair_output
            )
            if not repair_result:
                break
            if repair_result.get("fixed") and repair_result.get("level") != "L2":
                if retry_attempt < max_retries - 1:
                    print(f"  Repair:   re-submitting (attempt {retry_attempt + 1}/{max_retries})...")
                    ep = ExperimentPipeline(
                        project=project,
                        queue=queue,
                    )
                    result = ep.train_hpo_val_submit(
                        train_script=train_script,
                        eval_script=eval_script,
                        eval_params=eval_params if eval_params else None,
                        n_trials=trials,
                        parallel=parallel,
                        hpo_study_name=study_name,
                        objective_metric=metric,
                        direction=direction,
                        pruner=pruner,
                        pipeline_name=pipeline_name,
                        execution_queue=queue,
                        timeout=timeout if wait else None,
                        skip_steps=skip,
                    )
                    _print_result(result, wait, timeout)
                else:
                    print(f"  Repair:   max retries reached, escalating to manual inspection")
            else:
                break


# ── Shared printer ──


def _print_result(result: dict[str, Any], wait: bool, timeout: float | None) -> None:
    """Pretty-print a pipeline result."""
    mode = result.get("mode", "fast")
    print(f"Pipeline submitted: {result['name']} (mode: {mode})")
    print(f"  ID:       {result['pipeline_id']}")
    print(f"  Project:  {result['project']}")
    print(f"  Queue:    {result['queue']}")
    print(f"  Steps:    {len(result['steps'])}")
    for s in result["steps"]:
        parents_str = f" (depends on: {', '.join(s['parents'])})" if s.get("parents") else ""
        print(f"    - {s['name']} [{s['status']}]{parents_str}")
    if result.get("n_trials"):
        print(f"  HPO:      {result['n_trials']} trials, {result.get('parallel', '?')} parallel")
    print(f"  Status:   {result['status']}")
    if wait:
        print(f"  Wait:     completed (timeout={timeout or 'none'} min)")


# ── Repair integration ──


# Signal exit codes from clearml-agent or OS
_SIGNAL_CODES: dict[int, str] = {
    134: "SIGABRT",
    137: "SIGKILL (OOM)",
    139: "SIGSEGV",
    143: "SIGTERM",
}
_FAILURE_KEYWORDS = ("Traceback", "Error", "Failed", "Killed")


def _fetch_task_log(pipeline_id: str) -> tuple[str, int]:
    """Fetch pipeline task log from clearml with fallback.

    Returns:
        (task_log, exit_code). Empty log on failure.
    """
    task_log = ""
    exit_code = 1
    try:
        from clearml import Task

        task = Task.get_task(task_id=pipeline_id)
        console = task.get_reported_console_output()
        task_log = "\n".join(console) if console else ""
        status_msg = getattr(task, "status", "")
        status_str = str(status_msg).lower()
        if "failed" in status_str:
            exit_code = 1
        elif "killed" in status_str:
            exit_code = 137
    except ImportError:
        logger.warning("clearml SDK not available — cannot fetch task log")
    except Exception as e:
        logger.warning("Failed to fetch clearml task log for %s: %s", pipeline_id, e)

    if not task_log:
        logger.warning("clearml returned empty log for %s — repair will have no T context", pipeline_id)
    return task_log, exit_code


def _task_log_valid(task_log: str) -> bool:
    """Check if task_log has enough content for meaningful analysis."""
    if not task_log or not task_log.strip():
        return False
    lowered = task_log.lower()
    return any(kw.lower() in lowered for kw in _FAILURE_KEYWORDS)


def _exit_code_category(exit_code: int) -> str:
    """Categorise an exit code for repair diagnostics."""
    if exit_code == 0:
        return "success"
    if exit_code == 1:
        return "error"
    if exit_code in _SIGNAL_CODES:
        return f"signal ({_SIGNAL_CODES[exit_code]})"
    return f"unknown ({exit_code})"


def _resolve_repair_output(
    user_path: str | None,
    pipeline_id: str,
) -> str | None:
    """Resolve the repair output path with collision-safe defaults.

    If user_path is given, use it directly.
    Otherwise, generate a timestamped path under ~/.expflow/repair_pending/.
    If the resolved path exists, append .N suffix to prevent overwrite.
    Returns None if pipeline_id is empty (falls back to /tmp/ fallback).
    """
    import datetime
    import os

    if not pipeline_id:
        return None

    if user_path:
        resolved = user_path
    else:
        pending_dir = os.path.expanduser("~/.expflow/repair_pending")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
        resolved = os.path.join(pending_dir, f"{pipeline_id}_{ts}.json")

    # Collision detection — append .N if exists
    base = resolved
    counter = 0
    while os.path.exists(resolved):
        counter += 1
        resolved = f"{base}.{counter}"
    return resolved


def _maybe_repair_pipeline(
    result: dict[str, Any],
    enable_reflection: bool,
    queue: str,
    project: str,
    repair_output: str | None = None,
) -> dict[str, Any] | None:
    """Check pipeline status and attempt repair if failed.

    Called after --wait completes. If the pipeline finished with errors,
    runs RepairStage analysis and prints suggestions.

    When enable_reflection=True and repair_output is set, writes the full
    structured L2 result to a JSON file that Hermes can watch and consume
    via the l2-repair-executor skill (spawns a delegate_task subagent).

    Returns:
        The repair_result dict on failure, or None if no repair needed.
    """
    status = result.get("status", "")
    if status in ("completed", "success"):
        print("  Repair:   no repair needed (pipeline completed successfully)")
        return None

    pipeline_id = result.get("pipeline_id", "")
    if not pipeline_id:
        print("  Repair:   no pipeline_id — cannot check task status")
        return None

    print(f"  Repair:   pipeline failed (status={status}), analyzing...")

    # Fetch task log with fallback handling
    task_log, exit_code = _fetch_task_log(pipeline_id)

    from expflow_pde.pipeline import ExperimentPipeline

    ep = ExperimentPipeline(project=project, queue=queue)
    repair_result = ep.repair_task(
        task_log=task_log,
        exit_code=exit_code,
        enable_reflection=enable_reflection,
    )

    print(f"  Repair:   level={repair_result.get('level', '?')}")
    print(f"  Fixed:    {repair_result.get('fixed', False)}")
    print(f"  Action:   {repair_result.get('action', '?')[:200]}")
    ec_cat = _exit_code_category(exit_code)
    print(f"  Exit:     code={exit_code} ({ec_cat})")
    input_valid = repair_result.get("input_valid", True)
    if not input_valid:
        print(f"  Warning:  task_log empty or has no failure signal — repair context may be incomplete")
    if repair_result.get("history"):
        print(f"  History:  {len(repair_result['history'])} attempt(s)")

    # Resolve repair_output path — default to timestamped file in repair_pending/
    if repair_result.get("level") == "L2" and repair_result.get("input_valid", True):
        resolved_output = _resolve_repair_output(repair_output, pipeline_id)
    else:
        resolved_output = None

    # Write structured repair output to file for Hermes L2 executor
    if resolved_output:
        if not repair_result.get("input_valid", True):
            print(f"  Warning:  L2 input invalid — skipping repair_output write (use manual inspection)")
        else:
            import json
            import os
            os.makedirs(os.path.dirname(resolved_output) or ".", exist_ok=True)
            with open(resolved_output, "w") as f:
                json.dump(repair_result, f, indent=2, ensure_ascii=False)
            print(f"  L2 output: {resolved_output} (awaiting Hermes subagent)")

    return repair_result
