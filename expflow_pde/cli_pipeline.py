#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow pipeline CLI — pipeline sub-commands.

Hermes /goal compatible: use --json for machine-readable output,
no auto-retry loop (Hermes decides next action).
"""

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
    step_time_limit: Optional[float] = typer.Option(
        None, "--step-time-limit", help="Max minutes per step (clearml server kills overdue tasks)"
    ),
    skip: Optional[list[str]] = typer.Option(
        None,
        "--skip",
        help="Steps to skip: e.g. --skip eval --skip train",
    ),
    packages: Optional[list[str]] = typer.Option(
        None,
        "--packages",
        help="Packages for clearml Task.create (use '--packages' without value for empty list)",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output result as JSON (for Hermes /goal consumption)"
    ),
) -> None:
    """Submit a fast train -> eval pipeline (Mode B).

    Skips HPO, runs directly with given parameters. Best for competition
    sprint where you already know the best hyperparameters.
    No auto-retry — Hermes /goal drives the iteration loop.

    Examples:

        # Train only
        expflow pipeline submit train_task1.py --queue default

        # Train + eval with JSON output (for /goal)
        expflow pipeline submit train_task1.py --eval-script eval_task1.py \\
            --train-param epochs=80 --train-param lr=0.001 --json

        # Skip eval (train only)
        expflow pipeline submit train_task1.py --skip eval

        # No auto-installed packages (use conda env only)
        expflow pipeline submit train_task1.py --packages ''
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

    # Resolve packages: '' → empty list, None → None (clearml default)
    resolved_packages: list[str] | None = None
    if packages is not None:
        resolved_packages = [p for p in packages if p] if packages else []

    ep = ExperimentPipeline(
        project=project,
        queue=queue,
        docker=docker,
        abort_on_failure=abort_on_failure,
        packages=resolved_packages,
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
        step_time_limit=step_time_limit,
        skip_steps=skip,
    )

    # ── Post-submit: if --wait, poll for true completion ──
    # `pipe.wait(timeout=X)` returns on timeout even if pipeline is still running.
    # Use clearml task polling to check real status after wait returns.
    if wait and result.get("pipeline_id"):
        try:
            from expflow_pde.clearml import _get_pipeline_module, get_task

            PipelineController = _get_pipeline_module()
            pipe = PipelineController(
                name=(pipeline_name or result.get("name", pipeline_name)).replace(".py", ""),
                project=project,
            )
            pid = result["pipeline_id"]
            # Poll pipeline status up to 5 more minutes (60s intervals)
            import time as _time

            for _ in range(5):
                _time.sleep(60)
                task = get_task(pid)
                if task:
                    task_status = task.get("status", "")
                    if task_status in ("completed", "failed", "stopped"):
                        result["status"] = task_status
                        break
        except Exception:
            pass  # polling is best-effort; fall through to initial status

    # ── Post-submit: enrich result with eval task id ──
    # Read eval step's clearml task from pipeline controller for direct scalar access
    if eval_script and "eval" not in (skip or []):
        try:
            from expflow_pde.clearml import _get_pipeline_module

            PipelineController = _get_pipeline_module()  # noqa: N806
            pipe = PipelineController(
                name=(pipeline_name or result.get("name", pipeline_name)).replace(".py", ""),
                project=project,
                version=version or "0.1.0",
            )
            eval_step = pipe.get_step("eval")
            result["eval_task_id"] = eval_step.id if hasattr(eval_step, "id") else ""
        except Exception:
            result["eval_task_id"] = ""

    # ── Post-submit: write to DispatchDB for StagnationDetector ──
    if json_output and wait:
        try:
            from expflow_pde.dispatch_db import DispatchDB

            db = DispatchDB()
            db.register_experiment(
                script=f"pipeline submit {train_script}",
                args={"train_script": train_script, "eval_script": eval_script, "params": train_params},
                queue=queue,
                project=project,
                source="pipeline_cli",
                result_summary=result,
            )
        except Exception:
            pass  # non-critical; DispatchDB may not exist

    if json_output:
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_result(result, wait, timeout)


# ── Mode A (full): HPO -> Train -> Eval ──


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
    step_time_limit: Optional[float] = typer.Option(
        None, "--step-time-limit", help="Max minutes per step (clearml server kills overdue tasks)"
    ),
    skip: Optional[list[str]] = typer.Option(
        None,
        "--skip",
        help="Steps to skip: e.g. --skip hpo --skip eval",
    ),
    packages: Optional[list[str]] = typer.Option(
        None,
        "--packages",
        help="Packages for clearml Task.create (use '--packages' without value for empty list)",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output result as JSON (for Hermes /goal consumption)"
    ),
) -> None:
    """Submit a full HPO -> train -> eval pipeline (Mode A).

    Fully automated competition pipeline:
    1. HPO: runs N trials via clearml queue to find best hyperparams
    2. Train: trains with best params from HPO
    3. Eval: evaluates the best checkpoint

    No auto-retry — Hermes /goal drives the iteration loop.

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

    resolved_packages: list[str] | None = None
    if packages is not None:
        resolved_packages = [p for p in packages if p] if packages else []

    ep = ExperimentPipeline(
        project=project,
        queue=queue,
        packages=resolved_packages,
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
        step_time_limit=step_time_limit,
        skip_steps=skip,
    )

    # ── Post-submit: if --wait, poll for true completion ──
    if wait and result.get("pipeline_id"):
        try:
            from expflow_pde.clearml import _get_pipeline_module, get_task

            PipelineController = _get_pipeline_module()
            pipe = PipelineController(
                name=(pipeline_name or result.get("name", pipeline_name)).replace(".py", ""),
                project=project,
            )
            pid = result["pipeline_id"]
            import time as _time

            for _ in range(5):
                _time.sleep(60)
                task = get_task(pid)
                if task:
                    task_status = task.get("status", "")
                    if task_status in ("completed", "failed", "stopped"):
                        result["status"] = task_status
                        break
        except Exception:
            pass

    # ── Post-submit: enrich with eval task id ──
    if eval_script and "eval" not in (skip or []):
        try:
            from expflow_pde.clearml import _get_pipeline_module

            PipelineController = _get_pipeline_module()
            pipe = PipelineController(
                name=(pipeline_name or result.get("name", pipeline_name)).replace(".py", ""),
                project=project,
            )
            eval_step = pipe.get_step("eval")
            result["eval_task_id"] = eval_step.id if hasattr(eval_step, "id") else ""
        except Exception:
            result["eval_task_id"] = ""

    # ── Post-submit: write to DispatchDB ──
    if json_output and wait:
        try:
            from expflow_pde.dispatch_db import DispatchDB

            db = DispatchDB()
            db.register_experiment(
                script=f"pipeline submit-full {train_script}",
                args={"train_script": train_script, "eval_script": eval_script, "trials": trials},
                queue=queue,
                project=project,
                source="pipeline_cli",
                result_summary=result,
            )
        except Exception:
            pass

    if json_output:
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_result(result, wait, timeout)


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
