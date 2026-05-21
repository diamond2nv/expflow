#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow iterate — automatic experiment iteration: diagnose -> suggest -> submit."""

from typing import Any


def run_iteration(
    task_id: str | None = None,
    json_path: str | None = None,
    current_hparams: dict[str, Any] | None = None,
    train_script: str = "train_task1.py",
    eval_script: str = "eval_task1.py",
    project: str = "PDEBench",
    queue: str = "default",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one complete iteration: diagnose -> suggest -> submit.

    Args:
        task_id: Clearml task ID of the experiment to iterate from.
        json_path: Local eval JSON path (alternative).
        current_hparams: Current experiment's hyperparameters.
        train_script: Base clearml task name for training.
        eval_script: Base clearml task name for evaluation.
        project: Clearml project name.
        queue: Queue to submit the next iteration.
        dry_run: Print what would happen, don't execute.

    Returns:
        Dict with diagnosis, suggestion, and (if not dry_run) pipeline result.
    """
    from expflow_pde.analyze import diagnose_experiment, suggest_next_params

    diagnosis = diagnose_experiment(task_id=task_id, json_path=json_path)
    if diagnosis is None:
        return {"error": "Cannot load experiment metrics", "step": "diagnose"}

    suggestion = suggest_next_params(
        diagnosis,
        current_hparams=current_hparams or {},
    )

    if dry_run:
        return {
            "diagnosis": diagnosis,
            "suggestion": suggestion,
            "submitted": False,
        }

    from expflow_pde.pipeline import ExperimentPipeline

    ep = ExperimentPipeline(project=project, queue=queue)

    suggested = dict(suggestion.get("suggested_params", {}))
    suggested.pop("tag", None)

    pipe_result = ep.train_val_submit(
        train_script=train_script,
        train_params=suggested,
        eval_script=eval_script,
    )

    # Auto-repair if pipeline failed
    if pipe_result.get("status") not in ("completed", "success", "started"):
        from expflow_pde.repair import RepairStage

        task_log = pipe_result.get("error", "")
        stage = RepairStage(
            experiment_id=pipe_result.get("pipeline_id", ""),
            max_l1_attempts=2,
        )
        repair = stage.run(
            task_log=task_log,
            exit_code=1,
            enable_reflection=False,
        )
        pipe_result["repair"] = repair

    return {
        "diagnosis": diagnosis,
        "suggestion": suggestion,
        "submitted": True,
        "pipeline": pipe_result,
    }
