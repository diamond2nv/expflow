#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow pipeline — High-level ExperimentPipeline class.

Wraps low-level clearml PipelineController operations into streamlined
workflows for PDEBench experiments.

Three pipeline modes:

Mode A — **full** (HPO → Train → Eval):
    Automates the entire workflow: search hyperparameters, train with best
    params, then evaluate and prepare submission.

    ep = ExperimentPipeline(project="PDEBench", queue="gpu_queue")
    result = ep.train_hpo_val_submit(
        train_script="train_task1.py",
        eval_script="eval_task1.py",
        n_trials=50, parallel=4,
    )

Mode B — **fast** (Train → Eval):
    Skips HPO, directly trains with given parameters. Best for competition
    sprint: you already know the best params from a previous HPO run.

    result = ep.train_val_submit(
        train_script="train_task1.py",
        train_params={"lr": 1e-3, "epochs": 80},
        eval_script="eval_task1.py",
    )

Mode C — **custom skip**:
    Skips specified steps (e.g. train-only, eval-only).

    result = ep.train_val_submit(
        train_script="train_task1.py",
        skip_steps=["eval"],  # train only
    )
"""

from __future__ import annotations

from typing import Any

DEFAULT_PIPELINE_VERSION = "0.1.0"


class ExperimentPipeline:
    """High-level pipeline orchestrator for PDEBench workflows.

    Attributes:
        project: ClearML project name.
        queue: Target queue for pipeline controller and all steps.
        docker: Optional Docker image for remote execution.
        abort_on_failure: Stop all steps if one fails (default: False).
        add_run_number: Append run number to pipeline name (default: True).
    """

    def __init__(
        self,
        project: str = "PDEBench",
        queue: str = "default",
        docker: str | None = None,
        abort_on_failure: bool = False,
        add_run_number: bool = True,
        packages: list[str] | None = None,
    ) -> None:
        """Initialize the pipeline orchestrator."""
        self.project = project
        self.queue = queue
        self.docker = docker
        self.abort_on_failure = abort_on_failure
        self.add_run_number = add_run_number
        # ── 3-tier packages resolution ──
        # 1) Explicit constructor arg (highest priority)
        # 2) config.yaml pipeline.packages
        # 3) Default to [] (bare-metal conda env assumption)
        if packages is not None:
            resolved_packages = list(packages)
        else:
            try:
                from expflow_pde.config import get

                cfg_packages = get("pipeline.packages")
                if cfg_packages is not None:
                    resolved_packages = list(cfg_packages)
                else:
                    resolved_packages = []
            except Exception:
                resolved_packages = []
        self.packages = resolved_packages
        if not self.packages:
            import warnings

            warnings.warn(
                "packages=[]: clearml-agent will NOT auto-install any pip deps. "
                "Set 'pipeline.packages' in config.yaml or pass "
                "packages=['torch', 'numpy', ...] to ExperimentPipeline(). "
                "This default assumes bare-metal conda env with pre-installed deps.",
                UserWarning,
                stacklevel=2,
            )
        self._last_result: dict[str, Any] | None = None

    # ── Mode B (fast): train → eval ──

    def train_val_submit(
        self,
        train_script: str,
        train_params: dict[str, Any] | None = None,
        eval_script: str | None = None,
        eval_params: dict[str, Any] | None = None,
        pipeline_name: str | None = None,
        version: str | None = None,
        execution_queue: str | None = None,
        abort_on_failure: bool | None = None,
        timeout: float | None = None,
        step_time_limit: float | None = None,
        skip_steps: list[str] | None = None,
    ) -> dict[str, Any]:
        """Mode B (fast): Create and run a train → eval pipeline.

        Optionally skip steps via skip_steps: e.g. skip_steps=["eval"]
        produces a train-only pipeline. skip_steps=["train"] produces
        eval-only (unusual but supported).

        Args:
            train_script: Path to training script.
            train_params: Script arguments for the train step.
            eval_script: Path to evaluation script. If None, no eval step.
            eval_params: Script arguments for the eval step.
            pipeline_name: Pipeline name (auto-generated if None).
            version: Pipeline version string (default: 0.1.0).
            execution_queue: Queue for step execution (default: self.queue).
            abort_on_failure: Override instance-level abort_on_failure.
            timeout: Max minutes to wait for pipeline completion.
            step_time_limit: Max minutes per step (clearml server kills overdue tasks).
            skip_steps: List of step names to skip, e.g. ["eval"].

        Returns:
            Dict with pipeline_id, name, steps, status.
        """
        from expflow_pde.clearml import (
            pipeline_add_step,
            pipeline_create,
            pipeline_start,
        )

        skip = set(skip_steps or [])
        exec_queue = execution_queue or self.queue
        abort = abort_on_failure if abort_on_failure is not None else self.abort_on_failure

        if pipeline_name is None:
            script_stem = train_script.replace(".py", "").replace("/", "_")
            pipeline_name = f"pipeline_{script_stem}"

        create_result = pipeline_create(
            name=pipeline_name,
            project=self.project,
            version=version or DEFAULT_PIPELINE_VERSION,
            abort_on_failure=abort,
            add_pipeline_tags=True,
            add_run_number=self.add_run_number,
            docker=self.docker,
            packages=self.packages,
        )
        if "error" in create_result:
            raise RuntimeError(f"Pipeline creation failed: {create_result.get('error', 'unknown')}")

        steps: list[dict[str, Any]] = []
        pipeline_name_actual = create_result.get("name", pipeline_name)

        # Step: train
        if "train" not in skip and train_script:
            train_override: dict[str, Any] = {}
            if train_params:
                train_override["Args"] = {k: str(v) for k, v in train_params.items()}

            add_train_result = pipeline_add_step(
                pipeline_name=pipeline_name_actual,
                project=self.project,
                step_name="train",
                base_task_name=train_script,
                base_task_project=self.project,
                parameter_override=train_override if train_override else None,
                execution_queue=exec_queue,
                time_limit=step_time_limit,
            )
            steps.append(add_train_result)

        # Step: eval
        if "eval" not in skip and eval_script:
            eval_override: dict[str, Any] = {}
            if eval_params:
                eval_override["Args"] = {k: str(v) for k, v in eval_params.items()}

            add_eval_result = pipeline_add_step(
                pipeline_name=pipeline_name_actual,
                project=self.project,
                step_name="eval",
                base_task_name=eval_script,
                base_task_project=self.project,
                parents=["train"] if "train" not in skip else None,
                parameter_override=eval_override if eval_override else None,
                execution_queue=exec_queue,
                time_limit=step_time_limit,
            )
            steps.append(add_eval_result)

        start_result = pipeline_start(
            pipeline_name=pipeline_name_actual,
            project=self.project,
            version=version or DEFAULT_PIPELINE_VERSION,
            queue_name=exec_queue,
            timeout_minutes=timeout,
        )

        result: dict[str, Any] = {
            "pipeline_id": create_result.get("pipeline_id", ""),
            "name": pipeline_name_actual,
            "project": self.project,
            "version": version or DEFAULT_PIPELINE_VERSION,
            "steps": [
                {
                    "name": s.get("step_name", ""),
                    "parents": s.get("parents", []),
                    "status": s.get("status", "defined"),
                }
                for s in steps
            ],
            "status": start_result.get("status", "started"),
            "queue": exec_queue or "local",
            "mode": "fast",
        }
        self._last_result = result
        return result

    # ── Mode A (full): HPO → Train → Eval ──

    def train_hpo_val_submit(
        self,
        train_script: str,
        eval_script: str | None = None,
        eval_params: dict[str, Any] | None = None,
        n_trials: int = 50,
        parallel: int = 4,
        hpo_study_name: str | None = None,
        hpo_search_space: dict[str, dict[str, Any]] | None = None,
        objective_metric: str = "seg_total",
        direction: str = "maximize",
        pipeline_name: str | None = None,
        version: str | None = None,
        execution_queue: str | None = None,
        timeout: float | None = None,
        step_time_limit: float | None = None,
        pruner: str = "hyperband",
        skip_steps: list[str] | None = None,
    ) -> dict[str, Any]:
        """Mode A (full): Create and run a HPO → Train → Eval pipeline.

        This is the fully automated competition pipeline:
        1. HPO: runs N trials via clearml queue to find best hyperparams
        2. Train: trains with the best parameters from HPO
        3. Eval: evaluates the best checkpoint

        Steps can be individually skipped via skip_steps.

        Args:
            train_script: Path to training script.
            eval_script: Path to evaluation script (optional).
            eval_params: Script arguments for the eval step.
            n_trials: Number of HPO trials.
            parallel: Max concurrent HPO trials.
            hpo_study_name: Optuna study name (auto-generated if None).
            hpo_search_space: Hyperparameter search space (default if None).
            objective_metric: Which metric HPO should optimize.
            direction: 'maximize' or 'minimize'.
            pipeline_name: Pipeline name (auto-generated if None).
            version: Pipeline version.
            execution_queue: Queue for execution.
            timeout: Max minutes to wait.
            pruner: Optuna pruner ('hyperband', 'median', 'percentile').
            skip_steps: Steps to skip, e.g. ["hpo", "eval"].

        Returns:
            Dict with pipeline metadata and results.
        """
        from expflow_pde.clearml import (
            pipeline_add_step,
            pipeline_create,
            pipeline_start,
        )

        skip = set(skip_steps or [])
        exec_queue = execution_queue or self.queue

        if pipeline_name is None:
            script_stem = train_script.replace(".py", "").replace("/", "_")
            pipeline_name = f"pipeline_hpo_{script_stem}"

        create_result = pipeline_create(
            name=pipeline_name,
            project=self.project,
            version=version or DEFAULT_PIPELINE_VERSION,
            abort_on_failure=self.abort_on_failure,
            add_pipeline_tags=True,
            add_run_number=self.add_run_number,
            docker=self.docker,
            packages=self.packages,
        )
        if "error" in create_result:
            raise RuntimeError(f"Pipeline creation failed: {create_result.get('error', 'unknown')}")

        steps: list[dict[str, Any]] = []
        pipeline_name_actual = create_result.get("name", pipeline_name)

        # Step 1: HPO
        if "hpo" not in skip:
            hpo_name = hpo_study_name or f"auto_hpo_{script_stem}"
            hpo_params_override: dict[str, Any] = {
                "Args": {
                    "--n-trials": str(n_trials),
                    "--parallel": str(parallel),
                    "--study-name": hpo_name,
                    "--metric": objective_metric,
                    "--direction": direction,
                    "--pruner": pruner,
                    "--distributed": "true",
                    "--queue": exec_queue or self.queue,
                    "--project": self.project,
                }
            }

            add_hpo_result = pipeline_add_step(
                pipeline_name=pipeline_name_actual,
                project=self.project,
                step_name="hpo",
                base_task_name=train_script,
                base_task_project=self.project,
                parameter_override=hpo_params_override,
                execution_queue=exec_queue,
                time_limit=step_time_limit,
            )
            steps.append(add_hpo_result)

        # Step 2: Train (uses best params from HPO)
        if "train" not in skip:
            train_parents = ["hpo"] if "hpo" not in skip else None
            add_train_result = pipeline_add_step(
                pipeline_name=pipeline_name_actual,
                project=self.project,
                step_name="train",
                base_task_name=train_script,
                base_task_project=self.project,
                parents=train_parents,
                # Parameter override will reference HPO best params
                # via clearml pipeline variable syntax
                parameter_override={} if train_parents else None,
                execution_queue=exec_queue,
                time_limit=step_time_limit,
            )
            steps.append(add_train_result)

        # Step 3: Eval
        if "eval" not in skip and eval_script:
            eval_parents: list[str] = []
            if "train" not in skip:
                eval_parents.append("train")
            elif "hpo" not in skip:
                eval_parents.append("hpo")

            eval_override: dict[str, Any] = {}
            if eval_params:
                eval_override["Args"] = {k: str(v) for k, v in eval_params.items()}

            add_eval_result = pipeline_add_step(
                pipeline_name=pipeline_name_actual,
                project=self.project,
                step_name="eval",
                base_task_name=eval_script,
                base_task_project=self.project,
                parents=eval_parents if eval_parents else None,
                parameter_override=eval_override if eval_override else None,
                execution_queue=exec_queue,
                time_limit=step_time_limit,
            )
            steps.append(add_eval_result)

        start_result = pipeline_start(
            pipeline_name=pipeline_name_actual,
            project=self.project,
            version=version or DEFAULT_PIPELINE_VERSION,
            queue_name=exec_queue,
            timeout_minutes=timeout,
        )

        result: dict[str, Any] = {
            "pipeline_id": create_result.get("pipeline_id", ""),
            "name": pipeline_name_actual,
            "project": self.project,
            "version": version or DEFAULT_PIPELINE_VERSION,
            "steps": [
                {
                    "name": s.get("step_name", ""),
                    "parents": s.get("parents", []),
                    "status": s.get("status", "defined"),
                }
                for s in steps
            ],
            "status": start_result.get("status", "started"),
            "queue": exec_queue or "local",
            "mode": "full",
            "n_trials": n_trials if "hpo" not in skip else 0,
            "parallel": parallel if "hpo" not in skip else 0,
        }
        self._last_result = result
        return result

    # ── Best params lookup ──

    def _get_hpo_best_params(self, study_name: str) -> dict[str, Any] | None:
        """Look up best parameters from a completed Optuna study.

        Args:
            study_name: Optuna study name.

        Returns:
            Dict of best params, or None if not found.
        """
        try:
            from expflow_pde.hpo import get_study_best_params

            return get_study_best_params(study_name)
        except Exception:
            return None

    # ── Repair a failed task ──

    def repair_task(
        self,
        task_log: str,
        exit_code: int,
        enable_reflection: bool = False,
    ) -> dict[str, Any]:
        """Analyze a failed task and suggest/apply repair.

        Uses three-level repair (L0 rule engine → L1 traceback → L2 reflection).
        Result dict is compatible with the pipeline result format.

        Args:
            task_log: Console output from the failed task.
            exit_code: Process exit code.
            enable_reflection: Whether to allow L2 subagent reflection.

        Returns:
            Dict with keys: fixed, level, action, attempts, history.
        """
        from expflow_pde.repair import RepairStage

        stage = RepairStage(
            experiment_id=self._last_result.get("pipeline_id", "") if self._last_result else "",
            max_l1_attempts=2,
        )
        return stage.run(
            task_log=task_log,
            exit_code=exit_code,
            enable_reflection=enable_reflection,
        )

    @property
    def last_result(self) -> dict[str, Any] | None:
        """Return the result from the most recent pipeline call."""
        return self._last_result
