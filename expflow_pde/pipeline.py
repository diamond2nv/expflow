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

import os
from typing import Any


def _try_float(v: str) -> float | str:
    """Try parsing a string as float; return the string if it fails."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


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
        self._noise_db_path = os.path.expanduser("~/.expflow/noise_floor.jsonl")
        self._registry = None  # lazy import for DeadEndRegistry
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
        """Mode A (full): Run HPO locally, then create a train → eval pipeline.

        This is the fully automated competition workflow:

        Phase 1 — Hyperparameter search:
            Call run_hpo() (distributed or local) to find the best parameters.
            Depending on n_trials and parallel, uses clearml queue distribution
            (n_trials >= parallel) or local subprocess (n_trials < parallel).

        Phase 2 — Train with best params:
            Use train_val_submit() with the discovered best parameters.

        This replaces the previous "HPO step inside clearml pipeline" approach,
        which had multiple design issues (wrong base_task, params never
        propagated, Optuna SQLite inaccessible from pipeline steps).

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
                When "hpo" is skipped, no hyperparameter search is performed
                and train uses default parameters.

        Returns:
            Dict with keys: phase_1 (HPO result), phase_2 (pipeline result).
        """
        from expflow_pde.hpo import run_hpo

        skip = set(skip_steps or [])
        exec_queue = execution_queue or self.queue
        script_stem = train_script.replace(".py", "").replace("/", "_")
        train_params: dict[str, Any] | None = None

        # ── Phase 1: HPO ──
        hpo_result: dict[str, Any] | None = None
        if "hpo" not in skip:
            hpo_study_name = hpo_study_name or f"auto_hpo_{script_stem}"

            # Use distributed mode when trials >= parallel (more than 1 concurrent)
            use_distributed = parallel > 1 and n_trials > 1

            hpo_result = run_hpo(
                script=train_script,
                n_trials=n_trials,
                n_jobs=parallel,
                study_name=hpo_study_name,
                search_space=hpo_search_space,
                direction=direction,
                objective_metric=objective_metric,
                distributed=use_distributed,
                queue=exec_queue,
                project=self.project,
                pruner=pruner,
            )

            # Extract best params from HPO result
            if hpo_result and hpo_result.get("best_params"):
                raw = hpo_result["best_params"]
                train_params = {}
                for k, v in raw.items():
                    if isinstance(v, str):
                        v = _try_float(v)
                    train_params[k] = v
            else:
                # Fallback: try reading from Optuna study storage
                best = self._get_hpo_best_params(hpo_study_name)
                if best:
                    train_params = {
                        k: (v if not isinstance(v, str) else _try_float(v)) for k, v in best.items()
                    }

        # ── Phase 2: Train → Eval pipeline with best params ──
        pipe_result = self.train_val_submit(
            train_script=train_script,
            train_params=train_params,
            eval_script=eval_script,
            eval_params=eval_params,
            pipeline_name=pipeline_name,
            version=version,
            execution_queue=exec_queue,
            abort_on_failure=None,
            timeout=timeout,
            step_time_limit=step_time_limit,
            skip_steps=[s for s in ["hpo", "eval"] if s in skip],
        )

        result: dict[str, Any] = {
            "phase_1": hpo_result or {"study_name": hpo_study_name, "skipped": True},
            "phase_2": pipe_result,
            "mode": "full",
            "n_trials": n_trials if "hpo" not in skip else 0,
            "parallel": parallel if "hpo" not in skip else 0,
        }
        self._last_result = result
        return result

    # ── Best params lookup ──

    def _get_hpo_best_params(self, study_name: str) -> dict[str, Any] | None:
        """Look up best parameters from a completed Optuna study.

        Cleans the raw params: strips Args/ prefix, auto-casts string
        values to float where possible. This prevents downstream
        `**best_params` from receiving invalid keys like 'Args/--lr'.

        Returns:
            Dict of clean best params, or None if not found.
        """
        try:
            from expflow_pde.hpo import get_study_best_params

            raw = get_study_best_params(study_name)
            if raw is None:
                return None
            clean: dict[str, Any] = {}
            for k, v in raw.items():
                key = k.removeprefix("Args/").removeprefix("--")
                if isinstance(v, str):
                    clean[key] = _try_float(v)
                else:
                    clean[key] = v
            return clean
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

    # ── Experiment validation (noise-aware champion promotion) ──

    def validate_experiment(
        self,
        candidate_value: float,
        champion_value: float,
        sigma_multiplier: float = 2.0,
        metric_name: str = "seg_total",
        noise_db_path: str | None = None,
        pde_candidate_value: float | None = None,
        pde_champion_value: float | None = None,
        pde_relative_threshold: float | None = None,
        pde_absolute_floor: float | None = None,
    ) -> dict[str, Any]:
        """Validate a candidate experiment result against the champion.

        Uses noise-aware champion validation (AutoScientists-style).
        If noise_floor is not provided, attempts lazy calibration from
        the noise floor database.

        **PDE residual gate (scale-aware)**: When pde_candidate_value and
        pde_champion_value are both provided (e.g. 'rans_pde_total' metrics
        from physics-informed training), uses a combined relative + absolute
        threshold (Zhang2026 JFM):

            blocked if candidate > max(champion × (1 + relative_threshold),
                                       champion + absolute_floor)

        This is scale-aware: at any PDE residual magnitude, the gate triggers
        at a consistent *relative* increase, with an absolute floor preventing
        false blocks when champion residual is near zero.

        Args:
            candidate_value: Metric value from the candidate.
            champion_value: Current champion metric value.
            sigma_multiplier: Noise band width (default: 2.0, per AutoScientists).
            metric_name: Metric name for calibration lookup.
            noise_db_path: Override for noise floor DB path.
            pde_candidate_value: PDE residual from candidate experiment.
            pde_champion_value: PDE residual from current champion.
            pde_relative_threshold: Fractional increase allowed (default: 0.50).
                Pass 0.20 for stricter gating, 1.0 for relaxed.
            pde_absolute_floor: Min absolute threshold gap (default: 0.01).
                Prevents trivial blocks when residual ~0.

        Returns:
            Validation result dict with action/promote/confirm/reject.
            Includes 'pde_gate_blocked' key if PDE residual gate vetoed.
        """
        from expflow_pde.validate import noise_aware_validate, check_pde_residual_gate

        # ── Primary metric validation ──
        primary = noise_aware_validate(
            candidate_value=candidate_value,
            champion_value=champion_value,
            sigma_multiplier=sigma_multiplier,
            noise_db_path=noise_db_path or self._noise_db_path,
            metric_name=metric_name,
        )

        # ── PDE residual gate (scale-aware, Zhang2026) ──
        pde_blocked = False
        gate_result: dict[str, Any] = {}
        if pde_candidate_value is not None and pde_champion_value is not None:
            gate_result = check_pde_residual_gate(
                candidate_residual=pde_candidate_value,
                champion_residual=pde_champion_value,
                relative_threshold=pde_relative_threshold,
                absolute_floor=pde_absolute_floor,
            )
            if gate_result["blocked"]:
                pde_blocked = True
                primary["action"] = "reject"
                primary["message"] += f" [{gate_result['message']}]"

        primary["pde_gate_blocked"] = pde_blocked
        primary["pde_gate_details"] = gate_result
        if pde_candidate_value is not None:
            primary["pde_candidate"] = pde_candidate_value
        if pde_champion_value is not None:
            primary["pde_champion"] = pde_champion_value

        return primary

    # ── Dead-end registry ──

    @property
    def _dead_end_registry(self):
        """Lazy-loaded DeadEndRegistry instance."""
        if self._registry is None:
            from expflow_pde.registry import DeadEndRegistry

            self._registry = DeadEndRegistry()
        return self._registry

    def register_dead_end(
        self,
        script: str,
        axis: str,
        reason: str,
        args: dict[str, Any] | None = None,
        code_hash: str | None = None,
        metric_value: float | None = None,
        bucket: str | None = None,
        bucket_low: float | None = None,
        bucket_high: float | None = None,
    ) -> dict[str, Any]:
        """Register a failed experiment direction in the dead-end registry.

        Args:
            script: Script that was run.
            axis: Search axis that failed (learning_rate, architecture, etc.).
            reason: Failure reason.
            args: Hyperparameters used.
            code_hash: Git commit hash.
            metric_value: Final metric value if applicable.
            bucket: Sub-axis bucket for grouped matching
                (e.g. 'n_modes' groups n_modes=8 and n_modes=24).
            bucket_low: Numeric range low end (interval overlap).
            bucket_high: Numeric range high end.

        Returns:
            Dict with entry_id and timestamp.
        """
        return self._dead_end_registry.register(
            script=script,
            axis=axis,
            reason=reason,
            args=args,
            code_hash=code_hash,
            metric_value=metric_value,
            bucket=bucket,
            bucket_low=bucket_low,
            bucket_high=bucket_high,
        )

    def lookup_dead_end(
        self,
        script: str,
        axis: str,
        args: dict[str, Any] | None = None,
        exact: bool = False,
        bucket: str | None = None,
        bucket_value: float | None = None,
    ) -> list[dict[str, Any]]:
        """Check if an approach has been tried and failed before.

        Three lookup modes:
        - **Exact**: perfect hash match on script+args+axis.
        - **Bucket fuzzy** (exact=False, bucket=...): matches on
          (script, axis, bucket) — n_modes=8 and n_modes=24
          hit the same bucket.
        - **Wildcard** (exact=False, bucket=None): any entry on
          script+axis.

        Args:
            script: Script name.
            axis: Search axis.
            args: Hyperparameters (for exact match only).
            exact: If True, requires perfect (script+args+axis) match.
            bucket: Sub-axis bucket for fuzzy grouping.
            bucket_value: Numeric value for interval-overlap filter.

        Returns:
            List of matching dead-end entries.
        """
        return self._dead_end_registry.lookup(
            script=script,
            axis=axis,
            args=args,
            exact=exact,
            bucket=bucket,
            bucket_value=bucket_value,
        )
