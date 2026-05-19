#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow audit CLI sub-commands."""

from typing import Optional

import typer

audit_app = typer.Typer(
    name="audit",
    help="Experiment validation, compliance checking, report generation",
    no_args_is_help=True,
)


def get_audit_app() -> typer.Typer:
    return audit_app


@audit_app.command("validate")
def validate_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    competition_rules: bool = typer.Option(
        False, "--competition-rules", "-c", help="Include PDEBench competition rule checks"
    ),
    task_id: Optional[str] = typer.Option(
        None, "--task-id", "-t", help="clearml task ID (required with --competition-rules)"
    ),
) -> None:
    """Run validation checks on an experiment.\n
    Use --competition-rules --task-id <id> to check PDEBench competition rules.\n
    \n
    Example:\n
        expflow audit validate exp-001 --competition-rules --task-id abc123
    """
    from expflow_pde.audit import validate_experiment

    result = validate_experiment(experiment_id, config_snapshot={}, metrics={})
    print(f"Validation for: {result['experiment_id']}")
    print()
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['detail']}")
    print(f"\nTimestamp: {result['timestamp']}")

    # Competition rules check
    if competition_rules:
        if not task_id:
            print("\n  Error: --competition-rules requires --task-id <clearml_task_id>")
            raise typer.Exit(code=1)
        _check_competition_rules(task_id)


def _check_competition_rules(task_id: str) -> None:
    """Check PDEBench Task 1 competition rules against a clearml task."""
    print("  ── Competition Rules ──")
    print()

    # Fetch metrics and params from clearml
    metrics = _get_task_metrics(task_id)
    params = _get_task_params(task_id)

    from expflow_pde.audit import validate_competition_rules

    result = validate_competition_rules(
        task_metrics=metrics,
        task_params=params,
    )

    for check in result["checks"]:
        flag = "PASS" if check["passed"] else "FAIL"
        print(f"  [{flag:<4}] {check['label']}: {check['detail']}")

    all_pass = result["all_pass"]
    if all_pass:
        print("\n  Result: ALL COMPETITION RULES PASSED ✓")
    else:
        print("\n  Result: SOME COMPETITION RULES FAILED ✗")


def _get_task_metrics(task_id: str) -> dict[str, float]:
    """Try to fetch scalar metrics from a clearml task."""
    try:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        scalar_metrics = getattr(task, "get_last_scalar_metrics", lambda: {})()
        flat: dict[str, float] = {}
        for group_name, series_dict in scalar_metrics.items():
            if not isinstance(series_dict, dict):
                continue
            for series_name, metric_info in series_dict.items():
                if isinstance(metric_info, dict):
                    value = metric_info.get("last") or metric_info.get("value")
                    if value is not None:
                        try:
                            flat[series_name] = float(value)
                        except (ValueError, TypeError):
                            pass
        return flat
    except Exception:
        return {}


def _check_task_parameters(task_id: str) -> bool | None:
    """Check if sub_step parameter exists in clearml task."""
    try:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        params = task.get_parameters()
        # Check various possible parameter names for sub_step
        for key in params or {}:
            if "sub_step" in key.lower() or "substep" in key.lower():
                val = params[key]
                try:
                    return int(val) > 0
                except (ValueError, TypeError):
                    return val is not None
        return False
    except Exception:
        return None


def _get_task_params(task_id: str) -> dict[str, str] | None:
    """Try to fetch parameters from a clearml task."""
    try:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        return task.get_parameters()
    except Exception:
        return None


@audit_app.command("check-dataset")
def check_dataset_cmd(
    dataset_name: str = typer.Argument(..., help="Dataset name"),
    compliance: str = typer.Option(
        ...,
        "--compliance",
        "-c",
        help="Compliance status: allowed or forbidden",
    ),
) -> None:
    """Check dataset compliance."""
    from expflow_pde.audit import check_dataset_compliance

    result = check_dataset_compliance(dataset_name, compliance)
    status = "COMPLIANT" if result["compliant"] else "NON-COMPLIANT"
    print(f"Dataset:  {result['dataset_name']}")
    print(f"Status:   {status}")
    print(f"Allowed:  {result['compliant']}")


@audit_app.command("report")
def report_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
) -> None:
    """Generate an experiment report (Markdown)."""
    from expflow_pde.audit import generate_report
    from expflow_pde.dispatcher import get_experiment_status

    exp = get_experiment_status(experiment_id)
    if "error" in exp:
        print(f"Experiment '{experiment_id}' not found.")
        return

    report = generate_report(
        experiment_id,
        config={"command": exp.get("command", ""), "queue": exp.get("queue", "")},
        metrics={},
    )
    print(report["markdown"])
