#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow audit CLI sub-commands."""

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
) -> None:
    """Run validation checks on an experiment."""
    from expflow_pde.audit import validate_experiment

    result = validate_experiment(experiment_id, config_snapshot={}, metrics={})
    print(f"Validation for: {result['experiment_id']}")
    print()
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['detail']}")
    print(f"\nTimestamp: {result['timestamp']}")


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
