#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow optuna CLI sub-commands — lazy imports of optuna at call time."""

from typing import Optional

import typer

optuna_app = typer.Typer(
    name="optuna",
    help="Interact with Optuna hyperparameter optimization",
    no_args_is_help=True,
)


def get_optuna_app() -> typer.Typer:
    """Return the optuna sub-command group."""
    return optuna_app


# ── Study commands ──


@optuna_app.command("create-study")
def create_study_cmd(
    study_name: str = typer.Argument(..., help="Study name"),
    direction: str = typer.Option(
        "minimize",
        "--direction",
        "-d",
        help="Optimization direction: minimize or maximize",
    ),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Database URL (e.g. sqlite:///optuna.db)",
    ),
) -> None:
    """Create a new Optuna study."""
    from expflow_pde.optuna import create_study

    result = create_study(study_name, direction=direction, storage=storage)
    print(f"Study created: {result['name']}")
    print(f"  ID:        {result['study_id']}")
    print(f"  Direction: {result['direction']}")


@optuna_app.command("studies")
def list_studies_cmd(
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Database URL",
    ),
) -> None:
    """List all Optuna studies."""
    from expflow_pde.optuna import list_studies

    studies = list_studies(storage=storage)

    if not studies:
        print("No studies found.")
        return

    print(f"{'ID':<5} {'NAME':<30} {'DIRECTION':<12} {'BEST VALUE':<12} {'BEST TRIAL':<10}")
    print("-" * 69)
    for s in studies:
        bv = f"{s['best_value']:.6f}" if s["best_value"] is not None else "-"
        bt = str(s["best_trial"]) if s["best_trial"] is not None else "-"
        print(f"{s['study_id']:<5} {s['name']:<30} {s['direction']:<12} {bv:<12} {bt:<10}")


@optuna_app.command("study")
def get_study_cmd(
    study_name: str = typer.Argument(..., help="Study name"),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Database URL",
    ),
) -> None:
    """Show details for a study."""
    from expflow_pde.optuna import get_study

    s = get_study(study_name, storage=storage)
    print(f"Name:      {s['name']}")
    print(f"ID:        {s['study_id']}")
    print(f"Direction: {s['direction']}")
    if s["best_trial"]:
        bt = s["best_trial"]
        print(f"Best Trial #{bt['number']}: value={bt['value']}")
        for k, v in bt["params"].items():
            print(f"  {k}: {v}")
    print(f"Trials: {len(s['trials'])}")


@optuna_app.command("delete-study")
def delete_study_cmd(
    study_name: str = typer.Argument(..., help="Study name to delete"),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Database URL",
    ),
) -> None:
    """Delete a study."""
    from expflow_pde.optuna import delete_study

    result = delete_study(study_name, storage=storage)
    print(f"Study '{result['study_name']}' deleted.")


# ── Trial commands ──


@optuna_app.command("ask")
def ask_cmd(
    study_name: str = typer.Argument(..., help="Study name"),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Database URL",
    ),
) -> None:
    """Ask for next trial parameters."""
    from expflow_pde.optuna import ask_trial

    result = ask_trial(study_name, storage=storage)
    print(f"Trial #{result['trial_number']}:")
    for k, v in result["params"].items():
        print(f"  {k}: {v}")


@optuna_app.command("tell")
def tell_cmd(
    study_name: str = typer.Argument(..., help="Study name"),
    trial_number: int = typer.Argument(..., help="Trial number from ask"),
    value: float = typer.Argument(..., help="Objective value"),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Database URL",
    ),
) -> None:
    """Report a trial result."""
    from expflow_pde.optuna import tell_trial

    result = tell_trial(study_name, trial_number, value, storage=storage)
    print(f"Trial #{result['trial_number']}: value={result['value']} reported.")


# ── Plot commands ──


@optuna_app.command("plot")
def plot_cmd(
    study_name: str = typer.Argument(..., help="Study name"),
    plot_type: str = typer.Option(
        "history",
        "--type",
        "-t",
        help="Plot type: history, parallel_coordinate, slice, contour",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path",
    ),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Database URL",
    ),
) -> None:
    """Generate optimization visualization."""
    from expflow_pde.optuna import plot_study

    result = plot_study(
        study_name,
        plot_type=plot_type,
        output_path=output,
        storage=storage,
    )
    print(f"Plot saved to: {result['output_path']}")


# ── High-level HPO ──


@optuna_app.command("run")
def hpo_run_cmd(
    script: str = typer.Argument(..., help="Training script path"),
    trials: int = typer.Option(50, "--trials", "-t", help="Number of trials"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j", help="Parallel jobs"),
    study_name: Optional[str] = typer.Option(
        None,
        "--study-name",
        "-n",
        help="Study name (default: auto)",
    ),
) -> None:
    """Run hyperparameter optimization on a script. Wraps Optuna study + trials."""
    from expflow_pde.hpo import run_hpo

    result = run_hpo(
        script=script,
        n_trials=trials,
        n_jobs=n_jobs,
        study_name=study_name,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"HPO started: study={result['study_name']}")
    print(f"  Script:   {result['script']}")
    print(f"  Trials:   {result['n_trials']}")
    print(f"  Parallel: {result['n_jobs']}")
    print(f"  Best:     {result.get('best_value', 'pending')}")
