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
    n_jobs: int = typer.Option(
        1, "--n-jobs", "-j", help="Parallel jobs (local) or max concurrent (distributed)"
    ),
    study_name: Optional[str] = typer.Option(
        None, "--study-name", "-n", help="Study name (default: auto)"
    ),
    distributed: bool = typer.Option(
        False, "--distributed", "-d", help="Distribute trials via clearml ask/tell"
    ),
    optimizer: bool = typer.Option(
        False, "--optimizer", "-O", help="Use ClearML HyperParameterOptimizer (native integration)"
    ),
    queue: Optional[str] = typer.Option(
        None, "--queue", "-q", help="Target queue (required for distributed/optimizer)"
    ),
    project: str = typer.Option("PDEBench", "--project", "-p", help="ClearML project name"),
    direction: str = typer.Option(
        "maximize", "--direction", help="Optimization direction: maximize or minimize"
    ),
    metric: str = typer.Option(
        "seg_total",
        "--metric",
        "-m",
        help="Objective metric name from METRIC: lines or clearml scalars",
    ),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="Max runtime in minutes"),
    pruner: str = typer.Option(
        "hyperband", "--pruner", help="Optuna pruner: hyperband, median, percentile, none"
    ),
    loss: Optional[str] = typer.Option(
        None,
        "--loss",
        "-l",
        help="Loss function name for training script (e.g., l2_rel, h1_1d). "
        "Passed as --loss=<name> to the training script.",
    ),
    param_prefix: str = typer.Option(
        "Args/--",
        "--param-prefix",
        help="Parameter prefix for clearml task. Args/-- for argparse, Args/ for clearml Args section",
    ),
) -> None:
    """Run hyperparameter optimization on a script.

    Three modes:
    - **Local** (default): runs trials sequentially on this machine.
    - **Distributed** (--distributed, --queue): ask/tell + clearml Task clones.
    - **Optimizer** (--optimizer, --queue): ClearML HyperParameterOptimizer
      (native Optuna integration, recommended for production).

    The training script must output METRIC:<name>=<value> lines (local mode)
    or report clearml scalars (distributed/optimizer mode).
    """
    from expflow_pde.hpo import run_hpo

    pruner_val = pruner if pruner.lower() != "none" else None

    result = run_hpo(
        script=script,
        n_trials=trials,
        n_jobs=n_jobs,
        study_name=study_name,
        direction=direction,
        objective_metric=metric,
        timeout_minutes=timeout,
        distributed=distributed,
        queue=queue,
        project=project,
        pruner=pruner_val,
        use_hpo_optimizer=optimizer,
        loss=loss,
        param_prefix=param_prefix,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    method = result.get("method", "local" if not distributed and not optimizer else "distributed")
    print(f"HPO started: study={result['study_name']}")
    print(f"  Method:   {method}")
    print(f"  Script:   {result.get('script', script)}")
    print(f"  Trials:   {result['n_trials']}")
    print(f"  Completed: {result['completed']}")
    print(f"  Failed:    {result['failed']}")
    print(f"  Parallel:  {n_jobs}")
    if result.get("best_value") is not None:
        print(f"  Best value: {result['best_value']:.4f}")
        print(f"  Best params: {result['best_params']}")
    print(f"  Direction: {result['direction']}")
    print(f"  Duration:  {result['duration_sec']:.1f}s")
    if result.get("timeout_minutes"):
        print(f"  Timeout:   {result['timeout_minutes']}min")
