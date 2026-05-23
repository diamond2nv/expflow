#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow optuna CLI sub-commands — lazy imports of optuna at call time."""

import os
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
        "Args/",
        "--param-prefix",
        help="Parameter prefix for clearml task. Args/ for clearml Args section, Args/-- for -- style",
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


# ── Pareto visualization ──


@optuna_app.command("pareto")
def pareto_cmd(
    study_name: str = typer.Argument(..., help="Study name or auto prefix to search"),
    direction1: str = typer.Option(
        "maximize", "--direction1", "-d1", help="First objective direction"
    ),
    direction2: str = typer.Option(
        "minimize", "--direction2", "-d2", help="Second objective direction"
    ),
    storage: Optional[str] = typer.Option(
        None, "--storage", "-s", help="SQLite storage URL (default: search ~/.expflow/)"
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output Pareto data as JSON (for Hermes /goal)"
    ),
) -> None:
    # pylint: disable=too-many-locals
    """Display Pareto frontier from a completed multi-objective Optuna study.

    Shows a terminal scatter plot of all trials with the Pareto front highlighted.
    Requires a multi-objective study (2 objectives).

    Example: expflow optuna pareto auto_hpo_train_task1
    """
    from expflow_pde.optuna import get_study

    # Resolve storage if not provided
    resolved_storage: str | None = storage
    if resolved_storage is None:
        import glob as _glob

        db_pattern = os.path.expanduser(f"~/.expflow/optuna_{study_name}.db")
        matching = _glob.glob(db_pattern)
        if matching:
            resolved_storage = f"sqlite:///{matching[0]}"

    try:
        study_data = get_study(study_name, storage=resolved_storage)
    except Exception as e:
        print(f"Error loading study '{study_name}': {e}")
        raise typer.Exit(1)

    trials = study_data.get("trials", [])
    if len(trials) < 2:
        print(f"Not enough completed trials ({len(trials)}). Need at least 2.")
        raise typer.Exit(1)

    all_values = [t["values"] for t in trials if t.get("values") is not None]
    if not all_values or len(all_values[0]) < 2:
        print(
            "Trials found but no multi-objective values detected. "
            "This study may be single-objective. "
            f"Raw values sample: {all_values[:3] if all_values else 'none'}"
        )
        raise typer.Exit(1)

    # Compute Pareto frontier
    n_obj = len(all_values[0])
    sign1 = 1 if direction1 == "maximize" else -1
    sign2 = 1 if direction2 == "maximize" else -1

    def dominates(a: list[float], b: list[float]) -> bool:
        """Check if a dominates b (assuming all max)."""
        signs = [sign1] + ([sign2] if n_obj >= 2 else [])
        scaled_a = [a[i] * signs[i] if i < len(signs) else a[i] for i in range(len(a))]
        scaled_b = [b[i] * signs[i] if i < len(signs) else b[i] for i in range(len(b))]
        at_least_one = False
        for sa, sb in zip(scaled_a, scaled_b):
            if sa < sb:
                return False
            if sa > sb:
                at_least_one = True
        return at_least_one

    pareto_indices: list[int] = []
    for i in range(len(all_values)):
        dominated = False
        for j in range(len(all_values)):
            if i != j and dominates(all_values[j], all_values[i]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    # Build trial data for output
    pareto_trials = [
        {
            "number": trials[idx].get("number", idx),
            "values": all_values[idx],
            "params": trials[idx].get("params", {}),
        }
        for idx in pareto_indices
    ]

    if json_output:
        import json as _json

        print(
            _json.dumps(
                {
                    "study_name": study_name,
                    "n_trials": len(trials),
                    "n_pareto": len(pareto_indices),
                    "pareto_trials": pareto_trials,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    # Terminal scatter plot
    x_vals = [v[0] for v in all_values]
    y_vals = [v[1] if len(v) >= 2 else 0 for v in all_values]

    print(f"\n{'=' * 60}")
    print(f"  Pareto Frontier Analysis — '{study_name}'")
    print(f"  Trials: {len(trials)} | Pareto-optimal: {len(pareto_indices)}")
    print(f"  Objective 1: {direction1} | Objective 2: {direction2}")
    print(f"{'=' * 60}")

    # Simple ascii scatter
    if x_vals and y_vals:
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)
        x_range = max_x - min_x if max_x != min_x else 1
        y_range = max_y - min_y if max_y != min_y else 1

        pareto_set = set(pareto_indices)
        print("\n  Scatter (↓ = Pareto front marker):")
        print("  O = dominated, * = Pareto-optimal")
        print()
        for row in range(20, -1, -1):
            y_val = min_y + (y_range * row / 20)
            line = ""
            for col in range(40):
                x_val = min_x + (x_range * col / 40)
                # Find nearest trial
                closest = min(
                    range(len(x_vals)),
                    key=lambda k: ((x_vals[k] - x_val) ** 2 + (y_vals[k] - y_val) ** 2) ** 0.5,
                )
                if closest in pareto_set:
                    px, py_val = x_vals[closest], y_vals[closest]
                    dist = ((px - x_val) ** 2 + (py_val - y_val) ** 2) ** 0.5
                    normalized_dist = dist / ((x_range / 40) ** 2 + (y_range / 20) ** 2) ** 0.5
                    if normalized_dist < 0.8:
                        line += "*"
                    else:
                        line += "."
                else:
                    line += "."
            print(f"  {line}")

        print(f"  {'min':>10} {'':>30} {'max':>10}")
        print(f"  Obj1: {min_x:<10.4f} {'':>23} {max_x:>10.4f}")
        print(f"  Obj2: {min_y:<10.4f} {'':>23} {max_y:>10.4f}")

    # Print Pareto-optimal details
    print(f"\n  Pareto-optimal trials ({len(pareto_trials)}):")
    print(f"  {'Trial #':>8} {'Obj1':>12} {'Obj2':>12}  Params")
    print(f"  {'-' * 8:>8} {'-' * 12:>12} {'-' * 12:>12}  {'-' * 30}")
    for pt in sorted(pareto_trials, key=lambda x: -x["values"][0]):
        params_str = ", ".join(f"{k}={v}" for k, v in list(pt.get("params", {}).items())[:3])
        v1 = pt["values"][0]
        v2 = pt["values"][1] if len(pt["values"]) >= 2 else 0
        print(f"  {pt['number']:>8} {v1:>12.4f} {v2:>12.4f}  {params_str}")
