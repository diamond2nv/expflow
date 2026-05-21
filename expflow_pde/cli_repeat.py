"""expflow repeat CLI — repair diagnosis and resubmit.

This command group replaces the old --repair / --repair-reflection flags on
pipeline submit. Diagnosis and resubmit are separate steps — Hermes /goal
reads the JSON output and decides what to do next.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

import typer

logger = logging.getLogger("expflow_pde.cli_repeat")

repeat_app = typer.Typer(
    name="repeat",
    help="Repair diagnosis and resubmit (Hermes /goal compatible)",
    no_args_is_help=True,
)


def get_repeat_app() -> typer.Typer:
    """Return the repeat sub-command group."""
    return repeat_app


@repeat_app.command("diagnose")
def repeat_diagnose_cmd(
    target: str = typer.Argument(
        ...,
        help="Pipeline ID (clearml) or path to a JSON result file",
    ),
    reflection: bool = typer.Option(
        False,
        "--reflection",
        "-R",
        help="Enable L2 reflection subagent analysis",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON (for Hermes /goal consumption)",
    ),
) -> None:
    """Analyze a failed pipeline and return structured diagnosis.

    Fetches the clearml task log (when target is a pipeline ID) or reads
    from a local JSON file (when target is a file path). Runs RepairStage
    and prints structured diagnosis.

    Examples:
        expflow repeat diagnose pipe_abc123
        expflow repeat diagnose pipe_abc123 --reflection --json
        expflow repeat diagnose /tmp/failed_result.json
    """
    # Determine if target is a file or pipeline ID
    if os.path.isfile(target):
        with open(target) as f:
            result_data = json.load(f)
        task_log = result_data.get("task_log", "")
        exit_code = result_data.get("exit_code", 1)
        pipeline_id = result_data.get("pipeline_id", target)
    else:
        pipeline_id = target
        task_log, exit_code, fetch_success = _fetch_task_log(pipeline_id)
        if not fetch_success:
            print(
                "  Warning: clearml API call failed — diagnosis may be incomplete", file=sys.stderr
            )

    if not task_log:
        print(
            "  Warning: empty task log — diagnosis will have no traceback context", file=sys.stderr
        )

    from expflow_pde.repair import RepairStage

    stage = RepairStage(experiment_id=pipeline_id)
    result = stage.run(
        task_log=task_log,
        exit_code=exit_code,
        enable_reflection=reflection,
    )
    result["pipeline_id"] = pipeline_id

    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_diagnosis(result)


@repeat_app.command("resubmit")
def repeat_resubmit_cmd(
    train_script: str = typer.Argument(..., help="Training script path (e.g. train_task1.py)"),
    eval_script: Optional[str] = typer.Option(
        None, "--eval-script", "-e", help="Evaluation script path"
    ),
    queue: str = typer.Option("default", "--queue", "-q"),
    project: str = typer.Option("PDEBench", "--project", "-p"),
    train_param: Optional[list[str]] = typer.Option(
        None,
        "--train-param",
        "-t",
        help="Training parameter e.g. --train-param epochs=80",
    ),
    eval_param: Optional[list[str]] = typer.Option(
        None,
        "--eval-param",
        help="Evaluation parameter",
    ),
    docker: Optional[str] = typer.Option(None, "--docker"),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON",
    ),
    wait: bool = typer.Option(False, "--wait", "-w"),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="Max wait time in minutes",
    ),
    skip: Optional[list[str]] = typer.Option(
        None,
        "--skip",
        help="Steps to skip",
    ),
    packages: Optional[list[str]] = typer.Option(
        None,
        "--packages",
        help="Packages for clearml Task.create",
    ),
) -> None:
    """Submit a new experiment (replaces a failed one).

    This is exactly the same as 'pipeline submit' but exists as a dedicated
    command so Hermes /goal can construct a deterministic resubmit without
    relying on expflow's internal repair logic to modify params.

    Examples:
        expflow repeat resubmit train_task1.py \\
            --train-param n_modes=24 --queue default --json
    """
    from expflow_pde.pipeline import ExperimentPipeline

    # Forward to pipeline submit — same logic but explicit about resubmit

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

    resolved_packages: list[str] | None = None
    if packages is not None:
        resolved_packages = [p for p in packages if p] if packages else []

    ep = ExperimentPipeline(
        project=project,
        queue=queue,
        docker=docker,
        packages=resolved_packages,
    )
    result = ep.train_val_submit(
        train_script=train_script,
        train_params=train_params if train_params else None,
        eval_script=eval_script,
        eval_params=eval_params if eval_params else None,
        execution_queue=queue,
        timeout=timeout if wait else None,
        skip_steps=skip,
    )

    if json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Resubmitted: {result.get('name', '?')} (ID: {result.get('pipeline_id', '?')})")
        print(f"  Status:   {result.get('status', '?')}")
        print(f"  Queue:    {queue}")


# ── Internal helpers (copied from old cli_pipeline.py) ──


def _fetch_task_log(pipeline_id: str) -> tuple[str, int, bool]:
    """Fetch pipeline task log from clearml with fallback."""
    task_log = ""
    exit_code = 1
    fetch_success = False
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
        fetch_success = True
    except ImportError:
        logger.warning("clearml SDK not available — cannot fetch task log")
    except Exception as e:
        logger.warning("Failed to fetch clearml task log for %s: %s", pipeline_id, e)
    return task_log, exit_code, fetch_success


def _print_diagnosis(result: dict[str, Any]) -> None:
    """Pretty-print a repair diagnosis result."""
    level = result.get("level", "?")
    print(f"\n  Diagnosis: level={level}")
    print(f"  Fixed:     {result.get('fixed', False)}")
    print(f"  Action:    {result.get('action', '?')[:200]}")
    ec_cat = result.get("exit_code_category", "")
    if ec_cat:
        print(f"  Exit:      {ec_cat}")
    input_valid = result.get("input_valid", True)
    if not input_valid:
        print("  Warning:   task_log empty or has no failure signal")
    if result.get("exit_code"):
        print(f"  Exit code: {result['exit_code']}")
    if result.get("wiki_source"):
        print(f"  Wiki:      {result['wiki_source']} -> {result.get('wiki_paths', [])}")
    if result.get("history"):
        print(f"  History:   {len(result['history'])} attempt(s)")
    print()
