#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow iterate CLI — one-shot automated experiment iteration."""

from typing import Optional

import typer

iterate_app = typer.Typer(
    name="iterate",
    help="One-shot experiment iteration: diagnose -> suggest -> submit",
    no_args_is_help=True,
)


@iterate_app.command("run")
def iterate_run_cmd(
    task_id: Optional[str] = typer.Option(None, "--task", "-t",
        help="Clearml task ID of completed experiment"),
    json_path: Optional[str] = typer.Option(None, "--json", "-j",
        help="Path to eval JSON file (alternative)"),
    train_script: str = typer.Option("train_task1.py", "--train", "-T",
        help="Base clearml task name for training"),
    queue: str = typer.Option("default", "--queue", "-q",
        help="Clearml queue"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n",
        help="Preview suggestion without submitting"),
) -> None:
    """Diagnose -> suggest -> submit next experiment iteration."""
    from expflow_pde.iterate import run_iteration

    if not task_id and not json_path:
        print("ERROR: Provide --task <id> or --json <path>")
        raise typer.Exit(code=1)

    result = run_iteration(
        task_id=task_id,
        json_path=json_path,
        train_script=train_script,
        queue=queue,
        dry_run=dry_run,
    )

    if "error" in result:
        print(f"ERROR: {result['error']} (step: {result.get('step', '?')})")
        raise typer.Exit(code=1)

    diag = result.get("diagnosis", {})
    sugg = result.get("suggestion", {})

    print(f"\n  Diagnosis:")
    print(f"    Pattern: {diag.get('degradation_pattern', '?')}")
    for d in diag.get("diagnosis", []):
        print(f"    - {d}")

    params = sugg.get("suggested_params", {})
    print(f"\n  Suggested params:")
    for k, v in params.items():
        if k == "tag":
            continue
        print(f"    --{k}={v}")

    if dry_run:
        print(f"\n  [dry-run] Would submit to queue '{queue}'")
    else:
        pipe = result.get("pipeline", {})
        print(f"\n  Submitted: pipeline_id={pipe.get('pipeline_id', '?')}")
        print(f"  Queue: {queue}")
