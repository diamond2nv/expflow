#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow dummy CLI — Dummy experiment game commands.

Provides a simulated experiment loop for testing the diagnose → suggest →
submit → repair → iterate cycle without real GPUs or clearml.

Usage:
    expflow dummy start --task task1 --seed 42
    expflow dummy step --params '{"n_modes": 20, "sub_step": 5}'
    expflow dummy step --inject module_not_found
    expflow dummy status
    expflow dummy auto --max-steps 10
"""

from __future__ import annotations

import json
from typing import Any, Optional

import typer

from expflow_pde.dummy.game import DummyExperimentGame

dummy_app = typer.Typer(
    name="dummy",
    help="Simulated experiment game for testing diagnose→repair→iterate cycle.",
    no_args_is_help=True,
)

# Shared game instance (singleton for the CLI session)
_game: DummyExperimentGame | None = None


def _get_game() -> DummyExperimentGame:
    global _game
    if _game is None:
        _game = DummyExperimentGame()
    return _game


@dummy_app.command()
def start(
    task: str = typer.Option("task1", "--task", help="Competition task ID"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Start a new dummy game session."""
    global _game
    _game = DummyExperimentGame(task_id=task, seed=seed)
    result = _game.start()
    print(json.dumps(result, indent=2, ensure_ascii=False))


@dummy_app.command()
def step(
    params: Optional[str] = typer.Option(
        None, "--params", help="JSON: suggested parameter changes"
    ),
    strategy: Optional[str] = typer.Option(
        None, "--strategy", help="Strategy name for branch link"
    ),
    inject: Optional[str] = typer.Option(
        None, "--inject",
        help=(
            "Failure pattern to inject: "
            "git_not_found, module_not_found, cuda_oom, "
            "data_not_found, unknown_error"
        ),
    ),
) -> None:
    """Run one dummy game step (diagnose → suggest → submit)."""
    game = _get_game()
    parsed_params: dict[str, Any] = {}
    if params:
        try:
            parsed_params = json.loads(params)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            raise typer.Exit(code=1)

    result = game.step(
        suggested_params=parsed_params,
        strategy=strategy,
        inject_failure=inject,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


@dummy_app.command()
def status() -> None:
    """Show current game state."""
    game = _get_game()
    result = game.status()
    print(json.dumps(result, indent=2, ensure_ascii=False))


@dummy_app.command()
def reset(
    seed: Optional[int] = typer.Option(None, "--seed", help="New random seed"),
) -> None:
    """Reset the game to baseline."""
    game = _get_game()
    result = game.reset(seed=seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


@dummy_app.command()
def auto(
    max_steps: int = typer.Option(10, "--max-steps", help="Max auto steps"),
    repair: bool = typer.Option(True, "--repair/--no-repair", help="Enable repair on failure"),
) -> None:
    """Run full automatic game: iterate diagnose→suggest→step until ceiling."""
    game = _get_game()
    game.start()

    from expflow_pde.analyze import diagnose_experiment, suggest_next_params
    from expflow_pde.repair import RepairStage

    for i in range(max_steps):
        print(f"\n=== Step {i + 1}/{max_steps} ===")

        # Check if we need to diagnose a previous step
        last_exp = game._db.get_experiment(game._last_exp_id)
        if last_exp and last_exp.get("result_summary"):
            summary = json.loads(last_exp["result_summary"])
            _ = diagnose_experiment(  # side effect: warms up diagnose module
                task_id=game.task_id,
                json_path=None,
            )
            # We can't pass clearml task to diagnose, so construct manually
            manual_diagnosis = {
                "seg1": summary.get("seg1", 0),
                "seg2": summary.get("seg2", 0),
                "seg3": summary.get("seg3", 0),
                "total": sum(summary.get(k, 0) for k in ("seg1", "seg2", "seg3")),
                "total_mse": 0.0,
            }
            suggestion = suggest_next_params(manual_diagnosis, task_id=game.task_id)
            suggested = suggestion.get("suggested_params", {})
            print(f"  Diagnose: {suggestion.get('degradation_pattern', 'stable')}")
            print(f"  Suggest: {suggested}")
        else:
            suggested = {}
            strategy = None

        # Run step
        result = game.step(
            suggested_params=suggested or None,
            strategy=strategy,
        )
        print(f"  Result: status={result['status']}, seg={result['seg']}, total={result['total']:.1f}")

        # Repair if failed
        if result["status"] == "failed" and repair:
            inject = result.get("inject_failure", {})
            exit_code = inject.get("exit_code", 1)
            stage = RepairStage(experiment_id=result["experiment_id"])
            repair_result = stage.run(
                task_log=result.get("task_log", ""),
                exit_code=exit_code,
                enable_reflection=True,
            )
            print(f"  Repair: level={repair_result['level']}, fixed={repair_result['fixed']}")
            if repair_result["fixed"]:
                # Retry without failure injection
                retry = game.step(suggested_params=suggested, strategy=strategy)
                print(f"  Retry: status={retry['status']}, seg={retry['seg']}")

        # Check ceiling
        status_info = game.status()
        seg = status_info["current_seg"]
        ceil = status_info.get("steps_left_to_ceiling", 99)
        if ceil <= 0:
            print(f"\n=== Reached ceiling ({seg}) — game ends ===")
            break

    print(f"\n=== Game complete: {game._step_count} steps ===")
    stats = game._db.stats()
    print(f"Experiments: {stats['total_experiments']}")
    print(f"By status: {stats['by_status']}")


@dummy_app.command()
def list_failures() -> None:
    """List available failure patterns for --inject."""
    from expflow_pde.dummy.game import _FAILURE_TEMPLATES

    print("Available failure patterns:")
    print("  Pattern Name         | Expected Level | Exit Code")
    print("  " + "-" * 55)
    for name, (_, code, level) in sorted(_FAILURE_TEMPLATES.items()):
        print(f"  {name:22s} | {level:14s} | {code}")
    print()
    print("Use: expflow dummy step --inject <pattern_name>")
