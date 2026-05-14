#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow CLI — experiment workflow orchestration."""

from typing import Optional

import typer

# Version number — keep synced with expflow/__init__.py
_EXPFLOW_VERSION = "0.1.0"

app = typer.Typer(
    name="expflow",
    help="Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci",
    no_args_is_help=True,
)


def _lazy_register_clearml():
    """Lazily import and register clearml sub-command group."""
    from expflow.cli_clearml import clearml_app

    # Check if already registered
    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "clearml":
            return clearml_app

    app.add_typer(
        clearml_app,
        name="clearml",
        help="Interact with ClearML experiment management",
    )
    return clearml_app


def _lazy_register_optuna():
    """Lazily import and register optuna sub-command group."""
    from expflow.cli_optuna import optuna_app

    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "optuna":
            return optuna_app

    app.add_typer(
        optuna_app,
        name="optuna",
        help="Interact with Optuna hyperparameter optimization",
    )
    return optuna_app


def _lazy_register_langfuse():
    """Lazily import and register langfuse sub-command group."""
    from expflow.cli_langfuse import langfuse_app

    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "langfuse":
            return langfuse_app

    app.add_typer(
        langfuse_app,
        name="langfuse",
        help="Interact with Langfuse observability platform",
    )
    return langfuse_app


# Call at module level to register sub-command groups
_ = _lazy_register_clearml()
_ = _lazy_register_optuna()
_ = _lazy_register_langfuse()


@app.callback()
def callback(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """expflow: experiment workflow orchestration toolkit."""
    if config:
        from expflow.config import load_config

        load_config(config)
    if verbose:
        print("Verbose mode enabled")


@app.command()
def version() -> None:
    """Show expflow version."""
    print(f"expflow v{_EXPFLOW_VERSION}")


@app.command()
def info() -> None:
    """Show system and environment info."""
    import platform

    from expflow.config import get

    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Config root: {get('', 'N/A')}")
