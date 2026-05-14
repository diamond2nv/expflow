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


def _lazy_register_run():
    """Lazily import and register run sub-command group."""
    from expflow.cli_run import run_app

    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "run":
            return run_app

    app.add_typer(
        run_app,
        name="run",
        help="Submit and manage experiments",
    )
    return run_app


def _lazy_register_audit():
    """Lazily import and register audit sub-command group."""
    from expflow.cli_audit import audit_app

    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "audit":
            return audit_app

    app.add_typer(
        audit_app,
        name="audit",
        help="Experiment validation, compliance checking, report generation",
    )
    return audit_app


def _lazy_register_system():
    """Lazily import and register system sub-command group."""
    from expflow.cli_system import system_app

    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "system":
            return system_app

    app.add_typer(
        system_app,
        name="system",
        help="System monitoring, health checks, utilities",
    )
    return system_app


# Call at module level to register sub-command groups
# ── Backward-compat sub-command groups ──
_ = _lazy_register_clearml()
_ = _lazy_register_optuna()
_ = _lazy_register_langfuse()
_ = _lazy_register_run()
_ = _lazy_register_audit()
_ = _lazy_register_system()


# ── Top-level commands ──


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


@app.command()
def mcp() -> None:
    """Start MCP Server for Hermes Agent integration."""
    from expflow.mcp import start_mcp

    start_mcp()


@app.command()
def init() -> None:
    """Interactively configure expflow."""
    from expflow.init import run_init

    run_init()


@app.command()
def config() -> None:
    """Show current expflow configuration."""
    from expflow.config import load_config

    cfg = load_config()
    if not cfg:
        print("No config loaded.")
        return
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sk, sv in v.items():
                print(f"  {sk}: {sv}")
        else:
            print(f"{k}: {v}")
