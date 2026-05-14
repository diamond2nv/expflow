#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow CLI — experiment workflow orchestration."""

from typing import Optional

import typer

app = typer.Typer(
    name="expflow",
    help="Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci",
    no_args_is_help=True,
)


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
    from expflow import __version__

    print(f"expflow v{__version__}")


@app.command()
def info() -> None:
    """Show system and environment info."""
    import platform

    from expflow.config import get

    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Config root: {get('', 'N/A')}")
