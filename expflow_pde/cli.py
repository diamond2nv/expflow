#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow CLI — experiment workflow orchestration."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="expflow-pde",
    help="Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci",
    no_args_is_help=True,
)


def _lazy_register_clearml():
    """Lazily import and register clearml sub-command group."""
    from expflow_pde.cli_clearml import clearml_app

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
    from expflow_pde.cli_optuna import optuna_app

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
    from expflow_pde.cli_langfuse import langfuse_app

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
    from expflow_pde.cli_run import run_app

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
    from expflow_pde.cli_audit import audit_app

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
    from expflow_pde.cli_system import system_app

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
        from expflow_pde.config import load_config

        load_config(config)
    if verbose:
        print("Verbose mode enabled")


def _get_version() -> str:
    """Return package version from __init__."""
    from expflow_pde import __version__

    return __version__


def _get_git_info() -> dict[str, str]:
    """Get git describe and commit info, or empty dict if not a git repo."""
    repo = Path(__file__).resolve().parent.parent
    try:
        r = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo,
        )
        describe = r.stdout.strip() if r.returncode == 0 else "unknown"
        r2 = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo,
        )
        branch = r2.stdout.strip() if r2.returncode == 0 else "unknown"
        return {"describe": describe, "branch": branch}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"describe": "N/A", "branch": "N/A"}


@app.command()
def version(
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show build info"),
) -> None:
    """Show expflow version."""
    ver = _get_version()
    if verbose:
        git = _get_git_info()
        print(f"expflow v{ver}")
        print(f"  Build:     {git['describe']}")
        print(f"  Branch:    {git['branch']}")
    else:
        print(f"expflow v{ver}")


@app.command()
def info() -> None:
    """Show detailed system, build, and environment info."""
    from expflow_pde import __version__

    ver = __version__
    git = _get_git_info()

    # Python / OS
    py_impl = platform.python_implementation()
    py_ver = platform.python_version()

    os_ver = f"{platform.system()} {platform.release()}"

    # Module availability with version
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    modules: dict[str, str] = {}
    for mod in ("clearml", "optuna", "langfuse"):
        try:
            __import__(mod)
            try:
                pkg_ver = _pkg_version(mod)
            except PackageNotFoundError:
                pkg_ver = "?"
            modules[mod] = pkg_ver
        except ImportError:
            modules[mod] = ""

    # clearml config check
    clearml_ver = modules.get("clearml", "")
    clearml_ok = bool(clearml_ver)

    # Print
    print("=" * 60)
    print("  expflow — Experiment Workflow Orchestration")
    print("=" * 60)
    print()
    print("  Version:")
    print(f"    Package:      v{ver}")
    print(f"    Build:        {git['describe']}")
    print(f"    Branch:       {git['branch']}")
    print()
    print("  System:")
    print(f"    OS:           {os_ver}")
    print(f"    Python:       {py_impl} {py_ver}")
    print(f"    Config root:  {Path.cwd()}")
    print()
    print("  Optional SDKs:")
    for mod in sorted(modules.keys()):
        ver = modules[mod]
        if ver:
            print(f"    \u2713 {mod}")
            print(f"         ({ver})")
        else:
            print(f"    \u2013 {mod}")
    print()
    print("  ClearML:")
    if clearml_ok:
        print(f"    SDK:          {clearml_ver}")
        try:
            from clearml.config import running_remotely

            is_agent = running_remotely()
            print(f"    Running remotely: {'yes' if is_agent else 'no'}")
        except Exception:
            pass
    else:
        print("    SDK:          not installed")
    print("=" * 60)


@app.command()
def mcp() -> None:
    """Start MCP Server for Hermes Agent integration."""
    from expflow_pde.mcp import start_mcp

    start_mcp()


@app.command()
def init() -> None:
    """Interactively configure expflow."""
    from expflow_pde.init import run_init

    run_init()


@app.command()
def config() -> None:
    """Show current expflow configuration."""
    from expflow_pde.config import load_config

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
