#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow init — interactive configuration wizard + expflow home setup."""

import os
from pathlib import Path

import yaml

_EXPFLOW_HOME_ENV = "EXPFLOW_HOME"


def _get_expflow_home() -> Path:
    """Return ~/.expflow (or $EXPFLOW_HOME if set)."""
    return Path(os.environ.get(_EXPFLOW_HOME_ENV, os.path.expanduser("~/.expflow")))


def _ensure_expflow_home() -> None:
    """Create ~/.expflow/ with default data files if they don't exist."""
    home = _get_expflow_home()
    home.mkdir(parents=True, exist_ok=True)

    # task_meta.yaml — competition task metadata
    meta_path = home / "task_meta.yaml"
    if not meta_path.exists():
        from importlib.resources import files as _files

        try:
            tmpl = _files("expflow_pde.data").joinpath("task_meta_template.yaml").read_text()
        except Exception:
            # Fallback: minimal empty template
            tmpl = "# expflow task metadata — edit fields below\n"
        meta_path.write_text(tmpl, encoding="utf-8")
        print(f"  [OK] Created {meta_path}")

    # noise_floor.jsonl — lazy noise floor calibration DB (empty initially)
    noise_path = home / "noise_floor.jsonl"
    if not noise_path.exists():
        noise_path.touch()
        print(f"  [OK] Created {noise_path}")

    # .gitignore to prevent accidental commit
    gitignore_path = home / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n", encoding="utf-8")
        print(f"  [OK] Created {gitignore_path}")


def run_init() -> None:
    """Interactive setup for expflow configuration + expflow home init."""
    config_path = Path.cwd() / "config.yaml"

    print("expflow init — Measurement Plane Configuration")
    print()
    print("This wizard creates a config.yaml in the current directory")
    print("and initializes ~/.expflow/ with default data files.")
    print()

    # ── Interactive config ──
    clearml_api = input("ClearML API server URL [http://localhost:8008]: ").strip()
    if not clearml_api:
        clearml_api = "http://localhost:8008"

    clearml_web = input("ClearML Web UI URL [http://localhost:8080]: ").strip()
    if not clearml_web:
        clearml_web = "http://localhost:8080"

    langfuse_host = input("Langfuse host [http://localhost:3000]: ").strip()
    if not langfuse_host:
        langfuse_host = "http://localhost:3000"

    deadline_default = "2026-06-30T23:59:59+08:00"
    deadline_str = input(f"Competition deadline (ISO-8601, CST) [{deadline_default}]: ").strip()
    if not deadline_str:
        deadline_str = deadline_default

    config = {
        "clearml": {
            "api_host": clearml_api,
            "web_host": clearml_web,
        },
        "langfuse": {
            "host": langfuse_host,
        },
        "competition": {
            "deadline": deadline_str,
            "deadline_tz": "+08:00",
        },
    }

    if config_path.exists():
        overwrite = input("config.yaml exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            return

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  [OK] Config saved to {config_path}")

    # ── Expflow home initialization ──
    print()
    print("Initializing expflow home directory...")
    _ensure_expflow_home()

    print()
    print("Next steps:")
    print(f"  1. Edit ~/.expflow/task_meta.yaml with your task metadata")
    print(f"  2. Edit {config_path} with your credentials")
    print("  3. Run: expflow status  (verify connections)")
    print("  4. Run: expflow list    (list experiments)")
