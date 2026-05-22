#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow init — interactive configuration wizard."""

from pathlib import Path

import yaml


def run_init() -> None:
    """Interactive setup for expflow measurement plane configuration."""
    config_path = Path.cwd() / "config.yaml"

    print("expflow init — Measurement Plane Configuration")
    print()
    print("This wizard creates a config.yaml in the current directory.")
    print()

    # Ask for clearml server URL
    clearml_api = input("ClearML API server URL [http://localhost:8008]: ").strip()
    if not clearml_api:
        clearml_api = "http://localhost:8008"

    clearml_web = input("ClearML Web UI URL [http://localhost:8080]: ").strip()
    if not clearml_web:
        clearml_web = "http://localhost:8080"

    # Ask for langfuse URL
    langfuse_host = input("Langfuse host [http://localhost:3000]: ").strip()
    if not langfuse_host:
        langfuse_host = "http://localhost:3000"

    # Competition deadline
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

    print(f"Config saved to {config_path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {config_path} with your credentials")
    print("  2. Run: expflow status  (verify connections)")
    print("  3. Run: expflow list    (list experiments)")
