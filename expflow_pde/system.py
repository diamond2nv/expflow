#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow system — component health checks and system utilities."""


def check_health() -> dict[str, dict]:
    """Check measurement plane component health.

    Returns:
        Dict mapping component names to status dicts
        with 'available' bool and optional 'detail' str.
    """
    result: dict[str, dict] = {}

    # clearml
    try:
        import clearml  # noqa: F401

        result["clearml"] = {"available": True, "detail": "package installed"}
    except ImportError:
        result["clearml"] = {"available": False, "detail": "not installed"}

    # optuna
    try:
        import optuna  # noqa: F401

        result["optuna"] = {"available": True, "detail": "package installed"}
    except ImportError:
        result["optuna"] = {"available": False, "detail": "not installed"}

    # langfuse
    try:
        import langfuse  # noqa: F401

        result["langfuse"] = {"available": True, "detail": "package installed"}
    except ImportError:
        result["langfuse"] = {"available": False, "detail": "not installed"}

    return result
