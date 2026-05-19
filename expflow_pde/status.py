#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow status — measurement plane component health checks."""

from typing import Any


def check_status() -> dict[str, dict[str, Any]]:
    """Check health of all measurement plane components.

    Returns dict of {component_name: {"ok": bool, "detail": str}}.
    All checks are best-effort — failures are reported, not raised.
    """
    results: dict[str, dict[str, Any]] = {}

    # clearml server check
    try:
        import clearml  # noqa: F401
        from clearml import Task

        tasks = Task.get_tasks(project_name="expflow")
        results["clearml"] = {
            "ok": True,
            "detail": f"connected ({len(tasks)} tasks in expflow project)",
        }
    except ImportError:
        results["clearml"] = {"ok": False, "detail": "clearml SDK not installed"}
    except Exception as e:
        results["clearml"] = {"ok": False, "detail": str(e)}

    # optuna check
    try:
        import optuna  # noqa: F401

        results["optuna"] = {"ok": True, "detail": "SDK available"}
    except ImportError:
        results["optuna"] = {"ok": False, "detail": "optuna SDK not installed"}
    except Exception as e:
        results["optuna"] = {"ok": False, "detail": str(e)}

    # langfuse check
    try:
        from langfuse import Langfuse  # noqa: F401

        results["langfuse"] = {"ok": True, "detail": "SDK available"}
    except ImportError:
        results["langfuse"] = {"ok": False, "detail": "langfuse SDK not installed"}
    except Exception as e:
        results["langfuse"] = {"ok": False, "detail": str(e)}

    # W&B check
    import os

    if os.environ.get("WANDB_API_KEY"):
        results["wandb"] = {"ok": True, "detail": "WANDB_API_KEY set"}
    else:
        results["wandb"] = {"ok": False, "detail": "WANDB_API_KEY not set"}

    # TensorBoard check
    try:
        import tensorboard  # noqa: F401

        results["tensorboard"] = {"ok": True, "detail": "SDK available"}
    except ImportError:
        results["tensorboard"] = {"ok": False, "detail": "tensorboard not installed"}
    except Exception as e:
        results["tensorboard"] = {"ok": False, "detail": str(e)}

    return results
