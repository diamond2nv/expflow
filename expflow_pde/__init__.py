#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow: experiment workflow orchestration toolkit for PDEBench/Agentic4Sci.

Core CLI (`expflow`) works with zero optional dependencies installed.
SDK-specific features (clearml, optuna, langfuse) are loaded on demand.
"""

from __future__ import annotations

__version__ = "0.3.0"

# ── Lazy import helpers ──
# clearml/optuna/langfuse SDKs are optional dependencies.
# Their functions are re-exported here for convenience, but imported only
# at call time so `import expflow` never triggers SDK import errors.
# Type stubs are provided via __init__.pyi for IDE/type-checker support.


def __getattr__(name: str):
    """Lazily resolve public API members at attribute access time."""
    _lazy_map = {
        # clearml
        "annotate_compliance": ("expflow_pde.clearml", "annotate_compliance"),
        "dataset_download": ("expflow_pde.clearml", "dataset_download"),
        "dataset_lineage": ("expflow_pde.clearml", "dataset_lineage"),
        "dataset_upload": ("expflow_pde.clearml", "dataset_upload"),
        "dequeue_task": ("expflow_pde.clearml", "dequeue_task"),
        "enqueue_task": ("expflow_pde.clearml", "enqueue_task"),
        "get_queue_status": ("expflow_pde.clearml", "get_queue_status"),
        "get_task": ("expflow_pde.clearml", "get_task"),
        "init_tracking": ("expflow_pde.clearml", "init_tracking"),
        "list_datasets": ("expflow_pde.clearml", "list_datasets"),
        "list_queues": ("expflow_pde.clearml", "list_queues"),
        "list_tasks": ("expflow_pde.clearml", "list_tasks"),
        "list_workers": ("expflow_pde.clearml", "list_workers"),
        "model_list": ("expflow_pde.clearml", "model_list"),
        "model_upload": ("expflow_pde.clearml", "model_upload"),
        "pipeline_add_step": ("expflow_pde.clearml", "pipeline_add_step"),
        "pipeline_create": ("expflow_pde.clearml", "pipeline_create"),
        "pipeline_list": ("expflow_pde.clearml", "pipeline_list"),
        "pipeline_start": ("expflow_pde.clearml", "pipeline_start"),
        "pipeline_stop": ("expflow_pde.clearml", "pipeline_stop"),
        "scheduler_add_task": ("expflow_pde.clearml", "scheduler_add_task"),
        "scheduler_create": ("expflow_pde.clearml", "scheduler_create"),
        "scheduler_list": ("expflow_pde.clearml", "scheduler_list"),
        "scheduler_remove_task": ("expflow_pde.clearml", "scheduler_remove_task"),
        "scheduler_start": ("expflow_pde.clearml", "scheduler_start"),
        # pipeline
        "ExperimentPipeline": ("expflow_pde.pipeline", "ExperimentPipeline"),
    }
    if name in _lazy_map:
        mod_path, attr = _lazy_map[name]
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "ExperimentPipeline",
    "annotate_compliance",
    "dataset_download",
    "dataset_lineage",
    "dataset_upload",
    "dequeue_task",
    "enqueue_task",
    "get_queue_status",
    "get_task",
    "init_tracking",
    "list_datasets",
    "list_queues",
    "list_tasks",
    "list_workers",
    "model_list",
    "model_upload",
    "pipeline_add_step",
    "pipeline_create",
    "pipeline_list",
    "pipeline_start",
    "pipeline_stop",
    "scheduler_add_task",
    "scheduler_create",
    "scheduler_list",
    "scheduler_remove_task",
    "scheduler_start",
]
