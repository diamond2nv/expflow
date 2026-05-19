#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow: experiment workflow orchestration toolkit for PDEBench/Agentic4Sci."""

__version__ = "0.3.0"

# Core API — lazy import at call time, exposed for convenience
from expflow_pde.clearml import (
    annotate_compliance,
    dataset_download,
    dataset_lineage,
    dataset_upload,
    dequeue_task,
    enqueue_task,
    get_queue_status,
    get_task,
    init_tracking,
    list_datasets,
    list_queues,
    list_tasks,
    list_workers,
    model_list,
    model_upload,
    pipeline_add_step,
    pipeline_create,
    pipeline_list,
    pipeline_start,
    pipeline_stop,
    scheduler_add_task,
    scheduler_create,
    scheduler_list,
    scheduler_remove_task,
    scheduler_start,
)
from expflow_pde.pipeline import ExperimentPipeline

__all__ = [
    "__version__",
    "ExperimentPipeline",
    "list_tasks",
    "list_workers",
    "get_task",
    "enqueue_task",
    "dequeue_task",
    "list_queues",
    "get_queue_status",
    "annotate_compliance",
    "list_datasets",
    "dataset_upload",
    "dataset_download",
    "dataset_lineage",
    "model_list",
    "model_upload",
    "init_tracking",
    "scheduler_create",
    "scheduler_add_task",
    "scheduler_list",
    "scheduler_remove_task",
    "scheduler_start",
    "pipeline_create",
    "pipeline_add_step",
    "pipeline_start",
    "pipeline_stop",
    "pipeline_list",
]
