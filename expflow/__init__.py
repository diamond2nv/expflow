#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow: experiment workflow orchestration toolkit for PDEBench/Agentic4Sci."""

__version__ = "0.2.0"

# Core API — lazy import at call time, exposed for convenience
from expflow.clearml import (
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
    model_list,
    model_upload,
    pipeline_add_step,
    pipeline_create,
    pipeline_list,
    pipeline_start,
    pipeline_stop,
)

__all__ = [
    "__version__",
    "list_tasks",
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
    "pipeline_create",
    "pipeline_add_step",
    "pipeline_start",
    "pipeline_stop",
    "pipeline_list",
]
