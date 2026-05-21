#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow MCP Server — FastMCP-based tool server for Hermes Agent.

Provides 18 tools across experiment tracking, HPO, dataset management,
model management, compliance audit, and system health.

Start via: expflow mcp  (which calls this module)
"""

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore


def serve() -> None:
    """Build and start the MCP server with all tools."""
    if FastMCP is None:
        print("  [--] fastmcp not installed. Run: pip install fastmcp")
        return

    mcp = FastMCP("expflow-mcp")

    _register_experiment_tools(mcp)
    _register_hpo_tools(mcp)
    _register_dataset_tools(mcp)
    _register_model_tools(mcp)
    _register_pipeline_tools(mcp)
    _register_audit_tools(mcp)
    _register_system_tools(mcp)

    print("  [OK] FastMCP server with 24 tools")
    mcp.run(transport="stdio")


# ── Experiment tools ──


def _register_experiment_tools(mcp: "FastMCP") -> None:
    from expflow_pde.clearml import dequeue_task, enqueue_task, get_task, list_tasks
    from expflow_pde.compare import compare_scores

    @mcp.tool()
    def exp_list_runs(project: str = "PDEBench", limit: int = 20) -> list[dict]:
        """List recent experiments (clearml tasks)."""
        return list_tasks(project_name=project)[:limit]

    @mcp.tool()
    def exp_get_run(experiment_id: str) -> dict:
        """Get details for a single experiment."""
        return get_task(experiment_id)

    @mcp.tool()
    def exp_enqueue_run(task_id: str, queue: str = "default") -> dict:
        """Enqueue a task to a clearml queue."""
        return enqueue_task(task_id, queue_name=queue)

    @mcp.tool()
    def exp_dequeue_run(task_id: str) -> dict:
        """Dequeue a task."""
        return dequeue_task(task_id)

    @mcp.tool()
    def exp_compare_scores(
        project: str = "PDEBench",
        tags: list[str] | None = None,
        sort_by: str = "seg_total",
        ascending: bool = False,
        gates: list[dict] | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Rank experiments by metric score with optional gating.

        Each gate is a dict: {"metric": "pde_mean", "op": "lt", "value": 18.09}
        """
        return compare_scores(
            project=project,
            tags=tags,
            sort_by=sort_by,
            ascending=ascending,
            gates=gates,
            max_results=max_results,
        )


# ── HPO tools ──


def _register_hpo_tools(mcp: "FastMCP") -> None:
    from expflow_pde.optuna import get_study, list_studies

    @mcp.tool()
    def exp_list_studies() -> list[dict]:
        """List all Optuna HPO studies."""
        return list_studies()

    @mcp.tool()
    def exp_get_study(study_name: str) -> dict:
        """Get details for an Optuna study."""
        return get_study(study_name=study_name)


# ── Dataset tools ──


def _register_dataset_tools(mcp: "FastMCP") -> None:
    from expflow_pde.clearml import (
        dataset_download,
        dataset_lineage,
        dataset_upload,
        list_datasets,
    )

    @mcp.tool()
    def exp_dataset_upload(
        local_path: str,
        dataset_name: str,
        project: str = "PDEBench",
        version: str | None = None,
        parent_ids: list[str] | None = None,
        compliance: str | None = None,
    ) -> dict:
        """Upload local files to clearml Fileserver and register as a Dataset."""
        return dataset_upload(
            local_path=local_path,
            dataset_name=dataset_name,
            dataset_project=project,
            version=version,
            parent_dataset_ids=parent_ids,
            compliance=compliance,  # type: ignore
        )

    @mcp.tool()
    def exp_dataset_download(
        target_folder: str,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        project: str = "PDEBench",
        version: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Download a Dataset from clearml Fileserver to a local folder."""
        return dataset_download(
            target_folder=target_folder,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_project=project,
            dataset_version=version,
            overwrite=overwrite,
        )

    @mcp.tool()
    def exp_dataset_lineage(dataset_id: str, depth: int = 10) -> list[dict]:
        """Trace dataset lineage via parent chain."""
        return dataset_lineage(dataset_id=dataset_id, depth=depth)

    @mcp.tool()
    def exp_list_datasets(
        project: str | None = "PDEBench",
        compliance_filter: str | None = None,
    ) -> list[dict]:
        """List registered datasets with compliance info."""
        return list_datasets(
            compliance_filter=compliance_filter,  # type: ignore
        )


# ── Model tools ──


def _register_model_tools(mcp: "FastMCP") -> None:
    from expflow_pde.clearml import model_list, model_upload

    @mcp.tool()
    def exp_model_list(
        project: str | None = None,
        published_only: bool = False,
        max_results: int = 20,
    ) -> list[dict]:
        """List registered checkpoint models."""
        return model_list(
            project_name=project,
            only_published=published_only,
            max_results=max_results,
        )

    @mcp.tool()
    def exp_model_upload(
        local_path: str,
        task_id: str,
        framework: str = "PyTorch",
        model_name: str | None = None,
    ) -> dict:
        """Upload a model checkpoint to clearml Model store."""
        return model_upload(
            local_path=local_path,
            task_id=task_id,
            framework=framework,
            model_name=model_name,
        )


# ── Pipeline tools ──


def _register_pipeline_tools(mcp: "FastMCP") -> None:
    from expflow_pde.clearml import (
        pipeline_add_step,
        pipeline_create,
        pipeline_list,
        pipeline_start,
        pipeline_stop,
    )

    @mcp.tool()
    def exp_pipeline_create(
        name: str,
        project: str = "PDEBench",
        version: str | None = None,
        abort_on_failure: bool = False,
    ) -> dict:
        """Create a clearml PipelineController."""
        return pipeline_create(
            name=name,
            project=project,
            version=version,
            abort_on_failure=abort_on_failure,
        )

    @mcp.tool()
    def exp_pipeline_add_step(
        pipeline_name: str,
        step_name: str,
        project: str = "PDEBench",
        base_task_id: str | None = None,
        base_task_name: str | None = None,
        parents: list[str] | None = None,
        execution_queue: str | None = None,
        parameter_override: dict | None = None,
    ) -> dict:
        """Add a step to an existing pipeline controller."""
        return pipeline_add_step(
            pipeline_name=pipeline_name,
            project=project,
            step_name=step_name,
            base_task_id=base_task_id,
            base_task_name=base_task_name,
            parents=parents,
            execution_queue=execution_queue,
            parameter_override=parameter_override,
        )

    @mcp.tool()
    def exp_pipeline_start(
        pipeline_name: str,
        project: str = "PDEBench",
        queue_name: str | None = None,
        timeout_minutes: float | None = None,
    ) -> dict:
        """Start a pipeline controller execution."""
        return pipeline_start(
            pipeline_name=pipeline_name,
            project=project,
            queue_name=queue_name,
            timeout_minutes=timeout_minutes,
        )

    @mcp.tool()
    def exp_pipeline_stop(
        pipeline_name: str,
        project: str = "PDEBench",
    ) -> dict:
        """Stop a running pipeline controller."""
        return pipeline_stop(
            pipeline_name=pipeline_name,
            project=project,
        )

    @mcp.tool()
    def exp_pipeline_list(
        project: str | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """List pipeline controller tasks."""
        return pipeline_list(
            project_name=project,
            max_results=max_results,
        )


# ── Audit tools ──


def _register_audit_tools(mcp: "FastMCP") -> None:
    from expflow_pde.audit import generate_report

    @mcp.tool()
    def exp_generate_report(
        experiment_id: str,
        config: dict | None = None,
        metrics: dict | None = None,
    ) -> dict:
        """Generate an experiment report in Markdown."""
        return generate_report(
            experiment_id=experiment_id,
            config=config or {},
            metrics=metrics or {},
        )


# ── System tools ──


def _register_system_tools(mcp: "FastMCP") -> None:
    from expflow_pde.system import check_health

    @mcp.tool()
    def exp_config_status() -> dict:
        """Check measurement plane component health."""
        return check_health()

    @mcp.tool()
    def exp_list_workers(
        project: str | None = None,
        max_results: int = 50,
    ) -> list[dict]:
        """List registered clearml workers."""
        from expflow_pde.clearml import list_workers

        return list_workers(
            project_name=project,
            max_results=max_results,
        )

    @mcp.tool()
    def exp_db_stats() -> dict:
        """Get dispatch database statistics."""
        from expflow_pde.dispatch_db import DispatchDB

        return DispatchDB().stats()

    @mcp.tool()
    def exp_db_tree(root_id: str) -> dict:
        """Get experiment tree structure from dispatch database."""
        from expflow_pde.dispatch_db import DispatchDB

        return DispatchDB().get_experiment_tree(root_id)

    @mcp.tool()
    def exp_db_archive(before_date: str) -> dict:
        """Archive old experiments from dispatch database."""
        from expflow_pde.dispatch_db import DispatchDB

        return DispatchDB().archive(before_date)

    @mcp.tool()
    def exp_db_audit_log(
        experiment_id: str | None = None,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query audit log entries."""
        from expflow_pde.dispatch_db import DispatchDB

        return DispatchDB().get_audit_log(
            experiment_id=experiment_id,
            event_type=event_type,
            limit=limit,
        )

    @mcp.tool()
    def exp_db_metrics(
        experiment_id: str,
        metric_name: str | None = None,
    ) -> list[dict]:
        """Get metrics for an experiment from dispatch database."""
        from expflow_pde.dispatch_db import DispatchDB

        return DispatchDB().get_metrics(experiment_id, name=metric_name)
