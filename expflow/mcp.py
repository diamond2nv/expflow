#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow mcp — MCP Server for Hermes Agent integration.

Provides tools for experiment dispatch, HPO query, dataset management,
model management, and audit. Serves as the bridge between Hermes Agent
and clearml/optuna/langfuse.

Uses FastMCP for stdio transport. Start with: expflow mcp
"""


def start_mcp() -> None:
    """Start the MCP Server for Hermes Agent integration.

    Registers tools for experiment CRUD, HPO study query, dataset upload/download/lineage,
    model list/upload, compliance audit, and report generation.
    """
    print("expflow MCP server starting...")
    _check_backend("clearml")
    _check_backend("optuna")
    _check_backend("langfuse")
    _try_serve()
    print()
    print("MCP server ready on stdio.")


def _try_serve() -> None:
    """Attempt to serve MCP tools via FastMCP."""
    try:
        from expflow.mcp_server import serve

        serve()
    except ImportError:
        _list_tools_stub()


def _list_tools_stub() -> None:
    """Stub: print available tools when FastMCP is not available."""
    print()
    print("MCP tools registered:")
    print("  - exp_run_task             Submit a clearml task")
    print("  - exp_list_runs            List recent experiments")
    print("  - exp_get_metrics          Get metrics for a run")
    print("  - exp_compare_runs         Compare two runs")
    print("  - exp_start_hpo            Start HPO study")
    print("  - exp_get_study            Query HPO results")
    print("  - exp_hpo_plot             Generate HPO visualization")
    print("  - exp_register_dataset     Register dataset with compliance [deprecated]")
    print("  - exp_list_datasets        List registered datasets")
    print("  - exp_dataset_upload       Upload dataset to Fileserver")
    print("  - exp_dataset_download     Download dataset from Fileserver")
    print("  - exp_dataset_lineage      Trace dataset lineage")
    print("  - exp_model_list           List checkpoint models")
    print("  - exp_model_upload         Upload checkpoint model")
    print("  - exp_check_compliance     Check dataset compliance")
    print("  - exp_generate_report      Generate experiment report")
    print("  - exp_board_url            Get TensorBoard URL")
    print("  - exp_config_status        Check component health")


def _check_backend(name: str) -> None:
    """Check if a backend is available."""
    try:
        __import__(name)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [--] {name} (not installed)")
