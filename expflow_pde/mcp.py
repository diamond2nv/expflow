#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow mcp — MCP Server for Hermes Agent integration.

Provides tools for experiment dispatch, HPO query, dataset management,
model management, and audit. Serves as the bridge between Hermes Agent
and clearml/optuna/langfuse.

Uses FastMCP for stdio transport. Start with: expflow mcp
"""

import sys


def start_mcp() -> None:
    """Start the MCP Server for Hermes Agent integration.

    Registers tools for experiment CRUD, HPO study query, dataset upload/download/lineage,
    model list/upload, compliance audit, and report generation.
    """
    print("expflow MCP server starting...", flush=True)
    _check_backend("clearml")
    _check_backend("optuna")
    _check_backend("langfuse")
    _try_serve()
    print()
    print("MCP server ready on stdio.", flush=True)


def _try_serve() -> None:
    """Attempt to serve MCP tools via FastMCP."""
    try:
        from expflow_pde.mcp_server import serve

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
    print("  - exp_compare_scores       Rank experiments by metric with gating")
    print("  - exp_list_workers          List clearml workers")
    print("  - exp_db_stats              Get dispatch database statistics")
    print("  - exp_db_tree               Get experiment tree structure")
    print("  - exp_db_archive            Archive old experiments")
    print("  - exp_db_audit_log          Query audit log entries")
    print("  - exp_db_metrics            Get metrics for an experiment")
    print()
    print("  [--] fastmcp not installed. Run: pip install fastmcp")
    print()


def _check_backend(name: str) -> None:
    """Check if a backend is available."""
    try:
        __import__(name)
        print(f"  [OK] {name}", flush=True)
    except ImportError:
        print(f"  [--] {name} (not installed)", flush=True)


def main() -> None:
    """MCP server entry point with graceful shutdown on Ctrl+C.

    FastMCP's run(transport='stdio') blocks on stdin/stdout. When the
    parent process (Hermes Agent) disconnects or the user presses Ctrl+C,
    this handler ensures a clean exit without traceback pollution.
    """
    try:
        start_mcp()
    except KeyboardInterrupt:
        print(file=sys.stderr)
        print("MCP server stopped.", file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        # Parent process closed stdin/stdout — normal shutdown
        sys.exit(0)
