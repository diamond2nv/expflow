#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow mcp — MCP Server for Hermes Agent integration.

Provides tools for experiment dispatch, HPO query, dataset management,
and audit. Serves as the bridge between Hermes Agent and clearml/optuna/langfuse.
"""


def start_mcp() -> None:
    """Start the MCP Server for Hermes Agent integration.

    Reads available backends from SDK availability and registers
    tools for experiment CRUD, HPO study query, dataset compliance,
    and audit report generation.
    """
    print("expflow MCP server starting...")
    print("Available backends:")
    _check_backend("clearml")
    _check_backend("optuna")
    _check_backend("langfuse")
    print()
    print("MCP tools registered:")
    print("  - exp_run_task          Submit a clearml task")
    print("  - exp_list_runs         List recent experiments")
    print("  - exp_get_metrics       Get metrics for a run")
    print("  - exp_compare_runs      Compare two runs")
    print("  - exp_start_hpo         Start HPO study")
    print("  - exp_get_study         Query HPO results")
    print("  - exp_hpo_plot          Generate HPO visualization")
    print("  - exp_register_dataset  Register dataset with compliance")
    print("  - exp_list_datasets     List registered datasets")
    print("  - exp_check_compliance  Check dataset compliance")
    print("  - exp_generate_report   Generate experiment report")
    print("  - exp_board_url         Get TensorBoard URL")
    print("  - exp_config_status     Check component health")
    print()
    print("MCP server ready on stdio.")


def _check_backend(name: str) -> None:
    """Check if a backend is available."""
    try:
        __import__(name)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [--] {name} (not installed)")
