#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow system CLI sub-commands — monitoring, health checks, utilities."""

from typing import Optional

import typer

system_app = typer.Typer(
    name="system",
    help="System monitoring, health checks, utilities",
    no_args_is_help=True,
)


def get_system_app() -> typer.Typer:
    return system_app


@system_app.command("status")
def status_cmd() -> None:
    """Check health of measurement plane components."""
    from expflow_pde.status import check_status

    results = check_status()
    for name, info in results.items():
        icon = "OK" if info.get("ok") else "ERR"
        detail = info.get("detail", "")
        print(f"[{icon:>3}] {name:<20} {detail}")


@system_app.command("board")
def board_cmd(
    port: int = typer.Option(6006, "--port", "-p", help="TensorBoard port"),
    logdir: Optional[str] = typer.Option(
        None,
        "--logdir",
        "-l",
        help="Log directory (default: ./runs)",
    ),
) -> None:
    """Launch TensorBoard."""
    from expflow_pde.board import start_board

    result = start_board(port=port, logdir=logdir)
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    print(f"TensorBoard started at http://localhost:{port}")
    print(f"  Logdir: {result['logdir']}")
