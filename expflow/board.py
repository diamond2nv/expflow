#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow board — TensorBoard launcher."""

import subprocess
import sys
from pathlib import Path
from typing import Any


def start_board(port: int = 6006, logdir: str | None = None) -> dict[str, Any]:
    """Start TensorBoard server.

    Args:
        port: Port to listen on.
        logdir: TensorBoard log directory (default: ./runs).

    Returns:
        Dict with status, port, logdir, or error.
    """
    if logdir is None:
        logdir = str(Path.cwd() / "runs")

    log_path = Path(logdir)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)

    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return {"error": "tensorboard not installed. Run: pip install tensorboard"}

    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(log_path),
                "--port",
                str(port),
                "--host",
                "0.0.0.0",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "status": "started",
        "port": port,
        "logdir": str(log_path),
    }
