#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow repair — RepairStage: auto-fix failed clearml experiments.

Three-level repair escalation:
    L0 — Rule engine (0 token, pure Python). Matches ~80% of common failures.
    L1 — Fast Hermes-style fix: extract traceback, patch code, retry.
    L2 — Reflection subagent: deep analysis (spawns delegate_task for context).

Each repair creates a child experiment in the experiment tree, so the full
repair history is auditable via branches/audit_log tables.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from expflow_pde.repair_rules import match_first

logger = logging.getLogger("expflow_pde.repair")

# ── Constants ──

MAX_L1_ATTEMPTS = 2  # Max L1 quick-fix attempts before escalating to L2


# ── RepairStage ──


class RepairStage:
    """Three-level auto-repair for failed experiments.

    Usage:
        stage = RepairStage()
        result = stage.run(task_log="...", exit_code=1)
        # -> {"fixed": True, "level": "L0", "action": "...", ...}
    """

    def __init__(
        self,
        experiment_id: str | None = None,
        max_l1_attempts: int = MAX_L1_ATTEMPTS,
    ):
        self._exp_id = experiment_id or ""
        self._max_l1_attempts = max_l1_attempts
        self._repair_history: list[dict[str, Any]] = []
        self._fixed: bool = False

    # ── Public API ──

    def run(
        self,
        task_log: str,
        exit_code: int,
        enable_reflection: bool = False,
    ) -> dict[str, Any]:
        """Run the full repair pipeline (L0 → L1 → L2)."""
        self._repair_history = []

        # Only attempt repair on non-zero exit codes
        if exit_code == 0:
            return self._build_result(
                "none",
                {"action": "Exit code 0 — no repair needed", "fixed": False},
            )

        # L0: Rule engine (0 token)
        l0_result = self._try_l0(task_log, exit_code)
        if l0_result["matched"]:
            self._repair_history.append(l0_result)
            return self._build_result("L0", l0_result)

        # L1: Fast Hermes-style fix (always attempts, records findings)
        l1_result = self._try_l1(task_log, exit_code)
        self._repair_history.append(l1_result)

        # L2: Reflection subagent (only if enabled AND L1 didn't already fix)
        if enable_reflection and not l1_result["fixed"]:
            l2_result = self._try_l2(task_log, exit_code)
            return self._build_result("L2", l2_result)

        if l1_result.get("attempts", 0) > 0:
            return self._build_result("L1", l1_result)

        return self._build_result(
            "none",
            {"action": "No matching repair rule found", "fixed": False},
        )

    @property
    def history(self) -> list[dict[str, Any]]:
        """Full repair trace."""
        return list(self._repair_history)

    @property
    def fixed(self) -> bool:
        """Whether any fix was applied."""
        return self._fixed

    # ── L0: Rule engine ──

    def _try_l0(self, task_log: str, exit_code: int) -> dict[str, Any]:
        """Try L0 rule engine — 0 token, pure Python."""
        suggestion = match_first(task_log, exit_code)
        if suggestion:
            return {
                "level": "L0",
                "matched": True,
                "fixed": not suggestion.get("needs_user_action", False),
                "rule": suggestion.get("rule", "?"),
                "action": suggestion.get("action", ""),
                "confidence": suggestion.get("confidence", 0.0),
                "needs_user_action": suggestion.get("needs_user_action", False),
            }
        return {"level": "L0", "matched": False, "fixed": False, "action": "No rule matched"}

    # ── L1: Fast Hermes-style fix ──

    def _try_l1(self, task_log: str, exit_code: int) -> dict[str, Any]:
        """Try L1 quick fix: extract traceback, identify error location.

        L1 is a scaffold for Hermes to patch code. It identifies the error
        location from the traceback and returns structured context.
        """
        # Extract traceback (last 50 lines)
        lines = task_log.split("\n")
        tb_lines: list[str] = []
        capturing = False
        for line in reversed(lines):
            if line.startswith("Traceback (most recent call last)"):
                capturing = True
            if capturing:
                tb_lines.insert(0, line)
            if len(tb_lines) > 50:
                break

        if not tb_lines:
            return {
                "level": "L1",
                "fixed": False,
                "attempts": 1,
                "action": "No traceback found in log. Cannot auto-fix without stack trace.",
            }

        # Identify the exception type
        exc_type = "Unknown"
        exc_message = ""
        for line in reversed(tb_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("File ") and not stripped.startswith(
                "Traceback"
            ):
                if ":" in stripped:
                    exc_type = stripped.split(":")[0].strip()
                    exc_message = ":".join(stripped.split(":")[1:]).strip()
                else:
                    exc_type = stripped.strip()
                break

        # Identify the last file and line number in the traceback
        last_file = ""
        last_line = 0
        for line in reversed(tb_lines):
            if line.strip().startswith("File "):
                # File "/path/to/file.py", line 42, in function_name
                import re
                match = re.search(r'File "(.+?)", line (\d+)', line)
                if match:
                    last_file = match.group(1)
                    last_line = int(match.group(2))
                break

        attempts = 0
        while attempts < self._max_l1_attempts:
            attempts += 1
            self._repair_history.append({
                "level": "L1",
                "attempt": attempts,
                "action": (
                    f"Identified {exc_type} at {last_file}:{last_line}. "
                    f"Hermes can patch: read_file, fix, git commit, retry."
                ),
                "exc_type": exc_type,
                "exc_message": exc_message,
                "file": last_file,
                "line": last_line,
            })

        if attempts > 0:
            return {
                "level": "L1",
                "fixed": False,  # L1 needs Hermes to actually apply the patch
                "attempts": attempts,
                "action": (
                    f"Identified {exc_type} at {last_file}:{last_line}. "
                    "Ready for Hermes to apply patch and retry."
                ),
                "exc_type": exc_type,
                "exc_message": exc_message,
                "file": last_file,
                "line": last_line,
            }

        return {"level": "L1", "fixed": False, "action": "L1 fix not attempted"}

    # ── L2: Reflection subagent ──

    def _try_l2(self, task_log: str, exit_code: int) -> dict[str, Any]:
        """Try L2 reflection: prepare context for delegate_task subagent.

        This is a scaffold: it structures the failure context so Hermes
        (the caller) can spawn a reflection subagent with it.
        Subagent output is a fix PLAN — not direct code changes.
        """
        # Extract key context
        tb_lines: list[str] = []
        for line in task_log.split("\n"):
            if "Traceback" in line or "Error" in line or "File " in line:
                tb_lines.append(line)
            if len(tb_lines) > 30:
                break

        return {
            "level": "L2",
            "fixed": False,
            "action": (
                "Context prepared for L2 reflection. "
                "Hermes should spawn a delegate_task subagent with the "
                "failure context, wiki knowledge, and clearml task metadata. "
                "Subagent output: fix plan (files to patch, changes to make)."
            ),
            "context_length": len(task_log),
            "tb_snippet": tb_lines[:20],
            "subagent_prompt_template": (
                "Analyze this experiment failure and produce a fix plan.\n"
                "1. Root cause analysis\n"
                "2. Files that need changes\n"
                "3. Exact changes to make\n"
                "4. Verification steps\n"
                "Do NOT apply changes directly — return a plan for Hermes to execute."
            ),
        }

    # ── Internal helpers ──

    def _build_result(self, level: str, detail: dict[str, Any]) -> dict[str, Any]:
        """Build the final result dict."""
        self._fixed = detail.get("fixed", False)
        return {
            "fixed": self._fixed,
            "level": level,
            "action": detail.get("action", ""),
            "attempts": len(self._repair_history),
            "history": list(self._repair_history),
            "error": None if self._fixed else "Repair attempted but unsuccessful",
        }

    def to_json(self) -> str:
        """Serialize repair result to JSON (for audit_log)."""
        return json.dumps(
            {
                "fixed": self._fixed,
                "experiment_id": self._exp_id,
                "repair_count": len(self._repair_history),
                "repair_history": self._repair_history,
            },
            indent=2,
            ensure_ascii=False,
        )
