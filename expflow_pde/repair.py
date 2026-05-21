#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow repair — RepairStage: auto-fix failed clearml experiments.

Three-level repair escalation:
    L0 — Rule engine (0 token, pure Python). Matches common failures.
    L1 — Traceback extraction + error localization (fixed=False, Hermes patch).
    L2 — Reflection subagent (needs enable_reflection=True) or explicit needs_human.

P0 fix: L1 never claims "fixed" — it only provides structured error context.
P0 fix: L2 with enable_reflection=False returns needs_human=True instead of
silently ending repair without actionable output.
"""

from __future__ import annotations

import json
import logging
import re
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

    L0 may return "fixed": True (rule matched, auto-fix suggestion).
    L1 always returns "fixed": False (Hermes must apply the patch).
    L2 with enable_reflection=True spawns reflection; False returns needs_human.
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

        # L1: Traceback extraction (always records findings)
        l1_result = self._try_l1(task_log, exit_code)
        self._repair_history.append(l1_result)

        # L1 with exc_info means Hermes can act on it — mark as L1 even if not fixed
        has_exc_info = (
            l1_result.get("exc_type") and l1_result.get("exc_type") != "Unknown"
        ) or exit_code != 0

        # L2: Reflection subagent (only if enabled)
        if enable_reflection:
            l2_result = self._try_l2(task_log, exit_code)
            self._repair_history.append(l2_result)
            return self._build_result("L2", l2_result)

        # L2 disabled — return what L1 found
        if has_exc_info:
            return self._build_result(
                "L1",
                {
                    "fixed": False,
                    "needs_human": True,
                    "action": (
                        f"L1 identified {l1_result.get('exc_type', 'error')} at "
                        f"{l1_result.get('file', '?')}:{l1_result.get('line', '?')}. "
                        "Enable --repair-reflection for L2 deep analysis or fix manually."
                    ),
                    "exc_type": l1_result.get("exc_type", "Unknown"),
                    "file": l1_result.get("file"),
                    "line": l1_result.get("line"),
                },
            )

        return self._build_result(
            "L1",
            {
                "action": "No matching repair rule found. No traceback captured.",
                "fixed": False,
                "needs_human": True,
            },
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
            needs_user = suggestion.get("needs_user_action", False)
            return {
                "level": "L0",
                "matched": True,
                "fixed": not needs_user,
                "rule": suggestion.get("rule", "?"),
                "action": suggestion.get("action", ""),
                "confidence": suggestion.get("confidence", 0.0),
                "needs_user_action": needs_user,
            }
        return {"level": "L0", "matched": False, "fixed": False, "action": "No rule matched"}

    # ── L1: Traceback extraction (never claims fixed) ──

    def _try_l1(self, task_log: str, exit_code: int) -> dict[str, Any]:
        """Extract traceback, identify error location — no auto-fix applied."""
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
                "action": "No traceback found in log. Cannot localize error.",
                "exc_type": "Unknown",
                "exc_message": "",
                "file": "",
                "line": 0,
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
                match = re.search(r'File "(.+?)", line (\d+)', line)
                if match:
                    last_file = match.group(1)
                    last_line = int(match.group(2))
                break

        return {
            "level": "L1",
            "fixed": False,
            "needs_human": True,
            "action": (
                f"L1 extracted {exc_type} at {last_file}:{last_line}. "
                "Hermes should read_file, apply patch, then retry."
            ),
            "exc_type": exc_type,
            "exc_message": exc_message,
            "file": last_file,
            "line": last_line,
        }

    # ── L2: Reflection subagent────

    def _try_l2(self, task_log: str, exit_code: int) -> dict[str, Any]:
        """Prepare context for L2 reflection."""
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
                "L2 context prepared. Hermes should spawn a delegate_task subagent "
                "with failure context + wiki knowledge. Subagent produces a fix plan."
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
            "needs_human": detail.get("needs_human", False),
            "attempts": len(self._repair_history),
            "history": list(self._repair_history),
            "error": None if self._fixed else (
                detail.get("action", "Repair attempted but unsuccessful")
            ),
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
