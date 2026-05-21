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
import re as _re
from typing import Any

from expflow_pde.repair_rules import match_first

logger = logging.getLogger("expflow_pde.repair")

# ── Constants ──

MAX_L1_ATTEMPTS = 2  # Max L1 quick-fix attempts before escalating to L2

# Signal exit codes from clearml-agent or OS
SIGNAL_CODES: dict[int, str] = {
    134: "SIGABRT",
    137: "SIGKILL (OOM)",
    139: "SIGSEGV",
    143: "SIGTERM",
}

# ── Helpers ──

_FAILURE_KEYWORDS = ("Traceback", "Error", "Failed", "Killed")


def _log_has_failure_signal(task_log: str) -> bool:
    """Check if the task log contains any failure signal for analysis."""
    if not task_log or not task_log.strip():
        return False
    lowered = task_log.lower()
    return any(kw.lower() in lowered for kw in _FAILURE_KEYWORDS)


def _exit_code_category(exit_code: int) -> str:
    """Categorise an exit code for repair diagnostics."""
    if exit_code == 0:
        return "success"
    if exit_code == 1:
        return "error"
    if exit_code in SIGNAL_CODES:
        return f"signal ({SIGNAL_CODES[exit_code]})"
    return f"unknown ({exit_code})"


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
        """Run the full repair pipeline (L0 -> L1 -> L2)."""
        self._repair_history = []
        self._last_exit_code = exit_code

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
            l2_result = self._try_l2(task_log, exit_code, l1_result)
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
                "fix_params": suggestion.get("fix_params", {}),
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
            # Signal exit codes produce no Python traceback
            if exit_code in SIGNAL_CODES:
                sig_name = SIGNAL_CODES[exit_code]
                action = f"Process terminated by {sig_name} (exit code {exit_code})"
                # Check for OOM-killer signature in log
                if "killed" in task_log.lower() or exit_code == 137:
                    action += " — likely OOM (out of memory)"
                return {
                    "level": "L1",
                    "fixed": False,
                    "needs_human": True,
                    "action": action,
                    "exc_type": sig_name,
                    "exc_message": "",
                    "file": "",
                    "line": 0,
                    "exit_code_category": _exit_code_category(exit_code),
                }
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
                match = _re.search(r'File "(.+?)", line (\d+)', line)
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

    def _try_l2(self, task_log: str, exit_code: int,
               l1_result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Prepare structured L2 reflection context for Hermes subagent.

        Returns a machine-readable dict that Hermes can consume to spawn a
        delegate_task subagent. The subagent produces a fix plan (not code).

        Args:
            task_log: Console output from failed task.
            exit_code: Process exit code.
            l1_result: Output of _try_l1() — may have already identified
                       signal exit codes or extracted traceback info.
        """
        # Input validation — early exit if log has no useful signal
        if not _log_has_failure_signal(task_log):
            return {
                "level": "L2",
                "fixed": False,
                "needs_human": True,
                "input_valid": False,
                "action": "task_log empty or contains no failure signal — subagent cannot analyze",
                "fallback_instruction": (
                    "Check clearml task directly: `clearml tasks <task_id>` or "
                    "`clearml logs <task_id>` to fetch full stderr/stdout. "
                    "The task may have failed without a Python traceback."
                ),
                "exit_code": exit_code,
                "experiment_id": self._exp_id,
            }

        # Extract traceback lines for context
        tb_lines: list[str] = []
        for line in task_log.split("\n"):
            if "Traceback" in line or "Error" in line or "File " in line:
                tb_lines.append(line)
            if len(tb_lines) > 30:
                break

        # Reuse L1 results for signal exit codes (no Python traceback)
        exc_type = "Unknown"
        exc_message = ""
        if not tb_lines and l1_result:
            exc_type = l1_result.get("exc_type", "Unknown") or "Unknown"
            exc_message = l1_result.get("exc_message", "")

        if not tb_lines and exc_type == "Unknown":
            pass  # will fall through to regular extraction below

        # Identify exception type and message from traceback
        if not exc_type or exc_type == "Unknown":
            for line in reversed(tb_lines):
                stripped = line.strip()
                if stripped and "File" not in stripped and "Traceback" not in stripped:
                    if ":" in stripped:
                        exc_type = stripped.split(":")[0].strip()
                        exc_message = ":".join(stripped.split(":")[1:]).strip()
                    else:
                        exc_type = stripped.strip()
                    break

        # Extract file paths from traceback
        files_to_check: list[str] = []
        for line in tb_lines:
            m = _re.search(r'File "(.+?)"', line)
            if m:
                fp = m.group(1)
                if fp not in files_to_check:
                    files_to_check.append(fp)

        # Map exception type to relevant wiki pages
        # Use L1 signal info when available (overrides generic fallback)
        l1_ec_cat = (l1_result or {}).get("exit_code_category", "")
        if exit_code in SIGNAL_CODES and l1_ec_cat and l1_ec_cat.startswith("signal"):
            wiki_info = self._signal_to_wiki(exit_code, exc_type, exc_message)
        else:
            wiki_info = self._exc_type_to_wiki(exc_type, exc_message)

        wiki_paths = wiki_info.get("paths", [])
        wiki_source = wiki_info.get("source", "none")

        # Render the subagent prompt with structured context
        context = {
            "experiment_id": self._exp_id,
            "exit_code": exit_code,
            "exc_type": exc_type,
            "exc_message": exc_message,
            "files_to_check": files_to_check,
            "wiki_paths": wiki_paths,
            "tb_snippet": tb_lines[:20],
        }
        prompt = self._render_l2_prompt(context)

        return {
            "level": "L2",
            "fixed": False,
            "needs_human": False,
            "action": "L2 context ready — Hermes should spawn a delegate_task subagent.",
            "context_length": len(task_log),
            "exit_code": exit_code,
            "experiment_id": self._exp_id,
            "exc_type": exc_type or "Unknown",
            "exc_message": exc_message or "",
            "files_to_check": files_to_check or [],
            "wiki_paths": wiki_paths or [],
            "wiki_source": wiki_source,
            "tb_snippet": tb_lines[:20],
            "subagent_prompt": prompt,
            "subagent_schema": {
                "goal": "Analyze experiment failure and produce a fix plan",
                "role": "leaf",
                "toolsets": ["terminal", "file", "skills"],
            },
        }

    _EXC_TYPE_WIKI: list[dict[str, Any]] = [
        # Exact matches (highest priority)
        {"match": "exact", "key": "ModuleNotFoundError",
         "paths": ["~/wiki/troubleshooting/pip-dependencies.md"]},
        {"match": "exact", "key": "torch.cuda.OutOfMemoryError",
         "paths": ["~/wiki/troubleshooting/gpu-memory.md"]},
        {"match": "exact", "key": "FileNotFoundError",
         "paths": ["~/wiki/troubleshooting/data-paths.md"]},
        # Prefix matches
        {"match": "prefix", "key": "ImportError",
         "paths": ["~/wiki/troubleshooting/pip-dependencies.md"]},
        # Substring matches (for combined / clearml augmented messages)
        {"match": "substring", "key": "CUDA out of memory",
         "paths": ["~/wiki/troubleshooting/gpu-memory.md"]},
        {"match": "substring", "key": "DataLoader",
         "paths": ["~/wiki/troubleshooting/data-loader.md"]},
        # Signal- or content-based — matched via _CLASSIFY_EXC fallback
    ]

    def _exc_type_to_wiki(self, exc_type: str, exc_message: str = "") -> dict[str, Any]:
        """Map exception type to relevant wiki page paths.

        Returns dict with keys:
            paths: list[str] — wiki file paths
            source: str — "exact" | "prefix" | "substring" | "fallback" | "none"
        """
        if not exc_type or exc_type == "Unknown":
            return {"paths": [], "source": "none"}

        for entry in self._EXC_TYPE_WIKI:
            if entry["match"] == "exact" and entry["key"] == exc_type:
                return {"paths": list(entry["paths"]), "source": "exact"}
            if entry["match"] == "prefix" and exc_type.startswith(entry["key"]):
                return {"paths": list(entry["paths"]), "source": "prefix"}
            if entry["match"] == "substring" and entry["key"] in exc_type:
                return {"paths": list(entry["paths"]), "source": "substring"}

        # Fallback — content-based classification for no explicit key matched
        paths = self._CLASSIFY_EXC(exc_type, exc_message)
        if paths:
            return {"paths": paths, "source": "fallback"}
        return {"paths": [], "source": "none"}

    @staticmethod
    def _word_in(combined: str, word: str) -> bool:
        """Case-insensitive whole-word match (\\b boundaries)."""
        return bool(_re.search(r'\b' + _re.escape(word) + r'\b', combined))

    @staticmethod
    def _CLASSIFY_EXC(exc_type: str, exc_message: str = "") -> list[str]:
        """Content-based fallback classification for unmapped exception types.

        Uses whole-word matching to avoid short-keyword false positives.
        Checks both exc_type and exc_message for known keywords.
        """
        combined = (exc_type + " " + exc_message).lower()
        # Exclude list — exc_types known to trigger false positives on short keywords.
        type_only = exc_type.lower()
        _EXCLUDED_TYPES = ("diskerror", "diskioerror")  # "disk" in "diskerror" → skip

        if any(excl in type_only for excl in _EXCLUDED_TYPES):
            return []

        if (RepairStage._word_in(combined, "cuda")
                or RepairStage._word_in(combined, "out of memory")
                or RepairStage._word_in(combined, "oom")):
            return ["~/wiki/troubleshooting/gpu-memory.md"]
        if (RepairStage._word_in(combined, "killed")
                or RepairStage._word_in(combined, "sigkill")
                or RepairStage._word_in(combined, "sigterm")
                or RepairStage._word_in(combined, "sigabrt")
                or RepairStage._word_in(combined, "sigsegv")
                or RepairStage._word_in(combined, "signal 11")
                or RepairStage._word_in(combined, "signal 6")
                or RepairStage._word_in(combined, "signal 15")):
            return ["~/wiki/troubleshooting/oom-killer.md"]
        if (RepairStage._word_in(combined, "bus error")
                or RepairStage._word_in(combined, "shm")
                or RepairStage._word_in(combined, "/dev/shm")):
            return ["~/wiki/troubleshooting/shared-memory.md"]
        if (RepairStage._word_in(combined, "no space")
                or RepairStage._word_in(combined, "disk quota")
                or RepairStage._word_in(combined, "disk full")
                or RepairStage._word_in(combined, "storage exhausted")):
            return ["~/wiki/troubleshooting/disk-space.md"]
        if (RepairStage._word_in(combined, "timeout")
                or RepairStage._word_in(combined, "time out")
                or RepairStage._word_in(combined, "deadline")):
            return ["~/wiki/troubleshooting/timeouts.md"]
        return []

    @staticmethod
    def _signal_to_wiki(exit_code: int, sig_name: str,
                       exc_message: str = "") -> dict[str, Any]:
        """Map signal exit codes to specific troubleshooting wiki pages.

        137 (SIGKILL) → oom-killer.md
        139 (SIGSEGV) → segfault.md
        134 (SIGABRT) → abort.md
        143 (SIGTERM) → timeouts.md (most common cause: timeout-killed)
        """
        mapping = {
            137: "oom-killer.md",
            139: "segfault.md",
            134: "abort.md",
            143: "timeouts.md",
        }
        page = mapping.get(exit_code, "unknown-signal.md")
        return {"paths": [f"~/wiki/troubleshooting/{page}"], "source": "signal"}

    def _render_l2_prompt(self, context: dict) -> str:
        """Render the subagent prompt template with actual context values."""
        tb_str = "\n".join(context.get("tb_snippet", []))
        files_str = "\n".join(f"  - {f}" for f in context.get("files_to_check", []))
        wiki_str = "\n".join(f"  - {w}" for w in context.get("wiki_paths", []))

        return (
            "Analyze this experiment failure and produce a fix plan.\n"
            "\n"
            "## Failure Context\n"
            f"Experiment ID: {context.get('experiment_id', '?')}\n"
            f"Exit code: {context.get('exit_code', 1)}\n"
            f"Exception: {context.get('exc_type', 'Unknown')}\n"
            f"Message: {context.get('exc_message', '')}\n"
            "\n"
            "## Traceback\n"
            f"{tb_str}\n"
            "\n"
            "## Files Involved\n"
            f"{files_str}\n"
            "\n"
            "## Wiki Context\n"
            f"{wiki_str}\n"
            "\n"
            "## Task\n"
            "1. Identify the root cause.\n"
            "2. Determine which files need changes and what the changes are.\n"
            "3. Return a fix plan as VALID JSON using this EXACT schema:\n"
            "\n"
            "   {\n"
            '     "plan_type": "fix",\n'
            '     "files": [{"path": "train.py", "old": "...", "new": "..."}],\n'
            '     "reasoning": "one-line explanation"\n'
            "   }\n"
            "\n"
            "   If you CANNOT determine the fix, return:\n"
            "   {\n"
            '     "plan_type": "no_plan",\n'
            '     "reason": "why you cannot produce a fix"\n'
            "   }\n"
            "\n"
            "   OUTPUT ONLY THE JSON BLOCK — no markdown fences, no extra text.\n"
            "   Do NOT apply changes directly. Do NOT run code.\n"
        )

    # ── Internal helpers ──

    def _build_result(self, level: str, detail: dict[str, Any]) -> dict[str, Any]:
        """Build the final result dict."""
        self._fixed = detail.get("fixed", False)
        result: dict[str, Any] = {
            "fixed": self._fixed,
            "level": level,
            "action": detail.get("action", ""),
            "needs_human": detail.get("needs_human", False),
            "attempts": len(self._repair_history),
            "history": list(self._repair_history),
            "exit_code_category": _exit_code_category(getattr(self, "_last_exit_code", 1)),
            "error": None if self._fixed else (
                detail.get("action", "Repair attempted but unsuccessful")
            ),
        }
        # For L2 results, propagate structured context to top level
        # so Hermes doesn't need to dig into history[]
        if level == "L2":
            for key in ("exc_type", "exc_message", "files_to_check",
                       "wiki_paths", "wiki_source", "tb_snippet", "subagent_prompt",
                       "subagent_schema", "exit_code", "experiment_id",
                       "context_length", "input_valid", "fix_params"):
                if key in detail:
                    result[key] = detail[key]
        else:
            # Propagate fix_params for L0 results so re-submit can use them
            if level == "L0" and "fix_params" in detail:
                result["fix_params"] = detail["fix_params"]
        return result

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
