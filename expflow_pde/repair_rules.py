#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow repair rules — Level 0 rule-based failure diagnosis and fix suggestions.

L0 rules handle ~80% of clearml-agent failures without requiring any LLM calls.
Each rule is a pure function: inspect task log + exit code → match? → fix suggestion.

Rules are ordered by priority (lowest number = first to try).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Signal exit codes that may carry useful failure context for L0 rules
_SIGNAL_EXIT_CODES = {134, 137, 139, 143}


# ── Rule base ──


class RepairRule(ABC):
    """Base class for a deterministic repair rule.

    Subclasses must define:
    - level: int (0 = config/environment, 1 = code error)
    - priority: int (execution order, lowest first)
    - name: str (unique rule identifier)
    """

    level: int
    priority: int
    name: str

    @abstractmethod
    def matches(self, task_log: str, exit_code: int) -> bool:
        """Check if this rule applies to the given failure."""

    @abstractmethod
    def fix_suggestion(self, task_log: str) -> dict[str, Any]:
        """Return a fix suggestion dict.

        Returns:
            dict with keys:
            - "action": str — description of what to do
            - "file": str | None — file to patch (if applicable)
            - "command": str | None — shell command to run (if applicable)
            - "confidence": float — 0.0 to 1.0
            - "needs_user_action": bool — True if user must do something
        """


# ── Rule 1: Git project not found ──


class GitProjectNotFoundRule(RepairRule):
    """Git clone fails because repo URL prefix is wrong.

    Matches: exit_code=128 AND "project not found" or "Could not read" in log.
    Fix: Check clearml.conf force_git_ssh_protocol, remove extra namespace prefix.
    """

    level = 0
    priority = 1
    name = "git_project_not_found"

    def matches(self, task_log: str, exit_code: int) -> bool:
        # exit_code 128 = git remote error; also accept signal codes
        if exit_code not in {128, 134, 137, 139, 143}:
            return False
        lower_log = task_log.lower()
        return "project not found" in lower_log or "could not read" in lower_log \
            or "could not be found" in lower_log

    def fix_suggestion(self, task_log: str) -> dict[str, Any]:
        return {
            "action": (
                "Git clone failed with 'project not found'. "
                "Check that the task's repo URL does not have an extra namespace prefix "
                "(e.g. 'yourlab/Agentic4Sci/PDEBench' → 'Agentic4Sci/PDEBench'). "
                "Also verify clearml.conf has force_git_ssh_protocol: true "
                "and that the SSH deploy key matches the git remote."
            ),
            "file": None,
            "command": "ssh -T git@gitlab-pdebench",
            "confidence": 0.85,
            "needs_user_action": True,
        }


# ── Rule 2: Module not found (conda env isolation) ──


class ModuleNotFoundRule(RepairRule):
    """Agent isolated venv doesn't see conda-installed packages.

    Matches: exit_code=1 AND ModuleNotFoundError/ImportError in log.
    Fix: Set system_site_packages=true in clearml.conf,
         or start agent with CLEARML_AGENT_SKIP_PIP_VENV_INSTALL.
    """

    level = 0
    priority = 2
    name = "module_not_found"

    def matches(self, task_log: str, exit_code: int) -> bool:
        if exit_code not in {1, 134, 137, 139, 143}:
            return False
        lower_log = task_log.lower()
        return "modulenotfounderror" in lower_log or "importerror" in lower_log

    def fix_suggestion(self, task_log: str) -> dict[str, Any]:
        # Try to extract the missing module name
        missing_module = ""
        for line in task_log.split("\n"):
            if "modulenotfounderror" in line.lower() or "importerror" in line.lower():
                import re
                match = re.search(r"no module named ['\"]?(\w[\w.-]*)", line.lower())
                if match:
                    missing_module = match.group(1)
                    break

        action = (
            "Module import failed — the agent likely created an isolated venv "
            "that doesn't inherit conda-installed packages.\n"
            "Fix: set system_site_packages: true in clearml.conf, "
            "or start agent with CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/path/to/python "
            "and CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1."
        )
        if missing_module:
            action += f"\nMissing module: {missing_module}"

        return {
            "action": action,
            "file": None,
            "command": (
                "On the remote node: edit ~/clearml.conf → set "
                "agent.package_manager.system_site_packages: true, "
                "then restart clearml-agent daemon"
            ),
            "confidence": 0.80,
            "needs_user_action": True,  # Needs SSH to remote node or agent restart
        }


# ── Rule 3: Pip version conflict ──


class PipConflictRule(RepairRule):
    """Pip install fails due to version conflicts from auto-detected packages.

    Matches: exit_code=1 AND pip-related conflict text in log.
    Fix: Use packages=[] in Task.create() — rely on conda env instead.
    """

    level = 0
    priority = 3
    name = "pip_version_conflict"

    def matches(self, task_log: str, exit_code: int) -> bool:
        if exit_code not in {1, 134, 137, 139, 143}:
            return False
        lower_log = task_log.lower()
        return "pip" in lower_log and ("conflict" in lower_log or "resolution" in lower_log)

    def fix_suggestion(self, task_log: str) -> dict[str, Any]:
        return {
            "action": (
                "Pip install failed due to version conflicts. "
                "The auto-detected requirements (Task.create with packages=True) "
                "may include incompatible versions. "
                "Fix: use packages=[] in Task.create() and rely on "
                "the existing conda environment."
            ),
            "file": None,
            "command": None,
            "confidence": 0.90,
            "needs_user_action": False,
        }


# ── Rule registry ──

_RULES: list[RepairRule] = [
    GitProjectNotFoundRule(),
    ModuleNotFoundRule(),
    PipConflictRule(),
]


def get_rules() -> list[RepairRule]:
    """Return all registered L0 rules, sorted by priority."""
    return sorted(_RULES, key=lambda r: r.priority)


def match_first(task_log: str, exit_code: int) -> dict[str, Any] | None:
    """Find the first matching rule and return its fix suggestion.

    Args:
        task_log: Full console output from the failed task.
        exit_code: Exit code of the failed process.

    Returns:
        Fix suggestion dict if a rule matches, or None if no rule matches.
    """
    for rule in get_rules():
        if rule.matches(task_log, exit_code):
            result = rule.fix_suggestion(task_log)
            result["rule"] = rule.name
            return result
    return None
