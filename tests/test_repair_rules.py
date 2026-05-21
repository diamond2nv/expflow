#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.repair_rules — L0 deterministic fix suggestions."""


from expflow_pde.repair_rules import (
    GitProjectNotFoundRule,
    ModuleNotFoundRule,
    PipConflictRule,
    match_first,
)


class TestGitProjectNotFoundRule:
    """Rule matches git clone failures."""

    def setup_method(self):
        self.rule = GitProjectNotFoundRule()

    def test_name_and_priority(self):
        assert self.rule.name == "git_project_not_found"
        assert self.rule.priority == 1
        assert self.rule.level == 0

    def test_matches_project_not_found(self):
        log = (
            "Cloning into 'PDEBench'...\n"
            "remote: The project you were looking for could not be found.\n"
            "fatal: repository 'git@gitlab-pdebench:wrong/prefix/PDEBench.git/' not found\n"
        )
        assert self.rule.matches(log, 128)

    def test_matches_could_not_read(self):
        log = "Could not read from remote repository."
        assert self.rule.matches(log, 128)

    def test_does_not_match_other_exit(self):
        assert not self.rule.matches("some error", 1)
        assert not self.rule.matches("", 0)

    def test_does_not_match_unrelated(self):
        log = "ModuleNotFoundError: No module named 'h5py'"
        assert not self.rule.matches(log, 128)

    def test_fix_suggestion_contains_action(self):
        result = self.rule.fix_suggestion("project not found")
        assert "action" in result
        assert "confidence" in result
        assert result["confidence"] > 0.5


class TestModuleNotFoundRule:
    """Rule matches conda env isolation issues."""

    def setup_method(self):
        self.rule = ModuleNotFoundRule()

    def test_name_and_priority(self):
        assert self.rule.name == "module_not_found"
        assert self.rule.priority == 2

    def test_matches_modulenotfound(self):
        log = "Traceback (most recent call last):\n...\nModuleNotFoundError: No module named 'h5py'"
        assert self.rule.matches(log, 1)

    def test_matches_importerror(self):
        log = "ImportError: libcuda.so.1: cannot open shared object file"
        assert self.rule.matches(log, 1)

    def test_does_not_match_exit_128(self):
        log = "ModuleNotFoundError: No module named 'x'"
        assert not self.rule.matches(log, 128)

    def test_does_not_match_clean_exit(self):
        assert not self.rule.matches("All good", 0)

    def test_fix_suggestion_needs_user_action(self):
        result = self.rule.fix_suggestion("No module named 'h5py'")
        assert result["needs_user_action"]  # Needs SSH to remote node
        assert "system_site_packages" in result["action"]
        assert result["confidence"] > 0.5


class TestPipConflictRule:
    """Rule matches pip version conflicts."""

    def setup_method(self):
        self.rule = PipConflictRule()

    def test_name_and_priority(self):
        assert self.rule.name == "pip_version_conflict"
        assert self.rule.priority == 3

    def test_matches_pip_conflict(self):
        log = "pip: resolution impossible: package A requires B>=2.0 but C needs B<2.0"
        assert self.rule.matches(log, 1)

    def test_does_not_match_without_conflict_keyword(self):
        log = "pip install succeeded"
        assert not self.rule.matches(log, 0)

    def test_does_not_match_without_pip(self):
        log = "conflict in file paths"
        assert not self.rule.matches(log, 1)

    def test_fix_suggestion(self):
        result = self.rule.fix_suggestion("pip conflict")
        assert "packages=[]" in result["action"]
        assert not result["needs_user_action"]


class TestMatchFirst:
    """Integration-level: match_first picks the right rule."""

    def test_matches_git_first(self):
        log = "fatal: Could not read from remote repository."
        result = match_first(log, 128)
        assert result is not None
        assert result["rule"] == "git_project_not_found"

    def test_matches_module_second(self):
        log = "ModuleNotFoundError: No module named 'torch'"
        result = match_first(log, 1)
        assert result is not None
        assert result["rule"] == "module_not_found"

    def test_matches_pip_third(self):
        log = "pip install failed: conflict with numpy 1.26 vs 1.24"
        result = match_first(log, 1)
        assert result is not None
        assert result["rule"] == "pip_version_conflict"

    def test_no_match_returns_none(self):
        log = "Some unknown error occurred"
        result = match_first(log, 1)
        assert result is None

    def test_no_match_on_success(self):
        result = match_first("Training completed. Score: 57.09", 0)
        assert result is None


class TestSignalExitCodes:
    """L0 rules should match on signal exit codes (134/137/139/143)."""

    def test_git_rule_matches_signal_137(self):
        rule = GitProjectNotFoundRule()
        log = "fatal: Could not read from remote repository."
        assert rule.matches(log, 137)

    def test_git_rule_matches_signal_139(self):
        rule = GitProjectNotFoundRule()
        log = "The project you were looking for could not be found."
        assert rule.matches(log, 139)

    def test_module_rule_matches_signal_137(self):
        rule = ModuleNotFoundRule()
        log = "ModuleNotFoundError: No module named 'torch'"
        assert rule.matches(log, 137)

    def test_module_rule_matches_signal_134(self):
        rule = ModuleNotFoundRule()
        log = "ImportError: libcuda.so.1 cannot open"
        assert rule.matches(log, 134)

    def test_pip_rule_matches_signal_143(self):
        rule = PipConflictRule()
        log = "pip install impossible: package conflict"
        assert rule.matches(log, 143)

    def test_pip_rule_matches_signal_137(self):
        rule = PipConflictRule()
        log = "pip resolution failed: conflicting packages"
        assert rule.matches(log, 137)

    def test_match_first_signal_137_returns_same_as_exit_1(self):
        log = "ModuleNotFoundError: No module named 'h5py'"
        r1 = match_first(log, 1)
        r2 = match_first(log, 137)
        assert r1 is not None
        assert r2 is not None
        assert r1["rule"] == r2["rule"] == "module_not_found"
