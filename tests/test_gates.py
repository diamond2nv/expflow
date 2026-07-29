#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification gates for expflow-pde (public PyPI package).

Self-contained — zero dependency on hermes-verify or any internal package.
Uses standard pytest markers registered in pyproject.toml.

Gate legend:
  B10 — FSMGate: FSM state transitions, terminal states, invalid transitions
  B11 — ConfigGate: YAML config loading + env override
  C01 — CLIGate: CLI subcommand tree, lazy registration
  C02 — EquationsGate: PDE equation registry (11 equations)
  F01 — VersionGate: Package version consistency + imports + file structure
  F02 — AuditGate: Audit/validate module functionality
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ═════════════════════════════════════════════════════════════════
# B10 — FSMGate: 实验状态机验证 (needs fysom)
# ═════════════════════════════════════════════════════════════════

pytestmark_gate_b = pytest.mark.gate_b


class TestFSMGate:
    """B10: ExperimentFSM state transition correctness."""

    @staticmethod
    def _require_fsm():
        pytest.importorskip("fysom", reason="fysom required for FSM tests")

    def test_fsm_initial_state(self):
        self._require_fsm()
        from expflow_pde.fsm import ExperimentFSM
        fsm = ExperimentFSM(experiment_id="test-001")
        assert fsm.current == "created"
        assert not fsm.is_finished()

    def test_fsm_happy_path(self):
        self._require_fsm()
        from expflow_pde.fsm import ExperimentFSM
        fsm = ExperimentFSM(experiment_id="test-002")
        fsm.dispatch()
        assert fsm.current == "dispatched"
        fsm.queue(); assert fsm.current == "queued"
        fsm.start(); assert fsm.current == "running"
        fsm.complete(); assert fsm.current == "completed"
        assert fsm.is_finished()

    def test_fsm_fail_path(self):
        self._require_fsm()
        from expflow_pde.fsm import ExperimentFSM
        fsm = ExperimentFSM(experiment_id="test-003")
        fsm.dispatch(); fsm.queue(); fsm.start(); fsm.fail()
        assert fsm.current == "failed"
        assert fsm.is_finished()

    def test_fsm_cancel_from_created(self):
        self._require_fsm()
        from expflow_pde.fsm import ExperimentFSM
        fsm = ExperimentFSM(experiment_id="test-004")
        fsm.cancel()
        assert fsm.current == "cancelled"

    def test_fsm_invalid_transition_raises(self):
        self._require_fsm()
        from expflow_pde.fsm import ExperimentFSM
        fsm = ExperimentFSM(experiment_id="test-005")
        fsm.dispatch(); fsm.queue(); fsm.start()
        with pytest.raises((RuntimeError, Exception)):
            fsm.dispatch()

    def test_fsm_terminal_states(self):
        self._require_fsm()
        from expflow_pde.fsm import TERMINAL_STATES, STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED
        assert STATE_COMPLETED in TERMINAL_STATES
        assert STATE_FAILED in TERMINAL_STATES
        assert STATE_CANCELLED in TERMINAL_STATES


# ═════════════════════════════════════════════════════════════════
# B11 — ConfigGate: 配置加载验证 (zero deps)
# ═════════════════════════════════════════════════════════════════


class TestConfigGate:
    """B11: Config loading, dot-separated access, env override."""

    def test_config_loads_from_yaml(self, temp_workdir):
        cfg = temp_workdir / "config.yaml"
        cfg.write_text("key: value\nnested:\n  inner: 42\n")
        from expflow_pde.config import load_config, get
        assert load_config(str(cfg)).get("key") == "value"
        assert get("nested.inner") == 42

    def test_config_returns_empty_for_missing(self):
        from expflow_pde.config import load_config
        assert load_config("/tmp/nonexistent_cfg_xyz.yaml") == {}

    def test_config_get_default(self):
        from expflow_pde.config import get
        assert get("nonexistent.key", default="fb") == "fb"


# ═════════════════════════════════════════════════════════════════
# C01 — CLIGate: CLI 子命令树 (needs typer)
# ═════════════════════════════════════════════════════════════════

pytestmark_gate_c = pytest.mark.gate_c


class TestCLIGate:
    """C01: CLI subcommand registration tree with lazy loading."""

    @staticmethod
    def _require_typer():
        pytest.importorskip("typer", reason="typer required for CLI tests")

    def test_cli_app_imports(self):
        self._require_typer()
        from expflow_pde.cli import app
        assert app is not None

    def test_core_top_level_commands(self):
        self._require_typer()
        from expflow_pde.cli import app
        names = {c.name for c in app.registered_commands}
        for cmd in ("version", "info", "init", "config"):
            assert cmd in names, f"Missing: {cmd}"

    def test_mcp_command_exists(self):
        self._require_typer()
        from expflow_pde.cli import app
        assert "mcp" in {c.name for c in app.registered_commands}

    def test_lazy_registration_groups(self):
        self._require_typer()
        from expflow_pde.cli import app
        groups = {g for g in app.registered_groups}
        for grp in ("clearml", "optuna", "langfuse", "run",
                     "audit", "system", "pin", "analyze", "pipeline"):
            assert grp in groups, f"Missing group: {grp}"

    def test_version_runs(self):
        self._require_typer()
        from typer.testing import CliRunner
        from expflow_pde.cli import app
        r = CliRunner().invoke(app, ["version"])
        assert r.exit_code == 0
        assert "expflow v" in r.stdout


# ═════════════════════════════════════════════════════════════════
# C02 — EquationsGate: PDE 方程注册表 (zero deps)
# ═════════════════════════════════════════════════════════════════


class TestEquationsGate:
    """C02: PDE equation registry correctness."""

    def test_equations_count(self):
        from expflow_pde.equations import get_equations
        assert len(get_equations()) >= 10

    def test_burgers_present(self):
        from expflow_pde.equations import get_equation
        b = get_equation("burgers")
        assert b and "latex" in b and b["dim"] == 1

    def test_unknown_returns_none(self):
        from expflow_pde.equations import get_equation
        assert get_equation("nonexistent_eq_xyz") is None

    def test_required_fields(self):
        from expflow_pde.equations import get_equations
        for name, info in get_equations().items():
            assert "full_name" in info and "latex" in info and "dim" in info

    def test_metrics_no_crash(self):
        from expflow_pde.equations import get_equations, get_equation_metrics
        for name in get_equations():
            m = get_equation_metrics(name)
            assert isinstance(m, (dict, list, type(None))), f"{name}: {type(m)}"

    def test_list_for_task(self):
        from expflow_pde.equations import list_equations_for_task, get_equations
        task1 = list_equations_for_task("task1")
        assert isinstance(task1, list)
        if task1:
            expected = [n for n, i in get_equations().items()
                       if i.get("competition_task") == "task1"]
            assert len(task1) == len(expected)


# ═════════════════════════════════════════════════════════════════
# F01 — VersionGate: 版本一致性 + 导入结构
# ═════════════════════════════════════════════════════════════════

pytestmark_gate_f = pytest.mark.gate_f


class TestVersionGate:
    """F01: Package version consistency + import structure."""

    REPO_ROOT = Path(__file__).resolve().parent.parent

    def test_version_defined(self):
        from expflow_pde import __version__
        assert __version__ and isinstance(__version__, str)

    def test_version_matches_pyproject(self):
        import tomllib
        from expflow_pde import __version__
        data = tomllib.loads(
            (self.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert __version__ == data.get("project", {}).get("version", "")

    def test_core_modules_importable(self):
        for mod in ("expflow_pde.config", "expflow_pde.equations",
                     "expflow_pde.metrics", "expflow_pde.audit"):
            __import__(mod)

    def test_entry_point_defined(self):
        import tomllib
        data = tomllib.loads(
            (self.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert "expflow" in data.get("project", {}).get("scripts", {})

    def test_license_exists(self):
        assert (self.REPO_ROOT / "LICENSE").exists()

    def test_readme_exists(self):
        assert (self.REPO_ROOT / "README.md").exists()

    def test_no_internal_deps(self):
        import tomllib
        data = tomllib.loads(
            (self.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        for dep in data.get("project", {}).get("dependencies", []):
            assert "hermes-verify" not in dep
            assert "hermes_verify" not in dep


# ═════════════════════════════════════════════════════════════════
# F02 — AuditGate: 审计模块 (zero deps)
# ═════════════════════════════════════════════════════════════════


class TestAuditGate:
    """F02: Audit module can validate and produce compliance results."""

    def test_validate_experiment_imports(self):
        from expflow_pde.audit import validate_experiment
        assert callable(validate_experiment)

    def test_check_dataset_compliance_imports(self):
        from expflow_pde.audit import check_dataset_compliance
        assert callable(check_dataset_compliance)

    def test_generate_report_imports(self):
        from expflow_pde.audit import generate_report
        assert callable(generate_report)

    def test_validate_competition_rules_imports(self):
        from expflow_pde.audit import validate_competition_rules
        assert callable(validate_competition_rules)

    def test_noise_aware_validate_imports(self):
        from expflow_pde.validate import noise_aware_validate
        assert callable(noise_aware_validate)

    def test_validate_experiment_runs(self, temp_workdir):
        from expflow_pde.audit import validate_experiment
        result = validate_experiment(
            experiment_id="test-gate-001",
            config_snapshot={},
            metrics={},
        )
        assert isinstance(result, (dict, type(None)))
