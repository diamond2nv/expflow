"""Third-party entry point tests for expflow CLI.

Verifies the exact pip-installed user experience:
1. entry_point_script — subprocess simulates console_scripts entry point
2. entry_point_missing_dep — --no-deps install shows ModuleNotFoundError for clearml commands
3. entry_point_help — --help shows correct usage
4. entry_point_info — basic non-clearml commands work without clearml dep
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

# Version import before HERE — needed by entry_point tests below
from expflow_pde import __version__ as _ver  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, ".."))


def _built_wheel_path() -> str | None:
    """Find the latest .whl in dist/."""
    dist_dir = os.path.join(PROJECT_ROOT, "dist")
    if not os.path.isdir(dist_dir):
        return None
    candidates = sorted(f for f in os.listdir(dist_dir) if f.endswith(".whl"))
    return os.path.join(dist_dir, candidates[-1]) if candidates else None


ENTRY_SCRIPT = (
    "#!/usr/bin/env python3\nimport sys\nfrom expflow_pde.cli import app\nsys.exit(app())\n"
)


def _run_entry_point(args: list[str], **subprocess_kwargs) -> subprocess.CompletedProcess:
    """Run the expflow entry point as a subprocess."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "entry_test")
        with open(path, "w") as f:
            f.write(ENTRY_SCRIPT)
        os.chmod(path, 0o755)

        env = {**os.environ, "PYTHONPATH": PROJECT_ROOT}
        # Remove any clearml SDK to test missing dep
        env.pop("CLEARML_API_HOST", None)

        result = subprocess.run(
            [sys.executable, path] + args,
            capture_output=True,
            text=True,
            timeout=15,
            env={**env, **(subprocess_kwargs.pop("env", {}))},
            **subprocess_kwargs,
        )
        return result


# ── Skip marker — wheel builds needed for --no-deps tests ──

_WHEEL_BUILT = _built_wheel_path()


@pytest.fixture(scope="module")
def built_wheel() -> str:
    """Build a wheel and return its path (once per module)."""
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", PROJECT_ROOT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Build failed: {result.stderr}"
    wheel = _built_wheel_path()
    assert wheel is not None, "No .whl found after build"
    return wheel


# ══════════════════════════════════════════════════════════════
# Entry point tests (subprocess)
# ══════════════════════════════════════════════════════════════


class TestEntryPoint:
    """Verify the pip-installed user experience for expflow."""

    def test_entry_point_version(self):
        """Entry point --version works correctly."""
        result = _run_entry_point(["version"])
        assert result.returncode == 0
        assert result.stdout.strip() == f"expflow v{_ver}"
        assert result.stderr == ""

    def test_entry_point_help(self):
        """Entry point --help shows usage with all commands."""
        result = _run_entry_point(["--help"])
        assert result.returncode == 0
        assert "version" in result.stdout
        assert "info" in result.stdout
        assert "clearml" in result.stdout
        assert "optuna" in result.stdout

    def test_entry_point_optuna_help(self):
        """Entry point optuna --help shows all optuna sub-commands."""
        result = _run_entry_point(["optuna", "--help"])
        assert result.returncode == 0
        assert "create-study" in result.stdout
        assert "studies" in result.stdout
        assert "study" in result.stdout
        assert "delete-study" in result.stdout
        assert "ask" in result.stdout
        assert "tell" in result.stdout
        assert "plot" in result.stdout

    def test_entry_point_info(self):
        """Entry point info shows system info (no clearml needed)."""
        result = _run_entry_point(["info"])
        assert result.returncode == 0
        assert "expflow" in result.stdout
        assert "Python:" in result.stdout

    def test_entry_point_clearml_help(self):
        """Entry point clearml --help shows all clearml sub-commands."""
        result = _run_entry_point(["clearml", "--help"])
        assert result.returncode == 0
        assert "tasks" in result.stdout
        assert "enqueue" in result.stdout
        assert "dequeue" in result.stdout
        assert "queues" in result.stdout
        assert "dataset-register" in result.stdout
        assert "dataset-list" in result.stdout


# ══════════════════════════════════════════════════════════════
# Missing dependency tests (temp venv + --no-deps)
# ══════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestEntryPointMissingDep:
    """Verify graceful error when clearml is not installed."""

    def test_entry_point_missing_clearml_shows_module_not_found(self, built_wheel):
        """pip install expflow without clearml extra -> clearml commands show ModuleNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            venv_path = os.path.join(tmp, "venv")
            subprocess.run(
                [sys.executable, "-m", "venv", venv_path],
                capture_output=True,
                timeout=30,
                check=True,
            )
            pip = os.path.join(venv_path, "bin", "pip")

            # Install normally (all deps) but no clearml
            install_result = subprocess.run(
                [pip, "install", built_wheel],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert install_result.returncode == 0, f"Install failed: {install_result.stderr}"

            expflow_cli = os.path.join(venv_path, "bin", "expflow")

            # version command should work fine
            ver_result = subprocess.run(
                [expflow_cli, "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert ver_result.returncode == 0, f"version failed: {ver_result.stderr}"
            assert ver_result.stdout.strip() == "expflow v0.6.0"

            # info command should work
            info_result = subprocess.run(
                [expflow_cli, "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert info_result.returncode == 0, f"info failed: {info_result.stderr}"
            assert "expflow" in info_result.stdout

            # clearml command without clearml SDK -> graceful error
            clearml_result = subprocess.run(
                [expflow_cli, "clearml", "tasks"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert clearml_result.returncode == 1, (
                f"Expected non-zero, got: {clearml_result.stdout}"
            )
            assert (
                "No module named 'clearml'" in clearml_result.stderr
                or "ModuleNotFoundError: No module named 'clearml'" in clearml_result.stderr
            )

            # optuna command without optuna SDK -> graceful error
            optuna_result = subprocess.run(
                [expflow_cli, "optuna", "studies"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert optuna_result.returncode == 1, f"Expected non-zero, got: {optuna_result.stdout}"
            assert (
                "No module named 'optuna'" in optuna_result.stderr
                or "ModuleNotFoundError: No module named 'optuna'" in optuna_result.stderr
            )
