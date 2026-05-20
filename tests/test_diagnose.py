"""Tests for experiment diagnosis engine (diagnose_experiment + suggest_next_params)."""

import json

import pytest


# ── Fixtures ──


@pytest.fixture
def sample_eval_json(tmp_path):
    """Create a sample eval_task1 JSON fixture — long-term collapse case."""
    data = {
        "experiment_id": "eval_task1_test",
        "results": {
            "total_mse": 0.0035,
            "mean_rel_mse": 0.067,
        },
        "segmented_scores": {
            "seg1_score": 85.3,
            "seg2_score": 55.1,
            "seg3_score": 22.7,
            "total_segmented_score": 46.4,
        },
    }
    path = tmp_path / "eval_result.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


@pytest.fixture
def stable_eval_json(tmp_path):
    """Fixture with good scores — no degradation."""
    data = {
        "experiment_id": "stable_test",
        "segmented_scores": {
            "seg1_score": 92.0,
            "seg2_score": 88.0,
            "seg3_score": 85.0,
            "total_segmented_score": 87.5,
        },
    }
    path = tmp_path / "stable_result.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


# ── diagnose_experiment tests ──


def test_diagnose_basic_output_shape(sample_eval_json):
    """Returns dict with seg keys, diagnosis list, degradation pattern."""
    from expflow_pde.analyze import diagnose_experiment

    result = diagnose_experiment(json_path=sample_eval_json)
    assert result is not None
    assert "seg1" in result
    assert "seg2" in result
    assert "seg3" in result
    assert "total" in result
    assert "diagnosis" in result
    assert isinstance(result["diagnosis"], list)
    assert "degradation_pattern" in result
    assert "total_mse" in result


def test_diagnose_detects_long_term_collapse(sample_eval_json):
    """seg1=85, seg2=55, seg3=23 -> long_term pattern."""
    from expflow_pde.analyze import diagnose_experiment

    result = diagnose_experiment(json_path=sample_eval_json)
    assert result["degradation_pattern"] == "long_term"
    assert any("collapse" in d.lower() for d in result["diagnosis"])


def test_diagnose_unknown_path():
    """Non-existent file returns None."""
    from expflow_pde.analyze import diagnose_experiment

    result = diagnose_experiment(json_path="/nonexistent/file.json")
    assert result is None


def test_diagnose_stable(stable_eval_json):
    """All seg >80 -> stable pattern."""
    from expflow_pde.analyze import diagnose_experiment

    result = diagnose_experiment(json_path=stable_eval_json)
    assert result["degradation_pattern"] == "stable"
    assert any("No critical" in d for d in result["diagnosis"])


# ── suggest_next_params tests ──


def test_suggest_long_term_collapse():
    """Seg3 collapse -> n_modes + sub_step + wd."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "long_term",
        "seg1": 85.3, "seg2": 55.1, "seg3": 22.7, "total": 46.4,
    }
    hp = {"n_modes": 12, "hidden_channels": 20, "lr": 0.001}
    result = suggest_next_params(diagnosis, current_hparams=hp)
    assert result["suggested_params"]["n_modes"] == 16  # 12+4
    assert result["suggested_params"]["num_sub_steps"] == 5
    assert result["suggested_params"].get("weight_decay") == 1e-4
    assert "rationale" in result


def test_suggest_already_has_wd():
    """If weight_decay already set, don't override."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "long_term",
        "seg1": 85.0, "seg2": 50.0, "seg3": 20.0, "total": 40.0,
    }
    hp = {"n_modes": 12, "weight_decay": 1e-5}
    result = suggest_next_params(diagnosis, current_hparams=hp)
    assert "weight_decay" not in result["suggested_params"]


def test_suggest_mid_term():
    """Medium-term drop -> stability_lambda."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "mid_term",
        "seg1": 87.0, "seg2": 45.0, "seg3": 38.0, "total": 54.0,
    }
    result = suggest_next_params(diagnosis, current_hparams={"lr": 0.001})
    assert result["suggested_params"].get("stability_lambda") == 0.001


def test_suggest_stable():
    """Stable -> HPO round recommended."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "stable",
        "seg1": 90, "seg2": 85, "seg3": 80, "total": 85,
    }
    result = suggest_next_params(diagnosis, current_hparams={})
    assert "hpo" in result["suggested_params"].get("tag", "")


def test_suggest_short_term():
    """Short-term weak -> higher lr + more epochs."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "short_term",
        "seg1": 55.0, "seg2": 50.0, "seg3": 45.0, "total": 50.0,
    }
    result = suggest_next_params(diagnosis, current_hparams={"lr": 0.001, "epochs": 80})
    assert result["suggested_params"]["lr"] == 0.002
    assert result["suggested_params"]["epochs"] == 100


# ── CLI tests ──


def test_diagnose_cli_json(sample_eval_json):
    from expflow_pde.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "diagnose", "--json", sample_eval_json])
    assert result.exit_code == 0, result.output
    assert "Seg1" in result.output
    assert "Pattern" in result.output


def test_diagnose_cli_no_args():
    from expflow_pde.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "diagnose"])
    assert result.exit_code != 0
    assert "ERROR" in result.output


def test_suggest_cli(sample_eval_json):
    from expflow_pde.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "suggest", "--json", sample_eval_json])
    assert result.exit_code == 0, result.output
    assert "n_modes" in result.output
    assert "Rationale" in result.output


# ── Iterate tests ──


def test_iterate_dry_run(sample_eval_json):
    from expflow_pde.iterate import run_iteration

    result = run_iteration(
        json_path=sample_eval_json,
        current_hparams={"n_modes": 12, "lr": 0.001},
        dry_run=True,
    )
    assert not result.get("submitted", True)
    assert "diagnosis" in result
    assert "suggestion" in result
    assert result["diagnosis"]["degradation_pattern"] == "long_term"
