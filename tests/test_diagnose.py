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
    """seg1=85.3, seg2=55.1, seg3=22.7 -> compound_mid_long.
    Both mid_term (>25 gap) and long_term (<35) triggered."""
    from expflow_pde.analyze import diagnose_experiment

    result = diagnose_experiment(json_path=sample_eval_json)
    # Seg1-Seg2=30.2>25 (mid_term) + Seg3=22.7<35 (long_term) => compound
    assert result["degradation_pattern"] == "compound_mid_long"
    assert any("collapse" in d.lower() for d in result["diagnosis"])
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
    """Seg3 collapse -> n_modes bias + sub_step fixed + wd bias."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "long_term",
        "seg1": 85.3,
        "seg2": 55.1,
        "seg3": 22.7,
        "total": 46.4,
    }
    hp = {"n_modes": 12, "hidden_channels": 20, "lr": 0.001}
    result = suggest_next_params(diagnosis, current_hparams=hp)
    assert "search_bias" in result
    assert "fixed_params" in result
    # n_modes bias: low=16 (12+4), high=24
    assert result["search_bias"]["n_modes"]["low"] == 16
    assert result["search_bias"]["n_modes"]["high"] == 24
    assert result["fixed_params"]["num_sub_steps"] == 5
    assert "weight_decay" in result["search_bias"]  # bias, not fixed
    assert "rationale" in result


def test_suggest_already_has_wd():
    """If weight_decay already set, don't add to search_bias."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "long_term",
        "seg1": 85.0,
        "seg2": 50.0,
        "seg3": 20.0,
        "total": 40.0,
    }
    hp = {"n_modes": 12, "weight_decay": 1e-5}
    result = suggest_next_params(diagnosis, current_hparams=hp)
    assert "weight_decay" not in result["search_bias"]


def test_suggest_mid_term():
    """Medium-term drop -> stability_lambda bias."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "mid_term",
        "seg1": 87.0,
        "seg2": 45.0,
        "seg3": 38.0,
        "total": 54.0,
    }
    result = suggest_next_params(diagnosis, current_hparams={"lr": 0.001})
    assert "stability_lambda" in result["search_bias"]
    assert result["search_bias"]["stability_lambda"]["low"] == 0.0005
    assert result["search_bias"]["stability_lambda"]["high"] == 0.005


def test_suggest_stable():
    """Stable -> no bias, just rationale."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "stable",
        "seg1": 90,
        "seg2": 85,
        "seg3": 80,
        "total": 85,
    }
    result = suggest_next_params(diagnosis, current_hparams={})
    assert result["search_bias"] == {}
    assert result["fixed_params"] == {}
    assert "stable" in result["rationale"][0].lower()


def test_suggest_short_term():
    """Short-term weak -> higher lr bias range + epochs bias."""
    from expflow_pde.analyze import suggest_next_params

    diagnosis = {
        "degradation_pattern": "short_term",
        "seg1": 55.0,
        "seg2": 50.0,
        "seg3": 45.0,
        "total": 50.0,
    }
    result = suggest_next_params(diagnosis, current_hparams={"lr": 0.001, "epochs": 80})
    assert "lr" in result["search_bias"]
    assert result["search_bias"]["lr"]["low"] == 0.0005
    assert result["search_bias"]["lr"]["high"] == 0.002
    assert "epochs" in result["search_bias"]
    assert result["search_bias"]["epochs"]["low"] == 100


# ── CLI tests ──


def test_diagnose_cli_json(sample_eval_json):
    from typer.testing import CliRunner

    from expflow_pde.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "diagnose", "--json", sample_eval_json])
    assert result.exit_code == 0, result.output
    assert "Seg1" in result.output
    assert "Pattern" in result.output


def test_diagnose_cli_no_args():
    from typer.testing import CliRunner

    from expflow_pde.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "diagnose"])
    assert result.exit_code != 0
    assert "ERROR" in result.output


def test_suggest_cli(sample_eval_json):
    from typer.testing import CliRunner

    from expflow_pde.cli import app

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
    assert result["diagnosis"]["degradation_pattern"] in ("long_term", "compound_mid_long")
