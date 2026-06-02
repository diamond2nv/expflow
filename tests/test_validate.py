#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.validate — three-tier validation.

Tier 1: noise_aware_validate (AutoScientists champion promotion)
Tier 2: check_pde_residual_gate (Zhang2026 scale-aware physics gate)
Tier 3: arbitrate_agent_outputs (inter-rater reliability)
"""

from __future__ import annotations

import json
import math
import os
import tempfile

from expflow_pde.validate import (
    arbitrate_agent_outputs,
    calibrate_noise_floor,
    calibrate_sigma_multiplier,
    check_pde_residual_gate,
    noise_aware_validate,
    record_noise_entry,
)


# ══════════════════════════════════════════════════════════════
# Tier 1 — Noise-Aware Champion Validation (AutoScientists)
# ══════════════════════════════════════════════════════════════


class TestNoiseAwareValidate:
    """Tests for noise_aware_validate — AutoScientists champion promotion rule.

    Theory: Δ = candidate - champion. Promote if Δ > M·σ.
    σ = pooled seed stdev from duplicate runs (intra-experiment, NOT inter-agent).
    """

    def test_confident_improvement(self):
        """Δ > M·σ → promote (AutoScientists Eq.1)."""
        result = noise_aware_validate(
            candidate_value=100.0,
            champion_value=90.0,
            noise_floor=1.0,
            sigma_multiplier=2.0,
        )
        assert result["action"] == "promote"
        assert result["delta"] == 10.0
        assert not result["second_seed_needed"]

    def test_within_noise_band(self):
        """0 < Δ ≤ M·σ → confirm (within noise band)."""
        result = noise_aware_validate(
            candidate_value=91.0,
            champion_value=90.0,
            noise_floor=1.0,
            sigma_multiplier=2.0,
        )
        assert result["action"] == "confirm"
        assert result["delta"] == 1.0
        assert result["second_seed_needed"]

    def test_no_improvement(self):
        """Δ ≤ 0 → reject."""
        result = noise_aware_validate(
            candidate_value=89.0,
            champion_value=90.0,
            noise_floor=1.0,
            sigma_multiplier=2.0,
        )
        assert result["action"] == "reject"
        assert result["delta"] == -1.0

    def test_exact_zero_delta(self):
        """Exact equality → reject (Δ=0 ≤ 0)."""
        result = noise_aware_validate(
            candidate_value=90.0,
            champion_value=90.0,
            noise_floor=1.0,
        )
        assert result["action"] == "reject"

    def test_lazy_calibration_no_db(self):
        """Noise_floor=None and no DB → uses fallback sigma."""
        result = noise_aware_validate(
            candidate_value=100.5,
            champion_value=100.0,
        )
        assert result["action"] in ("promote", "confirm")
        assert result["noise_floor"] > 0

    def test_sigma_multiplier_effect(self):
        """Higher M → more conservative (more confirms)."""
        base = noise_aware_validate(
            candidate_value=91.0, champion_value=90.0,
            noise_floor=1.0, sigma_multiplier=0.5,
        )
        conservative = noise_aware_validate(
            candidate_value=91.0, champion_value=90.0,
            noise_floor=1.0, sigma_multiplier=5.0,
        )
        assert base["action"] == "promote"       # Δ=1 > 0.5*1
        assert conservative["action"] == "confirm"  # Δ=1 ≤ 5*1

    def test_worse_than_champion(self):
        """Δ < 0 → reject regardless of noise."""
        result = noise_aware_validate(
            candidate_value=80.0, champion_value=100.0,
            noise_floor=10.0, sigma_multiplier=2.0,
        )
        assert result["action"] == "reject"
        assert result["delta"] == -20.0

    def test_sigma_zero_uses_fallback(self):
        """Zero sigma → uses fallback estimation."""
        result = noise_aware_validate(
            candidate_value=91.0, champion_value=90.0,
            noise_floor=0.0,
        )
        assert result["action"] in ("promote", "confirm")
        assert result["noise_floor"] > 0

    def test_auto_scientists_default_M2(self):
        """AutoScientists uses M=2 as default."""
        # Δ=3 with σ=2, M=2: 3 > 4? No → confirm (not promote)
        result = noise_aware_validate(
            candidate_value=93.0, champion_value=90.0,
            noise_floor=2.0, sigma_multiplier=2.0,
        )
        assert result["action"] == "confirm"
        # Δ=5 with σ=2, M=2: 5 > 4? Yes → promote
        result2 = noise_aware_validate(
            candidate_value=95.0, champion_value=90.0,
            noise_floor=2.0, sigma_multiplier=2.0,
        )
        assert result2["action"] == "promote"

    def test_pde_training_high_M(self):
        """For PDE training (seed noise 0.01-0.1%), M=10 is recommended."""
        # Tiny sigma (0.1% of 100 scale), small delta
        result = noise_aware_validate(
            candidate_value=100.2, champion_value=100.0,
            noise_floor=0.1, sigma_multiplier=10.0,
        )
        # Δ=0.2 ≤ 10*0.1=1.0 → confirm
        assert result["action"] == "confirm"

    def test_pde_training_confident(self):
        """With M=10, need Δ > 10σ for promote."""
        result = noise_aware_validate(
            candidate_value=101.5, champion_value=100.0,
            noise_floor=0.1, sigma_multiplier=10.0,
        )
        # Δ=1.5 > 10*0.1=1.0 → promote
        assert result["action"] == "promote"


# ══════════════════════════════════════════════════════════════
# Tier 2 — Physics Residual Gate (Zhang2026 JFM)
# ══════════════════════════════════════════════════════════════


class TestPdeResidualGate:
    """Tests for check_pde_residual_gate — scale-aware physics constraint.

    Theory (Zhang2026 JFM Sec 3.3): use combined relative + absolute threshold.
    blocked if candidate > max(champion × (1+rel), champion + abs_floor)
    """

    def test_passed_when_small_increase(self):
        """Small relative increase passes the gate."""
        result = check_pde_residual_gate(
            candidate_residual=0.06,
            champion_residual=0.05,
            relative_threshold=0.50,
            absolute_floor=0.01,
        )
        # threshold = max(0.05*1.5, 0.05+0.01) = max(0.075, 0.06) = 0.075
        # 0.06 < 0.075 → passed
        assert not result["blocked"]

    def test_blocked_when_large_increase(self):
        """Large relative increase triggers the gate."""
        result = check_pde_residual_gate(
            candidate_residual=0.12,
            champion_residual=0.05,
            relative_threshold=0.50,
            absolute_floor=0.01,
        )
        # threshold = max(0.075, 0.06) = 0.075  (relative dominates)
        # 0.12 > 0.075 → blocked
        assert result["blocked"]

    def test_scale_consistency_large_residual(self):
        """At large absolute scale, relative threshold dominates consistently."""
        # Champion=10.0, rel=0.50, abs_floor=0.01
        # threshold = max(15.0, 10.01) = 15.0
        result = check_pde_residual_gate(
            candidate_residual=14.0,
            champion_residual=10.0,
            relative_threshold=0.50,
            absolute_floor=0.01,
        )
        # 14.0 < 15.0 → passed (same 40% increase as small-residual case)
        assert not result["blocked"]

        result2 = check_pde_residual_gate(
            candidate_residual=16.0,
            champion_residual=10.0,
            relative_threshold=0.50,
            absolute_floor=0.01,
        )
        # 16.0 > 15.0 → blocked (same 60% increase)
        assert result2["blocked"]

    def test_absolute_floor_near_zero(self):
        """When champion residual is near zero, absolute floor prevents false block."""
        result = check_pde_residual_gate(
            candidate_residual=0.005,
            champion_residual=0.001,
            relative_threshold=0.50,
            absolute_floor=0.01,
        )
        # threshold = max(0.001*1.5, 0.001+0.01) = max(0.0015, 0.011) = 0.011
        # 0.005 < 0.011 → passed (relative=400% would have blocked without floor)
        assert not result["blocked"]

    def test_strict_gating(self):
        """Lower relative_threshold = stricter gate."""
        result = check_pde_residual_gate(
            candidate_residual=0.065,
            champion_residual=0.05,
            relative_threshold=0.20,  # strict: only 20% increase allowed
            absolute_floor=0.01,
        )
        # threshold = max(0.05*1.2, 0.06) = max(0.06, 0.06) = 0.06
        # 0.065 > 0.06 → blocked (30% increase exceeds 20%)
        assert result["blocked"]

    def test_relaxed_gating(self):
        """Higher relative_threshold = relaxed gate."""
        result = check_pde_residual_gate(
            candidate_residual=0.065,
            champion_residual=0.05,
            relative_threshold=1.00,  # relaxed: 100% increase allowed
            absolute_floor=0.01,
        )
        # threshold = max(0.05*2.0, 0.06) = max(0.10, 0.06) = 0.10
        # 0.065 < 0.10 → passed
        assert not result["blocked"]

    def test_pde_gate_reports_relative_increase(self):
        """Gate result includes relative increase info."""
        result = check_pde_residual_gate(
            candidate_residual=0.12,
            champion_residual=0.05,
        )
        assert "residual_increase_rel" in result
        # (0.12-0.05)/0.05 = 1.4
        assert abs(result["residual_increase_rel"] - 1.4) < 1e-12

    def test_zero_champion_fallback(self):
        """When champion is 0, threshold uses absolute_floor only."""
        result = check_pde_residual_gate(
            candidate_residual=0.02,
            champion_residual=0.0,
            relative_threshold=0.50,
            absolute_floor=0.01,
        )
        # threshold = max(0*1.5, 0+0.01) = max(0, 0.01) = 0.01
        # 0.02 > 0.01 → blocked
        assert result["blocked"]


# ══════════════════════════════════════════════════════════════
# Tier 3 — Agent Arbiter (inter-rater reliability)
# ══════════════════════════════════════════════════════════════


def _agent(score: float, agent_id: str = "a") -> dict:
    return {"output": f"... {agent_id}", "score": score, "agent_id": agent_id}


class TestArbitrateAgentOutputs:
    """Tests for arbitrate_agent_outputs — inter-rater reliability arbiter.

    Designed after audit: this is NOT experimental noise gating.
    It measures rank-agreement among agents scoring the SAME proposal.
    Promotion is decided first; agreement is a secondary report, not a blocker.
    """

    # ── Champion comparison tests (primary decision) ──

    def test_promote_best_large_margin(self):
        """Large margin above champion → promote_best."""
        # Tight cluster far above champion
        result = arbitrate_agent_outputs(
            outputs=[_agent(95), _agent(94), _agent(93)],
            champion_score=40.0,
            sigma_multiplier=2.0,
        )
        assert result["action"] == "promote_best"
        assert result["champion_score"] == 95.0

    def test_confirm_best_small_margin(self):
        """Small margin within dispersion band → confirm_best."""
        result = arbitrate_agent_outputs(
            outputs=[_agent(82), _agent(81), _agent(80)],
            champion_score=80.0,
            sigma_multiplier=2.0,
        )
        assert result["action"] == "confirm_best"

    def test_request_rerun_all_below(self):
        """All scores below champion → request_rerun."""
        result = arbitrate_agent_outputs(
            outputs=[_agent(70), _agent(68), _agent(65)],
            champion_score=80.0,
        )
        assert result["action"] == "request_rerun"
        assert len(result["low_scorers"]) == 3

    def test_first_round_no_champion(self):
        """No champion_score → promote best as initial."""
        result = arbitrate_agent_outputs(
            outputs=[_agent(85), _agent(82), _agent(78)],
            champion_score=None,
        )
        assert result["action"] == "promote_best"
        assert result["champion_score"] == 85.0

    # ── Agreement reporting (secondary, no longer blocking) ──

    def test_high_dispersion_does_not_block_promotion(self):
        """Bug fix: high dispersion no longer blocks promotion (Problem 2)."""
        # Three agents: one excellent, two poor → high dispersion
        result = arbitrate_agent_outputs(
            outputs=[_agent(95, "a1"), _agent(50, "a2"), _agent(45, "a3")],
            champion_score=30.0,
        )
        # Best=95 >> champion=30, Δ=65, should promote despite σ/mean being huge
        assert result["action"] == "promote_best"
        assert result["agreement"]["agreement_level"] == "conflict"

    def test_agreement_reported_separately(self):
        """Agreement info is reported in result['agreement'] dict."""
        result = arbitrate_agent_outputs(
            outputs=[_agent(85), _agent(82), _agent(78)],
            champion_score=80.0,
        )
        assert "agreement" in result
        assert "agreement_level" in result["agreement"]
        assert "sigma_ratio" in result["agreement"]
        assert "champion_margin_ratio" in result["agreement"]

    def test_suspicious_agreement_flagged_not_blocked(self):
        """Suspicious agreement (σ/mean < 1%) flags but doesn't block promotion."""
        # Three nearly identical scores
        result = arbitrate_agent_outputs(
            outputs=[
                _agent(80.0001, "a1"),
                _agent(80.0002, "a2"),
                _agent(79.9999, "a3"),
            ],
            champion_score=70.0,
        )
        # Promotion should still happen since best > champion
        assert result["action"] == "promote_best"
        assert result["agreement"]["agreement_level"] == "suspicious"

    # ── Boundary tests ──

    def test_insufficient_agents(self):
        """Fewer than 2 outputs → insufficient_agents."""
        result = arbitrate_agent_outputs(outputs=[_agent(85)])
        assert result["action"] == "insufficient_agents"

    def test_empty_outputs(self):
        """Empty list → insufficient_agents."""
        result = arbitrate_agent_outputs(outputs=[])
        assert result["action"] == "insufficient_agents"

    def test_missing_score_key(self):
        """Outputs missing score_key → insufficient."""
        result = arbitrate_agent_outputs(
            outputs=[
                {"output": "...", "agent_id": "a1"},
                {"output": "...", "agent_id": "a2"},
            ],
            score_key="score",
        )
        assert result["action"] == "insufficient_agents"

    def test_custom_score_key(self):
        """Custom score_key works."""
        result = arbitrate_agent_outputs(
            outputs=[
                {"output": "...", "quality": 90, "agent_id": "a1"},
                {"output": "...", "quality": 85, "agent_id": "a2"},
                {"output": "...", "quality": 83, "agent_id": "a3"},
            ],
            score_key="quality",
            champion_score=80.0,
        )
        assert result["best_score"] == 90.0

    def test_partial_low_scorers(self):
        """Some agents below champ → low_scorers correct."""
        result = arbitrate_agent_outputs(
            outputs=[_agent(82), _agent(79), _agent(77)],
            champion_score=80.0,
        )
        assert result["action"] == "confirm_best"
        assert len(result["low_scorers"]) == 2

    # ── Noise separation tests (post-audit correctness) ──

    def test_experiment_noise_not_confused_with_agent_dispersion(self):
        """Verify that agent arbiter does NOT claim to measure experimental noise.

        This was Problem 1 from the audit: the old code conflated inter-agent
        score dispersion with AutoScientists σ. The new code clearly separates them.
        """
        # Agents with 1-point dispersion
        outputs = [_agent(85), _agent(84), _agent(84)]
        result = arbitrate_agent_outputs(
            outputs=outputs,
            champion_score=80.0,
        )

        # Verify sigma_ratio is reported as agreement metric, NOT noise floor
        sigma_ratio = result["agreement"]["sigma_ratio"]
        assert sigma_ratio > 0  # Agreement dispersion is present
        assert "sigma" not in result  # No top-level sigma field
        # The message should NOT claim this is experimental noise
        assert "noise" not in result["message"].lower()

    def test_sigma_never_zero_with_identical_scores(self):
        """Precision-safe: identical scores should not crash (fixes Problem 5)."""
        # Use direct float construction, no str() round-trip
        result = arbitrate_agent_outputs(
            outputs=[_agent(80.0), _agent(80.0), _agent(80.0)],
            champion_score=70.0,
        )
        # Should still work and produce a valid action
        assert result["action"] in ("promote_best", "confirm_best")

    def test_widely_different_agents_correct_scenario(self):
        """Review audit scenario: same fact, different interpretation.

        Three agents read the same experimental result (seg_total=72.3)
        but have different interpretations. Scores [90, 70, 50] with champion=80.

        Here Δ=10, σ=20, 2σ=40. Δ=10 < 40 → confirm_best. This is *correct*:
        when agents are this conflicted (σ/mean=28%), we should NOT confidently
        promote — the dispersion tells us there's genuine rater disagreement.
        The key fix is that agreement is advisory, not blocking: we got
        confirm_best (not raise_alarm), and agreement is reported separately.
        """
        result = arbitrate_agent_outputs(
            outputs=[
                _agent(90, "architecture_focused"),
                _agent(70, "overfit_concerned"),
                _agent(50, "missing_modes"),
            ],
            champion_score=80.0,
        )
        # Δ=10 < 2*σ (σ=20) → confirm_best (correct — dispersion is high)
        assert result["action"] == "confirm_best"
        # Agreement is separately reported as conflict (informational)
        assert result["agreement"]["agreement_level"] == "conflict"
        # Message should describe the situation without claiming "noise"
        assert "noise" not in result["message"].lower()

    def test_widely_different_but_large_margin_still_promotes(self):
        """Audit Problem 2 scenario: high dispersion but huge margin → promote.

        [95, 50, 45] vs champion=30: Δ=65 >> 2σ ≈ 54.8 → promote_best.
        Old code would raise_alarm first and never reach promotion logic.
        """
        result = arbitrate_agent_outputs(
            outputs=[
                _agent(95, "excellent"),
                _agent(50, "mediocre"),
                _agent(45, "bad"),
            ],
            champion_score=30.0,
        )
        assert result["action"] == "promote_best"
        assert result["agreement"]["agreement_level"] == "conflict"

    def test_all_agents_equal_under_champion(self):
        """All below champion → rerun is the only valid action."""
        result = arbitrate_agent_outputs(
            outputs=[_agent(55), _agent(54), _agent(53)],
            champion_score=80.0,
        )
        assert result["action"] == "request_rerun"


# ══════════════════════════════════════════════════════════════
# Noise Floor Calibration
# ══════════════════════════════════════════════════════════════


class TestCalibrateNoiseFloor:
    """Tests for calibrate_noise_floor — lazy sigma from duplicate runs."""

    def test_no_db_file(self):
        """Non-existent DB → no sigma."""
        result = calibrate_noise_floor("/tmp/nonexistent.jsonl", "test_metric")
        assert result["sigma"] is None
        assert result["n_samples"] == 0

    def test_single_group_qualifies(self):
        """Single duplicate group with >= min_samples."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            db_path = f.name
            for val in [100.0, 101.0, 99.0]:
                f.write(json.dumps({
                    "metric": "m1", "value": val,
                    "code_hash": "abc", "params_hash": "def",
                }) + "\n")

        try:
            result = calibrate_noise_floor(db_path, "m1", min_samples=3)
            assert result["sigma"] is not None
            assert result["sigma"] > 0
            assert result["n_samples"] >= 1
            assert not result["locked"]
        finally:
            os.unlink(db_path)

    def test_insufficient_samples(self):
        """Group has fewer than min_samples → no sigma."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            db_path = f.name
            for val in [100.0, 101.0]:
                f.write(json.dumps({
                    "metric": "m1", "value": val,
                    "code_hash": "abc", "params_hash": "def",
                }) + "\n")

        try:
            result = calibrate_noise_floor(db_path, "m1", min_samples=3)
            assert result["sigma"] is None
        finally:
            os.unlink(db_path)

    def test_multiple_groups_pooled(self):
        """Multiple duplicate groups → pooled variance."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            db_path = f.name
            for val in [100.0, 102.0, 101.0]:
                f.write(json.dumps({
                    "metric": "m1", "value": val,
                    "code_hash": "a", "params_hash": "b",
                }) + "\n")
            for val in [200.0, 203.0, 201.0]:
                f.write(json.dumps({
                    "metric": "m1", "value": val,
                    "code_hash": "c", "params_hash": "d",
                }) + "\n")

        try:
            result = calibrate_noise_floor(db_path, "m1", min_samples=3)
            assert result["sigma"] is not None
            assert result["n_samples"] == 2
        finally:
            os.unlink(db_path)

    def test_locked_threshold(self):
        """>= lock_count groups → locked."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            db_path = f.name
            for group_id in range(6):
                code = f"code_{group_id}"
                for val in [100.0, 101.0, 99.0]:
                    f.write(json.dumps({
                        "metric": "m1", "value": val,
                        "code_hash": code, "params_hash": "fixed",
                    }) + "\n")

        try:
            result = calibrate_noise_floor(db_path, "m1", min_samples=3, lock_count=5)
            assert result["locked"]
        finally:
            os.unlink(db_path)


class TestRecordNoiseEntry:
    """Tests for record_noise_entry — JSONL append."""

    def test_appends_to_file(self):
        """record_noise_entry creates JSONL entry."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            db_path = f.name

        try:
            record_noise_entry(value=99.5, metric="seg_total", seed=42,
                               code_hash="abc", params_hash="def", db_path=db_path)
            with open(db_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["metric"] == "seg_total"
            assert entry["value"] == 99.5
        finally:
            os.unlink(db_path)

    def test_append_multiple_entries(self):
        """Multiple entries are appended (not overwritten)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            db_path = f.name

        try:
            for i in range(5):
                record_noise_entry(value=100.0 + i, metric="m1", db_path=db_path)
            with open(db_path) as f:
                lines = f.readlines()
            assert len(lines) == 5
        finally:
            os.unlink(db_path)


# ══════════════════════════════════════════════════════════════
# Sigma Multiplier Calibration
# ══════════════════════════════════════════════════════════════


class TestCalibrateSigmaMultiplier:
    """Tests for calibrate_sigma_multiplier — empirical M estimation."""

    def test_returns_default_with_no_data(self):
        """No data → returns 2.0 (AutoScientists default)."""
        M = calibrate_sigma_multiplier(metric_values=None)
        assert M == 2.0

    def test_returns_default_with_few_values(self):
        """Fewer than 5 values → returns 2.0."""
        M = calibrate_sigma_multiplier(metric_values=[1.0, 2.0, 3.0])
        assert M == 2.0

    def test_higher_noise_leads_to_higher_M(self):
        """Noisier metric → larger M (more conservative)."""
        # Deliberately design: same mean (100), different sequential noise
        # Clean: tiny step changes (±0.5)
        values_clean_raw = [100.0, 100.3, 99.7, 100.5, 99.5, 100.2, 99.8, 100.4, 99.6, 100.1]
        # Noisy: large step changes (±5)
        values_noisy_raw = [100.0, 105.0, 95.0, 108.0, 92.0, 103.0, 97.0, 106.0, 94.0, 102.0]

        M_clean = calibrate_sigma_multiplier(
            metric_values=values_clean_raw, target_promote_rate=0.10,
        )
        M_noisy = calibrate_sigma_multiplier(
            metric_values=values_noisy_raw, target_promote_rate=0.10,
        )

        assert M_noisy >= M_clean

    def test_M_clamped_to_reasonable_range(self):
        """M is clamped between 0.5 and 50."""
        # Near-zero sigma would produce huge M
        values_tiny_noise = [100.0, 100.0, 100.0, 100.0001, 100.0]
        M = calibrate_sigma_multiplier(
            metric_values=values_tiny_noise,
            target_promote_rate=0.10,
        )
        assert 0.5 <= M <= 50.0

    def test_higher_promote_rate_lower_M(self):
        """Higher target_promote_rate → lower M (easier to promote)."""
        values = [100.0, 105.0, 95.0, 110.0, 90.0]

        M_strict = calibrate_sigma_multiplier(
            metric_values=values, target_promote_rate=0.05,
        )
        M_relaxed = calibrate_sigma_multiplier(
            metric_values=values, target_promote_rate=0.20,
        )
        assert M_strict >= M_relaxed
