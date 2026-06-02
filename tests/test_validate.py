#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.validate — noise-aware champion validation.

Tests cover:
- Confident improvement (delta > M*sigma) → promote
- Within noise band (0 < delta <= M*sigma) → confirm
- No improvement (delta <= 0) → reject
- Lazy sigma calibration from JSONL data
- Fallback sigma estimation
"""

from __future__ import annotations

import json
import os
import tempfile

from expflow_pde.validate import (
    calibrate_noise_floor,
    noise_aware_validate,
    record_noise_entry,
)

# ── noise_aware_validate ──


class TestNoiseAwareValidate:
    def test_confident_improvement(self):
        """Delta > M*sigma → promote."""
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
        """0 < delta <= M*sigma → confirm."""
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
        """delta <= 0 → reject."""
        result = noise_aware_validate(
            candidate_value=89.0,
            champion_value=90.0,
            noise_floor=1.0,
            sigma_multiplier=2.0,
        )
        assert result["action"] == "reject"
        assert result["delta"] == -1.0

    def test_exact_zero_delta(self):
        """Exact equality → reject."""
        result = noise_aware_validate(
            candidate_value=90.0,
            champion_value=90.0,
            noise_floor=1.0,
        )
        assert result["action"] == "reject"

    def test_lazy_calibration_no_db(self):
        """Noise_floor=None and no DB file → uses fallback sigma."""
        result = noise_aware_validate(
            candidate_value=100.5,
            champion_value=100.0,
        )
        assert result["action"] in ("promote", "confirm")
        assert result["noise_floor"] > 0

    def test_lazy_calibration_with_db(self):
        """Noise_floor=None but DB has data → calibrates from DB."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            db_path = f.name
            for val in [100.0, 101.0, 99.0]:
                f.write(
                    json.dumps(
                        {
                            "metric": "test_metric",
                            "value": val,
                            "code_hash": "abc123",
                            "params_hash": "def456",
                        },
                    )
                    + "\n",
                )

        try:
            result = noise_aware_validate(
                candidate_value=105.0,
                champion_value=100.0,
                noise_db_path=db_path,
                metric_name="test_metric",
            )
            assert result["noise_floor"] > 0
            # With sigma ~1.0, delta=5.0 > 2*sigma → promote
            assert result["action"] == "promote"
        finally:
            os.unlink(db_path)

    def test_sigma_multiplier_effect(self):
        """Higher sigma_multiplier → more conservative (more confirms)."""
        base = noise_aware_validate(
            candidate_value=91.0,
            champion_value=90.0,
            noise_floor=1.0,
            sigma_multiplier=0.5,
        )
        conservative = noise_aware_validate(
            candidate_value=91.0,
            champion_value=90.0,
            noise_floor=1.0,
            sigma_multiplier=5.0,
        )
        # With sigma=1, M=0.5: delta=1 > 0.5*1 → promote
        assert base["action"] == "promote"
        # With sigma=1, M=5: delta=1 <= 5*1 → confirm
        assert conservative["action"] == "confirm"

    def test_worse_than_champion(self):
        """Delta < 0 → always reject regardless of noise."""
        result = noise_aware_validate(
            candidate_value=80.0,
            champion_value=100.0,
            noise_floor=10.0,
            sigma_multiplier=2.0,
        )
        assert result["action"] == "reject"
        assert result["delta"] == -20.0

    def test_sigma_zero_uses_fallback(self):
        """Zero sigma → uses fallback estimation."""
        result = noise_aware_validate(
            candidate_value=91.0,
            champion_value=90.0,
            noise_floor=0.0,
        )
        assert result["action"] in ("promote", "confirm")
        assert result["noise_floor"] > 0


# ── calibrate_noise_floor ──


class TestCalibrateNoiseFloor:
    def test_no_db_file(self):
        """Non-existent DB → no sigma."""
        result = calibrate_noise_floor(
            "/tmp/nonexistent.jsonl", "test_metric",
        )
        assert result["sigma"] is None
        assert result["n_samples"] == 0

    def test_single_group_qualifies(self):
        """Single duplicate group with >= min_samples."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            db_path = f.name
            for val in [100.0, 101.0, 99.0]:
                f.write(
                    json.dumps(
                        {
                            "metric": "m1",
                            "value": val,
                            "code_hash": "abc",
                            "params_hash": "def",
                        },
                    )
                    + "\n",
                )

        try:
            result = calibrate_noise_floor(
                db_path, "m1", min_samples=3,
            )
            assert result["sigma"] is not None
            assert result["sigma"] > 0
            assert result["n_samples"] >= 1
            assert not result["locked"]
        finally:
            os.unlink(db_path)

    def test_insufficient_samples(self):
        """Group has fewer than min_samples → no sigma."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            db_path = f.name
            for val in [100.0, 101.0]:
                f.write(
                    json.dumps(
                        {
                            "metric": "m1",
                            "value": val,
                            "code_hash": "abc",
                            "params_hash": "def",
                        },
                    )
                    + "\n",
                )

        try:
            result = calibrate_noise_floor(
                db_path, "m1", min_samples=3,
            )
            assert result["sigma"] is None
        finally:
            os.unlink(db_path)

    def test_multiple_groups_pooled(self):
        """Multiple duplicate groups → pooled variance."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            db_path = f.name
            # Group 1: code=a, params=b
            for val in [100.0, 102.0, 101.0]:
                f.write(
                    json.dumps(
                        {
                            "metric": "m1",
                            "value": val,
                            "code_hash": "a",
                            "params_hash": "b",
                        },
                    )
                    + "\n",
                )
            # Group 2: code=c, params=d
            for val in [200.0, 203.0, 201.0]:
                f.write(
                    json.dumps(
                        {
                            "metric": "m1",
                            "value": val,
                            "code_hash": "c",
                            "params_hash": "d",
                        },
                    )
                    + "\n",
                )

        try:
            result = calibrate_noise_floor(
                db_path, "m1", min_samples=3,
            )
            assert result["sigma"] is not None
            assert result["n_samples"] == 2
        finally:
            os.unlink(db_path)

    def test_locked_threshold(self):
        """>= lock_count groups → locked."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            db_path = f.name
            for group_id in range(6):
                code = f"code_{group_id}"
                for val in [100.0, 101.0, 99.0]:
                    f.write(
                        json.dumps(
                            {
                                "metric": "m1",
                                "value": val,
                                "code_hash": code,
                                "params_hash": "fixed",
                            },
                        )
                        + "\n",
                    )

        try:
            result = calibrate_noise_floor(
                db_path, "m1", min_samples=3, lock_count=5,
            )
            assert result["locked"]
        finally:
            os.unlink(db_path)


# ── record_noise_entry ──


class TestRecordNoiseEntry:
    def test_appends_to_file(self):
        """record_noise_entry creates JSONL entry."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            db_path = f.name

        try:
            record_noise_entry(
                value=99.5,
                metric="seg_total",
                seed=42,
                code_hash="abc",
                params_hash="def",
                db_path=db_path,
            )
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            db_path = f.name

        try:
            for i in range(5):
                record_noise_entry(
                    value=100.0 + i,
                    metric="m1",
                    db_path=db_path,
                )
            with open(db_path) as f:
                lines = f.readlines()
            assert len(lines) == 5
        finally:
            os.unlink(db_path)
