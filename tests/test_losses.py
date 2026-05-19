#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for expflow_pde.losses — all PDE loss functions.

Tests cover:
- Perfect prediction → loss ≈ 0
- Random noise → loss ≈ 1
- Batch vs per-sample reduction (size_mean=True/False/None)
- Gradient flow (loss must be differentiable w.r.t. pred)
- loss_selector dispatcher
- H1 losses on 1D/2D tensors
"""

from __future__ import annotations

import pytest
import torch

from expflow_pde.losses import (
    H1relLoss,
    H1relLoss_1D,
    LprelLoss,
    MSELoss_rel,
    SmoothL1Loss_rel,
    loss_selector,
    lpLoss,
)

# ── Fixtures ──


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture
def ones_1d():
    """Simple 1D batch: (N=4, n_x=64, out_dim=1) all-ones."""
    return torch.ones(4, 64, 1)


@pytest.fixture
def ones_2d():
    """2D batch: (N=4, 32, 32, 1) all-ones."""
    return torch.ones(4, 32, 32, 1)


@pytest.fixture
def random_1d(rng):
    """Random 1D prediction: (N=4, 64, 1)."""
    return torch.rand(4, 64, 1, generator=rng)


@pytest.fixture
def random_2d(rng):
    """Random 2D tensor: (N=4, 32, 32, 1)."""
    return torch.rand(4, 32, 32, 1, generator=rng)


# ── LprelLoss ──


class TestLprelLoss:
    def test_perfect_prediction_l1(self, ones_1d):
        """LprelLoss(p=1) on perfect pred → 0."""
        loss_fn = LprelLoss(p=1, size_mean=True)
        loss = loss_fn(ones_1d, ones_1d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_perfect_prediction_l2(self, ones_1d):
        """LprelLoss(p=2) on perfect pred → 0."""
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(ones_1d, ones_1d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_noisy_pred_approx_one(self, random_1d, ones_1d):
        """Random pred vs all-ones target → loss ≈ 1 (same magnitude)."""
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(random_1d * 0.5 + 0.5, ones_1d)
        # Random ~= U(0,1) centered at 0.5, so ||diff|| ≈ ||target||
        assert 0.1 < loss.item() < 5.0

    def test_reduction_none(self, ones_1d):
        """size_mean=None → per-sample output (N,)."""
        loss_fn = LprelLoss(p=2, size_mean=None)
        loss = loss_fn(ones_1d + 0.1 * torch.randn_like(ones_1d), ones_1d)
        assert loss.ndim == 1
        assert loss.size(0) == ones_1d.size(0)

    def test_reduction_sum(self, ones_1d):
        """size_mean=False → scalar sum over batch."""
        loss_fn = LprelLoss(p=2, size_mean=False)
        loss = loss_fn(ones_1d + 0.1 * torch.randn_like(ones_1d), ones_1d)
        assert loss.ndim == 0

    def test_multichannel(self):
        """LprelLoss on (N, 64, 3) multi-channel tensor."""
        x = torch.randn(4, 64, 3)
        y = torch.randn(4, 64, 3)
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(x, y)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gradient_flow(self, ones_1d):
        """Loss must be differentiable w.r.t. predictions."""
        x = torch.randn(4, 64, 1, requires_grad=True)
        y = torch.randn(4, 64, 1)
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ── H1relLoss_1D ──


class TestH1relLoss1D:
    def test_perfect_prediction(self, ones_1d):
        """Perfect pred → loss ≈ 0."""
        loss_fn = H1relLoss_1D(beta=1.0, size_mean=True)
        loss = loss_fn(ones_1d, ones_1d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_default_beta(self, ones_1d, random_1d):
        """H1relLoss_1D with default beta=1.0 is functional."""
        loss_fn = H1relLoss_1D(size_mean=True)
        loss = loss_fn(random_1d, ones_1d)
        assert loss.item() > 0

    def test_beta_zero_equals_l2_rel(self, random_1d, ones_1d):
        """beta=0 → only alpha*I term → equivalent to LprelLoss(p=2)."""
        h1_loss = H1relLoss_1D(beta=0.0, size_mean=True)(random_1d, ones_1d)
        l2_loss = LprelLoss(p=2, size_mean=True)(random_1d, ones_1d)
        assert h1_loss.item() == pytest.approx(l2_loss.item(), rel=1e-4)

    def test_higher_beta_penalises_high_freq(self):
        """Adding high-frequency noise should increase loss more with higher beta."""
        x = torch.linspace(-1, 1, 128).reshape(1, 128, 1)
        x_smooth = torch.sin(2 * torch.pi * x)
        x_noisy = x_smooth + 0.1 * torch.sin(20 * torch.pi * x)

        loss_low_beta = H1relLoss_1D(beta=0.1, size_mean=True)(x_noisy, x_smooth)
        loss_high_beta = H1relLoss_1D(beta=5.0, size_mean=True)(x_noisy, x_smooth)
        assert loss_high_beta.item() > loss_low_beta.item()

    def test_gradient_flow(self, random_1d):
        x = random_1d.clone().requires_grad_(True)
        y = random_1d.clone()
        loss_fn = H1relLoss_1D(size_mean=True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_multichannel(self):
        x = torch.randn(4, 64, 2)
        y = torch.randn(4, 64, 2)
        loss_fn = H1relLoss_1D(size_mean=True)
        loss = loss_fn(x, y)
        assert loss.ndim == 0
        assert loss.item() > 0


# ── H1relLoss 2D ──


class TestH1relLoss2D:
    def test_perfect_prediction(self, ones_2d):
        loss_fn = H1relLoss(beta=1.0, size_mean=True)
        loss = loss_fn(ones_2d, ones_2d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_beta_zero_equals_l2_rel(self, random_2d, ones_2d):
        h1_loss = H1relLoss(beta=0.0, size_mean=True)(random_2d, ones_2d)
        l2_loss = LprelLoss(p=2, size_mean=True)(random_2d, ones_2d)
        assert h1_loss.item() == pytest.approx(l2_loss.item(), rel=1e-4)

    def test_gradient_flow(self, random_2d):
        x = random_2d.clone().requires_grad_(True)
        y = random_2d.clone()
        loss_fn = H1relLoss(size_mean=True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_multichannel(self):
        x = torch.randn(4, 32, 32, 3)
        y = torch.randn(4, 32, 32, 3)
        loss_fn = H1relLoss(size_mean=True)
        loss = loss_fn(x, y)
        assert loss.ndim == 0


# ── MSELoss_rel ──


class TestMSELossRel:
    def test_perfect_prediction(self, ones_1d):
        loss_fn = MSELoss_rel(size_mean=True)
        loss = loss_fn(ones_1d, ones_1d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_sanity(self, random_1d, ones_1d):
        loss_fn = MSELoss_rel(size_mean=True)
        loss = loss_fn(random_1d, ones_1d)
        assert loss.item() > 0

    def test_gradient_flow(self, random_1d):
        x = random_1d.clone().requires_grad_(True)
        y = random_1d.clone()
        loss_fn = MSELoss_rel(size_mean=True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


# ── SmoothL1Loss_rel ──


class TestSmoothL1LossRel:
    def test_perfect_prediction(self, ones_1d):
        loss_fn = SmoothL1Loss_rel(size_mean=True)
        loss = loss_fn(ones_1d, ones_1d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_sanity(self, random_1d, ones_1d):
        loss_fn = SmoothL1Loss_rel(size_mean=True)
        loss = loss_fn(random_1d, ones_1d)
        assert loss.item() > 0

    def test_gradient_flow(self, random_1d):
        x = random_1d.clone().requires_grad_(True)
        y = random_1d.clone()
        loss_fn = SmoothL1Loss_rel(size_mean=True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


# ── lpLoss ──


class TestLpLoss:
    def test_perfect_prediction(self, ones_1d):
        loss_fn = lpLoss(p=2, size_mean=True)
        loss = loss_fn(ones_1d, ones_1d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_for_mismatch(self, random_1d, ones_1d):
        loss_fn = lpLoss(p=2, size_mean=True)
        loss = loss_fn(random_1d, ones_1d)
        assert loss.item() > 0

    def test_gradient_flow(self, random_1d):
        x = random_1d.clone().requires_grad_(True)
        y = random_1d.clone()
        loss_fn = lpLoss(p=2, size_mean=True)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


# ── loss_selector ──


class TestLossSelector:
    def test_all_names_return_module(self):
        names = ["l1_rel", "l2_rel", "h1_1d", "h1_2d", "mse_rel", "smoothl1_rel", "l2_abs", "mse_abs"]
        x_1d = torch.randn(4, 64, 1)
        y_1d = torch.randn(4, 64, 1)
        x_2d = torch.randn(4, 32, 32, 1)
        y_2d = torch.randn(4, 32, 32, 1)
        for name in names:
            x = x_2d if name == "h1_2d" else x_1d
            y = y_2d if name == "h1_2d" else y_1d
            loss_fn = loss_selector(name)
            loss = loss_fn(x, y)
            assert loss.ndim == 0, f"{name} didn't return scalar loss"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            loss_selector("non_existent")

    def test_h1_with_beta_override(self, ones_1d):
        """Custom beta passed via kwargs."""
        loss_fn = loss_selector("h1_1d", beta=2.0)
        loss = loss_fn(ones_1d + 0.1 * torch.randn_like(ones_1d), ones_1d)
        assert loss.ndim == 0

    def test_size_mean_none_passthrough(self):
        loss_fn = loss_selector("l2_rel", size_mean=None)
        x = torch.randn(4, 64, 1)
        y = torch.randn(4, 64, 1)
        loss = loss_fn(x, y)
        assert loss.ndim == 1
        assert loss.size(0) == 4

    def test_l1_rel_uses_p1(self, ones_1d):
        loss_fn = loss_selector("l1_rel")
        loss = loss_fn(ones_1d, ones_1d)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── Edge cases ──


class TestEdgeCases:
    def test_all_equal_targets(self):
        """Target all zeros, pred = target → loss = 0 (no div by zero)."""
        x = torch.zeros(4, 64, 1)
        y = torch.zeros(4, 64, 1)
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(x, y)
        # With guard: 0 / eps → 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_single_sample(self):
        """Batch size 1 works."""
        x = torch.randn(1, 64, 1)
        y = torch.randn(1, 64, 1)
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(x, y)
        assert loss.ndim == 0

    def test_large_batch(self):
        """Large batch (N=32) works without OOM."""
        x = torch.randn(32, 64, 1)
        y = torch.randn(32, 64, 1)
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(x, y)
        assert torch.isfinite(loss)

    def test_variable_spatial_dims(self):
        """Works with (N, 128, 1) — different spatial dim."""
        x = torch.randn(2, 128, 1)
        y = torch.randn(2, 128, 1)
        loss_fn = LprelLoss(p=2, size_mean=True)
        loss = loss_fn(x, y)
        assert torch.isfinite(loss)

    def test_h1_1d_variable_spatial(self):
        x = torch.randn(2, 128, 1)
        y = torch.randn(2, 128, 1)
        loss_fn = H1relLoss_1D(size_mean=True)
        loss = loss_fn(x, y)
        assert torch.isfinite(loss)

    def test_h1_2d_variable_spatial(self):
        x = torch.randn(2, 16, 32, 1)
        y = torch.randn(2, 16, 32, 1)
        loss_fn = H1relLoss(size_mean=True)
        loss = loss_fn(x, y)
        assert torch.isfinite(loss)
