#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow PDE Loss Suite — PDE-specific loss functions for neural operator training.

Transplanted from HyperNOs (hypernos/loss_fun.py) with dependency cleanup:
- Removed beartype/jaxtyped annotations
- Removed Chebyshev variants (ChebyshevLprelLoss, H1relLoss_cheb_mp)
- Added proper nn.Module subclassing, PEP8 type hints, and top-level __all__
- Removed multi-patch / multi-output variants (single-channel focused)
- All classes inherit from nn.Module for clearml/optuna compatibility

Key classes:
    LprelLoss      — Relative L^p norm loss (core)
    H1relLoss      — H^1 Sobolev loss for 2D (FFT frequency-weighted)
    H1relLoss_1D   — H^1 Sobolev loss for 1D
    MSELoss_rel    — Relative MSE
    SmoothL1Loss_rel — Relative Smooth L1
    lpLoss         — Absolute L^p norm loss (baseline)
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "LprelLoss",
    "H1relLoss",
    "H1relLoss_1D",
    "MSELoss_rel",
    "SmoothL1Loss_rel",
    "lpLoss",
    "loss_selector",
]

# ── Relative L^p Loss (Core) ──


class LprelLoss(nn.Module):
    """Relative L^p norm loss for N-D functions.

    ``loss = ||pred - target||_p / ||target||_p``

    Averaged over output channels. This is the workhorse PDE loss —
    ``LprelLoss(p=2)`` equals the relative MSE used in PDEBench scoring.

    Args:
        p: Norm order (1 for L1, 2 for L2).
        size_mean: If True, return scalar mean over batch.
                   If False, return sum over batch.
                   If None, return per-sample tensor (N,).
    """

    def __init__(self, p: int = 2, size_mean: bool = True):
        super().__init__()
        self.p = p
        self.size_mean = size_mean

    def _rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute relative norm for a single output channel.

        x, y: (N, *spatial, 1) — single-channel tensors.
        Returns: (N,) tensor of per-sample relative errors.
        """
        num_examples = x.size(0)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
            p=self.p,
            dim=1,
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), p=self.p, dim=1)
        # Guard against division by zero
        y_norms = torch.where(y_norms < 1e-7, torch.full_like(y_norms, 1e-7), y_norms)
        return diff_norms / y_norms

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute L^p relative loss.

        Args:
            x: Predicted tensor, shape (N, *spatial, out_dim).
            y: Target tensor, same shape.

        Returns:
            Scalar (size_mean=True), sum (size_mean=False),
            or per-sample tensor (size_mean=None).
        """
        out_dim = x.size(-1)
        acc = 0.0
        for i in range(out_dim):
            acc += self._rel(x[..., [i]], y[..., [i]])
        loss = acc / out_dim

        if self.size_mean is True:
            return loss.mean()
        elif self.size_mean is False:
            return loss.sum()
        return loss  # (N,)


# ── Absolute L^p Loss (Baseline) ──


class lpLoss(nn.Module):
    """Absolute L^p norm loss (no normalization by target norm).

    ``loss = ||pred - target||_p``

    Args:
        p: Norm order.
        size_mean: Reduction mode.
    """

    def __init__(self, p: int = 2, size_mean: bool = True):
        super().__init__()
        self.p = p
        self.size_mean = size_mean

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff_norms = torch.norm(
            (x - y).reshape(x.size(0), -1),
            p=self.p,
            dim=1,
        )
        if self.size_mean is True:
            return diff_norms.mean()
        elif self.size_mean is False:
            return diff_norms.sum()
        return diff_norms


# ── Relative MSE ──


class MSELoss_rel(nn.Module):
    """Relative Mean Squared Error.

    ``loss = MSE(pred, target) / MSE(0, target)``

    Args:
        size_mean: Reduction mode.
    """

    def __init__(self, size_mean: bool = True):
        super().__init__()
        self.size_mean = size_mean

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x_flat = x.reshape(N, -1)
        y_flat = y.reshape(N, -1)

        diff_mse = (x_flat - y_flat).pow(2).mean(dim=1)
        y_mse = y_flat.pow(2).mean(dim=1)
        y_mse = torch.where(y_mse < 1e-7, torch.full_like(y_mse, 1e-7), y_mse)
        loss = diff_mse / y_mse

        if self.size_mean is True:
            return loss.mean()
        elif self.size_mean is False:
            return loss.sum()
        return loss


# ── Relative Smooth L1 ──


class SmoothL1Loss_rel(nn.Module):
    """Relative Smooth L1 Loss.

    ``loss = SmoothL1(pred, target) / SmoothL1(0, target)``

    More robust to outliers than MSELoss_rel.

    Args:
        size_mean: Reduction mode.
    """

    def __init__(self, size_mean: bool = True):
        super().__init__()
        self.size_mean = size_mean

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x_flat = x.reshape(N, -1)
        y_flat = y.reshape(N, -1)

        smooth_l1 = nn.SmoothL1Loss(reduction="none")
        diff_sl1 = smooth_l1(x_flat, y_flat).mean(dim=1)
        y_sl1 = smooth_l1(torch.zeros_like(y_flat), y_flat).mean(dim=1)
        y_sl1 = torch.where(y_sl1 < 1e-7, torch.full_like(y_sl1, 1e-7), y_sl1)
        loss = diff_sl1 / y_sl1

        if self.size_mean is True:
            return loss.mean()
        elif self.size_mean is False:
            return loss.sum()
        return loss


# ── H^1 Relative Loss 1D (FFT Frequency-Weighted) ──


class H1relLoss_1D(nn.Module):
    """Relative H^1 (Sobolev) loss for 1D functions via FFT.

    ``loss = ||alpha*(x-y) + beta*grad(x-y)||_2 / ||alpha*y + beta*grad(y)||_2``

    Uses FFT: weight = alpha + beta * k^2, where k = |k_x|.
    This penalises high-frequency errors — ideal for PDEs with smooth solutions.

    Args:
        beta: Derivative weight (default: 1.0).
        size_mean: Reduction mode.
        alpha: Identity weight (default: 1.0).
    """

    def __init__(self, beta: float = 1.0, size_mean: bool = True, alpha: float = 1.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.size_mean = size_mean

    def _rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute relative norm for a single output channel.

        x, y: (N, n_x, 1) — single-channel complex tensors (already FFT'd).
        """
        num_examples = x.size(0)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
            p=2,
            dim=1,
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)
        y_norms = torch.where(y_norms < 1e-7, torch.full_like(y_norms, 1e-7), y_norms)
        return diff_norms / y_norms

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute H^1 relative loss for 1D.

        Args:
            x: Predicted tensor, shape (N, n_x, out_dim).
            y: Target tensor, same shape.

        Returns:
            Scalar or per-sample tensor per ``size_mean``.
        """
        n_x, out_dim = x.size(1), x.size(-1)

        # Build frequency grid
        k_x = torch.cat(
            [
                torch.arange(start=0, end=n_x // 2, step=1),
                torch.arange(start=-n_x // 2, end=0, step=1),
            ],
            dim=0,
        ).to(x.device)
        k_x = torch.abs(k_x).float().reshape(1, n_x, 1)  # (1, n_x, 1)

        acc = 0.0
        for i in range(out_dim):
            x_f = torch.fft.fftn(x[..., [i]], dim=[1])  # (N, n_x, 1)
            y_f = torch.fft.fftn(y[..., [i]], dim=[1])
            weight = self.alpha * 1.0 + self.beta * (k_x**2)  # (1, n_x, 1)
            acc += self._rel(x_f * torch.sqrt(weight), y_f * torch.sqrt(weight))

        loss = acc / out_dim

        if self.size_mean is True:
            return loss.mean()
        elif self.size_mean is False:
            return loss.sum()
        return loss


# ── H^1 Relative Loss 2D (FFT Frequency-Weighted) ──


class H1relLoss(nn.Module):
    """Relative H^1 (Sobolev) loss for 2D functions via FFT.

    ``loss = ||alpha*(x-y) + beta*grad(x-y)||_2 / ||alpha*y + beta*grad(y)||_2``

    Uses 2D FFT: weight = alpha + beta * (k_x^2 + k_y^2).

    Args:
        beta: Derivative weight (default: 1.0).
        size_mean: Reduction mode.
        alpha: Identity weight (default: 1.0).
    """

    def __init__(self, beta: float = 1.0, size_mean: bool = True, alpha: float = 1.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.size_mean = size_mean

    def _rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute relative norm for a single output channel.

        x, y: (N, n_x, n_y, 1) — complex tensors (already FFT'd).
        """
        num_examples = x.size(0)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
            p=2,
            dim=1,
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)
        y_norms = torch.where(y_norms < 1e-7, torch.full_like(y_norms, 1e-7), y_norms)
        return diff_norms / y_norms

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute H^1 relative loss for 2D.

        Args:
            x: Predicted tensor, shape (N, n_x, n_y, out_dim).
            y: Target tensor, same shape.

        Returns:
            Scalar or per-sample tensor per ``size_mean``.
        """
        n_x, n_y, out_dim = x.size(1), x.size(2), x.size(-1)

        # Build 2D frequency grid
        k_x = (
            torch.cat(
                [
                    torch.arange(start=0, end=n_x // 2, step=1),
                    torch.arange(start=-n_x // 2, end=0, step=1),
                ],
                dim=0,
            )
            .reshape(1, n_x, 1, 1)
            .to(x.device)
        )
        k_y = (
            torch.cat(
                [
                    torch.arange(start=0, end=n_y // 2, step=1),
                    torch.arange(start=-n_y // 2, end=0, step=1),
                ],
                dim=0,
            )
            .reshape(1, 1, n_y, 1)
            .to(x.device)
        )
        k_x = torch.abs(k_x).float()
        k_y = torch.abs(k_y).float()

        weight = self.alpha * 1.0 + self.beta * (k_x**2 + k_y**2)  # (1, n_x, n_y, 1)

        acc = 0.0
        for i in range(out_dim):
            x_f = torch.fft.fftn(x[..., [i]], dim=[1, 2])
            y_f = torch.fft.fftn(y[..., [i]], dim=[1, 2])
            acc += self._rel(x_f * torch.sqrt(weight), y_f * torch.sqrt(weight))

        loss = acc / out_dim

        if self.size_mean is True:
            return loss.mean()
        elif self.size_mean is False:
            return loss.sum()
        return loss


# ── Loss Selector ──


def loss_selector(
    name: str,
    size_mean: bool = True,
    **kwargs,
) -> nn.Module:
    """Select and configure a loss function by name.

    Args:
        name: One of 'l1_rel', 'l2_rel', 'h1_1d', 'h1_2d',
              'mse_rel', 'smoothl1_rel', 'l2_abs', 'mse_abs'.
        size_mean: Default reduction mode.
        **kwargs: Additional parameters passed to the loss constructor
                  (e.g., ``beta=2.0`` for H1 losses, ``p=1`` for LprelLoss).

    Returns:
        Configured loss module.

    Raises:
        ValueError: If name is not recognised.
    """
    registry: dict[str, type] = {
        "l1_rel": LprelLoss,
        "l2_rel": LprelLoss,
        "h1_1d": H1relLoss_1D,
        "h1_2d": H1relLoss,
        "mse_rel": MSELoss_rel,
        "smoothl1_rel": SmoothL1Loss_rel,
        "l2_abs": lpLoss,
        "mse_abs": nn.MSELoss,
    }

    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Options: {list(registry.keys())}")

    cls = registry[name]

    if name == "l1_rel":
        return cls(p=kwargs.get("p", 1), size_mean=size_mean)
    elif name == "l2_rel":
        return cls(p=kwargs.get("p", 2), size_mean=size_mean)
    elif name in ("h1_1d", "h1_2d"):
        return cls(
            beta=kwargs.get("beta", 1.0),
            alpha=kwargs.get("alpha", 1.0),
            size_mean=size_mean,
        )
    elif name == "mse_abs":
        return cls()  # nn.MSELoss has no size_mean
    else:
        return cls(size_mean=size_mean)
