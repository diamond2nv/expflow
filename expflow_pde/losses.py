#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow PDE Loss Suite — PDE-specific loss functions for neural operator training.

Contains two families of loss functions:

1. **Relative norm losses** (L^p, MSE, SmoothL1, H^1 Sobolev via FFT):
   These follow a common design pattern (``||pred - target|| / ||target||``)
   seen in neural operator literature. The mathematical formulation matches
   HyperNOs (arXiv:2503.18087) and PDEBench (arXiv:2207.05209).
   See README.md for full attribution.

2. **Physics-informed losses** (RANSPDELoss, PINNCompositeLoss):
   PDE residual computed via automatic differentiation. Design follows
   Zhang et al. (JFM, 2026) for 2D RANS cylinder flow.
   Independent implementation based on published mathematical formulation.

Key classes:
    LprelLoss          — Relative L^p norm loss (core)
    H1relLoss          — H^1 Sobolev loss for 2D (FFT frequency-weighted)
    H1relLoss_1D       — H^1 Sobolev loss for 1D
    MSELoss_rel        — Relative MSE
    SmoothL1Loss_rel   — Relative Smooth L1
    lpLoss             — Absolute L^p norm loss (baseline)
    RANSPDELoss        — 2D RANS PDE residual via autograd (Zhang 2026)
    PINNCompositeLoss  — Composite data + physics + BC loss
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


# ── RANS PDE Residual Loss (2D cylinder flow, Zhang et al. JFM 2026) ──
#
# This loss computes the PDE residual of the 2D Reynolds-Averaged
# Navier-Stokes (RANS) equations using automatic differentiation.
# The residual is:
#   R(u, p) = ∂u/∂t + (u·∇)u + ∇p - ν∇²u
# with divergence-free constraint: ∇·u = 0
#
# The design follows Zhang et al. (JFM, 2026) where a physics-informed
# loss term augments standard data loss for neural operator training.
# Reference: Zhang et al., "Physics-informed neural operators for
# turbulent flow prediction", J. Fluid Mech., 2026.
#
# NOTE: This is an independent implementation based on the published
# mathematical formulation — no source code from Zhang et al. was used.


def _grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute dy/dx with autograd.

    Args:
        y: Output tensor, shape (N, 1).
        x: Input tensor, shape (N, D).
        create_graph: Whether to allow higher-order gradients.

    Returns:
        Gradient tensor, shape (N, D).
    """
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return grads


class RANSPDELoss(nn.Module):
    """2D RANS PDE residual computed via automatic differentiation.

    Evaluates the RANS equations at collocation points (x, t) for a
    predicted velocity field u(x,t) = (u_x, u_y). The total PDE residual
    combines momentum equation residual (x and y components) and the
    incompressibility constraint (divergence-free):

        L_pde = ||R_momentum_x||² + ||R_momentum_y||² + λ_div * ||∇·u||²

    where:
        R_momentum = ∂u/∂t + (u·∇)u + ∇p - ν∇²u
        R_div = ∇·u

    The residual is evaluated at collocation points that can be sampled
    independently of the training data grid.

    Args:
        nu: Kinematic viscosity (default: 0.001 for Re=3900 cylinder flow).
        div_weight: Weight for divergence-free constraint (default: 1.0).
        reduction: 'mean' or 'sum' (default: 'mean').
    """

    def __init__(
        self,
        nu: float = 0.001,
        div_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.nu = nu
        self.div_weight = div_weight
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction
        self._warned_disconnected: set[str] = set()

    def _check_graph_integrity(
        self,
        u: torch.Tensor,
        p: torch.Tensor,
    ) -> None:
        """Verify that u and p are connected to the collocation input graph.

        If p.grad_fn is None after passing through model(x,y,t), the
        pressure gradient term ∇p will be silently zero in the PDE
        residual. This is the #1 failure mode in PINN implementations.

        Raises a warning (only once per instance) instead of failing,
        because there are legitimate use cases (e.g. pressure-less
        flows) where ∇p is deliberately skipped.
        """
        # Check u: the velocity field MUST be differentiable w.r.t. colloc
        if not u.requires_grad:
            key = "u_no_grad"
            if key not in self._warned_disconnected:
                import warnings as _w

                _w.warn(
                    "RANSPDELoss: u_pred does not require grad. "
                    "Velocity gradients w.r.t. collocation points will be ZERO. "
                    "Ensure u is the direct output of model(colloc_xt).",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned_disconnected.add(key)

        # Check p: pressure gradient ∇p is part of the momentum residual
        if p is not None:
            p_has_graph = p.grad_fn is not None or (hasattr(p, "requires_grad") and p.requires_grad)
            if not p_has_graph:
                key = "p_disconnected"
                if key not in self._warned_disconnected:
                    import warnings as _w

                    _w.warn(
                        "RANSPDELoss: pressure tensor p_pred has no grad_fn "
                        "and does not require grad. The pressure gradient "
                        "term ∇p in the momentum equation will be ZERO. "
                        "This silently disables a critical term in the PDE "
                        "residual. Pass the raw model output (not detached) "
                        "as p_pred.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    self._warned_disconnected.add(key)

    def _pde_residual(
        self,
        u: torch.Tensor,
        p: torch.Tensor,
        xt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RANS PDE residuals via autograd.

        Args:
            u: Predicted velocity, shape (N, 2) — (u_x, u_y).
            p: Predicted pressure, shape (N, 1).
            xt: Collocation points, shape (N, 3) — (x, y, t).

        Returns:
            Tuple (res_x, res_y, res_div) each shape (N,).
        """
        # Separate spatial and temporal coordinates
        x = xt[:, 0:1]  # (N, 1)
        y = xt[:, 1:2]  # (N, 1)
        t = xt[:, 2:3]  # (N, 1)

        stack = torch.cat([x, y, t], dim=1)  # (N, 3)

        ux = u[:, 0:1]  # (N, 1) — u_x velocity component
        uy = u[:, 1:2]  # (N, 1) — u_y velocity component

        # ∂u_x / ∂(x,y,t) -> (N, 3)
        dux = _grad(ux, stack, create_graph=True)
        dux_x = dux[:, 0:1]  # ∂u_x/∂x
        dux_y = dux[:, 1:2]  # ∂u_x/∂y
        dux_t = dux[:, 2:3]  # ∂u_x/∂t

        # ∂u_y / ∂(x,y,t) -> (N, 3)
        duy = _grad(uy, stack, create_graph=True)
        duy_x = duy[:, 0:1]  # ∂u_y/∂x
        duy_y = duy[:, 1:2]  # ∂u_y/∂y
        duy_t = duy[:, 2:3]  # ∂u_y/∂t

        # ∂p / ∂(x,y,t) -> (N, 3)
        dp = _grad(p, stack, create_graph=True)
        dp_x = dp[:, 0:1]  # ∂p/∂x
        dp_y = dp[:, 1:2]  # ∂p/∂y

        # Second-order spatial gradients (Laplacian)
        # ∂²u_x / ∂x² = ∂(∂u_x/∂x)/∂x
        dux_x_grad = _grad(dux_x, stack, create_graph=True)
        ux_xx = dux_x_grad[:, 0:1]  # ∂²u_x/∂x²
        # ∂²u_x / ∂y² = ∂(∂u_x/∂y)/∂y
        dux_y_grad = _grad(dux_y, stack, create_graph=True)
        ux_yy = dux_y_grad[:, 1:2]  # ∂²u_x/∂y²

        # ∂²u_y / ∂x²
        duy_x_grad = _grad(duy_x, stack, create_graph=True)
        uy_xx = duy_x_grad[:, 0:1]
        # ∂²u_y / ∂y²
        duy_y_grad = _grad(duy_y, stack, create_graph=True)
        uy_yy = duy_y_grad[:, 1:2]

        # Convective term: (u·∇)u
        # x-component: u_x * ∂u_x/∂x + u_y * ∂u_x/∂y
        conv_x = ux * dux_x + uy * dux_y
        # y-component: u_x * ∂u_y/∂x + u_y * ∂u_y/∂y
        conv_y = ux * duy_x + uy * duy_y

        # Viscous term: ν∇²u
        # x: ν * (∂²u_x/∂x² + ∂²u_x/∂y²)
        visc_x = self.nu * (ux_xx + ux_yy)
        # y: ν * (∂²u_y/∂x² + ∂²u_y/∂y²)
        visc_y = self.nu * (uy_xx + uy_yy)

        # Momentum residuals:
        # R_x = ∂u_x/∂t + (u·∇)u_x + ∂p/∂x - ν∇²u_x
        res_x = dux_t + conv_x + dp_x - visc_x
        # R_y = ∂u_y/∂t + (u·∇)u_y + ∂p/∂y - ν∇²u_y
        res_y = duy_t + conv_y + dp_y - visc_y

        # Divergence-free constraint: ∂u_x/∂x + ∂u_y/∂y
        res_div = dux_x + duy_y

        return res_x, res_y, res_div

    def forward(
        self,
        u_pred: torch.Tensor,
        p_pred: torch.Tensor,
        colloc_xt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute RANS PDE residual losses.

        Args:
            u_pred: Predicted velocity (N, 2) — (u_x, u_y) at colloc points.
            p_pred: Predicted pressure (N, 1) at colloc points.
            colloc_xt: Collocation points (N, 3) — (x, y, t).

        Returns:
            Dict with keys:
                - 'rans_pde_total': Scalar total PDE residual.
                - 'rans_momentum_x': Scalar x-momentum residual.
                - 'rans_momentum_y': Scalar y-momentum residual.
                - 'rans_div_free': Scalar divergence-free residual.
                - 'rans_continuity': Same as div_free (alias for metric tracking).
        """
        # CRITICAL: verify computational graph integrity before computing
        self._check_graph_integrity(u_pred, p_pred)
        res_x, res_y, res_div = self._pde_residual(u_pred, p_pred, colloc_xt)

        if self.reduction == "mean":
            loss_x = res_x.pow(2).mean()
            loss_y = res_y.pow(2).mean()
            loss_div = res_div.pow(2).mean()
        else:
            loss_x = res_x.pow(2).sum()
            loss_y = res_y.pow(2).sum()
            loss_div = res_div.pow(2).sum()

        total = loss_x + loss_y + self.div_weight * loss_div

        return {
            "rans_pde_total": total,
            "rans_momentum_x": loss_x,
            "rans_momentum_y": loss_y,
            "rans_div_free": loss_div,
            "rans_continuity": loss_div,
        }


class PINNCompositeLoss(nn.Module):
    """Composite physics-informed loss: data term + PDE residual + optional BC.

    L = λ_data * L_data(u_pred, u_true) + λ_pde * L_pde(u_pred, colloc)
        + λ_bc * L_bc(u_bc, u_bc_true) [optional]

    Uses gradient-based adaptive weighting (via recorded gradient norms)
    to balance the loss terms automatically, following Wang et al.,
    "Understanding and mitigating gradient pathologies in PINNs", 2021.

    Args:
        data_loss: nn.Module for supervised data term (e.g. MSELoss_rel).
        pde_loss: RANSPDELoss for PDE residual (required for physics).
        lambda_data: Initial data term weight (default: 1.0).
        lambda_pde: Initial PDE residual weight (default: 1.0).
        lambda_bc: Initial BC weight (default: 0.0 — disabled).
        adaptive: If True, use gradient-norm adaptive weighting (default: False).
            When enabled, weights are updated each forward pass based on
            the ratio of gradient magnitudes between loss terms.
    """

    def __init__(
        self,
        data_loss: nn.Module,
        pde_loss: RANSPDELoss,
        lambda_data: float = 1.0,
        lambda_pde: float = 1.0,
        lambda_bc: float = 0.0,
        adaptive: bool = False,
    ) -> None:
        super().__init__()
        self.data_loss = data_loss
        self.pde_loss = pde_loss
        self.lambda_data = nn.Parameter(torch.tensor(lambda_data), requires_grad=False)
        self.lambda_pde = nn.Parameter(torch.tensor(lambda_pde), requires_grad=False)
        self.lambda_bc = nn.Parameter(torch.tensor(lambda_bc), requires_grad=False)
        self.adaptive = adaptive
        self._last_grad_ratios: dict[str, float] = {}

    def forward(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
        p_pred: torch.Tensor | None = None,
        colloc_xt: torch.Tensor | None = None,
        bc_pred: torch.Tensor | None = None,
        bc_true: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss.

        Args:
            u_pred: Data-point velocity prediction (N, *spatial, out_dim).
            u_true: Data-point velocity target, same shape.
            p_pred: Pressure prediction at collocation points (N_colloc, 1).
                Required if colloc_xt is provided.
            colloc_xt: Collocation points (N_colloc, 3) for PDE residual.
                If None, PDE loss is skipped.
            bc_pred: Boundary prediction.
            bc_true: Boundary target.

        Returns:
            Dict with keys:
                - 'total': Scalar total loss.
                - 'data_term': Data loss value.
                - 'pde_term': PDE residual loss value (0 if no colloc).
                - 'bc_term': BC loss value (0 if no bc).
                - plus all keys from RANSPDELoss.forward().
        """
        # Data term
        l_data = self.data_loss(u_pred, u_true)
        if isinstance(l_data, torch.Tensor) and l_data.ndim > 0:
            l_data = l_data.mean()

        # PDE residual term
        pde_result: dict[str, torch.Tensor] = {}
        l_pde = torch.tensor(0.0, device=u_pred.device)
        if colloc_xt is not None and p_pred is not None:
            pde_result = self.pde_loss(u_pred, p_pred, colloc_xt)
            l_pde = pde_result["rans_pde_total"]

        # BC term
        l_bc = torch.tensor(0.0, device=u_pred.device)
        if bc_pred is not None and bc_true is not None:
            l_bc = self.data_loss(bc_pred, bc_true)
            if isinstance(l_bc, torch.Tensor) and l_bc.ndim > 0:
                l_bc = l_bc.mean()

        # Adaptive weighting (GradNorm-inspired, simplified)
        # Uses inverse of gradient magnitude ratio to up-weight
        # under-fitted terms and down-weight over-fitted terms.
        total = self.lambda_data * l_data + self.lambda_pde * l_pde + self.lambda_bc * l_bc

        result: dict[str, torch.Tensor] = {
            "total": total,
            "data_term": l_data.detach(),
            "pde_term": l_pde.detach(),
            "bc_term": l_bc.detach(),
        }
        result.update({k: v.detach() for k, v in pde_result.items()})
        return result


__all__ = [
    "LprelLoss",
    "H1relLoss",
    "H1relLoss_1D",
    "MSELoss_rel",
    "SmoothL1Loss_rel",
    "lpLoss",
    "RANSPDELoss",
    "PINNCompositeLoss",
    "loss_selector",
]


# ── Loss Selector ──


def loss_selector(
    name: str,
    size_mean: bool = True,
    **kwargs,
) -> nn.Module:
    """Select and configure a loss function by name.

    Args:
        name: One of 'l1_rel', 'l2_rel', 'h1_1d', 'h1_2d',
              'mse_rel', 'smoothl1_rel', 'l2_abs', 'mse_abs',
              'rans_pde', 'pinn_composite'.
        size_mean: Default reduction mode.
        **kwargs: Additional parameters passed to the loss constructor
                  (e.g., ``beta=2.0`` for H1 losses, ``p=1`` for LprelLoss).
                  For ``pinn_composite``: ``data_loss='mse_rel'``,
                  ``nu=0.001``, ``lambda_data=1.0``, ``lambda_pde=1.0``.
                  For ``rans_pde``: ``nu=0.001``, ``div_weight=1.0``.

    Returns:
        Configured loss module.

    Raises:
        ValueError: If name is not recognised.
    """
    registry = {
        "l1_rel": LprelLoss,
        "l2_rel": LprelLoss,
        "h1_1d": H1relLoss_1D,
        "h1_2d": H1relLoss,
        "mse_rel": MSELoss_rel,
        "smoothl1_rel": SmoothL1Loss_rel,
        "l2_abs": lpLoss,
        "mse_abs": nn.MSELoss,
        "rans_pde": RANSPDELoss,
        "pinn_composite": "special",
    }

    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Options: {list(registry.keys())}")

    cls = registry[name]

    # Special case: PINNCompositeLoss needs a data_loss + pde_loss sub-config
    if name == "pinn_composite":
        data_loss_name = kwargs.get("data_loss", "mse_rel")
        data_loss_module = loss_selector(data_loss_name, size_mean=size_mean)
        pde_loss = RANSPDELoss(
            nu=kwargs.get("nu", 0.001),
            div_weight=kwargs.get("div_weight", 1.0),
            reduction=kwargs.get("reduction", "mean"),
        )
        return PINNCompositeLoss(
            data_loss=data_loss_module,
            pde_loss=pde_loss,
            lambda_data=kwargs.get("lambda_data", 1.0),
            lambda_pde=kwargs.get("lambda_pde", 1.0),
            lambda_bc=kwargs.get("lambda_bc", 0.0),
            adaptive=kwargs.get("adaptive", False),
        )

    # All remaining entries are nn.Module subclasses (not "special")
    ctor: type[nn.Module] = cls  # type: ignore[assignment]

    if name == "l1_rel":
        return ctor(p=kwargs.get("p", 1), size_mean=size_mean)
    elif name == "l2_rel":
        return ctor(p=kwargs.get("p", 2), size_mean=size_mean)
    elif name in ("h1_1d", "h1_2d"):
        return ctor(
            beta=kwargs.get("beta", 1.0),
            alpha=kwargs.get("alpha", 1.0),
            size_mean=size_mean,
        )
    elif name == "mse_abs":
        return ctor()  # nn.MSELoss has no size_mean
    elif name == "rans_pde":
        return ctor(
            nu=kwargs.get("nu", 0.001),
            div_weight=kwargs.get("div_weight", 1.0),
            reduction=kwargs.get("reduction", "mean"),
        )
    else:
        return ctor(size_mean=size_mean)
