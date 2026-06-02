#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expflow equations — PDE equation registry for PDEBench/Agentic4Sci.

Provides a structured registry of PDE equations supported by the competition
and evaluation framework. Each equation has metadata including LaTeX expression,
dimensions, viscosity parameters, and associated metrics.

Usage:
    from expflow_pde.equations import (
        get_equations,
        get_equation,
        list_equations_for_task,
        get_equation_metrics,
    )

    # List all equations
    eqs = get_equations()
    for name, info in eqs.items():
        print(f"{name}: {info['latex_short']}")

    # Get specific equation
    burgers = get_equation("burgers")
    print(burgers["latex"])

    # What metrics are relevant for Task 1?
    metrics = get_equation_metrics("burgers", task="task1")
"""

from typing import Any

# ── Equation descriptor schema ──
#
# Each entry:
#   name: unique key, lowercase
#   full_name: human-readable
#   latex: LaTeX expression of the PDE
#   latex_short: compact LaTeX for table display
#   dim: spatial dimensions (1, 2, or 3)
#   time_dependent: bool
#   competition_task: "task1", "task2", "task3", or None
#   viscosity_params: relevant nu/viscosity parameters
#   nu_values: list of nu values available in data
#   description: plain-text explanation
#   metrics: metrics relevant to this equation
#   references: list of paper/URL references
#   solver: reference solver method if known

EQUATIONS: dict[str, dict[str, Any]] = {
    "burgers": {
        "full_name": "Burgers' Equation",
        "latex": (
            r"\partial_t u(x,t) + u \cdot \partial_x u = \nu \partial_{xx} u,"
            r"\quad x \in [0,1],\; t \in [0,2]"
        ),
        "latex_short": r"\partial_t u + u\partial_x u = \nu\partial_{xx}u",
        "dim": 1,
        "time_dependent": True,
        "competition_task": "task1",
        "viscosity_params": "nu",
        "nu_values": [0.001, 0.01, 0.1],
        "competition_nu": 0.001,
        "description": (
            "1D Burgers equation — the primary competition PDE for Task 1. "
            "Models shock formation and viscous dissipation in 1D fluid flow. "
            "Only nu=0.001 is allowed for competition submission."
        ),
        "metrics": [
            "seg_total",
            "seg1",
            "seg2",
            "seg3",
            "val_mse",
            "val_relmse",
            "pde_mean",
            "pde_seg1",
            "pde_seg2",
            "pde_seg3",
            "train_time_min",
            "arch_params",
            "epochs",
        ],
        "references": [
            "arXiv:2207.05209 (PDEBench)",
            "https://github.com/pdebench/PDEBench",
        ],
        "solver": "Burgers solver (Cole-Hopf transformation + FFT)",
        "data_samples": 10000,
        "ic_distribution": "Gaussian random field",
        "competition_info": {
            "task": "task1",
            "max_score": 150,
            "prediction_score_max": 75,
            "train_time_score_max": 35,
            "inference_score_max": 40,
            "observation_steps": 10,
            "prediction_steps": 190,
            "total_steps": 200,
            "grid_points": 256,
            "segment_1": {"steps": "0-47", "weight": 0.25},
            "segment_2": {"steps": "47-95", "weight": 0.25},
            "segment_3": {"steps": "95-190", "weight": 0.50},
            "train_time_60min": 35,
            "train_time_120min": 25,
            "train_time_300min": 20,
            "train_time_500min": 10,
            "train_time_over_500": 0,
            "inference_time_limit_min": 2,
            "inference_time_bonus_at_zero_min": 40,
            "total_time_limit_hours": 12,
            "gpu_coefficients": {
                "A100_80GB": 1.0,
                "H100_80GB": 2.0,
                "H800_80GB": 1.8,
                "RTX5090": 0.75,
                "RTX4090": 0.7,
                "RTX3090": 0.45,
                "RTX3080": 0.3,
            },
        },
    },
    "burgers_multi_nu": {
        "full_name": "Burgers' Equation (Multi-nu)",
        "latex": (
            r"\partial_t u(x,t) + u \cdot \partial_x u = \nu \partial_{xx} u,"
            r"\quad \nu \in [0.001, 0.1]"
        ),
        "latex_short": r"\partial_t u+u\partial_x u=\nu\partial_{xx}u,\;\nu\in[0.001,0.1]",
        "dim": 1,
        "time_dependent": True,
        "competition_task": "task2",
        "viscosity_params": "nu (multiple values)",
        "nu_values": [0.001, 0.01, 0.1],
        "description": (
            "Multi-nu Burgers equation — Task 2 of the AI4S CNS challenge. "
            "Requires the model to generalize across multiple viscosity coefficients. "
            "Training data provides nu as a conditional input; test data does not."
        ),
        "metrics": [
            "seg_total",
            "seg1",
            "seg2",
            "seg3",
            "val_mse",
            "val_relmse",
            "train_time_min",
            "arch_params",
            "epochs",
        ],
        "references": [
            "arXiv:2207.05209 (PDEBench)",
            "https://competition.ai4s.com.cn/race/7/introduction",
        ],
        "solver": "Burgers solver (Cole-Hopf transformation + FFT)",
        "data_samples": 10000,
        "ic_distribution": "Gaussian random field",
        "competition_info": {
            "task": "task2",
            "max_score": 150,
            "prediction_score_multiplier": 1.5,
            "observation_steps": 10,
            "prediction_steps": 190,
            "total_steps": 200,
            "grid_points": 256,
            "segment_1": {"steps": "0-47", "weight": 0.25},
            "segment_2": {"steps": "47-95", "weight": 0.25},
            "segment_3": {"steps": "95-190", "weight": 0.50},
            "inference_time_limit_min": 2,
            "total_time_limit_hours": 12,
            "train_from_scratch": True,
            "no_pretrained_weights": True,
            "nu_provided_in_train": True,
            "nu_provided_in_test": False,
        },
    },
    "advection": {
        "full_name": "Advection Equation",
        "latex": (
            r"\partial_t u(x,t) + \beta \cdot \partial_x u = 0,"
            r"\quad x \in [0,1],\; t \in [0,2]"
        ),
        "latex_short": r"\partial_t u + \beta\partial_x u = 0",
        "dim": 1,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "beta",
        "nu_values": [0.1, 0.5, 1.0, 2.0, 5.0],
        "description": (
            "1D linear advection equation. Models scalar transport "
            "at constant velocity beta. Used as a baseline PDE for "
            "benchmarking neural operators."
        ),
        "metrics": ["seg_total", "val_mse", "val_relmse", "train_time_min"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Upwind / WENO scheme",
        "data_samples": 10000,
    },
    "diffusion_reaction_1d": {
        "full_name": "1D Diffusion-Reaction Equation",
        "latex": (
            r"\partial_t u = D \partial_{xx} u + k u^2,"
            r"\quad x \in [0,1],\; t \in [0,2]"
        ),
        "latex_short": r"\partial_t u = D\partial_{xx}u + k u^2",
        "dim": 1,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "D, k",
        "nu_values": [0.001, 0.01],
        "description": (
            "1D diffusion-reaction equation with quadratic reaction term. "
            "Combines diffusive transport with nonlinear source term."
        ),
        "metrics": ["seg_total", "val_mse", "val_relmse", "pde_mean"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Crank-Nicolson + operator splitting",
        "data_samples": 10000,
    },
    "diffusion_reaction_2d": {
        "full_name": "2D Diffusion-Reaction Equation",
        "latex": (
            r"\partial_t u = D \nabla^2 u + k u^2,"
            r"\quad x,y \in [0,1],\; t \in [0,2]"
        ),
        "latex_short": r"\partial_t u = D\nabla^2 u + k u^2",
        "dim": 2,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "D, k",
        "nu_values": [0.001, 0.01],
        "description": (
            "2D diffusion-reaction equation — the 2D generalization of "
            "the 1D diffusion-reaction equation. Tests the ability of "
            "neural operators to handle 2D spatial dynamics."
        ),
        "metrics": ["seg_total", "val_mse", "val_relmse", "pde_mean"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Alternating Direction Implicit (ADI)",
        "data_samples": 1000,
    },
    "compressible_navier_stokes_1d": {
        "full_name": "1D Compressible Navier-Stokes",
        "latex": (
            r"\partial_t \mathbf{U} + \partial_x \mathbf{F} = 0,"
            r"\quad \mathbf{U} = (\rho, \rho u, E)^\top"
        ),
        "latex_short": r"\partial_t\mathbf{U} + \partial_x\mathbf{F} = 0",
        "dim": 1,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "Reynolds number",
        "nu_values": [100, 1000, 10000],
        "description": (
            "1D compressible Navier-Stokes equations for an ideal gas. "
            "Models shock-tube problems (Sod-type) with conservative "
            "variables density, momentum, and energy."
        ),
        "metrics": ["seg_total", "val_mse", "val_relmse"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Riemann solver + MUSCL scheme",
        "data_samples": 10000,
    },
    "compressible_navier_stokes_2d": {
        "full_name": "2D Compressible Navier-Stokes",
        "latex": (
            r"\partial_t \mathbf{U} + \nabla \cdot \mathbf{F} = 0,"
            r"\quad \mathbf{U} = (\rho, \rho u, \rho v, E)^\top"
        ),
        "latex_short": r"\partial_t\mathbf{U} + \nabla\cdot\mathbf{F}=0",
        "dim": 2,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "Reynolds number",
        "nu_values": [100, 1000],
        "description": (
            "2D compressible Navier-Stokes — 2D generalization of the "
            "1D Euler/Navier-Stokes with an additional velocity component."
        ),
        "metrics": ["seg_total", "val_mse"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Riemann solver + MUSCL scheme",
        "data_samples": 1000,
    },
    "incompressible_navier_stokes_2d": {
        "full_name": "2D Incompressible Navier-Stokes",
        "latex": (
            r"\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u}"
            r" = -\nabla p + \nu \nabla^2 \mathbf{u},"
            r"\quad \nabla\cdot\mathbf{u}=0"
        ),
        "latex_short": (
            r"\partial_t\mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u}"
            r"=-\nabla p + \nu\nabla^2\mathbf{u}"
        ),
        "dim": 2,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "nu (kinematic viscosity)",
        "nu_values": [0.001, 0.01, 0.1],
        "description": (
            "2D incompressible Navier-Stokes — models vorticity dynamics "
            "in 2D fluid flow with divergence-free constraint. Tests "
            "whether neural operators respect incompressibility."
        ),
        "metrics": ["seg_total", "val_mse", "pde_mean", "div_free_residual"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Pseudospectral / projection method",
        "data_samples": 1000,
    },
    "darcy_flow": {
        "full_name": "Darcy Flow (2D Steady-State)",
        "latex": (
            r"-\nabla \cdot (a(x)\nabla u) = f(x),"
            r"\quad x \in (0,1)^2"
        ),
        "latex_short": r"-\nabla\cdot(a(x)\nabla u)=f",
        "dim": 2,
        "time_dependent": False,
        "competition_task": None,
        "viscosity_params": "Permeability field a(x)",
        "nu_values": None,
        "description": (
            "2D steady-state Darcy flow — elliptic PDE for fluid "
            "flow in porous media. Maps permeability field a(x) to "
            "pressure field u(x). Time-independent (steady-state) problem."
        ),
        "metrics": ["val_mse", "val_relmse", "pde_mean"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Finite element method (FEM)",
        "data_samples": 10000,
    },
    "shallow_water": {
        "full_name": "2D Shallow Water Equations",
        "latex": (
            r"\partial_t h + \partial_x(hu) + \partial_y(hv) = 0,"
            r"\quad \partial_t(hu) + \partial_x(hu^2 + \frac12 gh^2)"
            r" + \partial_y(huv) = 0"
        ),
        "latex_short": r"\partial_t h + \nabla\cdot(h\mathbf{u}) = 0",
        "dim": 2,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "g (gravity)",
        "nu_values": [9.81],
        "description": (
            "2D shallow water equations — models free-surface flow "
            "under gravity in the shallow water approximation. "
            "Includes dam break simulations and wave propagation."
        ),
        "metrics": ["seg_total", "val_mse"],
        "references": ["arXiv:2207.05209 (PDEBench)"],
        "solver": "Finite volume / Godunov scheme",
        "data_samples": 1000,
    },
    "cylinder_rans": {
        "full_name": "2D RANS Cylinder Wake (Zhang 2026)",
        "latex": (
            r"\partial_t \bar{u}_i + \bar{u}_j \partial_j \bar{u}_i = "
            r"-\partial_i \bar{p} + \nu \partial_{jj} \bar{u}_i"
            r" + \partial_j \tau_{ij},\; \partial_i \bar{u}_i = 0,"
            r"\quad (x,y) \in [-10,25] \times [-5,5]"
        ),
        "latex_short": (
            r"\partial_t\bar{u}+(\bar{u}\cdot\nabla)\bar{u}"
            r"=-\nabla\bar{p}+\nu\nabla^2\bar{u}+\nabla\cdot\tau,\;"
            r"\nabla\cdot\bar{u}=0"
        ),
        "dim": 2,
        "time_dependent": True,
        "competition_task": None,
        "viscosity_params": "nu (kinematic viscosity)",
        "nu_values": [0.001],
        "description": (
            "2D Reynolds-Averaged Navier-Stokes (RANS) for flow past a circular "
            "cylinder at Re=3900 (Zhang et al., JFM 2026). Benchmark problem for "
            "physics-informed neural operator training. The RANS equations model "
            "the mean velocity field (u_bar) with Reynolds stress tensor tau "
            "capturing turbulent fluctuations. Training uses both data loss "
            "(on observed velocity fields) and PDE residual loss (RANS equations "
            "enforced via automatic differentiation). Domain: x in [-10, 25], "
            "y in [-5, 5] with cylinder of diameter D=1 centered at (0,0)."
        ),
        "metrics": [
            "val_mse",
            "val_relmse",
            "rans_pde_total",
            "rans_div_free",
            "rans_continuity",
            "rans_momentum_x",
            "rans_momentum_y",
            "train_time_min",
            "arch_params",
            "epochs",
        ],
        "references": [
            "Zhang et al., JFM 2026 — RANS PINN cylinder flow",
            "JFM benchmark: cylinder Re=3900, D=1, domain [-10,25]x[-5,5]",
        ],
        "solver": "DNS reference (JFM benchmark data)",
        "data_samples": "Reference DNS dataset",
        "ic_distribution": "Developed turbulent wake (steady-state mean)",
        "competition_info": None,
    },
    "kuramoto_sivashinsky": {
        "full_name": "Kuramoto-Sivashinsky Equation",
        "latex": (
            r"\partial_t u + u \cdot \partial_x u + \lambda_2 \partial_{xx} u"
            r" + \partial_{xxxx} u = 0,"
            r"\quad x \in \mathcal{D},\; t \in [0, T]"
        ),
        "latex_short": (
            r"\partial_t u + u\partial_x u + \lambda_2\partial_{xx}u"
            r" + \partial_{xxxx}u = 0"
        ),
        "dim": 1,
        "time_dependent": True,
        "competition_task": "task3",
        "viscosity_params": "lambda_2 (diffusion coefficient, energy injection)",
        "nu_values": [1.0, 1.5],
        "description": (
            "Kuramoto-Sivashinsky (KS) equation — a canonical nonlinear PDE "
            "that exhibits spatiotemporal chaos. "
            "Used as Task 3 (bonus) in the AI4S CNS challenge. "
            "The lambda_2 controls energy injection and chaotic strength. "
            "Requires neural operators to predict 380-step chaotic trajectories "
            "from only 20 observation steps, without lambda_2 at inference."
        ),
        "metrics": [
            "seg_total",
            "seg1",
            "seg2",
            "seg3",
            "val_mse",
            "val_relmse",
            "train_time_min",
            "arch_params",
            "epochs",
        ],
        "references": [
            "https://competition.ai4s.com.cn/race/7/introduction",
            "PDEBench (arXiv:2207.05209)",
        ],
        "solver": "Pseudo-spectral / exponential time differencing (ETD)",
        "data_samples": 2100,
        "ic_distribution": "Chaotic / unknown initialization",
        "competition_info": {
            "task": "task3",
            "max_score": 350,
            "scoring_formula": "max(plan_a, plan_b)",
            "plan_a": "task1(150) + task2(150) + seg_score*0.5(max 50) = 350",
            "plan_b": "task1(150) + seg_score*2(max 200) = 350",
            "observation_steps": 20,
            "prediction_steps": 380,
            "total_steps": 400,
            "grid_points": 256,
            "dt_stored": 0.5,
            "segment_1": {
                "steps": "20-49 (30 steps)",
                "physical_time": "t∈[10,24.5]",
                "weight": 0.25,
            },
            "segment_2": {
                "steps": "50-199 (150 steps)",
                "physical_time": "t∈[25,99.5]",
                "weight": 0.25,
            },
            "segment_3": {
                "steps": "200-399 (200 steps)",
                "physical_time": "t∈[100,199.5]",
                "weight": 0.50,
            },
            "inference_time_limit_min": 2,
            "total_time_limit_hours": 12,
            "train_from_scratch": True,
            "lambda_2_provided_in_train": True,
            "lambda_2_provided_in_test": False,
        },
    },
}

# ── Task-level equation groupings ──

_TASK_EQUATIONS: dict[str, list[str]] = {
    "task1": ["burgers"],
    "task2": ["burgers_multi_nu"],
    "task3": ["kuramoto_sivashinsky"],
}

# ── Public API ──


def get_equations() -> dict[str, dict[str, Any]]:
    """Return a copy of the full equation registry.

    Returns:
        Dict of {equation_name: metadata} for all known PDEs.
    """
    return dict(EQUATIONS)


def get_equation(name: str) -> dict[str, Any] | None:
    """Get metadata for a specific equation by name.

    Args:
        name: Equation name key (e.g. 'burgers', 'advection').

    Returns:
        Equation metadata dict, or None if not found.
    """
    info = EQUATIONS.get(name)
    return dict(info) if info else None


def list_equations_for_task(task: str) -> list[dict[str, Any]]:
    """List equations that belong to a competition task.

    Args:
        task: 'task1', 'task2', 'task3', or None for all non-competition equations.

    Returns:
        List of equation metadata dicts.
    """
    names = _TASK_EQUATIONS.get(task, [])
    result = []
    for name in names:
        info = EQUATIONS.get(name)
        if info:
            result.append({"name": name, **info})
    return result


def get_equation_metrics(name: str, task: str | None = None) -> list[str]:
    """Get the list of standard metrics relevant for a given equation.

    Args:
        name: Equation name key.
        task: Optional task filter ('task1', 'task2', 'task3').
            If provided, only returns metrics relevant to that task.

    Returns:
        List of metric names.
    """
    info = EQUATIONS.get(name)
    if info is None:
        return []
    metrics = list(info.get("metrics", []))
    if task:
        from expflow_pde.metrics import get_registered_metrics

        all_metrics = get_registered_metrics()
        metrics = [m for m in metrics if m in all_metrics]
    return metrics


def list_equation_names() -> list[str]:
    """Return sorted list of all equation names."""
    return sorted(EQUATIONS.keys())


def list_competition_equations() -> list[str]:
    """Return list of equations assigned to any competition task."""
    result = set()
    for names in _TASK_EQUATIONS.values():
        result.update(names)
    return sorted(result)
