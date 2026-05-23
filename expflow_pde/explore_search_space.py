#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explore-mode hyperparameter search space — wide-coverage Define-by-Run.

This module provides a secondary search space designed for the **explore**
mode (5090 GPU node, no strict time budget).  It extends beyond the
competition-verified ranges in ``_DEFAULT_SEARCH_SPACE`` and uses Optuna's
native Define-by-Run pattern (``trial.suggest_categorical`` + ``if/else``)
to support multiple architectures and training strategies.

Usage::

    from expflow_pde.explore_search_space import get_explore_objective
    import optuna

    study = optuna.create_study(direction="maximize")
    study.optimize(get_explore_objective(train_script), n_trials=200)

Key differences from ``_DEFAULT_SEARCH_SPACE``:

+-------------------+---------------------+---------------------+
| Dimension         | Competition (Task1) | Explore (5090)      |
+===================+=====================+=====================+
| Architecture      | FNO only            | FNO / FNO+attn /   |
|                   |                     | DeepONet / PINO     |
+-------------------+---------------------+---------------------+
| n_modes           | [8, 32]             | [8, 64]             |
| width             | [16, 128]           | [16, 256]           |
| n_layers          | [2, 6]              | [2, 8]              |
| lr                | [1e-4, 1e-2]        | [1e-5, 1e-2]        |
| weight_decay      | [0, 1e-5]           | [0, 1e-3]           |
| batch_size        | [64, 256]           | [32, 512]           |
| epochs            | [40, 120]           | [40, 500]           |
| scheduler         | cosine (fixed)      | cosine / StepLR /   |
|                   |                     | OneCycleLR / None   |
| loss_fn           | l2_rel (fixed)      | l2_rel / h1_1d /    |
|                   |                     | smoothl1_rel        |
| n-step AR unroll  | 1 (competition)     | [1, 20] (PBDL rec)  |
| freeze mode       | none                | none / spectral /   |
|                   |                     | staged              |
+-------------------+---------------------+---------------------+

Inspired by:
- HyperNOs best Burgers config (arXiv:2503.18087): wd=1e-4, StepLR,
  AdamW, width=64, modes=16, n_layers=4
- IS-FNO (arXiv:2512.19439): n-step=20, gradient clipping, GELU
- PINNs (Raissi 2019): physics-informed training, 8x20 MLP
- PBDL book Ch.5: multi-step AR unrolling for long-horizon stability
"""

from __future__ import annotations

from typing import Any, Callable


# ── Architecture registry ──

ARCHITECTURES = frozenset({
    "FNO",
    "FNO_attention",
    "FNO_deep",
    "DeepONet",
    "DeepONet_wide",
    "PINO",
    "PINN_MLP",
})

"""Supported architectures in explore mode.

Each maps to a training script / model class:

- ``FNO`` — Standard FNO as in ``train_task1.py`` (modes=8..64, width=16..256)
- ``FNO_attention`` — FNO with channel-attention skip (HyperNOs style)
- ``FNO_deep`` — Deep FNO (n_layers=6..8, width=64..128)
- ``DeepONet`` — Standard DeepONet (branch=2..6 layers, trunk=2..6 layers)
- ``DeepONet_wide`` — Wide DeepONet (width=128..256, for long-horizon)
- ``PINO`` — FNO + PDE physics loss (weights 0.01..0.5)
- ``PINN_MLP`` — Pure physics-informed MLP (Raissi-style, 4..8 layers)
"""


# ── Shared hyperparameter distributions ──


def suggest_shared(trial: Any, name: str) -> Any:
    """Suggest a shared training hyperparameter.

    These are sampled regardless of architecture choice.
    """
    suggestions = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_int("batch_size", 32, 512, step=32),
        "n_train": trial.suggest_int("n_train", 500, 2000, step=500),
        "seed": trial.suggest_int("seed", 1, 100, step=1),
    }
    return suggestions[name]


def suggest_optimizer(trial: Any) -> dict[str, Any]:
    """Suggest optimizer + scheduler combination (shared across archs)."""
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    sched_name = trial.suggest_categorical(
        "scheduler",
        ["cosine", "StepLR", "OneCycleLR", "none"],
    )

    optim = {"name": opt_name}

    # weight_decay: uniform across optimizer types to avoid Optuna
    # distribution conflict (same name, different log setting).
    # Range: HyperNOs recommends 1e-4, competition proved 0 optimal
    # under 60min limit; explore mode has no such limit.
    optim["weight_decay"] = trial.suggest_float(
        "weight_decay", 0.0, 1e-3, log=False
    )

    # Scheduler parameters
    if sched_name == "StepLR":
        optim["scheduler_step"] = trial.suggest_int("scheduler_step", 10, 200, step=10)
        optim["scheduler_gamma"] = trial.suggest_float("scheduler_gamma", 0.5, 0.99)
    elif sched_name == "cosine":
        optim["scheduler_T_max"] = trial.suggest_int("T_max", 20, 200, step=10)
    elif sched_name == "OneCycleLR":
        optim["max_lr"] = trial.suggest_float("max_lr", 1e-4, 5e-3, log=True)

    return optim


# ── Architecture-specific suggestion functions ──


def suggest_fno_params(trial: Any) -> dict[str, Any]:
    """Suggest FNO-specific architecture parameters.

    Range sources:
    - HyperNOs best Burgers: modes=16, width=64, layers=4
    - IS-FNO: modes up to 128, 4 layers
    - Competition evidence: modes 16-24 optimal, width 64 sweet spot
    """
    return {
        "n_modes": trial.suggest_int("n_modes", 8, 64, step=4),
        "width": trial.suggest_int("width", 16, 256, step=16),
        "n_layers": trial.suggest_int("n_layers", 2, 8, step=1),
        "activation": trial.suggest_categorical(
            "activation", ["GELU", "LeakyReLU", "ReLU", "tanh"]
        ),
        "padding": trial.suggest_int("padding", 0, 12, step=1),
        # Multi-step AR unrolling: PBDL Ch.5 recommends n>=8 for stability
        "ar_steps": trial.suggest_int("ar_steps", 1, 20, step=1),
        # Freeze strategy: spectral layer freezing for long-horizon
        "freeze": trial.suggest_categorical(
            "freeze", ["none", "spectral", "staged"]
        ),
    }


def suggest_fno_attention_params(trial: Any) -> dict[str, Any]:
    """Suggest FNO with channel-attention (HyperNOs-style)."""
    base = suggest_fno_params(trial)
    base["attn_heads"] = trial.suggest_int("attn_heads", 2, 8, step=2)
    return base


def suggest_deeponet_params(trial: Any) -> dict[str, Any]:
    """Suggest DeepONet-specific architecture parameters.

    Range sources:
    - PDEBench experiments: branch=2-4, trunk=3-6, width=64-96
    - DeepONet wide variant for long-horizon stability
    """
    is_wide = trial.suggest_categorical("deeponet_variant", ["standard", "wide"])

    if is_wide == "wide":
        return {
            "branch_layers": trial.suggest_int("branch_layers", 2, 6, step=1),
            "trunk_layers": trial.suggest_int("trunk_layers", 2, 8, step=1),
            "width": trial.suggest_int("width", 128, 256, step=32),
        }
    return {
        "branch_layers": trial.suggest_int("branch_layers", 2, 6, step=1),
        "trunk_layers": trial.suggest_int("trunk_layers", 2, 6, step=1),
        "width": trial.suggest_int("width", 32, 128, step=16),
    }


def suggest_pino_params(trial: Any) -> dict[str, Any]:
    """Suggest PINO (Physics-Informed Neural Operator) parameters.

    FNO backbone + PDE physics loss.  physics_weight range references
    the experimental finding that pw >= 0.001 makes phys_loss dominate
    by 3000x.  Explore mode can try wider range because there's no
    60-minute limit to fight.
    """
    base = suggest_fno_params(trial)
    base["physics_weight"] = trial.suggest_float(
        "physics_weight", 1e-5, 0.5, log=True
    )
    return base


def suggest_pinn_mlp_params(trial: Any) -> dict[str, Any]:
    """Suggest PINN MLP (Raissi 2019 style) parameters.

    Pure physics-informed: no training data, only IC/BC/PDE residual.
    """
    return {
        "n_layers": trial.suggest_int("pinn_layers", 4, 12, step=1),
        "width": trial.suggest_int("pinn_width", 16, 128, step=16),
        "n_iters": trial.suggest_int("pinn_iters", 10000, 100000, step=10000),
        "n_res_points": trial.suggest_int("n_res_points", 1000, 10000, step=1000),
        "activation": trial.suggest_categorical(
            "pinn_activation", ["tanh", "GELU", "silu"]
        ),
        "lr_scheduler": trial.suggest_categorical(
            "pinn_scheduler", ["cosine", "StepLR", "none"]
        ),
    }


# ── Loss function suggestion ──


def suggest_loss(trial: Any) -> str:
    """Suggest a loss function for the training script.

    References:
    - l2_rel: default, robust
    - h1_1d: Sobolev-weighted, suppresses high-freq oscillations
    - smoothl1_rel: robust to outliers
    """
    return trial.suggest_categorical(
        "loss_fn", ["l2_rel", "h1_1d", "smoothl1_rel", "mse_rel"]
    )


# ── Main objective factory ──


def get_explore_objective(
    train_script: str | Callable | None = None,
) -> Callable:
    """Return an Optuna-compatible objective for explore-mode HPO.

    The objective samples an architecture, its parameters, shared
    training params, and a loss function, then runs the training
    script and returns the metric value.

    Args:
        train_script: Path to training script, or a callable
            ``f(params: dict) -> float``.  If None, returns a
            *mock* objective for testing (returns a simulated score).

    Returns:
        An ``objective(trial) -> float`` callable suitable for
        ``optuna.study.optimize(objective, ...)``.

    Example::

        import optuna
        from expflow_pde.explore_search_space import get_explore_objective

        study = optuna.create_study(direction="maximize")
        study.optimize(get_explore_objective("./train_task1.py"), n_trials=50)
    """
    def _default_eval(params: dict[str, Any]) -> float:
        """Mock evaluation for testing — returns a synthetic score."""
        import math
        # Fake score centred around 100 ± 30, skewed by param quality
        lr = params.get("lr", 1e-3)
        width = params.get("width", 64)
        n_modes = params.get("n_modes", 16)
        score = 100.0 \
            + 15.0 * math.exp(-((math.log10(lr) + 3.0) ** 2) / 0.5) \
            + 10.0 * (1.0 - math.exp(-width / 64.0)) \
            + 5.0 * (1.0 - math.exp(-n_modes / 16.0))
        return max(0.0, min(150.0, score))

    if train_script is None:
        fn: Callable = _default_eval
    elif isinstance(train_script, str):
        def fn(params: dict[str, Any]) -> float:
            from expflow_pde.hpo import _run_trial_local
            metric = params.get("_metric", "seg_total")
            result = _run_trial_local(train_script, params, objective_metric=metric)
            return result or 0.0
    else:
        fn = train_script

    def objective(trial: Any) -> float:
        """Define-by-Run objective for explore-mode HPO.

        Samples architecture, its specific params, shared params,
        loss function, then evaluates.
        """
        # Decide what to try
        arch = trial.suggest_categorical("architecture", sorted(ARCHITECTURES))

        # Shared params
        params: dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_int("batch_size", 32, 512, step=32),
            "epochs": trial.suggest_int("epochs", 40, 500, step=10),
            "n_train": trial.suggest_int("n_train", 500, 2000, step=500),
            "loss_fn": trial.suggest_categorical(
                "loss_fn", ["l2_rel", "h1_1d", "smoothl1_rel", "mse_rel"]
            ),
        }
        optim = suggest_optimizer(trial)
        params.update(optim)

        # Architecture-specific params
        if arch in ("FNO",):
            params.update(suggest_fno_params(trial))
            params["architecture"] = "FNO"
        elif arch in ("FNO_attention",):
            params.update(suggest_fno_attention_params(trial))
            params["architecture"] = "FNO_attention"
        elif arch in ("FNO_deep",):
            params.update({
                "n_modes": trial.suggest_int("n_modes", 8, 64, step=4),
                "n_layers_deep": trial.suggest_int("n_layers_deep", 4, 8, step=1),
                "width_deep": trial.suggest_int("width_deep", 64, 256, step=32),
                "activation": trial.suggest_categorical(
                    "activation", ["GELU", "LeakyReLU", "ReLU", "tanh"]
                ),
                "padding": trial.suggest_int("padding", 0, 12, step=1),
                "ar_steps": trial.suggest_int("ar_steps", 1, 20, step=1),
                "freeze": trial.suggest_categorical(
                    "freeze", ["none", "spectral", "staged"]
                ),
            })
            params["architecture"] = "FNO_deep"
        elif arch in ("DeepONet", "DeepONet_wide"):
            params.update(suggest_deeponet_params(trial))
            params["architecture"] = arch
        elif arch in ("PINO",):
            params.update(suggest_pino_params(trial))
            params["architecture"] = "PINO"
        elif arch in ("PINN_MLP",):
            params.update(suggest_pinn_mlp_params(trial))
            params["architecture"] = "PINN_MLP"

        return fn(params)

    return objective


# ── Search space introspection ──


def describe_explore_space() -> dict[str, Any]:
    """Return a human-readable description of the explore search space.

    Useful for logging / Hermes context::

        from expflow_pde.explore_search_space import describe_explore_space
        space_info = describe_explore_space()
        # -> {\"architectures\": [...], \"param_count\": ..., \"description\": \"...\"}
    """
    return {
        "architectures": sorted(ARCHITECTURES),
        "arch_count": len(ARCHITECTURES),
        "param_categories": [
            "architecture",
            "learning_rate", "batch_size", "epochs", "n_train",
            "optimizer", "weight_decay", "scheduler",
            "loss_fn",
        ],
        "inspired_by": [
            "HyperNOs (arXiv:2503.18087)",
            "IS-FNO (arXiv:2512.19439)",
            "Raissi PINNs (2019)",
            "PBDL Book Ch.5 — multi-step AR unrolling",
        ],
        "description": (
            "Explore-mode search space with 7 architectures, "
            "Define-by-Run conditional sampling, and wide ranges "
            "for 5090 GPU node.  Covers FNO, FNO+attention, "
            "DeepONet, PINO, and PINN-MLP."
        ),
    }
