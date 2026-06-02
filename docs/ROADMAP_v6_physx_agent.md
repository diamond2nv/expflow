# expflow v6 — Physical Constraints + Agent-Guided Search Integration

> **Version targeting**: v0.7.0
> **Core thesis**: Two orthogonal robustness layers — training-level (PINN physics loss, Zhang2026) and experiment-level (noise gating + stagnation detection, AutoScientists) — fused into expflow's pipeline architecture.
> **License obligation**: HyperNOs-derived code retains arXiv:2503.18087 citation in README acknowledgements. AutoScientists (arXiv:2605.28655) and Zhang2026 (JFM, 2026) contributions are independent rewrites based on published mathematical formulations.

---

## §1. Why Fuse? — The Two-Gap Model

The expflow pipeline currently orchestrates experiments (HPO → train → eval → submit) but has **no intrinsic notion of physical correctness** or **experimental rigor**:

```
Current pipeline (Mode A):
  HPO → best_params → Train(nn.MSELoss) → eval → submit
                         ^                ^
                         |                └── scalar metrics only, no PDE residual check
                         └── no physics-informed loss term
```

**Two gaps to fill:**

| Layer | Gap | Source | Mechanism | Integration Point |
|:------|:----|:------|:----------|:-----------------|
| **L1: Training loss** | Model doesn't know PDE | Zhang2026 (JFM) | `RANSPDELoss` — RANS PDE residual term + data loss | `losses.py` → `loss_selector()` → training script |
| **L2: Experiment rigor** | Can't tell real improvement from noise | AutoScientists | Noise gating (lazy σ, dual-seed) + stagnation detection + dead-end registry | `pipeline.py` validation step + cron monitor |

### Orthogonality proof

```
Zhang2026:  Loss = λ_data * ||û - u||² + λ_pde * ||R(û)||²
            ↑ solves "is my prediction physically plausible?"

AutoScientists:  promote(p') = 
                   true if Δ > Mσ       ← confident
                   confirm(p') if 0<Δ≤Mσ  ← noise band, dual-seed
                   false if Δ ≤ 0
                 ↑ solves "is my improvement real?"
```

**These address different failure modes** — a model can have low MSE but violate the PDE (Zhang2026 catches this), or have an apparent 2% improvement that's just seed noise (AutoScientists catches this). They compose multiplicatively.

---

## §2. Architecture

```
                     Hermes Agent / CLI User
                            │
                    expflow CLI (Typer)
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    Pipeline Layer     Loss Layer         Monitor Layer
    (pipeline.py)     (losses.py)        (cron + dispatch)
         │                  │                  │
    ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
    │ Train   │        │ DataLoss│        │Stagnation│
    │ → Eval  │        │ +RANSPDE│        │Detector  │
    │ → Submit│        │ Loss    │        │(KEEP-cnt)│
    └────┬────┘        └─────────┘        └────┬────┘
         │                                     │
    ┌────┴─────────────────────────────────────┴────┐
    │  Noise-Aware Validation Gate                   │
    │  (lazy σ calibration + dual-seed confirm)     │
    └──────────────────┬───────────────────────────┘
                       │
              ┌────────┴────────┐
              │ Dead-end Registry│
              │ (cross-exp fail) │
              └─────────────────┘
```

### Integration points (file-by-file)

| File | Existing | New | License |
|:-----|:---------|:----|:--------|
| `expflow_pde/equations.py` | 11 PDEs | Add `cylinder_rans` (2D RANS) | Original |
| `expflow_pde/metrics.py` | ~25 metrics | Add RANS-specific PDE residual metrics | Original |
| `expflow_pde/losses.py` | 6 loss classes (HyperNOs-derived) | Add `RANSPDELoss`, `PINNCompositeLoss`, `PhysicsInformedLoss` | HyperNOs via arXiv:2503.18087; new losses original via Zhang2026 JFM 2026 |
| `expflow_pde/pipeline.py` | 3 modes (fast/full/custom) | Add noise gating step, `--validate` flag | Original |
| `expflow_pde/dispatch.py` | jsonl registry | Add dead-end registry table | Original |
| `expflow/monitor/` | — | New: stagnation detection cron script | Original |
| `README.md` | — | Add Acknowledgements section | — |
| `AGENTS.md` | — | Update with new capabilities | — |

---

## §3. Layer 1: Physics-Informed Training Loss (Zhang2026)

### §3.1 Zhang2026 Core Idea

Zhang et al. (JFM, 2026) propose training neural operators for 2D RANS cylinder flow using a **composite loss**:

```
L_total = λ_data * L_data(u_pred, u_true) + λ_pde * L_pde(R(u_pred))
```

Where:
- `L_data` — standard supervised loss (MSE/Rel-MSE) on observed data
- `L_pde` — RANS PDE residual: `R(u) = ∂u/∂t + (u·∇)u + ∇p - ν∇²u`
- `λ_data, λ_pde` — adaptive weighting (gradient normalization, not fixed)

### §3.2 cylinder_rans Equation Entry

New entry in `equations.py`:

```python
"cylinder_rans": {
    "full_name": "2D Reynolds-Averaged Navier-Stokes (Cylinder Wake)",
    "latex": (
        r"\\partial_t \\bar{u}_i + \\bar{u}_j \\partial_j \\bar{u}_i = "
        r"-\\partial_i \\bar{p} + \\nu \\partial_{jj} \\bar{u}_i"
        r" + \\partial_j \\tau_{ij},\\; \\partial_i \\bar{u}_i = 0"
    ),
    "latex_short": r"RANS cylinder wake — 2D",
    "dim": 2,
    "time_dependent": True,
    "competition_task": None,
    "viscosity_params": "nu (kinematic viscosity)",
    "nu_values": [0.001],
    ...
}
```

### §3.3 RANSPDELoss Design

The loss computes the RANS PDE residual on a **collocation grid** (not training grid):

```python
class RANSPDELoss(nn.Module):
    """RANS PDE residual loss for 2D cylinder flow.

    L_pde = ||∂u/∂t + (u·∇)u + ∇p - ν∇²u||²

    Uses automatic differentiation (torch.autograd.grad) for spatial
    and temporal derivatives. Operates on collocation points sampled
    independently of the training data grid.

    Args:
        nu: Kinematic viscosity (default: 0.001).
        reduction: 'mean' or 'sum'.
    """

    def forward(self, u_pred, x_colloc, t_colloc):
        """Compute RANS PDE residual.

        Args:
            u_pred: callable or stored (N_colloc, 2) velocity field.
            x_colloc: (N, 2) collocation points in space.
            t_colloc: (N, 1) collocation time points.

        Returns:
            Scalar PDE residual loss.
        """
```

**Key design decisions:**
1. **Autograd-based** (not finite difference) — compatible with modern neural operator training
2. **Collocation sampling** — separate from training data grid to prevent overfitting to mesh
3. **Incompressibility constraint** — `∇·u = 0` term as soft penalty (no pressure solver needed)

### §3.4 PINNCompositeLoss

Orchestrates data loss + PDE loss with adaptive weighing:

```python
class PINNCompositeLoss(nn.Module):
    """Composite loss: data term + physics residual + optional BC/IC.

    L = λ_data * L_data + λ_pde * L_pde + λ_bc * L_bc

    Args:
        data_loss: nn.Module for supervised data term.
        pde_loss: RANSPDELoss for physics residual.
        lambda_data: Data term weight.
        lambda_pde: PDE residual weight.
        lambda_bc: Boundary condition weight.
        adaptive: Use gradient-based adaptive weighting (GradNorm-style).
    """
```

### §3.5 RANS-Specific Metrics

New metrics in `STANDARD_METRICS`:

```python
"rans_div_free": {"type": "scalar", "group": "RANS", "higher_is_better": False},
"rans_continuity": {"type": "scalar", "group": "RANS", "higher_is_better": False},
"rans_momentum_x": {"type": "scalar", "group": "RANS", "higher_is_better": False},
"rans_momentum_y": {"type": "scalar", "group": "RANS", "higher_is_better": False},
"rans_pde_total": {"type": "scalar", "group": "RANS", "higher_is_better": False},
```

These are tracked during training (alongside standard `val_mse`, `val_relmse`) and visualized in clearml.

---

## §4. Layer 2: Experiment Rigor (AutoScientists)

### §4.1 Noise-Aware Champion Validation

AutoScientists' champion promotion rule translated into expflow's pipeline validation step:

```python
def noise_aware_validate(
    candidate_value: float,
    champion_value: float,
    noise_floor: float | None = None,
    sigma: float = 2.0,       # M in the paper
    noise_db_path: str | None = None,
) -> dict:
    """Validate a candidate against champion with noise awareness.

    The promotion rule (AutoScientists Eq. 1, arXiv:2605.28655):
        promote(p') = true     if Δ > Mσ   ← confident improvement
                      confirm  if 0 < Δ ≤ Mσ  ← within noise band
                      false    if Δ ≤ 0    ← no improvement

    Args:
        candidate_value: Metric value from candidate experiment.
        champion_value: Current best metric value.
        noise_floor: Pre-calibrated noise σ. If None, uses lazy calibration.
        sigma: Noise band multiplier M (default: 2.0).
        noise_db_path: Path to JSONL noise calibration data.

    Returns:
        Dict with keys:
            - action: 'promote', 'confirm', 'reject'
            - delta: candidate - champion
            - noise_floor: Used σ value
            - second_seed_needed: Boolean
            - seed_suggestions: Dict of seed params if confirm needed
    """
```

**Integration with pipeline:**
- After train step, the champion comparison becomes noise-aware
- If `confirm` needed → pipeline auto-launches a dual-seed verification task
- If `promote` → champion updated, pipeline continues
- If `reject` → experiment logged to dead-end registry, pipeline suggests alternative

### §4.2 Lazy Noise Floor Calibration

Instead of dedicated seed probes, σ is estimated from passive experiment data:

```python
def calibrate_noise_floor(log_path: str, min_samples: int = 3) -> dict:
    """Lazy σ calibration from completed experiment duplicates.

    Reads JSONL noise log, groups by (metric_a, metric_b, code_hash),
    computes pooled stdev when n >= min_samples. Locks when n >= 5.

    Returns:
        Dict with keys:
            - sigma: Pooled standard deviation.
            - n_samples: Number of duplicate pairs used.
            - locked: Whether sigma is locked (n >= 5).
    """
```

Stored at `~/.expflow/noise_floor.jsonl` — append-only log of duplicate seed runs.

### §4.3 Stagnation Detection

```python
def detect_stagnation(
    experiment_history: list[dict],
    keep_window: int = 5,        # how many recent experiments to check
    max_keep_no_progress: int = 3,  # max accepted improvements without real gain
    single_axis_threshold: int = 8,  # DISCARDs on same axis → stagnation
) -> dict:
    """Detect stagnation in an experiment tree.

    Two modes (AutoScientists §2.3):
    1. KEEP-count: If >= max_keep_no_progress consecutive KEEP decisions
       without meaningful metric gain → stagnation.
    2. Single-axis exhaustion: If >= single_axis_threshold DISCARDs
       on ≤3 axes with no paired probe → stagnation.

    Returns:
        Dict with keys:
            - stagnant: Boolean.
            - reason: Description of stagnation mode.
            - suggested_action: 'regroup', 'explore_new_axis', 'terminate'.
            - axis_distribution: Dict of axis → discard count.
    """
```

Integrated as a **cron job** alongside the experiment polling cron.

### §4.4 Dead-End Registry

```python
class DeadEndRegistry:
    """Cross-session dead-end registry — prevents redundant exploration.

    Each entry: {approach_hash, axis, reason, attempted_at, code_hash}
    Queried before launching any new experiment — if approach_hash
    matches a recorded dead end, pipeline warns and suggests alternatives.

    Storage: ~/.expflow/dead_ends.db (SQLite) — separate from dispatch.db
    to allow independent lifecycle.
    """

    def register(approach: str, axis: str, reason: str, code_hash: str) -> dict: ...
    def lookup(approach: str) -> list[dict]: ...
    def list_recent(limit: int = 20) -> list[dict]: ...
```

---

## §5. Implementation Plan

### Phase A — Equation + Metrics (estimated: 1.5h)
| # | Task | Files | 
|:-:|------|-------|
| A1 | Add `cylinder_rans` to `equations.py` | `equations.py` |
| A2 | Add RANS-specific metrics to `STANDARD_METRICS` | `metrics.py` |
| A3 | Tests for equation entry + metrics | `tests/test_equations.py` |

### Phase B — Physics Losses (estimated: 2h)
| # | Task | Files |
|:-:|------|-------|
| B1 | `RANSPDELoss` — autograd-based PDE residual for 2D RANS | `losses.py` |
| B2 | `PINNCompositeLoss` — adaptive λ weighting | `losses.py` |
| B3 | Update `loss_selector()` with 'rans_pde', 'pinn_composite' | `losses.py` |
| B4 | Tests: gradient flow, perfect solution → 0, PDE violation penalty | `tests/test_losses.py` |

### Phase C — Noise Gating (estimated: 2h)
| # | Task | Files |
|:-:|------|-------|
| C1 | `noise_aware_validate()` — champion promotion logic | `expflow_pde/validate.py` (new) |
| C2 | `calibrate_noise_floor()` — lazy σ estimation | `validate.py` |
| C3 | `DeadEndRegistry` — SQLite-backed failure log | `expflow_pde/registry.py` (new) |
| C4 | Pipeline integration: `--validate` flag on `train_val_submit()` | `pipeline.py` |
| C5 | Tests | `tests/test_validate.py`, `tests/test_registry.py` |

### Phase D — Stagnation Monitor (estimated: 1h)
| # | Task | Files |
|:-:|------|-------|
| D1 | `detect_stagnation()` — KEEP-count + axis exhaustion | `expflow_pde/monitor.py` (new) |
| D2 | Cron job registration (weekly) | Hermes cron |
| D3 | Output to QQ report | — |

### Phase E — Cleanup (estimated: 0.5h)
| # | Task | Files |
|:-:|------|-------|
| E1 | README.md: Acknowledgements section with arXiv IDs | `README.md` |
| E2 | README.md: Remove "Transplanted from HyperNOs" from docstring | `losses.py` |
| E3 | AGENTS.md: New sections for RANS/noise-gating/monitor | `AGENTS.md` |

---

## §6. License & Attribution Strategy

| Component | Origin | License Status | Action |
|:----------|:-------|:---------------|:-------|
| `LprelLoss`, `H1relLoss`, `MSELoss_rel`, etc. | HyperNOs (arXiv:2503.18087) | Cite paper | README: "Loss suite inspired by HyperNOs" + arXiv ID; code already rewritten from scratch |
| `RANSPDELoss`, `PINNCompositeLoss` | Zhang2026 (JFM, 2026) | Math/physics — not copyrightable | README: "Physics-informed losses follow Zhang et al. (JFM, 2026)" |
| `noise_aware_validate`, `calibrate_noise_floor` | AutoScientists (arXiv:2605.28655) | Algorithmic description — not copyrightable | README: "Experiment validation gating inspired by AutoScientists" + arXiv ID |
| `DeadEndRegistry`, `detect_stagnation` | AutoScientists (conceptual) | Independent implementation | Same as above |

All external code is **rewritten from scratch** — only the mathematical formulation is referenced. This is the safest posture for third-party license compliance.

---

## §7. Non-Goals (explicitly excluded)

- ❌ No LLM-based experiment design (that's Hermes ML Ops, not expflow)
- ❌ No replacement of clearml's execution layer
- ❌ No pressure-Poisson solver (the `∇·u=0` soft penalty avoids this)
- ❌ No full PINN training framework (only loss function component)
- ❌ No ClawInstitute/Claude Code agent protocol (that's AutoScientists' infrastructure)
