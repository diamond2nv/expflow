# Competition Integration Guide

## Overview

expflow-pde provides tools specifically designed for the **AI4S Neural Operator PDE
Competition** (Race 7 on [competition.ai4s.com.cn](https://competition.ai4s.com.cn/race/7)).
It orchestrates the full experiment lifecycle: training, HPO, evaluation, submission
packaging, competition rules audit, and strategic planning.

## Competition Tasks

| Task | PDE | Max Score | Status |
|:----:|-----|:---------:|:------:|
| 1 | Burgers (nu=0.001) | 150 | `in_progress` |
| 2 | Burgers (multi-nu) | 150 | `not_started` |
| 3 | Kuramoto-Sivashinsky | 350 | `not_started` |

## Competition Pipeline

The recommended workflow uses expflow's three pipeline modes:

### Phase 1: Exploration (find best hyperparams)

```bash
expflow pipeline submit-full train_task1.py \
    --queue default \
    --trials 50 --parallel 4 \
    --eval-script eval_task1.py \
    --metric seg_total --direction maximize \
    --pruner hyperband
```

### Phase 2: Sprint (known params, iterate fast)

```bash
expflow pipeline submit train_task1.py \
    --queue default \
    --train-param lr=0.001 --train-param epochs=80 \
    --train-param sub_step=5 \
    --eval-script eval_task1.py
```

### Phase 3: Competition audit

```bash
# Validate metrics against competition rules
expflow audit validate exp-001 \
    --competition-rules \
    --task-id <clearml_task_id>

# Compare all runs with gating
expflow clearml compare-scores \
    --project PDEBench --tags task1 \
    --sort-by seg_total --gate pde_mean:lt:18.09
```

## Standard Metrics Registry

All competition metrics follow the `Group/Metric` naming convention:

| Group | Metric | Higher Is Better | Competition Threshold |
|-------|--------|:----------------:|:---------------------:|
| `Score` | `Seg Total` | ✅ | — |
| `Score` | `Seg1` | ✅ | — |
| `Score` | `Seg2` | ✅ | — |
| `Score` | `Seg3` | ✅ | — |
| `Loss` | `Val MSE` | ❌ | — |
| `Loss` | `Val RelMSE` | ❌ | — |
| `PDE` | `Mean Residual` | ❌ | 18.09 |
| `Time` | `Train Time Min` | ❌ | 60.0 |
| `Time` | `Inference Time Min` | ❌ | 2.0 |

Add these to your training script (see [USAGE.md](USAGE.md) for the full pattern):

```python
if clearml_logger is not None:
    clearml_logger.report_scalar('Score', 'Seg Total', seg_total, iteration=epoch)
    clearml_logger.report_scalar('Score', 'Seg1', seg1, iteration=epoch)
    clearml_logger.report_scalar('Score', 'Seg2', seg2, iteration=epoch)
    clearml_logger.report_scalar('Score', 'Seg3', seg3, iteration=epoch)
    clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=epoch)
    clearml_logger.report_scalar('Loss', 'Val RelMSE', val_rel, iteration=epoch)
    clearml_logger.report_scalar('PDE', 'Mean Residual', pde_mean, iteration=epoch)
```

## Competition Rules Audit

The `expflow audit validate --competition-rules` command checks:

| Check | Rule | Scoring |
|-------|------|---------|
| `seg_total` | Must be reported (no gating) | Primary competition score |
| `pde_mean` | Must be < 18.09 | PDE residual compliance |
| `train_time_min` | Must be < 60 | Training within 60-min limit |
| `sub_step` | Must exist and be > 0 | dt mismatch fix required |

### Python API

```python
from expflow_pde.audit import validate_competition_rules

result = validate_competition_rules(
    task_metrics={
        "seg_total": 57.09,
        "pde_mean": 15.0,
        "train_time_min": 45.5,
    },
    task_params={"Args/--sub_step": "5"},
)

print(f"All pass: {result['all_pass']}")
for check in result["checks"]:
    print(f"  {check['name']}: {'✓' if check['passed'] else '✗'} ({check['detail']})")
```

## Task Intelligence

The `expflow analyze` command group provides strategic competition advice:

```bash
# Overall status
expflow analyze status

# Task 1 deep analysis
expflow analyze task task1

# Strategic recommendation
expflow analyze advise

# Equation reference
expflow analyze equations --task competition
```

### Score Estimation

```python
from expflow_pde.analyze import estimate_score_potential, get_strategic_recommendation

# Per-task score projections
estimates = estimate_score_potential("task1")
# Returns:
# {
#   "optimistic": 148,
#   "expected": 145,
#   "conservative": 140,
#   "confidence": "high",
# }

# Full strategy
rec = get_strategic_recommendation()
# Returns:
# {
#   "primary_focus": "task1",
#   "remaining_headroom": {...},
#   "suggested_schedule": {
#     "day_1_2": "Task 1: HPO on lambda_stab + longer epochs",
#     "day_3_4": "...",
#   },
# }
```

## Proven Strategies (Task 1)

Archived from actual experiments on `token_arena/PDEBench`:

| Strategy | Seg Improvement | Details |
|----------|:--------------:|---------|
| `sub_step=5` | +11.37 | dt mismatch fix (dt=0.01 vs 0.05) |
| Stability FT | +23.45 | Step-wise variance penalty (3 lines of code) |
| P2 architecture (16/32, 50K params) | — | Optimal model size |
| FT lr≈1e-7 | — | Preserves pretrained features |

## Competition Constraints Summary

| Constraint | Task 1 | Task 2 | Task 3 |
|------------|:------:|:------:|:------:|
| Training time | < 60 min | < 60 min | N/A |
| Inference time | < 2 min | < 2 min | < 2 min |
| Total time | < 12 h | < 12 h | < 12 h |
| Observation steps | 10 | 10 | 20 |
| Prediction steps | 190 | 190 | 380 |
| Nu known at test? | ✅ | ❌ | N/A |
| λ₂ known at test? | N/A | N/A | ❌ |
| Pretrained allowed? | ❌ | ❌ | ❌ |

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture
- [USAGE.md](USAGE.md) — CLI reference
- [DATA_LAYER.md](DATA_LAYER.md) — Data layer design
- Package skill `competition-task-intelligence` for agent guidance
