---
name: competition-task-intelligence
title: Competition Task Intelligence — PDE Equation Registry, Task Analysis, and Strategic Advising
description: >
  Build and maintain a structured PDE equation registry, analyze competition tasks
  (difficulty, bottlenecks, score projections), generate strategic recommendations
  for research focus, and expose this intelligence via CLI and MCP tools.
category: mlops
tags: [mlops, competition, strategy, equations, analysis, planning, pde, task-intelligence]
related_skills:
  - agent4pde-competition-scoring
  - analyze-experiment-autoregressive-degradation
  - pde-competition-solver-strategy
  - experiment-lifecycle-governance
created: 2026-05-19
updated: 2026-05-19
---

# Competition Task Intelligence

## Overview

System for structured PDE equation management and competition task analysis. Provides:

1. **PDE Equation Registry** — structured metadata (LaTeX, dimensions, params, datasets) for all known PDEs
2. **Task Analysis** — per-task difficulty assessment, bottleneck identification, proven strategy catalog
3. **Score Projection** — optimistic/expected/conservative score estimates with confidence levels
4. **Strategic Advising** — which task to focus on, suggested schedule, rationale
5. **CLI + MCP** — `expflow analyze` command group and MCP tools

## Architecture

```
expflow_pde/equations.py     ──── PDE equation static registry (11+ equations)
        │
expflow_pde/analyze.py       ──── Analysis engine (task intelligence, strategy)
        │
expflow_pde/cli_analyze.py   ──── CLI: analyze task/equations/status/advise
        │
expflow_pde/mcp_server.py    ──── MCP: exp_compare_scores, exp_list_workers (partial)
```

## Data Flow

```
User/Agent: expflow analyze advise
  → analyze.get_strategic_recommendation()
    → reads _TASK_META (hardcoded intelligence)
    → looks up equations via equations.list_equations_for_task()
    → returns strategy dict with primary_focus, suggested_schedule
```

## 1. PDE Equation Registry

### Data Schema

Each equation entry in `EQUATIONS` dict:

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `full_name` | str | ✅ | Human-readable name |
| `latex` | str | ✅ | Full LaTeX PDE expression |
| `latex_short` | str | ✅ | Compact LaTeX for tables |
| `dim` | int | ✅ | Spatial dimensions (1/2/3) |
| `time_dependent` | bool | ✅ | Time-dependent or steady-state |
| `competition_task` | str\|None | ✅ | `task1`, `task2`, `task3`, or `None` |
| `viscosity_params` | str | ✅ | Key parameters (nu, beta, lambda₂, etc.) |
| `nu_values` | list\|None | ✅ | Available parameter values |
| `description` | str | ✅ | Plain-text explanation of the PDE |
| `metrics` | list[str] | ✅ | Relevant metric names from STANDARD_METRICS |
| `references` | list[str] | ✅ | Paper/URL references |
| `solver` | str | ✅ | Reference numerical solver |
| `data_samples` | int | ✅ | Total data samples available |
| `competition_info` | dict\|None | — | Optional: scoring formula, observation steps, segments, limits |

### Example: Kuramoto-Sivashinsky

```python
"kuramoto_sivashinsky": {
    "full_name": "Kuramoto-Sivashinsky Equation",
    "latex": r"\partial_t u + u \cdot \partial_x u + \lambda_2 \partial_{xx} u + \partial_{xxxx} u = 0",
    "latex_short": r"\partial_t u+u\partial_x u+\lambda_2\partial_{xx}u+\partial_{xxxx}u=0",
    "dim": 1,
    "time_dependent": True,
    "competition_task": "task3",
    "viscosity_params": "lambda_2 (diffusion coefficient, energy injection)",
    "nu_values": [1.0, 1.5],
    "metrics": ["seg_total", "seg1", "seg2", "seg3", "val_mse", ...],
    "data_samples": 2100,
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
        "inference_time_limit_min": 2,
        "total_time_limit_hours": 12,
        "train_from_scratch": True,
        "lambda_2_provided_in_train": True,
        "lambda_2_provided_in_test": False,
    },
}
```

### API

```python
from expflow_pde.equations import (
    get_equations(),                    # All 11+ equations
    get_equation(name),                 # Single equation
    list_equations_for_task(task_id),   # task1/task2/task3
    get_equation_metrics(name, task),   # Relevant STANDARD_METRICS
    list_equation_names(),              # Sorted names
    list_competition_equations(),       # Only competition equations
)
```

### Extending the Registry

When adding a new PDE equation:

1. Add entry to `EQUATIONS` dict with all required fields
2. If it's a competition equation, add to `_TASK_EQUATIONS` mapping
3. Add any new metric names to `STANDARD_METRICS` in `metrics.py`
4. Update tests in `test_equations.py`
5. Sync wiki: create entity page + update index.md + log.md

## 2. Task-Level Intelligence (_TASK_META)

Each task has structured metadata:

```python
_TASK_META = {
    "task1": {
        "label": "Task 1 — Burgers (fixed nu=0.001)",
        "max_score": 150,
        "difficulty": "medium",
        "priority": "high",
        "status": "in_progress",
        "current_best_seg": 57.09,
        "current_best_total": 142,
        "estimated_ceiling": 150,
        "remaining_headroom": 8,
        "key_bottlenecks": [
            "Seg3 long-horizon stability (190-step AR rollout)",
            "IC distribution mismatch between train/val",
            "Training time <60min to get full 35/35 time score",
        ],
        "proven_strategies": [
            "sub_step=5: +11.37 Seg (dt mismatch fix)",
            "Stability FT: +23.45 Seg",
            "P2 architecture (16/32, 50K params): optimal size",
            "FT lr≈1e-7: preserves pretrained features",
        ],
        "next_steps": [
            "HPO on lambda_stab (0.0001-0.01)",
            "Stability FT more epochs (20-30)",
            "P3 (24/32) baseline + sub_step=5 + stability FT",
        ],
    },
    "task2": { ... },
    "task3": { ... },
}
```

### Difficulty Classification

| Label | Icon | Example | Meaning |
|-------|:----:|---------|---------|
| `easy` | 🟢 | Baseline tasks | High confidence, proven methods exist |
| `medium` | 🟡 | Task 1 | Known bottlenecks, clear path forward |
| `hard` | 🔴 | Task 2 | Multiple unknown challenges, needs baseline evaluation |
| `very_hard` | 🔥 | Task 3 (KS) | Chaotic dynamics, exponential error growth, order-of-magnitude harder |

### Score Estimation

Each task has optimistic/expected/conservative estimates with confidence levels:

```python
{
    "optimistic": 148,
    "expected": 145,
    "conservative": 140,
    "confidence": "high",  # or "low"/"medium"
}
```

**Confidence rules:**
- `high` = Baseline exists + proven strategies work + clear ceiling
- `medium` = Partial evaluation done + some unknowns
- `low` = No baseline yet + fundamental uncertainty (e.g. chaotic KS dynamics scaling)

## 3. Strategic Recommendation System

### Logic

```python
def get_strategic_recommendation() -> dict:
    # 1. Read remaining headroom per task
    # 2. Consider difficulty + status + time remaining
    # 3. Recommend primary_focus (highest ROI in remaining time)
    # 4. Return with rationale and suggested_schedule
```

**Decision factors (in order):**
- Remaining headroom × (1 / difficulty)
- Knowledge transfer potential between tasks
- Time constraints (competition deadline, submissions/day limit)
- Current status (in_progress > not_started for same ROI)

### Suggested Schedule Template

```python
{
    "day_1_2": "...",
    "day_3_4": "...",
    "day_5_6": "...",
    "day_7_8": "...",
}
```

Days are paired to reflect: day 1 = research/experiment, day 2 = iteration.

## 4. CLI Commands

### `expflow analyze` Command Group

```bash
# Strategic advising (primary CLI entry point)
expflow analyze advise

# Per-task deep analysis
expflow analyze task task1
expflow analyze task task3

# PDE equation listing
expflow analyze equations                # All equations
expflow analyze equations --task competition  # Only competition
expflow analyze equations kuramoto_sivashinsky  # Single equation

# Loss function catalog (expflow v0.4.0+)
expflow analyze losses                   # List all 8 PDE loss functions
expflow analyze losses h1_1d             # Show details + params

# Competition overview
expflow analyze status
```

### Output Format Guidelines

All analyze CLI output uses:
- Unicode box-drawing where safe (`─` separators)
- Emoji icons for status/difficulty: 🟢 🟡 🔴 🔥 ⚪
- Priority strip at the right edge
- LaTeX as raw text (not rendered)
- Collaborative scoring formulas on separate lines

## 5. Testing Patterns

### Test Coverage Requirements

| Test area | Min tests | What to cover |
|-----------|:---------:|---------------|
| `list_task_summaries()` | 5 | Returns 3 tasks, required keys, valid IDs, positive scores |
| `analyze_task(task_id)` | 6 | Each task details, unknown task, bottlenecks, ceilings |
| `estimate_score_potential()` | 3 | All 3 tasks, optimistic>expected>conservative |
| `get_strategic_recommendation()` | 2 | Returns valid recommendation, schedule has day keys |
| `get_equation_analysis()` | 4 | Burgers, KS, unknown, non-competition |
| `list_all_equations_summary()` | 2 | Contains known equations, has required fields |

### Key Assertion Patterns

```python
# Task metadata integrity
assert r["max_score"] == 150
assert r["difficulty"] == "medium"
assert r["current_best"] == 57.09  # Known baseline value

# Score projection ordering
assert e["optimistic"] > e["expected"] > e["conservative"]

# Strategy return shape
assert r["primary_focus"] in ("task1", "task2", "task3")
assert "_" in list(r["suggested_schedule"].keys())[0]
```

## 6. Integration with Other Systems

### With experiment-lifecycle-governance

The `compare-scores` gating from governance builds on equation metrics from this system. When adding a new equation, its metrics must exist in `STANDARD_METRICS` for gating to work.

### With analyze-experiment-autoregressive-degradation

This system's `analyze advise` recommends which model/experiment to debug next. The degradation analysis skill provides the diagnostic methodology. Chain: `analyze advise → decide task → run experiment → analyze-experiment-autoregressive-degradation → feed results back to _TASK_META`.

### With wiki

Equation registry should stay in `expflow_pde/equations.py` (LLM-readable). Wiki pages are for human reference with richer prose. Keep them synced:
- `entities/<equation-name>.md` — human docs with full LaTeX, datasets, competition info
- entities/burgers-equation.md links to competition-scoring-rules.md

## Pitfalls

### 1. _TASK_META Becomes Stale

The `_TASK_META` dict contains hardcoded score estimates, bottlenecks, and strategies. As experiments progress, these MUST be updated:
- `current_best_seg` and `current_best_total` after each submission
- `key_bottlenecks` as problems are solved
- `proven_strategies` as new methods work
- `estimated_ceiling` as understanding deepens

**Without updates, `expflow analyze advise` gives misleading recommendations.**

### 2. Competition Deadline Hardcoded

The `get_strategic_recommendation()` function has `remaining_days` calculated from a hardcoded deadline (`2026-05-27 14:00 UTC+8`). For future competitions, this needs to be a parameter or config value.

### 3. Equation Registry ≠ Exhaustive

The COMPETITION equations (task1, task2, task3 in `_TASK_EQUATIONS`) are monitored. The other 8 benchmark equations (advection, darcy, etc.) are for reference only — their `competition_task` is `None`.

### 4. Scoring Formula Duplication

Task 3 scoring formulae (plan_a, plan_b) are stored in both `equations.py` (kuramoto_sivashinsky.competition_info) and `analyze.py` (_TASK_META.task3). Keep them in sync — the equations.py version is the authoritative source.

### 5. Multiprocess numpy/GIL

Do NOT import clearml in the analyze module (or equations module). analyze.py uses only `expflow_pde.equations` (pure Python/stdlib). This ensures fast CLI startup even without GPU stack loaded.

## Related Wiki Pages

- `~/wiki/entities/kuramoto-sivashinsky-equation.md` — KS equation full doc
- `~/wiki/entities/burgers-equation.md` — Burgers equation + task mapping
- `~/wiki/concepts/competition-scoring-rules.md` — Task 3 scoring rules
- `~/wiki/entities/pdebench.md` — Competition Tasks Overview table
