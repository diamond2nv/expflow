---
name: competition-task-intelligence
description: Generated from skills/competition-task-intelligence/SKILL.md (package reference copy)
---

     1|---
     2|name: competition-task-intelligence
     3|title: Competition Task Intelligence — PDE Equation Registry, Task Analysis, and Strategic Advising
     4|description: >
     5|  Build and maintain a structured PDE equation registry, analyze competition tasks
     6|  (difficulty, bottlenecks, score projections), generate strategic recommendations
     7|  for research focus, and expose this intelligence via CLI and MCP tools.
     8|category: mlops
     9|author: Li Shen
    10|version: 1.0.0
    11|tags: [mlops, competition, strategy, equations, analysis, planning, pde, task-intelligence]
    12|metadata:
    13|  hermes:
    14|    tags: [mlops, pde, competition, strategy, equations, analysis, ai4s]
    15|    homepage: https://github.com/diamond2nv/expflow
    16|    related_skills:
    17|      - expflow-pipeline-hpo
    18|      - experiment-lifecycle-governance
    19|      - clearml-metrics-logging-pattern
    20|      - agent4pde-competition-scoring
    21|      - pde-experiment-hyperparameters
    22|created: 2026-05-19
    23|updated: 2026-05-19
    24|---
    25|
    26|# Competition Task Intelligence
    27|
    28|## Overview
    29|
    30|System for structured PDE equation management and competition task analysis. Provides:
    31|
    32|1. **PDE Equation Registry** — structured metadata (LaTeX, dimensions, params, datasets) for 11+ PDEs
    33|2. **Task Analysis** — per-task difficulty assessment, bottleneck identification, proven strategy catalog
    34|3. **Score Projection** — optimistic/expected/conservative score estimates with confidence levels
    35|4. **Strategic Advising** — which task to focus on, suggested schedule, rationale
    36|5. **CLI + MCP** — `expflow analyze` command group and MCP tools
    37|
    38|## Installation
    39|
    40|```bash
    41|pip install expflow-pde
    42|```
    43|
    44|## Architecture
    45|
    46|```
    47|expflow_pde/equations.py     ──── PDE equation static registry (11+ equations)
    48|expflow_pde/analyze.py       ──── Analysis engine (task intelligence, strategy)
    49|expflow_pde/cli_analyze.py   ──── CLI: analyze task/equations/status/advise
    50|expflow_pde/mcp_server.py    ──── MCP: exp_compare_scores, exp_list_workers
    51|```
    52|
    53|## 1. PDE Equation Registry
    54|
    55|Each equation entry in `EQUATIONS` dict includes: full name, LaTeX, dimensions, parameters, competition task mapping, metrics, solver, data samples, and competition info.
    56|
    57|### API
    58|
    59|```python
    60|from expflow_pde.equations import (
    61|    get_equations(),                    # All 11+ equations
    62|    get_equation(name),                 # Single equation
    63|    list_equations_for_task(task_id),   # task1/task2/task3
    64|    get_equation_metrics(name, task),   # Relevant STANDARD_METRICS
    65|    list_equation_names(),              # Sorted names
    66|    list_competition_equations(),       # Only competition equations
    67|)
    68|```
    69|
    70|## 2. Task-Level Intelligence
    71|
    72|### CLI
    73|
    74|```bash
    75|# Strategic advising (primary entry point)
    76|expflow analyze advise
    77|
    78|# Per-task analysis
    79|expflow analyze task task1
    80|expflow analyze task task3
    81|
    82|# Equation reference
    83|expflow analyze equations --task competition
    84|
    85|# Competition overview
    86|expflow analyze status
    87|```
    88|
    89|### Example Output
    90|
    91|```
    92|expflow analyze status
    93|
    94|Task     Score              Difficulty     Status         Priority
    95|  ────────────────────────────────────────────────────────────────────
    96|  task1    142/150            🟡 medium       🔴 In Progress  high
    97|  task2    -/150              🔴 hard         ⚪ Not Started  low
    98|  task3    -/350              🔥 very_hard    ⚪ Not Started  medium
    99|
   100|  总分: 142/650  (508 pts remaining)
   101|```
   102|
   103|### Score Estimation
   104|
   105|```python
   106|from expflow_pde.analyze import estimate_score_potential, get_strategic_recommendation
   107|
   108|estimates = estimate_score_potential("task1")
   109|# Returns: {"optimistic": 148, "expected": 145, "conservative": 140, "confidence": "high"}
   110|
   111|rec = get_strategic_recommendation()
   112|# Returns: {"primary_focus": "task1", "remaining_headroom": {...}, "suggested_schedule": {...}}
   113|```
   114|
   115|### Difficulty Classification
   116|
   117|| Label | Icon | Example | Meaning |
   118||-------|:----:|---------|---------|
   119|| easy | 🟢 | Baseline tasks | High confidence, proven methods exist |
   120|| medium | 🟡 | Task 1 | Known bottlenecks, clear path forward |
   121|| hard | 🔴 | Task 2 | Multiple unknown challenges |
   122|| very_hard | 🔥 | Task 3 (KS) | Chaotic dynamics, exponential error growth |
   123|
   124|## Integration with Other Systems
   125|
   126|### With experiment-lifecycle-governance
   127|`compare-scores` gating builds on equation metrics from this system. When adding a new equation, its metrics must exist in `STANDARD_METRICS` for gating to work.
   128|
   129|### With analyze-experiment-autoregressive-degradation
   130|Chain: `analyze advise → decide task → run experiment → analyze degradation → feed back to _TASK_META`.
   131|
   132|## Pitfalls
   133|
   134|1. **`_TASK_META` becomes stale** — hardcoded scores must be updated after each submission
   135|2. **Competition deadline hardcoded** — `get_strategic_recommendation()` has `remaining_days` from `2026-05-27`
   136|3. **Scoring formula duplication** — Task 3 formulae are in both `equations.py` and `analyze.py`; keep synced
   137|4. **No clearml import in analyze** — `analyze.py` uses only pure Python/stdlib for fast CLI startup
   138|
