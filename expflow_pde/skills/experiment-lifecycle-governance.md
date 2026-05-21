---
name: experiment-lifecycle-governance
description: Generated from skills/experiment-lifecycle-governance/SKILL.md (package reference copy)
---

     1|---
     2|name: experiment-lifecycle-governance
     3|title: Experiment Lifecycle Governance — PIN, Metrics Registry, Compare-Scores, Audit
     4|description: Add governance to experiment workflows — PIN-protected destructive ops, standardized metrics registry with thresholds, compare-scores ranking with gating, and competition rules audit. Builds on clearml-agent-dispatch and fysom-fsm-integration.
     5|category: mlops
     6|author: Li Shen
     7|version: 1.0.0
     8|tags: [governance, pin, metrics, compare, audit, guard, competition, safety]
     9|metadata:
    10|  hermes:
    11|    tags: [mlops, pde, governance, clearml, experiment, audit, safety]
    12|    homepage: https://github.com/diamond2nv/expflow
    13|    related_skills: [expflow-pipeline-hpo, clearml-metrics-logging-pattern, competition-task-intelligence]
    14|---
    15|
    16|# Experiment Lifecycle Governance
    17|
    18|## Overview
    19|
    20|Governance layer for experiment workflows: protect destructive operations, standardize metrics, rank experiments with gating, and audit against competition rules.
    21|
    22|Three sub-systems:
    23|1. **PIN Protection** — 4-digit PIN guard for cancel/stop/delete operations
    24|2. **Metrics Registry** — Standardized metric definitions with thresholds
    25|3. **Compare-Scores** — Multi-model ranking with gating
    26|
    27|## Installation
    28|
    29|```bash
    30|pip install expflow-pde
    31|```
    32|
    33|## 1. PIN Protection Pattern
    34|
    35|### Architecture
    36|
    37|```
    38|~/.expflow/pin.hash          # SHA-256 hash of 4-digit PIN (never plaintext)
    39|~/.expflow/experiments.jsonl # Experiment registry (each line = JSON record)
    40|```
    41|
    42|### Module Design
    43|
    44|```python
    45|# pin.py — 4 components:
    46|# 1. init_pin(pin: str) -> hash          # Validate + hash + write
    47|# 2. verify_pin(pin: str) -> bool         # Hash comparison
    48|# 3. pin_is_set() -> bool                 # Check if PIN configured
    49|# 4. guard(action_description) -> bool    # Interactive prompt
    50|
    51|# sha256 hash — never store raw PIN
    52|def _hash_pin(pin: str) -> str:
    53|    return hashlib.sha256(pin.encode()).hexdigest()
    54|
    55|# Validate exactly 4 digits
    56|def _validate_pin(pin: str) -> None:
    57|    if not pin.isdigit() or len(pin) != 4:
    58|        raise ValueError("PIN must be exactly 4 digits (0-9)")
    59|```
    60|
    61|### CLI Commands
    62|
    63|```bash
    64|expflow pin init 1234          # Set PIN (SHA-256 stored)
    65|expflow pin check              # Interactive verify
    66|expflow pin clear [--force]    # Remove PIN
    67|expflow pin status             # Show if active
    68|
    69|# Guarded commands (require PIN unless --force):
    70|expflow run cancel <id>            # Interactive PIN prompt
    71|expflow run cancel <id> --force    # Skip PIN
    72|```
    73|
    74|## 2. Standardized Metrics Registry
    75|
    76|### Structure
    77|
    78|```python
    79|STANDARD_METRICS = {
    80|    "seg_total": {
    81|        "type": "scalar", "group": "Score",
    82|        "higher_is_better": True,
    83|        "description": "Total segment score (primary competition metric)",
    84|    },
    85|    "pde_mean": {
    86|        "type": "scalar", "group": "PDE",
    87|        "higher_is_better": False,
    88|        "threshold": 18.09,  # Competition gate
    89|    },
    90|    "train_time_min": {
    91|        "type": "scalar", "group": "Time",
    92|        "higher_is_better": False,
    93|        "threshold": 60,  # Competition limit
    94|    },
    95|    # ... 13 total metrics across Score/Loss/PDE/Time/Model/Training groups
    96|}
    97|```
    98|
    99|### report_standard()
   100|
   101|```python
   102|def report_standard(task: Any | None = None, **kwargs: float) -> dict[str, float]:
   103|    reported = {}
   104|    for name, value in kwargs.items():
   105|        info = STANDARD_METRICS.get(name)
   106|        if info is None:
   107|            raise ValueError(f"Unknown metric '{name}'...")
   108|        reported[name] = float(value)
   109|        if task is not None:
   110|            task.report_scalar(title=info["group"], series=name, value=float(value), iteration=0)
   111|    return reported
   112|```
   113|
   114|## 3. Compare-Scores: Multi-Model Ranking
   115|
   116|### CLI
   117|
   118|```bash
   119|expflow clearml compare-scores \
   120|    --project PDEBench --tags task1 \
   121|    --sort-by pde_mean --ascending \
   122|    --gate pde_mean:lt:18.09 --gate train_time_min:lt:60
   123|```
   124|
   125|### Gate Format
   126|
   127|Gates use `metric:op:value` triplets:
   128|- `pde_mean:lt:18.09` — PDE mean < 18.09
   129|- `train_time_min:le:60` — Training time ≤ 60 min
   130|- `seg_total:ge:50` — Score ≥ 50
   131|
   132|Operators: `lt`, `le`, `gt`, `ge`.
   133|
   134|## 4. Competition Rules Audit
   135|
   136|### CLI
   137|
   138|```bash
   139|expflow audit validate exp-001 --competition-rules --task-id abc123
   140|```
   141|
   142|### Python API
   143|
   144|```python
   145|from expflow_pde.audit import validate_competition_rules
   146|
   147|result = validate_competition_rules(
   148|    task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5},
   149|    task_params={"Args/--sub_step": "5"},
   150|)
   151|print(f"All pass: {result['all_pass']}")
   152|```
   153|
   154|### Validation Checks
   155|
   156|| Check | Condition | Details |
   157||-------|-----------|---------|
   158|| `seg_total` | Primary competition score (no gating) | Reported, not gated |
   159|| `pde_mean` | Must be < 18.09 | Threshold from STANDARD_METRICS |
   160|| `train_time_min` | Must be < 60 | Threshold from STANDARD_METRICS |
   161|| `sub_step` parameter | Must exist and be > 0 | Searches case-insensitive |
   162|
   163|## Testing Patterns
   164|
   165|### PIN Tests (36 tests)
   166|- Hash consistency: same input → same hash
   167|- Validation rejects: wrong length, non-numeric, empty
   168|- Init → file exists with correct hash
   169|- Guard mock: correct → True, quit → False
   170|
   171|### Metrics Tests
   172|- Registry structure: each metric has type, group, higher_is_better
   173|- report_standard: returns dict of reported metrics
   174|
   175|### Compare Tests
   176|- _apply_gate: all 4 operators (lt/le/gt/ge) with passing and failing cases
   177|
   178|## Pitfalls
   179|
   180|### 1. YAML/Env vs File Storage for PIN
   181|PIN hash must NOT go into `config.yaml` (risk of git commit). Use `~/.expflow/pin.hash`.
   182|Precedence: `pin.hash` file > `.env EXPFLOW_PIN_HASH` > `config.yaml pin.hash`.
   183|
   184|### 2. `get_last_scalar_metrics()` clearml API
   185|Returns nested dict: `{"Score": {"seg_total": {"last": 57.09, ...}}, ...}`. Flatten to `{"seg_total": 57.09}` for compare_scores.
   186|
   187|### 3. `--force` Flag for Script Calls
   188|Always provide `--force` / `-f` on PIN-guarded commands for CI/automation.
   189|
   190|### 4. Interactive `getpass` vs Non-Interactive
   191|`getpass.getpass()` works in terminals but fails in piped commands, CI, or subagent calls. Always provide `--pin` or `--force` as alternative paths.
   192|
