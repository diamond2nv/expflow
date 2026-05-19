---
name: experiment-lifecycle-governance
title: Experiment Lifecycle Governance — PIN, Metrics Registry, Compare-Scores, Audit
description: Add governance to experiment workflows — PIN-protected destructive ops, standardized metrics registry with thresholds, compare-scores ranking with gating, and competition rules audit. Builds on clearml-agent-dispatch and fysom-fsm-integration.
category: mlops
tags: [governance, pin, metrics, compare, audit, guard, competition, safety]
created: 2026-05-19
updated: 2026-05-19
---

# Experiment Lifecycle Governance

## Overview

Governance layer for experiment workflows: protect destructive operations, standardize metrics, rank experiments with gating, and audit against competition rules.

Three sub-systems:
1. **PIN Protection** — 4-digit PIN guard for cancel/stop/delete operations
2. **Metrics Registry** — Standardized metric definitions with thresholds
3. **Compare-Scores** — Multi-model ranking with gating

## 1. PIN Protection Pattern

### Architecture

```
~/.expflow/pin.hash         # SHA-256 hash of 4-digit PIN (never plaintext)
~/.expflow/experiments.jsonl # Experiment registry (each line = JSON record)
```

### Module Design

```python
# pin.py — 4 components:
# 1. init_pin(pin: str) -> hash          # Validate + hash + write
# 2. verify_pin(pin: str) -> bool         # Hash comparison
# 3. pin_is_set() -> bool                 # Check if PIN configured
# 4. guard(action_description) -> bool    # Interactive prompt

# sha256 hash — never store raw PIN
def _hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()

# Validate exactly 4 digits
def _validate_pin(pin: str) -> None:
    if not pin.isdigit() or len(pin) != 4:
        raise ValueError("PIN must be exactly 4 digits (0-9)")
```

### File Storage

PIN hash goes to `~/.expflow/pin.hash` (NOT config.yaml, to avoid git-committing hashes):

```python
def _write_pin_hash(pin_hash: str | None) -> None:
    path = _pin_hash_path()  # os.path.join(_get_pin_dir(), "pin.hash")
    if pin_hash is None:
        if os.path.isfile(path): os.remove(path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f: f.write(pin_hash + "\n")
```

### Test Isolation for PIN

Use a module-level `_PIN_DIR` variable that tests can override:

```python
_PIN_DIR: str | None = None  # Set to tmp_path in tests

def _get_pin_dir() -> str:
    if _PIN_DIR is not None: return _PIN_DIR
    return os.path.expanduser("~/.expflow")
```

Fixtures use `monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))`.

### CLI Integration

```python
@run_app.command("cancel")
def cancel_cmd(experiment_id: str, force: bool = typer.Option(False, "--force", "-f")):
    from expflow_pde.pin import guard
    if not force:
        if not guard(f"cancel experiment {experiment_id}"):
            raise typer.Exit(code=1)
    # proceed with cancellation...
```

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| SHA-256, not plaintext | Even if pin.hash leaks, PIN can't be reversed |
| `~/.expflow/pin.hash`, not config.yaml | Avoid accidental git commit of hash |
| `--force` bypass | Script/CI pipelines can skip interactive prompt |
| `guard()` returns bool, not raise | Composable — integrate into any CLI command |
| No PIN = no guard | Zero friction for new users |
| `guard()` uses `getpass.getpass()` | Hidden input for security |

## 2. Standardized Metrics Registry

### Structure

```python
STANDARD_METRICS = {
    "seg_total": {
        "type": "scalar", "group": "Score",
        "higher_is_better": True,
        "description": "Total segment score (primary competition metric)",
    },
    "pde_mean": {
        "type": "scalar", "group": "PDE",
        "higher_is_better": False,
        "threshold": 18.09,  # Competition gate
    },
    "train_time_min": {
        "type": "scalar", "group": "Time",
        "higher_is_better": False,
        "threshold": 60,  # Competition limit
    },
    # New in v0.4.0 — PDEBench 6-metric suite
    "val_rmse":       {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_nrmse":      {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_max_err":    {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_bd_err":     {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_csv_err":    {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_fourier_low":{"type": "scalar", "group": "Fourier", "higher_is_better": False},
    "val_fourier_mid":{"type": "scalar", "group": "Fourier", "higher_is_better": False},
    "val_fourier_high":{"type": "scalar", "group": "Fourier", "higher_is_better": False},
    # New in v0.4.0 — HyperNOs-style training losses
    "val_lprel":      {"type": "scalar", "group": "Loss", "higher_is_better": False},
    "val_h1rel":      {"type": "scalar", "group": "Loss", "higher_is_better": False},
    # ... 21+ total metrics across Score/Loss/Error/Fourier/PDE/Time/Model/Training groups
}
```

### report_standard() — Training Script Integration

```python
def report_standard(task: Any | None = None, **kwargs: float) -> dict[str, float]:
    """Report metrics to clearml or return dict if no task.
    
    Args:
        task: Optional clearml Task instance.
        **kwargs: Metric name=value pairs (unknown names are logged as warning, not error).
    """
    reported = {}
    for name, value in kwargs.items():
        info = STANDARD_METRICS.get(name)
        if info is None:
            raise ValueError(f"Unknown metric '{name}'...")
        reported[name] = float(value)
        if task is not None:
            task.report_scalar(title=info["group"], series=name, value=float(value), iteration=0)
    return reported
```

### validate_metric_threshold()

```python
def validate_metric_threshold(name: str, value: float) -> dict:
    """Returns {name, value, threshold, passed, detail}.
    Unknown metrics and metrics without thresholds always pass.
    """
```

## 3. Compare-Scores: Multi-Model Ranking

### Logic

```python
def compare_scores(
    project="PDEBench", tags=None,
    sort_by="seg_total", ascending=False,
    gates=None, max_results=20,
) -> list[dict]:
    # 1. Fetch clearml tasks by project + tags
    # 2. For each task, get_last_scalar_metrics() → flatten
    # 3. Apply gates (metric:op:value, e.g. pde_mean:lt:18.09)
    # 4. Sort by sort_by metric
    # 5. Return top N
```

### Gate Format

Gates use metric:op:value triplets:
- `pde_mean:lt:18.09` — PDE mean < 18.09
- `train_time_min:le:60` — Training time ≤ 60 min
- `seg_total:ge:50` — Score ≥ 50

Support operators: `lt`, `le`, `gt`, `ge`.

### CLI

```bash
expflow clearml compare-scores \
    --project PDEBench --tags task1 \
    --sort-by pde_mean --ascending \
    --gate pde_mean:lt:18.09 --gate train_time_min:lt:60
```

Output:
```
Rank  ID           Name                     pde_mean    Gates
 1    a1b2c3d4     P2+sub5                 18.09       ✓
 2    e5f6g7h8     P2+sub5+n2000           27.58       ✗ | pde_mean=27.58
```

### MCP Tool

```python
@mcp.tool()
def exp_compare_scores(project, tags, sort_by, ascending, gates, max_results):
    """Rank experiments by metric score with optional gating."""
    return compare_scores(...)
```

## 4. Workers Command

```bash
expflow clearml workers

NAME                          STATUS       QUEUE            GPUS         IP
5090-node-gpu0                online       gpu_queue        1            10.15.8.93
3080-node-gpu0                online       default          1            10.15.8.94
```

Implemented via `clearml.Worker.get_workers()`.

## 5. Competition Rules Audit

### CLI

```bash
expflow audit validate exp-001 --competition-rules --task-id abc123
```

### Python API (`audit.validate_competition_rules()`)

```python
from expflow_pde.audit import validate_competition_rules

result = validate_competition_rules(
    task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5},
    task_params={"Args/--sub_step": "5"},
)

result["all_pass"]  # True/False
result["checks"]    # List of per-rule dicts
```

Function signature:
- `task_metrics: dict[str, float]` — Flat metric names → values
- `task_params: dict[str, str] | None` — clearml task parameters (for sub_step check)

Return dict:
```
{
  "all_pass": bool,
  "checks": [  # 4 items
    {"name": "seg_total", "label": "...", "value": 57.09, "passed": True, "detail": "57.09 (rule: >= 0)"},
    {"name": "pde_mean", "label": "...", "value": 15.0, "passed": True, "detail": "15.0 (rule: <= 18.09)"},
    {"name": "train_time_min", "label": "...", "value": 45.5, "passed": True, "detail": "45.5 (rule: <= 60)"},
    {"name": "sub_step", "label": "...", "value": True, "passed": True, "detail": "present and > 0"},
  ],
  "metrics": {"seg_total": ..., "pde_mean": ..., "train_time_min": ...},
}
```

Checks:
| Check | Condition | Details |
|-------|-----------|---------|
| `seg_total` | Primary competition score (reported, no gating) | Uses STANDARD_METRICS threshold if defined |
| `pde_mean` | Must be < 18.09 | Threshold from STANDARD_METRICS, higher_is_better=False |
| `train_time_min` | Must be < 60 | Threshold from STANDARD_METRICS, higher_is_better=False |
| `sub_step` parameter | Must exist and be > 0 | Searches case-insensitive for "sub_step" or "substep" in params |

### Function Location

The core validation logic lives in `audit.validate_competition_rules()` — **not** in the CLI layer. The CLI (`cli_audit.py`) is just a thin wrapper that fetches clearml data and prints results.

```python
# expflow_pde/audit.py — standalone, no clearml dependency
def validate_competition_rules(
    task_metrics: dict[str, float],
    task_params: dict[str, str] | None = None,
) -> dict[str, Any]:
```

Return dict:
```python
{
  "all_pass": bool,        # ALL checks passed
  "checks": [...],          # 4 items: seg_total, pde_mean, train_time_min, sub_step
  "metrics": {...},        # Original input metrics (pass-through)
}
```

### CLI helper functions (in `cli_audit.py`)

```python
_get_task_metrics(task_id: str) -> dict[str, float]
# Fetches clearml Task scalars, flattens nested dict

_get_task_params(task_id: str) -> dict[str, str] | None
# Fetches clearml Task parameters (for sub_step check)
```

### Test Coverage

8 tests in `TestValidateCompetitionRules` class (in `tests/test_audit.py`):
- All rules pass (happy path)
- PDE mean exceeds threshold
- Training time exceeds limit
- Missing metric (all_pass=False)
- Missing sub_step (all_pass=False)
- sub_step=0 fails
- Empty params dict fails sub_step
- Result includes original metrics

### Usage Pattern

```bash
# Via CLI (fetches from clearml automatically)
expflow audit validate <exp_id> --competition-rules --task-id <clearml_task_id>

# Via Python API (standalone, no clearml needed)
from expflow_pde.audit import validate_competition_rules

result = validate_competition_rules(
    task_metrics={"seg_total": 57.09, "pde_mean": 15.0, "train_time_min": 45.5},
    task_params={"Args/--sub_step": "5"},
)
```

## 6. Queue Status Command

```bash
expflow clearml queue-status <queue_name>
```

Shows queue depth, pending count, and running count. Uses `clearml.get_queue_status()`.

## Dependencies

```toml
# pyproject.toml — no new deps for PIN (uses stdlib hashlib/getpass)
# metrics uses stdlib only
# compare-scores uses clearml SDK (lazy import)
```

## Testing Patterns

### PIN Tests
- Hash consistency: same input → same hash
- Validation rejects: wrong length, non-numeric, empty, special chars
- Init → file exists with correct hash
- Clear → file removed
- Verify: correct → True, incorrect → False, no PIN → False
- Guard mock: correct → True, quit → False, KeyboardInterrupt → False

### Metrics Tests
- Registry structure: each metric has type, group, higher_is_better
- report_standard: returns dict of reported metrics
- report_standard unknown metric raises ValueError
- validate_metric_threshold: passing/failing/borderline/none

### Compare Tests
- _apply_gate: all 4 operators (lt/le/gt/ge) with passing and failing cases
- Unknown operator returns True (pass through)

## Pitfalls

### 1. YAML/Env vs File Storage for PIN

PIN hash must NOT go into `config.yaml` (risk of git commit). Use `~/.expflow/pin.hash` instead. But also support `.env` `EXPFLOW_PIN_HASH` for CI/automated environments.

Precedence: `pin.hash` file > `.env EXPFLOW_PIN_HASH` > `config.yaml pin.hash`.

### 2. `get_last_scalar_metrics()` clearml API

The clearml SDK `task.get_last_scalar_metrics()` returns a nested dict:
```python
{"Score": {"seg_total": {"last": 57.09, "min": ..., "max": ...}}, ...}
```
Flatten to `{"seg_total": 57.09}` for compare_scores.

Import note: this method's availability varies by clearml SDK version. Always use `getattr(task, "get_last_scalar_metrics", lambda: {})()` as fallback.

### 3. Worker.get_workers() Availability

The `clearml.Worker` class may not be available on all clearml SDK versions. Wrap the import in try/except and return empty list on failure.

### 4. numpy Import Error in CI Tests

When clearml IS installed but numpy has C extension loading issues (e.g., after pip multiple-imports warnings), tests that trigger clearml imports will fail with `ImportError: cannot load module more than once per process`. Mark tests requiring clearml as `@pytest.mark.integration`.

### 5. `--force` Flag for Script Calls

Always provide `--force` / `-f` on commands guarded by PIN so CI pipelines and automated scripts can bypass the interactive prompt.

### 6. Interactive `getpass` vs Non-Interactive

`getpass.getpass()` works in terminals but will fail/block in:
- Piped commands (`echo "1234" | expflow pin check`)
- CI environments without TTY
- Hermes Agent subagent calls

Always provide `--pin` flag or `--force` as alternative paths.

## Related Skills

- `clearml-agent-experiment-dispatch` — covers the actual experiment submission flow
- `fysom-fsm-integration` — covers the FSM lifecycle (cancel transitions)
- `clearml-sdk-wrapper-pattern` — covers the three-layer architecture for clearml wrappers
- `competition-task-intelligence` — PDE equation registry, task analysis, and strategic advising (which equations to run, what metrics to track, where to focus)
