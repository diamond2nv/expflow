---
name: expflow-pipeline-hpo
description: >
  PDEBench competition workflow orchestration with expflow —
  three pipeline modes (full/fast/skip), distributed HPO, Pruner integration,
  and Clearml HyperParameterOptimizer native mode.
---

# expflow PDEBench Pipeline & HPO

Orchestrate experiment workflows for the AI4S PDE competition using expflow.
Three modes for three competition phases.

## Triggers

- User says "run HPO", "submit pipeline", "distributed experiment"
- User says "competition sprint" or "fast iterate"
- User asks about automating the train→eval→submit loop
- User mentions needing to find best hyperparams

## Available Pipeline Modes

Three pipeline modes, each mapped to a CLI command:

### Mode A — Full (HPO → Train → Eval)

For the **exploration phase** of a competition task. Optuna finds best params
via distributed clearml-agent trials, trains with best, then evaluates.

```bash
expflow pipeline submit-full train_task1.py \
    --queue default \
    --trials 50 --parallel 4 \
    --eval-script eval_task1.py \
    --metric seg_total --direction maximize
```

Flags used:
- `--trials N`: total HPO trials
- `--parallel M`: max concurrent trials (use GPU node count)
- `--metric`: objective metric name prefixed `METRIC:` in script stdout
- `--pruner hyperband|median|percentile`: early-stop poor trials
- `--study-name`: Optuna study name (auto if omitted; persists to SQLite)
- `--skip hpo --skip eval`: run train only within full skeleton

### Mode B — Fast (Train → Eval)

For the **competition sprint** phase. You already know best params. Skip HPO,
run directly with fixed args.

```bash
expflow pipeline submit train_task1.py \
    --queue default \
    --train-param lr=0.001 --train-param epochs=80 \
    --eval-script eval_task1.py \
    --eval-param sub_step=5
```

Flags:
- `--skip eval`: train-only (just submit checkpoint)
- `--train-param key=val`: injected as `--key=val` to training script
- `--eval-param key=val`: injected as `--key=val` to eval script

### Mode C — Flexible Skip

Override step inclusion on either mode:

```bash
expflow pipeline submit-full train_task1.py \
    --skip hpo --skip eval          # = train only
expflow pipeline submit-full train_task1.py \
    --skip train --skip eval         # = HPO only
```

## HPO: Three Execution Modes

HPO (`expflow optuna run`) has three backends:

| Mode | Flag | Description | Best for |
|------|------|-------------|----------|
| Local | (default) | subprocess serial on CPU | ≤20 trials, quick test |
| Distributed | `--distributed` | ask/tell + clearml Task clone| Multi-GPU, custom control|
| Optimizer | `--optimizer -O` | Clearml `HyperParameterOptimizer` | Production, 50-200+ trials |

Key flags across all HPO modes:
- `--pruner hyperband|median|percentile|none`: ASHA pruner saves ~40% GPU time
- `--metric <name>`: reads `METRIC:<name>=<value>` from script stdout
- `--direction maximize|minimize`
- `--timeout <min>`: safety cutoff
- `--loss <name>` (expflow v0.4.0+): inject `--loss=<name>` into training script (e.g. `l2_rel`, `h1_1d`)

## PDE-Specialized Loss Functions (expflow v0.4.0+)

Starting from expflow v0.4.0, PDE-specific loss functions are integrated
directly into `expflow_pde/losses.py` — use the standardised CLI and API
instead of hand-copying from HyperNOs.

### Available Loss Functions

| CLI Name | Class | Description | Use Case |
|:--------:|:-----:|-------------|----------|
| `l1_rel` | `LprelLoss(p=1)` | Relative L1 norm | Robust to outliers |
| `l2_rel` | `LprelLoss(p=2)` | **Relative L2 (default)** | Competition Rel-MSE squared root |
| `h1_1d` | `H1relLoss_1D` | FFT-weighted Sobolev 1D | Suppress high-freq oscillations |
| `h1_2d` | `H1relLoss` | FFT-weighted Sobolev 2D | 2D PDE smoothness (Navier-Stokes) |
| `mse_rel` | `MSELoss_rel` | Normalised MSE | Simple & effective |
| `smoothl1_rel` | `SmoothL1Loss_rel` | Robust Smooth L1 | Noisy data |
| `l2_abs` | `lpLoss(p=2)` | Absolute L2 baseline | No normalisation |
| `mse_abs` | `nn.MSELoss()` | Standard Torch MSE | Original PDEBench default |

### CLI interaction

```bash
# List all available loss functions
expflow analyze losses

# Show details for a specific loss (params, use case)
expflow analyze losses h1_1d

# Use with HPO — loss name is injected as --loss=<name> to training script
expflow optuna run train_task1.py --loss l2_rel --trials 50 --distributed --queue gpu_queue

# Custom H1 parameters via loss_selector kwargs (from Python)
from expflow_pde.losses import loss_selector
loss_fn = loss_selector("h1_1d", beta=2.0, size_mean=True)
```

### Training Script Integration Pattern

Training scripts should accept `--loss` CLI argument:

```python
from expflow_pde.losses import loss_selector

parser.add_argument("--loss", default="l2_rel", help="Loss function name")
loss_fn = loss_selector(args.loss, size_mean=True)

# During training
output = model(x)
loss = loss_fn(output, y)
loss.backward()
```

### Key Differences from HyperNOs Original

| Aspect | HyperNOs | expflow v0.4.0 |
|--------|----------|----------------|
| Base class | Plain class | `nn.Module` (clearml/optuna compatible) |
| Type annotations | `beartype` + `jaxtyped` | Standard PEP8 type hints |
| Chebyshev variants | 4 classes | ❌ Omitted (not needed for Burgers) |
| Multi-patch variants | 3 classes | ❌ Omitted |
| Multi-output variants | 3 classes | Omitted (single-channel focused) |
| `size_mean` default | `False` (sum) | `True` (mean) — matches typical ML practice |
| Division-by-zero guard | Raises `ValueError` | Clamps to 1e-7, silently continues |

### Legacy HyperNOs Library

If you need loss functions not provided by expflow (Chebyshev variants,
multi-patch, physics residuals), the original HyperNOs source is at
`~/Gitlab/Agentic4Sci/HyperNOs/hypernos/loss_fun.py` (17 classes).

## Script Requirements

The training/eval script must:
1. Accept hyperparams as `--key=value` CLI arguments
2. Output `METRIC:<name>=<value>` to stdout for objective capture (local mode)
3. Report clearml scalars for distributed/optimizer mode:
   ```python
   Task.current_task().report_scalar("Score", "seg_total", value, iteration=epoch)
   ```

## Pitfalls

- **Pruner needs `trial.report()` calls during training.** If the script only
  reports at the end, the pruner has nothing to prune on. Call
  `trial.report(val_loss, epoch)` at least every 10 epochs.
- **HyperParameterOptimizer needs the metric name in `Title/Series` format.**
  If your metric is `seg_total`, it becomes `title=seg_total, series=seg_total`.
  If your clearml report_scalar is `report_scalar("Score", "seg_total", v)`,
  pass `--metric Score/seg_total`.
- **Clearml-agent must be running on GPU nodes** before submitting. Verify
  with `expflow clearml workers` or check Web UI at `clear-ml-internal-server:8080`.
- **In pipeline mode, don't expose `@optuna_app.command("run")` as `def hpo_run_cmd`**
  — ensures Typer registers it under `optuna run` subcommand.
- **Clearml Task.create with `add_task_init_call=True`** is zero-modification
  submission — the SDK auto-injects `Task.init()` into the script.
- **`_collect_one_trial` polls every 5s** — waits up to 60min per trial.
  If trials are expected to run longer, increase `timeout_minutes`.

## Architecture Reference

See `~/Gitlab/Agentic4Sci/expflow/PLAN_v2.md` Section I-J for design doc.

Key files:
- `expflow_pde/hpo.py` — 3-mode HPO runner (local/distributed/optimizer)
- `expflow_pde/pipeline.py` — ExperimentPipeline class (fast/full modes)
- `expflow_pde/cli_pipeline.py` — `pipeline submit` + `pipeline submit-full`
- `expflow_pde/cli_optuna.py` — `optuna run` with all three backends

## Related

- `agentic4sci-system-report` — overall Agentic4Sci architecture diagrams
- `pde-experiment-hyperparameters` — PDEBench-specific hyperparameter reference
- `multi-agent-distributed-experiment-workflow` — Hermes → OpenCode → clearml
- `experiment-lifecycle-governance` — PIN, metrics registry, compare-scores, competition rules audit
- `references/pypi-readiness-checklist.md` — PyPI publication assessment criteria and preflight steps

## Langfuse ↔ expflow Integration

### Trace ID & Session ID Architecture

Two types of IDs bridge clearml and Langfuse:

| ID | Source | Pre-computable? | Format |
|----|--------|:--------------:|--------|
| Langfuse trace_id | Langfuse server (auto) | ❌ Created on `client.trace()` | Server-assigned string |
| Langfuse session_id | User-defined string | ✅ Yes | `my_hpo_run` or auto `exp:snow_<id>` |
| clearml task_id | clearml server (auto) | ❌ Created on `Task.create()` | Server-assigned string |

### Session ID: Three-Tier Fallback

`trace_experiment()` resolves the session_id in this order:

```
Tier 1 — Explicit: caller passes --session <id>                  (highest priority)
Tier 2 — Inheritance: parent clearml Task's metadata             (keeps child under same session)
         expflow:langfuse_session_id from parent_task_id
Tier 3 — Auto-generate: expflow snowflake ID prefixed exp:snow_  (every trace gets a session)
```

```python
# Implementation in expflow_pde/langfuse.py
def _resolve_session_id(
    session_id: str | None = None,
    parent_task_id: str | None = None,
) -> str:
```

### Bidirectional Linking

```python
# trace_experiment writes two-directional metadata:
# 1. clearml Task metadata:
#    expflow:langfuse_trace_id        → "lf_abc123"
#    expflow:langfuse_session_id       → "exp:snow_9876543210"
#    expflow:langfuse_parent_trace_id  → "lf_hermes_xyz" (if parent_trace_id given)
#
# 2. Langfuse trace metadata:
#    "source": "expflow"
#    "task_id": "cm_xyz"
#    "parent_trace_id": "lf_hermes_xyz" (if given)
```

### CLI Usage

```bash
# Auto session (Tier 3) — every experiment gets a Langfuse session
expflow langfuse trace-experiment cm_abc123

# Explicit session (Tier 1)
expflow langfuse trace-experiment cm_abc123 --session pdebench:hpo_v2

# Inherit from parent clearml Task (Tier 2)
expflow langfuse trace-experiment child_task --parent-task-id parent_task

# Link to Hermes Agent decision trace
expflow langfuse trace-experiment cm_abc123 --parent-trace lf_hermes_xyz
```

### Snowflake ID Generator

The snowflake ID generator lives in `expflow_pde/snowflake.py` — ported from hfpapers-crawler's yitter snowflake drift algorithm (same M1 implementation). Thread-safe, time-rollback-tolerant.

- worker_id=1 reserved for expflow
- worker_id_bit_length=6, seq_bit_length=6
- base_time aligned with hfpapers (2024-10-04) for cross-tool consistency
- Output format: `exp:snow_<19-digit-int>` (~30 chars, well within Langfuse's <200-char session_id limit)
