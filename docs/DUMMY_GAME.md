# expflow-pde Dummy Experiment Game

> **Test the entire experiment lifecycle — diagnose, suggest, submit, fail, repair, iterate — without GPUs, ClearML, Optuna, or torch.**

## Overview

The Dummy Experiment Game is a **zero-dependency, fully self-contained simulation** of the expflow experiment loop. It replaces real ML training (which needs GPUs, ClearML server, and Optuna databases) with a synthetic model that produces plausible seg scores from hyperparameter changes, injects realistic failure modes, and records every step into `DispatchDB` just like real experiments.

This serves three purposes:

| Purpose | What it proves |
|---------|---------------|
| **System integration test** | diagnose → suggest → submit → fail → repair → iterate: does the full pipeline work end-to-end? |
| **Repair verification** | Does L0 catch `git_not_found`? Does L1 extract CUDA OOM traceback? Does L2 produce reflection context? |
| **Onboarding / demo** | Show how expflow works to a new user without any infrastructure — just `pip install` and run. |

## Quick Start

```bash
# Start a new game (task1, Burgers scenario)
expflow dummy start --task task1 --seed 42

# Run one iteration step
expflow dummy step

# Run step with a failure injection (tests repair)
expflow dummy step --inject git_not_found

# Run step simulating a "suggested fix" from Hermes
expflow dummy step --params '{"n_modes": 20, "num_sub_steps": 5}'

# Check game state
expflow dummy status

# Run fully automated loop (diagnose → suggest → step → repair)
expflow dummy auto --max-steps 10 --repair
```

## How It Works

### Simulation Model

The game maintains an internal state of `{seg1, seg2, seg3}` scores per step.

- **Baseline**: Starts at `{55, 30, 20}` for task1 (Burgers FNO).
- **Effects**: Each hyperparameter change affects seg scores additively:

| Parameter Change | seg1 | seg2 | seg3 | Models |
|-----------------|:----:|:----:|:----:|--------|
| `n_modes +4`    |  0   | +3   | +8   | More Fourier modes capture high-frequency dynamics |
| `num_sub_steps=5` | +2 | +3 | +5 | Finer temporal resolution fixes dt mismatch |
| `lr ×2`        | +5   | -2   | -1   | Higher LR helps short-term but hurts long-term stability |
| `stability_lambda=0.001` | -1 | +6 | 0 | Stability penalty suppresses mid-term drift |
| `width +16`    | +2   | +2   | +3   | Wider network increases capacity |
| `weight_decay=1e-4` | 0 | +1 | +3   | Regularisation improves long-term generalisation |
| `epochs +40`   | +1   | +1   | +2   | More training helps convergence |

- **Noise**: Each step adds Gaussian noise (±2 stdev) to simulate experiment variance.
- **Ceiling**: Scores are capped at per-task maxima (`{70, 60, 45}` for task1, `{25, 18, 12}` for task3) to model the diminishing-returns nature of real ML improvement.

### Task Profiles

| Task | Baseline | Ceiling | Characteristic |
|------|:--------:|:-------:|---------------|
| `task1` (Burgers) | 55/30/20 | 70/60/45 | Moderate gap between seg segments |
| `task2` | 40/20/10 | 55/45/30 | Harder, wider gaps |
| `task3` (KS) | 15/8/4 | 25/18/12 | Chaotic dynamics, much lower baseline |

### Failure Injection

The game can inject realistic failures to test the repair pipeline:

| Failure Pattern | Exit Code | Expected Level | Log Content |
|----------------|:---------:|:--------------:|-------------|
| `git_not_found` | 128 | **L0** — rule match | `"ERROR: Repository not found"` |
| `module_not_found` | 1 | **L0** — rule match | `"ModuleNotFoundError: No module named 'torch'"` |
| `cuda_oom` | 1 | **L1** — traceback | `"torch.cuda.OutOfMemoryError"` at `train.py:42` |
| `data_not_found` | 1 | **L1** — traceback | `"FileNotFoundError: dataset.hdf5"` at `eval.py:15` |
| `unknown_error` | 1 | **L2** — deep reflection | No traceback, opaque error code |

By default ~30% of steps fail randomly (the game picks one of the 5 patterns). Use `--inject <name>` to force a specific failure, or `--inject none` to force success.

## All CLI Commands

| Command | Description |
|---------|-------------|
| `expflow dummy start` | Start a new game session. Options: `--task`, `--seed` |
| `expflow dummy step` | Run one iteration. Options: `--params`, `--strategy`, `--inject` |
| `expflow dummy status` | Show current game state, remaining steps to ceiling |
| `expflow dummy reset` | Reset to baseline. Option: `--seed` |
| `expflow dummy auto` | Run full automated loop. Options: `--max-steps`, `--repair/--no-repair` |
| `expflow dummy list-failures` | List all injectable failure patterns |

## Querying Game History

Because every step creates real records in `DispatchDB`, you can inspect the experiment tree:

```bash
# Show the full experiment tree
expflow dispatch tree <root_experiment_id>

# Get database statistics
expflow dispatch stats

# View audit log for repair events
expflow dispatch audit-log --event-type repair

# List recent experiments
expflow dispatch list --limit 20
```

## Automated Testing

The Dummy Game comes with **20 pytest tests** covering:

- Basic lifecycle (`start`, `step`, `status`, `reset`)
- All 5 failure patterns inject correctly
- L0 repair matches `git_not_found` and `module_not_found`
- L1 extraction for `cuda_oom` and `data_not_found`
- Ceiling convergence: scores plateau at expected maxima
- `diagnose_experiment()` + `suggest_next_params()` integration

Run them:

```bash
python -m pytest tests/test_dummy_game.py -v
```

All 20 tests pass with zero external dependencies (no torch, no clearml, no GPU).

## Comparison to Real Experiments

| Aspect | Real Experiment | Dummy Game |
|--------|:---------------:|:----------:|
| GPU required | ✅ Yes | ❌ No |
| ClearML server | ✅ Required | ❌ Not needed |
| Optuna database | ✅ Required | ❌ Not needed |
| torch / CUDA | ✅ Required | ❌ Not needed |
| Data files (HDF5) | ✅ Required | ❌ Not needed |
| Training time | Hours | Milliseconds |
| seg scores | Real evaluation | Synthetic (effect + noise + ceiling) |
| DispatchDB records | ✅ Same | ✅ Same |
| Repair pipeline | ✅ Same | ✅ Same |
| diagnose → suggest | ✅ Same | ✅ Same |

The Dummy Game produces the **same DispatchDB schema, the same audit trails, the same branch trees, and the same repair interfaces** as real experiments. The only difference is what drives the seg scores.

## Related

- [USAGE.md — `expflow dummy` CLI section](USAGE.md#dummy-experiment-game)
- [ARCHITECTURE.md — System layers](ARCHITECTURE.md)
- [DEVELOPMENT.md — Testing guidelines](DEVELOPMENT.md)
- [DispatchDB design — `expflow_pde/dispatch_db.py`](../expflow_pde/dispatch_db.py)
