---
name: dummy-experiment-game
description: Generated from skills/dummy-experiment-game/SKILL.md (package reference copy)
---

# Dummy Experiment Game

> **Test the entire experiment lifecycle without GPUs, ClearML, or torch.**

## Overview

The Dummy Experiment Game replaces real GPU training with a synthetic seg-score
model, so you can verify the full experiment loop anywhere.

### What it proves

| What | How |
|------|-----|
| **Pipeline integrity** | diagnose → suggest → submit → fail → repair → iterate works end-to-end |
| **Repair level correct** | L0 catches git/clone errors, L1 extracts CUDA OOM, L2 flags unknown errors |
| **DispatchDB correctness** | Experiment tree branches, audit log, archive all work without clearml |
| **Onboarding** | First-time user can see expflow work in 30 seconds — no GPU needed |

## Quick Start

```bash
# Start a game
expflow dummy start --task task1 --seed 42

# Run a step (synthetic seg scores are generated)
expflow dummy step

# Inject a failure and verify repair
expflow dummy step --inject cuda_oom
expflow dummy step --inject git_not_found

# Simulate Hermes suggesting hyperparams
expflow dummy step --params '{"n_modes": 20, "num_sub_steps": 5}'

# Check how far from ceiling
expflow dummy status

# Fully automated loop
expflow dummy auto --max-steps 10 --repair
```

## Failure Patterns

| Pattern | Exit | Repair | Log |
|---------|:----:|:------:|-----|
| `git_not_found` | 128 | L0 (rule) | "ERROR: Repository not found" |
| `module_not_found` | 1 | L0 (rule) | "ModuleNotFoundError: No module named 'torch'" |
| `cuda_oom` | 1 | L1 (traceback) | "torch.cuda.OutOfMemoryError" at train.py:42 |
| `data_not_found` | 1 | L1 (traceback) | "FileNotFoundError: dataset.hdf5" at eval.py:15 |
| `unknown_error` | 1 | L2 (reflection) | Opaque error code, no traceback |

Use `expflow dummy list-failures` to see all patterns.

## CLI Reference

| Command | Description |
|---------|-------------|
| `expflow dummy start` | New game session. `--task`, `--seed` |
| `expflow dummy step` | One iteration. `--params`, `--strategy`, `--inject` |
| `expflow dummy status` | Game state, steps-to-ceiling |
| `expflow dummy reset` | Reset to baseline. `--seed` |
| `expflow dummy auto` | Full automated loop. `--max-steps`, `--repair` |
| `expflow dummy list-failures` | All injectable failure patterns |

## Comparison to Real Experiments

| Aspect | Real | Dummy |
|--------|:----:|:-----:|
| GPU | ✅ | ❌ |
| ClearML server | ✅ | ❌ |
| torch / CUDA | ✅ | ❌ |
| Training time | Hours | ms |
| DispatchDB | ✅ Same | ✅ Same |
| Repair pipeline | ✅ Same | ✅ Same |
