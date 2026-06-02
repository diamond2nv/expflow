<p align="center">
  <a href="README.md">English</a> | <a href="docs/cn/USAGE.zh-CN.md">简体中文</a>
</p>

# expflow-pde

[![PyPI version](https://img.shields.io/pypi/v/expflow-pde)](https://pypi.org/project/expflow-pde/)
[![Python versions](https://img.shields.io/pypi/pyversions/expflow-pde)](https://pypi.org/project/expflow-pde/)
[![License](https://img.shields.io/github/license/diamond2nv/expflow)](https://github.com/diamond2nv/expflow/blob/master/LICENSE)

**Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci.**

Manage ML experiments across ClearML, Optuna, and Langfuse from a single CLI — training, HPO, distributed dispatch, compliance checks, and observability.

> ⚠️ **Alpha**: Core functionality works. APIs may change as we stabilize the feature set.

---

## Quick Start

### Install

```bash
# Core CLI (no external SDKs needed)
pip install expflow-pde

# With all SDK integrations
pip install "expflow-pde[all]"

# Individual extras
pip install "expflow-pde[clearml]"   # Task/queue/dataset management
pip install "expflow-pde[optuna]"    # Hyperparameter optimization
pip install "expflow-pde[langfuse]"  # LLM observability traces
pip install "expflow-pde[mcp]"       # MCP server + all SDKs
```

### Verify

```bash
expflow version
expflow info
```

---

## What expflow-pde Solves

Running PDEBench or Agentic4Sci experiments involves coordinating multiple tools:

| Problem | How expflow-pde Helps |
|---------|----------------------|
| Train → eval → submit loop | `expflow pipeline submit` — 3 modes (full/fast/skip) |
| Hyperparameter search | `expflow optuna run` — local, distributed, or clearml-native |
| Experiment tracking | `expflow clearml tasks` — list, enqueue, dequeue, compare |
| Competition compliance | `expflow audit validate` — PIN, metrics, rules, dataset lineage |
| LLM call observability | `expflow langfuse trace` — trace cost, session management |
| Multi-machine dispatch | `expflow run submit` — git worktree + clearml-agent queue |
| System monitoring | `expflow system status` — component health checks |
| Experiment simulation | `expflow dummy` — full lifecycle test without GPU/clearml |

### Built-in Experiment Simulator

**`expflow dummy`** — simulate the entire experiment loop (diagnose → suggest → submit → fail → repair → iterate) **without GPUs, ClearML, or torch**. Inject realistic failures (git clone error, CUDA OOM, missing module), verify L0/L1/L2 repair, and inspect the full experiment tree in DispatchDB — all from a single CLI command on your laptop.

```bash
expflow dummy start                        # Start a game
expflow dummy step --inject cuda_oom       # Test L1 traceback repair
expflow dummy auto --max-steps 10 --repair # Full automated loop
```

[Full documentation →](docs/DUMMY_GAME.md) | [中文文档 →](docs/cn/DUMMY_GAME.zh-CN.md)

### Non-Goals

- Not a general-purpose experiment manager (use ClearML directly for that)
- Not a PDE solver (use PDEBench / PhysicsNeMo)
- Not a replacement for your existing experiment tracking

---

## CLI Overview

```
expflow
├── version / info       ← Package info, system overview (no SDK deps)
├── init                 ← Interactive config wizard
├── clearml              ← Task/queue/dataset CRUD [needs clearml extra]
├── optuna               ← HPO study/trial/plot [needs optuna extra]
├── langfuse             ← Trace/session/cost [needs langfuse extra]
├── run                  ← Local experiment submit/list/status/cancel
├── audit                ← Validate, compare, compliance report
├── system               ← Health checks, TensorBoard
├── pin                  ← PIN-protect destructive operations
| analyze              ← Task intelligence, equation registry, strategy
├── dummy               ← Experiment simulator (no GPU needed)
├── dispatch            ← Local SQLite experiment registry
├── iterate             ← One-shot: diagnose → suggest → submit
└── pipeline             ← Train → eval → submit pipeline
```

## Pipeline Modes

### Full (HPO → Train → Eval)

```bash
expflow pipeline submit-full train_task1.py \
    --queue default \
    --trials 50 --parallel 4 \
    --eval-script eval_task1.py \
    --metric seg_total --direction maximize
```

### Fast (Train → Eval, skip HPO)

```bash
expflow pipeline submit train_task1.py \
    --queue default \
    --train-param lr=0.001 --train-param epochs=80 \
    --eval-script eval_task1.py
```

### Flexible Skip

```bash
expflow pipeline submit-full train_task1.py --skip hpo --skip eval  # train only
expflow pipeline submit-full train_task1.py --skip train --skip eval  # HPO only
```

## Hermes Agent Integration

expflow-pde ships with four Hermes Agent skills for AI-assisted experiment management.
Skills live in the repository's `skills/` directory — install via URL:

```bash
# Install individual skills
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/expflow-pipeline-hpo/SKILL.md
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/experiment-lifecycle-governance/SKILL.md
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/clearml-metrics-logging-pattern/SKILL.md
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/competition-task-intelligence/SKILL.md

# Or tap the repo for easier access
hermes skills tap add diamond2nv/expflow
hermes skills install expflow-pipeline-hpo
```

This adds 4 agent skills:
| Skill | Purpose |
|-------|---------|
| `expflow-pipeline-hpo` | Competition pipeline orchestration (HPO → train → eval) |
| `experiment-lifecycle-governance` | PIN protection, metrics registry, compare-scores |
| `clearml-metrics-logging-pattern` | Standardized ClearML metric naming & reporting |
| `competition-task-intelligence` | PDE equation registry, task analysis, strategic advising |

### MCP Server

```bash
expflow mcp                              # Start MCP server (stdio)
```

Register in `~/.hermes/config.yaml` for agent access to all expflow tools:

```yaml
mcp:
  servers:
    expflow:
      command: "expflow"
      args: ["mcp"]
```

After registration, the agent can: list tasks, enqueue experiments, compare scores, and more — directly from chat.

### Agent Instructions

The `AGENTS.md` in the repo root provides self-contained agent instructions (project map, development commands, testing conventions, pitfalls) for any AI coding assistant reading the project.

---

## Prerequisites

- **Python 3.11+**
- **ClearML server** (optional) — for distributed experiment dispatch
- **Optuna** (optional) — for hyperparameter optimization
- **Langfuse server** (optional) — for LLM trace observability

Configure via `expflow init` or by placing `config.yaml` / `.env` in your project root.

## Optional Dependencies

```bash
# Minimal: pip install expflow-pde
pip install expflow-pde                   # core CLI only

# Extras:
pip install "expflow-pde[clearml]"        # ClearML SDK integration
pip install "expflow-pde[optuna]"         # Optuna HPO
pip install "expflow-pde[langfuse]"       # Langfuse tracing
pip install "expflow-pde[pipeline]"       # pipeline mode (needs clearml)
pip install "expflow-pde[mcp]"            # MCP server (all above)
pip install "expflow-pde[all]"            # everything
pip install "expflow-pde[dev]"            # development tooling
```

## Development

```bash
git clone https://github.com/diamond2nv/expflow.git
cd expflow
python -m venv venv && source venv/bin/activate
pip install -e ".[all,dev]"
```

```bash
ruff format .                          # Format code
ruff check --fix .                     # Lint + auto-fix
pyright .                              # Type check
python -m pytest tests/ -v             # Run tests
python -m build                        # Build package
```

## Acknowledgements

This project builds on ideas and mathematical formulations from several research works:

- **HyperNOs** (arXiv:2503.18087) — Relative norm loss formulation (L^p, H^1 Sobolev via FFT) used in `losses.py` as the design pattern for the relative norm loss family.
- **AutoScientists** (arXiv:2605.28655, Harvard/MIMS 2026) — Noise-aware champion validation, lazy sigma calibration, stagnation detection, and dead-end registry concepts implemented in `validate.py`, `registry.py`, and `monitor.py`. Independent implementation based on published algorithmic description.
- **Zhang et al. (JFM, 2026)** — Physics-informed RANS PDE residual loss (`RANSPDELoss`, `PINNCompositeLoss`) design follows the physics-informed neural operator training methodology for 2D cylinder flow. Independent implementation from published mathematical formulation.
- **PDEBench** (arXiv:2207.05209) — Standard evaluation metrics and PDE equation definitions used across the metric registry.

All code is original and written from scratch. Only the mathematical/algorithmic ideas are referenced.

## License

MIT

## Links

- [Full Usage Guide](docs/USAGE.md) (English) | [中文使用指南](docs/cn/USAGE.zh-CN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Developer Guide](docs/DEVELOPMENT.md)
- [Data Layer Design](docs/DATA_LAYER.md)
- [Competition Integration](docs/COMPETITION.md)
- [Dummy Experiment Game](docs/DUMMY_GAME.md) (English) | [虚拟实验游戏](docs/cn/DUMMY_GAME.zh-CN.md)
- [Hermes Agent Skills](skills/) — 4 skills for MLOps experiment orchestration
