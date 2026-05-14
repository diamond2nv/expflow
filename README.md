# expflow

Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci.

[简体中文](docs/cn/README.zh-CN.md)

## Installation

```bash
pip install expflow
```

For clearml support:

```bash
pip install expflow[clearml]
```

## What is expflow?

expflow is the **measurement-plane unified interface** for the Agent4PDE three-plane architecture.
It wraps clearml (experiment tracking) / clearml-data (dataset versioning) / optuna (HPO) / langfuse (observability)
into a single CLI and MCP interface.

### Key commands

```bash
# Experiment tracking
expflow clearml tasks --project pdebench    # List experiments
expflow clearml compare <id1> <id2>         # Side-by-side comparison

# Hyperparameter optimization
expflow optuna run train.py --trials 50    # Full HPO workflow
expflow optuna plot <study> --type history  # HPO visualization

# Dataset compliance
expflow clearml dataset-register <name> --compliance allowed
expflow clearml dataset-list

# Audit
expflow audit validate <experiment_id>
expflow audit report <experiment_id>

# MCP Server (for Hermes Agent)
expflow mcp
```

### Architecture decisions

| Decision | Detail |
|----------|--------|
| Data versioning | clearml-data (not DVC) — built-in, no extra MinIO |
| Experiment tracking | clearml (clearml-mcp is read-only; expflow needs writes) |
| Pipeline orchestration | clearml PipelineController (not expflow dispatcher) |
| Dataset upload | wraps clearml SDK (not reimplemented — clearml handles hash/chunk/cache) |
| TensorBoardX | automatically captured by clearml (zero-code) |

See [PLAN.md](PLAN.md) for the full roadmap, [AGENTS.md](AGENTS.md) for AI agent development guide.

## License

MIT
