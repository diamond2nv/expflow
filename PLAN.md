# expflow — Macro Roadmap

> Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci

## Vision

expflow bridges PDEBench experiments with ML experiment management systems (clearml,
optuna, langfuse) via CLI and MCP interfaces. It provides:

1. **Experiment dispatch** — launch, track, cancel experiments across distributed workers
2. **Hyperparameter optimization** — clearml-native + optuna backend integration
3. **Dataset compliance** — MD5 verification, source provenance, train/val/test audit
4. **Agent-friendly MCP tools** — read/write experiment metadata, query optimization studies

## Phases

### Phase 0: Skeleton (当前)
- [x] Package skeleton: pyproject.toml, __init__.py, config.py, cli.py
- [x] AGENTS.md + PLAN.md + .gitignore
- [ ] Initial commit + tag v0.1.0

### Phase 1: clearml Integration
- [ ] `expflow/clearml.py` — task CRUD, queue management, agent control
- [ ] `expflow/clearml_mcp.py` — MCP tools for Agent experiment dispatch
- [ ] CLI: `expflow clearml list-tasks`, `expflow clearml enqueue`, `expflow clearml status`
- [ ] Dataset compliance API — clearml dataset registration with `--compliance allowed/forbidden`

### Phase 2: optuna Integration
- [ ] `expflow/optuna.py` — study/trial management, optuna-mcp CLI wrapper
- [ ] `expflow/optuna_mcp.py` — MCP tools for Agent HPO
- [ ] CLI: `expflow optuna create-study`, `expflow optuna ask/tell`, `expflow optuna plot`
- [ ] Pipeline integration: expflow `run --tune` auto-launches optuna study

### Phase 3: langfuse Integration
- [ ] `expflow/langfuse.py` — trace query, cost analysis, session search
- [ ] CLI: `expflow langfuse traces`, `expflow langfuse cost`, `expflow langfuse sessions`
- [ ] Agent audit log integration (mirror traces to Langfuse)
- [ ] Web dashboard annotation (MCP tools for Agent self-annotation)

### Phase 4: Distributed Experiment Dispatch
- [ ] `expflow/dispatcher.py` — Hermes -> expflow CLI -> (clearml agent | OpenCode) routing
- [ ] `expflow/dispatcher_mcp.py` — Agent orchestration MCP tools
- [ ] Multi-GPU task scheduling with clearml queue management
- [ ] 7x24 background experiment mode (Hermes sleeps, expflow monitors)

### Phase 5: Audit Pipeline
- [ ] `expflow/audit.py` — experiment result cross-validation, reproducibility check
- [ ] `expflow/report.py` — auto-generated experiment report (Markdown + PDF)
- [ ] Langfuse <-> clearml trace linking
- [ ] Competition compliance checker (training data provenance, checkpoint lineage)
