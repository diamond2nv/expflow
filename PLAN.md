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

### Phase 0: Skeleton ✅
- [x] Package skeleton: pyproject.toml, __init__.py, config.py, cli.py
- [x] AGENTS.md + PLAN.md + .gitignore
- [x] Initial commit + tag v0.1.0

### Phase 1: clearml Integration ✅
- [x] `expflow/clearml.py` — task CRUD, queue management, dataset compliance
- [x] CLI: `expflow clearml tasks/task/enqueue/dequeue/queues/dataset-register/dataset-list`
- [x] Dataset compliance API — clearml dataset registration with `--compliance allowed/forbidden`
- [x] 41 tests (unit + CLI mock + entry point subprocess)
- [ ] `expflow/clearml_mcp.py` — MCP tools for Agent experiment dispatch (stretch/next phase)

### Phase 2: optuna Integration ✅
- [x] `expflow/optuna.py` — study/trial management (create, list, get, delete, ask, tell, plot)
- [x] CLI: `expflow optuna create-study/studies/study/delete-study/ask/tell/plot`
- [x] 15 unit tests + 9 CLI tests + entry point coverage — all mock optuna SDK
- [ ] `expflow/optuna_mcp.py` — MCP tools for Agent HPO (stretch/next phase)
- [ ] Pipeline integration: expflow `run --tune` auto-launches optuna study (stretch)

### Phase 3: langfuse Integration (next)
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
