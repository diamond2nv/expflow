# expflow — Macro Roadmap

> Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci

## Vision

expflow bridges PDEBench experiments with ML experiment management systems (clearml,
optuna, langfuse) via CLI and MCP interfaces. It provides:

1. **Experiment dispatch** — launch, track, cancel experiments across distributed workers
2. **Hyperparameter optimization** — clearml-native + optuna backend integration
3. **Dataset compliance** — MD5 verification, source provenance, train/val/test audit
4. **Agent-friendly MCP tools** — read/write experiment metadata, query optimization studies

## Status — 98 tests, 6 CLI command groups, 5 phases complete

```
expflow [OPTIONS] COMMAND

  version              Show expflow version
  info                 Show system and environment info
  clearml              Interact with ClearML experiment management
    tasks              List ClearML tasks
    task               Show details for a single task
    enqueue            Enqueue a task to a queue
    dequeue            Dequeue a task
    queues             List all available queues
    dataset-register   Register a PDEBench dataset with compliance annotation
    dataset-list       List registered datasets with compliance info

  optuna               Interact with Optuna hyperparameter optimization
    create-study       Create a new Optuna study
    studies            List all Optuna studies
    study              Show details for a study
    delete-study       Delete a study
    ask                Ask for next trial parameters
    tell               Report a trial result
    plot               Generate optimization visualization

  langfuse             Interact with Langfuse observability platform
    traces             List Langfuse traces
    trace              Show details for a trace
    trace-cost         Show cost breakdown for a trace
    sessions           List Langfuse sessions
    session            Show details for a session
    metrics            Show aggregated usage/cost metrics

  run                  Submit and manage experiments
    submit             Submit an experiment
    list               List all experiments
    status             Show experiment status
    cancel             Cancel an experiment

  audit                Experiment validation, compliance checking, report generation
    validate           Run validation checks on an experiment
    check-dataset      Check dataset compliance
    report             Generate an experiment report (Markdown)
```

## Test Coverage

| Package | Tests | Scope |
|---------|-------|-------|
| expflow.clearml | 24 unit | Task CRUD, queue mgmt, dataset compliance |
| expflow.optuna | 15 unit | Study CRUD, ask/tell, plot |
| expflow.langfuse | 10 unit | Trace list/get, cost, sessions, metrics |
| expflow.dispatcher | 9 unit | Experiment submit/list/status/cancel |
| expflow.audit | 13 unit | Validation, compliance check, report |
| CLI (CliRunner) | 21 | All 5 command groups + entry point subprocess |
| **Total** | **98 tests, all pass** | `ruff` 0 errors, 5 commits |

## Stretch Goals (not implemented)

- `expflow/clearml_mcp.py` — MCP tools for Agent experiment dispatch
- `expflow/optuna_mcp.py` — MCP tools for Agent HPO
- `expflow/dispatcher_mcp.py` — Agent orchestration MCP tools
- `expflow/run --tune` — auto-launch optuna study alongside experiment
- Multi-GPU task scheduling with clearml queue management
- Langfuse <-> clearml trace linking
- 7x24 background experiment mode
