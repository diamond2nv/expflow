# expflow — Macro Roadmap

> Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci

## Vision

expflow bridges PDEBench experiments with ML experiment management systems (clearml,
optuna, langfuse) via CLI and MCP interfaces. It provides:

1. **Experiment dispatch** — launch, track, cancel experiments across distributed workers
2. **Hyperparameter optimization** — clearml-native + optuna backend integration
3. **Dataset & model management** — upload, version, lineage, compliance via clearml Fileserver
4. **Compliance auditing** — competition-legal dataset verification, provenance chain check
5. **Agent-friendly MCP tools** — read/write experiment metadata, query optimization studies

## Data Layer — Design Decision

**Clearl ML Fileserver（内置）替代 DVC + 独立 MinIO**，详见 [docs/data_layer_design.md](docs/data_layer_design.md)。

核心理由：
- clearml-server docker compose 已内置 fileserver（基于文件系统的 MinIO 兼容存储）
- `clearml.Dataset` 类原生提供版本管理、血缘追踪、元数据标注
- Hermes Agent 通过 expflow MCP 直接调用 clearml API，无需额外 DVC CLI wrapper
- 实验数据自动随 Task 记录，无需手动 `dvc push`

## Status — 98 tests, 6 CLI command groups, 5 phases complete

```
expflow [OPTIONS] COMMAND

  version              Show expflow version
  info                 Show system and environment info
  mcp                  Start MCP server for Hermes Agent integration
  init                 Initialize expflow configuration
  config               Read/write expflow configuration

  clearml              Interact with ClearML experiment management
    tasks              List ClearML tasks
    task               Show details for a single task
    enqueue            Enqueue a task to a queue
    dequeue            Dequeue a task
    queues             List all available queues
    compare            Compare two or more tasks
    dataset-register   Register a PDEBench dataset with compliance annotation
    dataset-list       List registered datasets with compliance info
    dataset-upload     Upload dataset to clearml Fileserver [PLANNED]
    dataset-download   Download dataset from clearml Fileserver [PLANNED]
    dataset-lineage    Trace dataset lineage via parent chain [PLANNED]
    model-list         List checkpoint models [PLANNED]
    model-upload       Upload checkpoint model [PLANNED]

  optuna               Interact with Optuna hyperparameter optimization
    create-study       Create a new Optuna study
    studies            List all Optuna studies
    study              Show details for a study
    delete-study       Delete a study
    ask                Ask for next trial parameters
    tell               Report a trial result
    plot               Generate optimization visualization
    run                Run HPO search on a training script

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

  system               System health and monitoring
    status             Check clearml/optuna/langfuse service health
    board              Launch clearml compare board
```

## Test Coverage

| Package | Tests | Scope |
|---------|-------|-------|
| expflow.clearml | 24 unit | Task CRUD, queue mgmt, dataset compliance |
| expflow.optuna | 15 unit | Study CRUD, ask/tell, plot |
| expflow.langfuse | 10 unit | Trace list/get, cost, sessions, metrics |
| expflow.dispatcher | 9 unit | Experiment submit/list/status/cancel |
| expflow.audit | 13 unit | Validation, compliance check, report |
| CLI (CliRunner) | 21 | All 6 command groups + entry point subprocess |
| **Total** | **98 tests, all pass** | `ruff` 0 errors, 9 commits |

## Implementation Phases

### Phase 0 — Skeleton ✅
- [x] `pyproject.toml`, `.gitignore`, `README.md`, `AGENTS.md`, `PLAN.md`
- [x] `expflow/__init__.py`, `expflow/config.py`
- [x] `expflow/cli.py` — Typer app with lazy-import command groups
- [x] Entry point `expflow = "expflow.cli:app"`

### Phase 1 — ClearML Integration ✅
- [x] `expflow/clearml.py` — Task CRUD, queue mgmt, dataset compliance
- [x] `expflow/cli_clearml.py` — 8 subcommands + `compare`
- [x] 24 unit tests + 12 CLI tests (mocked clearml SDK)

### Phase 2 — Optuna Integration ✅
- [x] `expflow/optuna.py` — Study CRUD, ask/tell, plot, HPO run
- [x] `expflow/cli_optuna.py` — 8 subcommands + `run` wrapper
- [x] 15 unit tests + 9 CLI tests (mocked optuna SDK)

### Phase 3 — Langfuse Integration ✅
- [x] `expflow/langfuse.py` — Trace/session/metrics
- [x] `expflow/cli_langfuse.py` — 6 subcommands
- [x] 10 unit tests (mocked langfuse SDK)

### Phase 4 — Dispatcher ✅
- [x] `expflow/dispatcher.py` — Submit/list/status/cancel
- [x] `expflow/cli_run.py` — 4 subcommands
- [x] 9 unit tests (mocked clearml SDK)

### Phase 5 — Audit ✅
- [x] `expflow/audit.py` — Validate, provenance check, report
- [x] `expflow/cli_audit.py` — 3 subcommands
- [x] 13 unit tests (mocked clearml SDK)

### Phase 6 — Measurement Plane Alignment ✅
- [x] `expflow/cli_system.py` — system status/board
- [x] `expflow/cli.py` — mcp/init/config top-level commands
- [x] `expflow/mcp.py` — MCP server stub
- [x] `expflow/system.py` — health check, tensorboard launch
- [x] `AGENTS.md` — Hermes Agent Usage Guide (4 scenarios + MCP ref)
- [x] `docs/data_layer_design.md` — clearml Fileserver 数据层设计

### Phase 7 — Data Layer: clearml Fileserver Upload/Download [PLANNED]
- [ ] `clearml.dataset_upload()` — upload local HDF5 to Fileserver
- [ ] `clearml.dataset_download()` — download from Fileserver with MD5 check
- [ ] `clearml.dataset_lineage()` — trace parent chain
- [ ] `clearml.model_list()` / `model_upload()`
- [ ] CLI subcommands: dataset-upload/download/lineage, model-list/upload
- [ ] Tests: 15+ unit tests (mocked clearml SDK Dataset API)
- [ ] MCP tools: dataset_upload/download/list/lineage, model_list/upload

### Phase 8 — End-to-End Integration & Deployment [PLANNED]
- [ ] Deploy clearml-server (port 8082:80)
- [ ] Real integration tests vs clearml + fileserver
- [ ] End-to-end experiment: Hermes → expflow → clearml → fileserver loop
- [ ] Multi-GPU task scheduling with clearml queue management

## Stretch Goals (not implemented)

- `expflow/run --tune` — auto-launch optuna study alongside experiment
- Langfuse <-> clearml trace linking
- 7x24 background experiment mode
- Dataset auto-preview (MD5 diff, sample count, time step range)
