# expflow — Macro Roadmap

> Experiment workflow orchestration toolkit for PDEBench/Agentic4Sci

## Vision

expflow is the **measurement-plane unified interface** for the Agent4PDE three-plane architecture.
It bridges PDEBench experiments with clearml (experiment tracking), clearml-data (dataset versioning),
optuna (HPO), and langfuse (observability) via CLI and MCP interfaces.

### What expflow provides:

1. **Experiment dispatch & tracking** — launch, track, cancel experiments via clearml
2. **Hyperparameter optimization** — optuna backend integration + clearml Task tracking
3. **Dataset & model management** — upload, version, lineage, compliance via clearml-data
4. **Compliance auditing** — competition-legal dataset verification, provenance chain check
5. **Agent-friendly MCP tools** — read/write experiment metadata, query optimization studies

### What expflow does NOT do (leverages clearml native):

| Not needed | Because |
|-----------|---------|
| Data file transfer / hash / chunking | clearml-data SDK handles all of it |
| Job scheduling / queue management | clearml-agent daemon is production-ready |
| Pipeline orchestration | clearml PipelineController covers it |
| Task init / auto-logging | `Task.init()` is one line of code |
| Step caching / parallel execution | clearml Pipeline does it natively |

expflow's **real value** is in the gaps clearml doesn't fill:
- Competition compliance annotation & audit
- Unified CLI (clearml + optuna + langfuse + audit)
- Hermes MCP tool integration
- Dataset lineage + compliance provenance chain

---

## Data Layer — Design Decision

**ClearlML Fileserver（内置）替代 DVC + 独立 MinIO**，详见 [docs/data_layer_design.md](docs/data_layer_design.md)。

核心理由：
- clearml-data（`clearml.Dataset` class）是 DVC 的超集：版本管理+血缘追踪+差异化存储+自动缓存
- 与 clearml Task 原生集成——训练脚本自动记录 Dataset ID
- Hermes Agent 通过 expflow MCP 直接调 clearml-data API
- 详见 [~/wiki/clearml/clearml_data_vs_dvc.md](~/wiki/clearml/clearml_data_vs_dvc.md)

---

## Status — 98 tests, 7 CLI command groups, 6 phases complete, 2 phases planned

### CLI Tree

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
    dataset-register   Register a dataset with compliance annotation [metadata-only]
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

34 sub-commands across 7 command groups, 5 top-level commands.

---

## Test Coverage

| Package | Tests | Scope |
|---------|-------|-------|
| expflow.clearml | 24 unit | Task CRUD, queue mgmt, dataset compliance |
| expflow.optuna | 15 unit | Study CRUD, ask/tell, plot |
| expflow.langfuse | 10 unit | Trace list/get, cost, sessions, metrics |
| expflow.dispatcher | 9 unit | Experiment submit/list/status/cancel |
| expflow.audit | 13 unit | Validation, compliance check, report |
| CLI (CliRunner) | 21 | All 7 command groups + entry point subprocess |
| System tests | 6 | status/board/mcp/init/config + edge cases |
| **Total** | **98 tests, all pass** | `ruff` 0 errors, 9 commits |

---

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
- [x] `docs/data_layer_design.md` — clearml Fileserver 数据层设计（clearml 官方文档调研后更新）

### Phase 7 — Data Layer: clearml-data Dataset API [PLANNED]
- [ ] `clearml.dataset_upload()` — upload local HDF5 to Fileserver (wraps clearml Dataset.create+add_files+upload+finalize)
- [ ] `clearml.dataset_download()` — download from Fileserver (wraps Dataset.get+get_mutable_local_copy)
- [ ] `clearml.dataset_lineage()` — recursive parent chain (wraps Dataset.get(id).parent)
- [ ] `clearml.model_list()` / `model_upload()` — Model.query_models + InputModel/OutputModel
- [ ] CLI subcommands: dataset-upload/download/lineage, model-list/upload
- [ ] Tests: 15+ unit (mocked clearml Dataset API)
- [ ] MCP tools: dataset_upload/download/list/lineage, model_list/upload
- [ ] Old `register_dataset()` → `annotate_compliance()` refactor

**Prerequisite:** clearml-server must be deployed first. clearml-data requires an active apiserver.

### Phase 8 — Pipeline Integration (clearml Pipeline) [PLANNED]
- [ ] clearml PipelineController encapsulation for train → validate workflows
- [ ] Automatic dataset ID injection into pipeline parameters
- [ ] Compliance check step injection
- [ ] MCP tools: pipeline_submit / list / status

---

## Design Decisions

| Decision | Date | Reason |
|----------|------|--------|
| Use clearml-data instead of DVC | 2026-05-13 | clearml-data is DVC's superset; no extra MinIO; natively integrated with Task/Model |
| Use clearml SDK, not clearml-mcp | 2026-05-13 | clearml-mcp is read-only; expflow needs write access (create/run/enqueue) |
| Use clearml Pipeline over native dispatcher | 2026-05-13 | PipelineController already handles DAG/caching/parallelism |
| dataset_upload wraps clearml SDK (not reimplement) | 2026-05-13 | clearml-data handles hash/chunk/cache natively; 20 lines vs 100 lines |
| clearml automatically captures TensorBoardX | 2026-05-13 | Zero-code integration — Task.init() before PyTorch import is all needed |

---

## Stretch Goals (not implemented)

- `expflow/run --tune` — auto-launch optuna study alongside experiment
- Langfuse <-> clearml trace linking
- 7x24 background experiment mode
- Dataset auto-preview (MD5 diff, sample count, time step range)
- `expflow system board` → clearml compare-board integration
