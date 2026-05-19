# expflow-pde System Architecture

> **Package name**: `expflow-pde` (PyPI: `expflow-pde`, CLI: `expflow`)
> **Repository**: [github.com/diamond2nv/expflow](https://github.com/diamond2nv/expflow)

## Project Overview

expflow-pde is an experiment workflow orchestration toolkit for PDEBench/Agentic4Sci.
It bridges the gap between PDE training scripts and production-grade MLOps by providing
a unified CLI over ClearML (experiment tracking), Optuna (hyperparameter optimization),
and Langfuse (LLM observability).

The toolkit is designed with **optional dependencies** — the core CLI works with zero
external SDKs installed, and SDK-specific features are loaded on demand.

## Architecture Layers

```
┌──────────────────────────────────────────────────────────────────┐
│                       CLI (Typer)                                 │
│  expflow version | info | clearml | optuna | langfuse | run      │
│  expflow audit | system | pin | analyze | pipeline | mcp | init  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    MCP Server (stdio)                              │
│  18+ MCP tools: exp_compare_scores, exp_list_workers,            │
│  exp_list_tasks, exp_trace_experiment, exp_submit_experiment...   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    Lazy Import Layer                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  clearml.py    │  │  optuna.py     │  │  langfuse.py       │  │
│  │  (Task/Queue/  │  │  (Study/Trial/ │  │  (Trace/Session/   │  │
│  │   Dataset/     │  │   Plot/HPO)    │  │   Cost/Metrics)    │  │
│  │   Worker/      │  │                │  │                    │  │
│  │   Pipeline/    │  │                │  │                    │  │
│  │   Scheduler)   │  │                │  │                    │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  pin.py        │  │  metrics.py    │  │  fsm.py            │  │
│  │  (SHA-256 PIN) │  │  (Standard     │  │  (7-state          │  │
│  │                │  │   Metric       │  │   Experiment FSM)  │  │
│  │                │  │   Registry)    │  │                    │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  equations.py  │  │  analyze.py    │  │  compare.py        │  │
│  │  (11 PDE       │  │  (Competition  │  │  (Score            │  │
│  │   Equations)   │  │   Intelligence)│  │   Comparison)      │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    Supporting Modules                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  config.py     │  │  dispatcher.py │  │  pipeline.py       │  │
│  │  (YAML + .env) │  │  (In-memory    │  │  (3-mode Pipeline) │  │
│  │                │  │   Experiment   │  │                    │  │
│  │                │  │   Registry)    │  │                    │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  hpo.py        │  │  worktree.py   │  │  snowflake.py     │  │
│  │  (3-mode HPO:  │  │  (Git Worktree │  │  (ID Generator)   │  │
│  │   Local/Dist/  │  │   for Experiment│ │                    │  │
│  │   Optimizer)   │  │   Isolation)   │  │                    │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐                           │
│  │  status.py     │  │  board.py      │                           │
│  │  (Component    │  │  (TensorBoard  │                           │
│  │   Health)      │  │   Launcher)    │                           │
│  └────────────────┘  └────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Responsibility | Dependencies | CLI Group |
|--------|---------------|:------------:|-----------|
| `cli.py` | Typer CLI, 8 command groups, 5 top-level commands | typer | `expflow` |
| `clearml.py` | Task/Queue/Dataset/Worker/Scheduler CRUD (all lazy imports) | clearml (optional) | `expflow clearml` |
| `optuna.py` | Study/Trial/Plot management (lazy import) | optuna (optional) | `expflow optuna` |
| `langfuse.py` | Trace/Session/Cost/Metrics (lazy import) | langfuse (optional) | `expflow langfuse` |
| `hpo.py` | 3-mode hyperparameter optimization runner | optuna, clearml (optional) | `expflow optuna run` |
| `pipeline.py` | Competition pipeline: train → eval → submit | clearml (optional) | `expflow pipeline` |
| `dispatcher.py` | In-memory experiment registry + lifecycle | None | `expflow run` |
| `fsm.py` | 7-state finite state machine for experiments | fysom | — |
| `pin.py` | PIN protection (SHA-256) for destructive ops | None | `expflow pin` |
| `metrics.py` | Standardized metric registry with thresholds | None | — |
| `compare.py` | Multi-model score ranking with gating | clearml (optional) | `expflow clearml compare` |
| `equations.py` | PDE equation registry (11 equations) | None | — |
| `analyze.py` | Competition task intelligence & strategy | None | `expflow analyze` |
| `audit.py` | Competition rules validation & compliance | None | `expflow audit` |
| `config.py` | YAML + .env config loader | pyyaml, python-dotenv | — |
| `worktree.py` | Git worktree experiment isolation | git | — |
| `snowflake.py` | Snowflake ID generator | None | — |
| `status.py` | Component health checks | clearml/optuna/langfuse (optional) | `expflow system status` |
| `board.py` | TensorBoard launcher | tensorboard (optional) | `expflow system board` |
| `mcp.py` | MCP server entry point | clearml, optuna, langfuse (optional) | `expflow mcp` |
| `mcp_server.py` | 18+ MCP tool definitions | clearml, optuna, langfuse, mcp (optional) | — |
| `cli_clearml.py` | CLI group for clearml (14 commands) | clearml (optional) | `expflow clearml` |
| `cli_optuna.py` | CLI group for optuna (8 commands) | optuna (optional) | `expflow optuna` |
| `cli_langfuse.py` | CLI group for langfuse (6 commands) | langfuse (optional) | `expflow langfuse` |
| `cli_audit.py` | CLI group for audit (3 commands) | clearml (optional) | `expflow audit` |
| `cli_analyze.py` | CLI group for analyze (4 commands) | None | `expflow analyze` |
| `cli_pin.py` | CLI group for PIN (4 commands) | None | `expflow pin` |
| `cli_pipeline.py` | CLI group for pipeline (1 command) | clearml (optional) | `expflow pipeline` |
| `cli_run.py` | CLI group for experiment dispatch (4 commands) | None | `expflow run` |
| `cli_system.py` | CLI group for system (2 commands) | clearml/optuna/langfuse (optional) | `expflow system` |

## Module Dependency Chain

```
cli.py (Typer) — 8 command groups + 5 top-level commands
  ├── clearml.py        → Task/Queue/Dataset SDK wrappers
  ├── optuna.py          → Study/Trial/Plot SDK wrappers
  ├── langfuse.py        → Trace/Session/Cost SDK wrappers
  ├── dispatcher.py      → Experiment dispatch (in-memory registry)
  ├── audit.py           → Validation + compliance + report
  ├── system.py          → Component health checks + board
  └── mcp.py             → MCP Server entry point

All SDK imports are LAZY — cleared at module-level import time,
loaded only when the corresponding command group is invoked.
```

## Config Loading

```python
from expflow_pde.config import load_config, get

cfg = load_config()            # Load YAML + .env
val = get("clearml.api")       # Dot-separated access
```

Config search order: CWD `config.yaml` → parent dirs → `.env` (env-only overrides API keys).

## Data Flow

### Experiment Lifecycle

```
expflow run submit <script.py>       # Register experiment
         │
         ▼
dispatcher.py: add_experiment()      # In-memory registry
         │
         ▼
fsm.py: EXPERIMENT_FSM               # DRAFT → ENQUEUED → RUNNING
         │                               → COMPLETED / FAILED / CANCELLED
         ▼
expflow run cancel <id>              # FSM: CANCEL_PENDING → CANCELLED
         │                              (PIN-guarded unless --force)
         ▼
expflow clearml compare-scores       # Fetch clearml metrics
         │                              (if clearml integration is set up)
         ▼
expflow audit validate <id>          # Compliance check against competition rules
```

### Pipeline Flow (train → eval → submit)

```
expflow pipeline submit-full <script.py>
         │
         ▼
  Stage 1: HPO (Optuna trials — optional)
         ├── local: subprocess serial on CPU
         ├── distributed: ask/tell + clearml Task clone
         └── optimizer: clearml HyperParameterOptimizer
         │
         ▼
  Stage 2: Train (best params)
         └── clearml-agent queue → GPU node
         │
         ▼
  Stage 3: Eval (generate pred.hdf5)
         └── clearml-agent queue → GPU node
         │
         ▼
  Result: JSON summary + clearml task IDs
```

### HPO Execution Modes

```
expflow optuna run <script.py>
         │
    ┌────┴──────────┐
    │               │
    ▼               ▼
  local           distributed
  (subprocess)    (clearml queue)
    │               │
    ▼               ▼
  SQLite DB      clearml Task per trial
  ~/.expflow/    parent study = controller task
  optuna_<name>.db
```

## CLI Command Tree

```
expflow
├── version / info / mcp / init / config         ← top-level (no SDK deps)
├── clearml     (14 sub-cmds)                    ← lazy import clearml
│   ├── tasks / task / enqueue / dequeue / queues / compare / workers
│   ├── dataset-register / dataset-list / dataset-upload / dataset-download
│   ├── pipeline-create / pipeline-add-step / pipeline-start / pipeline-stop / pipeline-list
│   └── scheduler-create / scheduler-start / scheduler-add-task / scheduler-list / scheduler-remove-task
├── optuna      (8 sub-cmds)                     ← lazy import optuna
│   ├── create-study / studies / study / delete-study / ask / tell / plot / run
├── langfuse    (6 sub-cmds)                     ← lazy import langfuse
│   ├── traces / trace / trace-cost / sessions / session / metrics
├── run         (4 sub-cmds)                     ← no SDK deps
│   ├── submit / list / status / cancel
├── audit       (3 sub-cmds)                     ← no SDK deps
│   ├── validate / check-dataset / report
├── system      (2 sub-cmds)                     ← lazy import per check
│   ├── status (health checks) / board (TensorBoard)
├── pin         (4 sub-cmds)                     ← no SDK deps
│   ├── init / check / clear / status
├── analyze     (4 sub-cmds)                     ← no SDK deps
│   ├── task / equations / status / advise
└── pipeline    (1 sub-cmd)                      ← lazy import clearml
    └── submit (train → eval → submit pipeline)
```

## Key Design Decisions

### 1. Optional SDK Dependencies

All three major SDKs (clearml, optuna, langfuse) are **optional extras**.
The `__init__.py` uses `__getattr__` for lazy resolution with `.pyi` type stubs:

```python
# expflow_pde/__init__.py
def __getattr__(name: str):
    _lazy_map = {
        "list_tasks": ("expflow_pde.clearml", "list_tasks"),
        # ...
    }
    if name in _lazy_map:
        mod_path, attr = _lazy_map[name]
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
```

### 2. FSM-Driven Experiment Lifecycle

Experiments follow a 7-state FSM (fysom):
- `DRAFT` → `ENQUEUED` → `RUNNING` → `COMPLETED`
- `RUNNING` → `FAILED`
- `CANCEL_PENDING` → `CANCELLED`
- Cancellation is PIN-guarded (unless `--force`)

### 3. PIN Protection for Destructive Operations

SHA-256 hashed PIN stored in `~/.expflow/pin.hash` (never config.yaml).
Supports `--force` bypass for CI/automation.

### 4. Standardized Metrics Registry

Metrics use `Group/Metric` naming (e.g., `Score/Seg Total`, `Loss/Val MSE`)
for compatibility with `expflow clearml compare-scores` gating system.

### 5. Snowflake IDs

Thread-safe yitter snowflake M1 implementation in `snowflake.py`.
worker_id=1 reserved for expflow. Format: `exp:snow_<19-digit-int>`.

## Related

- [USAGE.md](USAGE.md) — Installation and CLI reference
- [DEVELOPMENT.md](DEVELOPMENT.md) — Developer guide and testing
- [DATA_LAYER.md](DATA_LAYER.md) — ClearML data layer design
- [COMPETITION.md](COMPETITION.md) — PDE competition integration
