# expflow — AI Agent Development Guide

This file is for AI coding assistants (Hermes Agent, OpenCode, Claude Code, etc.)
working on this project. It describes the project structure, key patterns, pitfalls, and constraints.

## Quick Navigation

```
~/Gitlab/Agentic4Sci/expflow/
├── expflow_pde/         # Main Python package (16 modules)
│   ├── __init__.py      # Version, exports
│   ├── config.py        # YAML + .env config loader
│   ├── cli.py           # Typer CLI — 5 top-level commands + 8 command groups
│   ├── clearml.py       # ClearML task/queue/dataset CRUD (SDK lazy import)
│   ├── optuna.py        # Optuna study/trial/plot (SDK lazy import)
│   ├── langfuse.py      # Langfuse trace/session/metrics (SDK lazy import)
│   ├── dispatcher.py    # Experiment submit/list/status/cancel
│   ├── audit.py         # Validation, compliance, report generation
│   ├── system.py        # Component health checks + TensorBoard
│   ├── mcp.py           # MCP Server for Hermes Agent
│   ├── mcp_server.py    # FastMCP tool registry (18+ tools)
│   ├── pipeline.py      # ExperimentPipeline class (train→eval→submit)
│   ├── fsm.py           # 7-state FSM for experiment lifecycle
│   ├── pin.py           # PIN protection module
│   ├── metrics.py       # Standard metric registry
│   ├── compare.py       # Model comparison & score ranking
│   ├── analyze.py       # PDE competition intelligence
│   ├── equations.py     # PDE formula registry (11 equations)
│   ├── worktree.py      # Git worktree for experiment isolation
│   ├── cli_clearml.py   # clearml command group (14 sub-commands)
│   ├── cli_optuna.py    # optuna command group (8 sub-commands)
│   ├── cli_langfuse.py  # langfuse command group (6 sub-commands)
│   ├── cli_run.py       # run command group (4 sub-commands)
│   ├── cli_audit.py     # audit command group (3 sub-commands)
│   ├── cli_system.py    # system command group (2 sub-commands)
│   ├── cli_pin.py       # pin command group (4 sub-commands)
│   ├── cli_analyze.py   # analyze command group (4 sub-commands)
│   └── cli_pipeline.py  # pipeline command group (submit command)
├── tests/               # pytest tests (292+ tests)
│   ├── cli_optuna.py    # optuna command group (8 sub-commands)
│   ├── cli_langfuse.py  # langfuse command group (6 sub-commands)
│   ├── cli_run.py       # run command group (4 sub-commands)
│   ├── cli_audit.py     # audit command group (3 sub-commands)
│   └── cli_system.py    # system command group (2 sub-commands)
├── tests/               # pytest tests (329 tests, 18 files)
├── docs/                # Design documentation
│   └── data_layer_design.md   # clearml-data data layer architecture
├── config.yaml          # Optional project config
├── pyproject.toml       # Package config (setuptools)
├── AGENTS.md            # ← This file
├── PLAN.md              # Macro roadmap
└── .gitignore
```

## Core Architecture

### Module Dependency Chain

```
|cli.py (Typer) — 8 command groups + 5 top-level commands
  ├── clearml.py        → Task/Queue/Dataset SDK wrappers
  ├── optuna.py          → Study/Trial/Plot SDK wrappers
  ├── langfuse.py        → Trace/Session/Metrics SDK wrappers
  ├── dispatcher.py      → Experiment dispatch (in-memory registry)
  ├── audit.py           → Validation + compliance + report
  ├── system.py          → Component health checks + board
  └── mcp.py             → MCP Server stubs

All SDK imports are LAZY — cleared at module-level import time,
loaded only when the corresponding command group is invoked.
```

### CLI Command Tree

```
expflow
├── version / info / mcp / init / config           ← top-level (no SDK deps)
├── clearml     (14 sub-cmds)                      ← lazy import clearml SDK
│   ├── tasks / task / enqueue / dequeue / queues / compare / workers
│   ├── dataset-register / dataset-list / dataset-upload / dataset-download
│   ├── pipeline-create / pipeline-add-step / pipeline-start / pipeline-stop / pipeline-list
│   └── scheduler-create / scheduler-start / scheduler-add-task / scheduler-list / scheduler-remove-task
├── optuna      (8 sub-cmds)                       ← lazy import optuna SDK
│   ├── create-study / studies / study / delete-study / ask / tell / plot / run
├── langfuse    (6 sub-cmds)                       ← lazy import langfuse SDK
│   ├── traces / trace / trace-cost / sessions / session / metrics
├── run         (4 sub-cmds)                       ← no SDK deps (in-memory)
│   ├── submit / list / status / cancel
├── audit       (3 sub-cmds)                       ← no SDK deps
│   ├── validate / check-dataset / report
├── system      (2 sub-cmds)                       ← lazy import per check
│   ├── status (health checks) / board (TensorBoard)
├── pin         (4 sub-cmds)                       ← no SDK deps
│   ├── init / check / clear / status
├── analyze     (4 sub-cmds)                       ← no SDK deps
│   ├── task / equations / status / advise
└── pipeline    (1 sub-cmd)                        ← lazy import clearml SDK
    └── submit (train → eval pipeline)
```

### Config Loading

```python
from expflow_pde.config import load_config, get

cfg = load_config()            # Load YAML + .env
val = get("clearml.api")       # Dot-separated access
```

Config search order: CWD `config.yaml` → parent dirs → `.env` (env-only overrides API keys)

---

## Development Commands

```bash
source venv/bin/activate    # Must activate
ruff format .               # Format (line-length=100, double quotes)
ruff check --fix .          # Lint + auto-fix
pyright .                   # Type check (0 errors)
python -m pytest tests/ -v  # Run tests
python -m build             # Build package
```

## Testing Guidelines

### Test Strategy

| Category | Description | External Dependencies |
|----------|-------------|----------------------|
| Unit | Config CRUD, CLI parsing | None |
| Unit | Module-level logic | None |
| Integration | Config ↔ filesystem | Filesystem |
| Integration | Future: clearml/optuna/langfuse | External services (marked @integration) |

Creating new tests:
1. `tests/test_<module>.py`
2. Use `tmp_path` fixture for filesystem isolation
3. Mock external services (requests / subprocess)
4. Mark integration tests with `@pytest.mark.integration`

## Developer Conventions

### PEP8 Internationalization Standards

All Python files MUST be 100% English-only:
- **Comments** — English only (docstrings, inline comments, block comments)
- **Strings** — English only (print, log, error messages, CLI output)
- **Variable/function/class names** — English only (PEP8 naming)
- **No Chinese characters, emoji, or box-drawing characters** in .py/.yaml/.sh/.md files

Why: `conda` environment has `LC_ALL=C` which causes `UnicodeEncodeError` on non-ASCII output.

Every `.py` file must have header:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```

Exceptions (Chinese allowed):
| Location | What | Reason |
|----------|------|--------|
| `docs/cn/` | Chinese documentation | Intended for Chinese readers |
| `.hermes/plans/` | Hermes agent plans | Internal tooling, not user-facing |
| `README.md` | `简体中文` navigation link only | One-line label |

### PyPI Package Release Checklist

Before tagging a release:

```bash
# 1. Format & lint
ruff format .
ruff check --fix .

# 2. Type check
pyright .

# 3. Test
python -m pytest tests/ -v

# 4. Verify version alignment
grep __version__ expflow_pde/__init__.py  # e.g. '0.1.0'
grep ^version pyproject.toml           # Must match

# 5. Build + verify
python -m build
twine check dist/*

# 6. Tag
git tag v0.1.0
git push --tags

# 7. Publish
twine upload dist/*
```

### Testing Before Release

- `ruff check .` must pass with **zero errors** (including tests/)
- `pytest` must pass all tests
- `pyright` zero errors in `expflow_pde/` package

## Git Conventions

```bash
git add <files>
git commit -m "<type>: <description>"
git tag v0.1.0          # Semantic versioning
```

`.gitignore` covers: `__pycache__/`, `*.egg-info/`, `dist/`, `venv/`, `.ruff_cache/`, etc.

## Versioning

**Current version: 0.3.0** (pre-release)
- Semantic versioning with 0.x.y — x=feature iteration, y=fix/minor
- Don't bump to 1.0.0 before official release
- Version defined in `expflow_pde/__init__.py` `__version__`
- Sync `pyproject.toml` version field
- Tag: `git tag v0.x.y && git push --tags`

## Pitfalls

### Config Cache Is Global

`_config_cache` in `config.py` is a module-level global, shared across all imports.
Tests must reset cache between runs. Use pytest fixtures:

```python
@pytest.fixture(autouse=True)
def reset_config():
    from expflow_pde import config
    config._config_cache.clear()
    yield
```

### Config Search in CWD

`_find_config()` searches CWD → parents for `config.yaml`. If tests change CWD,
config resolution may pick up wrong file. Use explicit `load_config(path=...)` in tests.

---

## Hermes Agent Usage Guide

expflow is the measurement-plane CLI for PDEBench's three-plane architecture.
Hermes uses expflow for experiment dispatch, HPO, dataset compliance, and audit.

### Quick Command Reference

| Category | Command | What it does |
|----------|---------|-------------|
| **Experiment** | `expflow clearml tasks --project pdebench` | List experiments |
| | `expflow clearml task <id>` | Show experiment details |
| | `expflow clearml compare <id1> <id2>` | Side-by-side comparison |
| | `expflow clearml enqueue <id> --queue gpu0` | Submit to GPU queue |
| | `expflow clearml dequeue <id>` | Stop queued experiment |
| | `expflow run submit <script> --queue gpu0` | Submit new experiment |
| **HPO** | `expflow optuna create-study <name> --direction maximize` | Create HPO study |
| | `expflow optuna run <script> --trials 50 --n-jobs 2` | Full HPO workflow |
| | `expflow optuna studies` | List all HPO studies |
| | `expflow optuna study <name>` | Get best result |
| | `expflow optuna plot <name> --type history` | Generate HPO viz |
| **Dataset** | `expflow clearml dataset-register <name> --compliance allowed` | Add compliance metadata |
| | `expflow clearml dataset-list` | List registered datasets |
| | `expflow audit check-dataset <name> --compliance allowed` | Check compliance |
| **Monitoring** | `expflow system status` | Check measurement plane health |
| | `expflow system board --port 6006` | Start TensorBoard |
| **Observability** | `expflow langfuse traces --limit 20` | List Langfuse traces |
| | `expflow langfuse trace-cost <id>` | Get trace cost |
| | `expflow langfuse sessions` | List sessions |
| **Report** | `expflow audit validate <id>` | Run reproducibility checks |
| | `expflow audit report <id>` | Generate Markdown report |
| **Config** | `expflow init` | Interactive config wizard |
| | `expflow config` | Show current config |
| **MCP** | `expflow mcp` | Start MCP Server |

### clearml-data Knowledge (from clearml official docs)

clearml-data (`clearml.Dataset` class) is the data versioning layer. Key patterns:

```python
# Upload: create → add_files → upload → finalize
ds = Dataset.create(
    dataset_name="1D_Burgers", dataset_project="PDEBench",
    parent_datasets=[PARENT_ID],         # lineage: inherit parent's content
    dataset_version="1.0",                # semantic version, auto-increment if omitted
    output_uri=None,                       # default: clearml fileserver
)
ds.add_files(path="/data/burgers.hdf5")   # auto-hash, compare with parent, upload diff
ds.upload()                                # upload to fileserver (chunked, parallel)
ds.finalize()                              # close → immutable

# Download
local_path = Dataset.get(
    dataset_id="abc123"
).get_mutable_local_copy(
    target_folder="./data/",
    overwrite=True,
)  # auto-cached in ~/.clearml/cache/

# Metadata (used for compliance annotation)
ds.set_metadata("expflow:compliance", "allowed")
ds.set_metadata("expflow:md5", "a1b2c3d4e5...")
```

**Key invariant:** Once finalized, a dataset is read-only. To add/remove files,
create a child dataset with `parent_datasets`.

### TensorBoardX Integration Knowledge

clearml **automatically captures** all `torch.utils.tensorboard.SummaryWriter` output.
No code changes needed beyond `Task.init()`:

```python
# DO this (before PyTorch imports):
from clearml import Task
Task.init(project_name="PDEBench", task_name="FNO_burgers")

# THEN import PyTorch and use SummaryWriter as usual:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('loss/train', loss, epoch)  # clearml auto-captures
```

Controls:
```python
Task.init(auto_connect_frameworks={
    'tensorboard': True,    # enables TensorBoardX auto-capture
    'pytorch': True,        # enables model checkpoint auto-logging
    'matplotlib': False,    # disable Matplotlib capture
})
```

### Key Design Principles for Agent Workflows

1. **Lazy SDK imports** — import time is O(1). `expflow version|info|init|config`
   never trigger SDK imports. SDK import only happens when you call the
   corresponding command group.

2. **All output is JSON-serializable** — every `expflow/*.py` function returns
   plain dicts/lists. CLI handlers format them for terminal output.

3. **Mockable by design** — 98 unit tests use `sys.modules` patch to mock SDKs,
   zero external dependencies. All new modules should follow this pattern.

### MCP Tools (via `expflow mcp`)

When `expflow mcp` is started, these tools are available:

| Tool | Function | Backend |
|------|----------|---------|
| `exp_run_task` | Submit clearml task + enqueue | clearml SDK |
| `exp_list_runs` | List recent experiments | clearml SDK |
| `exp_get_metrics` | Get metrics for a run | clearml SDK |
| `exp_compare_runs` | Compare two runs | clearml SDK |
| `exp_start_hpo` | Start HPO study | optuna SDK |
| `exp_get_study` | Query HPO results | optuna SDK |
| `exp_hpo_plot` | Generate HPO visualization | optuna + plotly |
| `exp_register_dataset` | Register dataset with compliance | clearml Dataset |
| `exp_list_datasets` | List registered datasets | clearml Dataset |
| `exp_check_compliance` | Check dataset compliance | audit module |
| `exp_generate_report` | Generate experiment report | audit module |
| `exp_board_url` | Get TensorBoard URL | board module |
| `exp_config_status` | Component health check | status module |

### Hermes Scenario Workflows

#### Scenario 1: Task parameter scan

Steps:
1. Hermes reflects on past results: `expflow clearml compare <v1> <v2>`
2. Hermes searches wiki for context: `read ~/wiki/concepts/...`
3. Hermes designs HPO: `expflow optuna run train.py --trials 24`
4. Hermes checks results: `expflow optuna study <name>`
5. Hermes writes findings to wiki

#### Scenario 2: Dataset compliance check

Steps:
1. Register a dataset with compliance: `expflow clearml dataset-register --compliance allowed`
2. Verify registration: `expflow clearml dataset-list`
3. Before submission: `expflow audit check-dataset <name> --compliance allowed`

#### Scenario 3: 7×24 autonomous loop

Cron-triggered session uses expflow to:
1. Collect results: `expflow clearml tasks --status completed`
2. Analyze: `expflow clearml compare <best> <second>`
3. Design: `expflow optuna run <script> --trials 50`
4. Submit: `expflow run submit --queue gpu0`
5. Record: update `~/wiki/log.md`

#### Scenario 4: Full audit report
```
expflow audit validate <experiment_id>
expflow audit report <experiment_id>
expflow langfuse trace-cost <trace_id>
expflow system status
```

## External Wiki References

For deep clearml knowledge, refer to `~/wiki/clearml/`:

| File | Content |
|------|---------|
| `data_management.md` | clearml-data CLI + SDK exhaustive API reference |
| `sdk.md` | Task/Model/HPO core classes |
| `agent_and_serving.md` | Agent daemon / Pipeline / Scheduler |
| `deploy.md` | Server deployment, docker-compose, clearml.conf |
| `advanced.md` | PipelineController / TaskScheduler / TriggerScheduler |
| `clearml_data_vs_dvc.md` | clearml-data = DVC superset, concept mapping |
| `tensorboardx_integration.md` | clearml × TensorBoardX auto-capture, zero-code |
