# expflow — AI Agent Development Guide

This file is for AI coding assistants (Hermes Agent, OpenCode, Claude Code, etc.)
working on this project. It describes the project structure, key patterns, pitfalls, and constraints.

## Quick Navigation

```
~/Gitlab/Agentic4Sci/expflow/
├── expflow/             # Main Python package (12 modules)
│   ├── __init__.py      # Version, exports
│   ├── config.py        # YAML + .env config loader
│   ├── cli.py           # Typer CLI — 11 top-level commands + 6 command groups
│   ├── clearml.py       # ClearML task/queue/dataset CRUD (SDK lazy import)
│   ├── optuna.py        # Optuna study/trial/plot (SDK lazy import)
│   ├── langfuse.py      # Langfuse trace/session/metrics (SDK lazy import)
│   ├── dispatcher.py    # Experiment submit/list/status/cancel
│   ├── audit.py         # Validation, compliance, report generation
│   ├── compare.py       # Side-by-side task comparison
│   ├── hpo.py           # High-level HPO wrapper
│   ├── status.py        # Component health checks
│   ├── board.py         # TensorBoard launcher
│   ├── mcp.py           # MCP Server for Hermes Agent
│   ├── init.py          # Interactive configuration wizard
│   ├── cli_clearml.py   # clearml command group (8 sub-commands)
│   ├── cli_optuna.py    # optuna command group (8 sub-commands)
│   ├── cli_langfuse.py  # langfuse command group (6 sub-commands)
│   ├── cli_run.py       # run command group (4 sub-commands)
│   ├── cli_audit.py     # audit command group (3 sub-commands)
│   └── cli_system.py    # system command group (2 sub-commands)
├── tests/               # pytest tests (98 tests)
├── scripts/             # Utility scripts
├── docs/                # English documentation
│   └── cn/              # 中文文档 (Chinese docs)
├── config.yaml          # Optional project config
├── pyproject.toml       # Package config (setuptools)
├── AGENTS.md            # ← This file
├── PLAN.md              # Macro roadmap
└── .gitignore
```

## Core Architecture

### Module Dependency Chain

```
cli.py (Typer) — 6 command groups + 5 top-level commands
  ├── clearml.py        → Task/Queue/Dataset SDK wrappers
  ├── optuna.py          → Study/Trial/Plot SDK wrappers
  ├── langfuse.py        → Trace/Session/Metrics SDK wrappers
  ├── dispatcher.py      → Experiment dispatch (in-memory registry)
  ├── audit.py           → Validation + compliance + report
  ├── compare.py         → Side-by-side comparison
  ├── hpo.py             → High-level HPO wrapper
  ├── status.py          → Component health checks
  ├── board.py           → TensorBoard launcher
  ├── mcp.py             → MCP Server stubs
  └── init.py            → Configuration wizard

All SDK imports are LAZY — cleared at module-level import time,
loaded only when the corresponding command group is invoked.

### CLI Command Tree

```
expflow
├── version / info / mcp / init / config        ← top-level (no SDK deps)
├── clearml     (8 sub-cmds)                     ← lazy import clearml SDK
│   ├── tasks / task / enqueue / dequeue / queues / compare
│   └── dataset-register / dataset-list
├── optuna      (8 sub-cmds)                     ← lazy import optuna SDK
│   ├── create-study / studies / study / delete-study / ask / tell / plot
│   └── run                                      ← high-level HPO wrapper
├── langfuse    (6 sub-cmds)                     ← lazy import langfuse SDK
│   ├── traces / trace / trace-cost / sessions / session / metrics
├── run         (4 sub-cmds)                     ← no SDK deps (in-memory)
│   ├── submit / list / status / cancel
├── audit       (3 sub-cmds)                     ← no SDK deps
│   ├── validate / check-dataset / report
└── system      (2 sub-cmds)                     ← lazy import per check
    ├── status (health checks)
    └── board  (TensorBoard)
```

### Config Loading

```python
from expflow.config import load_config, get

cfg = load_config()            # Load YAML + .env
val = get("clearml.api")       # Dot-separated access
```

Config search order: CWD `config.yaml` → parent dirs → `.env` (env-only overrides API keys)

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
| Unit | Future module logic | None |
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
| `AGENTS.md` | `中文文档` directory reference only | One-line comment |

### Chinese Documentation Convention

- Chinese docs live in `docs/cn/*.zh-CN.md`
- Must be **line-to-line translations** of English originals (same line count)
- This enables: diff tracking, side-by-side editing, automated sync checks
- Update English first, then mirror edits to Chinese version

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
grep __version__ expflow/__init__.py  # e.g. '0.1.0'
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
- `pyright` zero errors in `expflow/` package

## Git Conventions

```bash
git add <files>
git commit -m "<type>: <description>"
git tag v0.1.0          # Semantic versioning
```

`.gitignore` covers: `__pycache__/`, `*.egg-info/`, `dist/`, `venv/`, `.ruff_cache/`, etc.

## Versioning

**Current version: 0.1.0** (pre-release)
- Semantic versioning with 0.x.y — x=feature iteration, y=fix/minor
- Don't bump to 1.0.0 before official release
- Version defined in `expflow/__init__.py` `__version__`
- Sync `pyproject.toml` version field
- Tag: `git tag v0.x.y && git push --tags`

## File Operations (AI Assistant)

- X Don't use `cat`/`grep`/`sed`/`ls` — use `read_file`/`search_files`/`patch`
- ✅ Use `write_file` for creating files, `terminal` for running commands
- ✅ Use `search_files(target="files")` instead of `ls`
- ✅ Use `search_files(pattern="content")` instead of `grep`

## Pitfalls

### Config Cache Is Global

`_config_cache` in `config.py` is a module-level global, shared across all imports.
Tests must reset cache between runs. Use pytest fixtures:

```python
@pytest.fixture(autouse=True)
def reset_config():
    from expflow import config
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
| **Dataset** | `expflow clearml dataset-register <name> --path <path> --compliance allowed/forbidden` | Register dataset |
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
| `exp_list_runs` | List recent experiments | clearml SDK / clearml-mcp |
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

#### Scenario 1: Task 2 parameter scan

```
1. Hermes reflects on past results:
   expflow clearml compare <task_v1> <task_v2>

2. Hermes searches wiki for context:
   read ~/wiki/concepts/ar-step-and-val-segments.md

3. Hermes searches session history:
   session_search: "nu=1.0 ar_steps"

4. Hermes designs HPO experiment:
   expflow optuna run train_task2.py \\
       --trials 24 --n-jobs 2 --study-name hpo_task2_v3

5. Hermes checks results:
   expflow optuna study hpo_task2_v3
   expflow optuna plot hpo_task2_v3 --type history

6. Hermes writes findings to wiki:
   update ~/wiki/concepts/ar-step-and-val-segments.md
```

#### Scenario 2: Dataset compliance check

```
1. Register a dataset:
   expflow clearml dataset-register burgers_nu0.001 \\
       --path data/burgers.hdf5 --compliance allowed

2. Verify it's registered:
   expflow clearml dataset-list

3. Before submission, run compliance audit:
   expflow audit check-dataset burgers_nu0.001 --compliance allowed
   expflow audit validate --experiment <id>
```

#### Scenario 3: 7x24 autonomous experiment loop

Cron-triggered Hermes session uses expflow to drive the full cycle:

```
1. Collect:    expflow clearml tasks --status completed --project pdebench
2. Analyze:    expflow clearml compare <best> <second>
3. Research:   read ~/wiki/concepts/ + session_search
4. Design:     expflow optuna run <script> --trials 50
5. Submit:     expflow run submit <cmd> --queue gpu0 --tag v3
6. Record:     update ~/wiki/log.md
```

#### Scenario 4: Full audit report

```
expflow audit validate <experiment_id>
expflow audit report <experiment_id>
expflow langfuse trace-cost <trace_id>
expflow system status
```
