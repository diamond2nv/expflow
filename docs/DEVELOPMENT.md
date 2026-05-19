# Developer Guide

## Development Environment

```bash
# Clone and install
git clone https://github.com/diamond2nv/expflow.git
cd expflow
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Verify
expflow --help
```

## Code Standards

| Standard | Value |
|----------|-------|
| **Language** | Python 3.11+ |
| **Formatter** | Ruff (line-length=100, double quotes) |
| **Linter** | Ruff (E, W, F, I, N) |
| **Type checker** | Pyright (strict mode on `expflow_pde/`) |
| **Test framework** | Pytest (329+ tests) |
| **Package manager** | pip + setuptools |

### PEP8 Internationalization

All Python files must be 100% English-only:
- **Comments**: English only (docstrings, inline comments, block comments)
- **Strings**: English only (print, log, error messages, CLI output)
- **Variable/function/class names**: English only (PEP8 naming)
- **No Chinese characters, emoji, or box-drawing characters** in `.py`/`.yaml`/`.sh` files

Why: `conda` environment may have `LC_ALL=C` which causes `UnicodeEncodeError` on non-ASCII output.

Exceptions (Chinese allowed):
| Location | What | Reason |
|----------|------|--------|
| `docs/cn/` | Chinese documentation | Intended for Chinese readers |
| `README.md` | `简体中文` navigation link only | One-line label |
| `.hermes/` | Hermes agent plans | Internal tooling |

### Chinese Documentation Convention

As established in the hfpapers-crawler project, Chinese docs must be **line-to-line translations**
of English originals. This enables diff tracking, side-by-side editing, and automated sync checks.
Always update the English version first, then mirror edits to Chinese.

## Running Commands

```bash
# Format code
ruff format .

# Lint + auto-fix
ruff check --fix .

# Type check
pyright .

# Run tests
python -m pytest tests/ -v            # All tests, verbose
python -m pytest tests/ -q            # Quiet mode
python -m pytest tests/test_pin.py    # Single test file
python -m pytest tests/ -x -v         # Stop on first failure

# Run tests with coverage
python -m pytest tests/ --cov=expflow_pde

# Build package
python -m build

# Verify package
twine check dist/*
```

## Testing Guidelines

### Test Strategy

| Category | Description | External Dependencies |
|----------|-------------|----------------------|
| Unit | Config CRUD, CLI parsing | None |
| Unit | Module-level logic (pin, metrics, compare, equations, analyze) | None |
| Unit | FSM (fysom state machine) | fysom |
| Integration | Config ↔ filesystem | Filesystem |
| Integration | Test entry point (`built_wheel` fixture) | Build isolation |
| E2E (marked) | clearml/optuna/langfuse interactions | External services |

### Creating New Tests

```bash
# 1. Create test file
touch tests/test_<module>.py

# 2. Use tmp_path for filesystem isolation
# 3. Mock external SDK imports with monkeypatch
# 4. Mark integration tests with @pytest.mark.integration
```

### Key Fixture Patterns

```python
# PIN test with isolated directory
@pytest.fixture(autouse=True)
def setup(self, tmp_path, monkeypatch):
    monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))

# FSM test
@pytest.fixture
def fsm():
    from expflow_pde.fsm import create_experiment_fsm
    return create_experiment_fsm()

# Entry point test (uses built_wheel fixture from conftest.py)
def test_entry_point(built_wheel):
    result = subprocess.run(...)
    assert result.returncode == 0
```

## Lazy Import Pattern

All clearml/optuna/langfuse SDK imports must be **lazy** — inside function bodies, not at module level:

```python
# ✅ CORRECT — lazy import inside function
def list_tasks(project=None, tags=None):
    from clearml import Task
    tasks = Task.get_tasks(project_name=project or "PDEBench")
    return [_serialize_task(t) for t in tasks]

# ❌ WRONG — module-level import triggers SDK dependency at import time
from clearml import Task  # Never do this
```

The `__init__.py` also uses `__getattr__` for lazy re-exports:

```python
# expflow_pde/__init__.py
def __getattr__(name: str):
    _lazy_map = {
        "list_tasks": ("expflow_pde.clearml", "list_tasks"),
    }
    if name in _lazy_map:
        mod_path, attr = _lazy_map[name]
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(...)
```

## Package Structure

```
expflow/
├── expflow_pde/                  # Main Python package (33 modules)
│   ├── __init__.py               # Version + lazy re-exports
│   ├── __init__.pyi              # Type stubs for IDE/type-checker
│   ├── cli.py                    # Typer CLI (8 command groups)
│   ├── clearml.py                # ClearML SDK wrapper (~1K lines)
│   ├── optuna.py                 # Optuna SDK wrapper
│   ├── langfuse.py               # Langfuse SDK wrapper
│   ├── hpo.py                    # 3-mode HPO runner
│   ├── pipeline.py               # Competition pipeline
│   ├── dispatcher.py             # Experiment registry
│   ├── fsm.py                    # 7-state FSM
│   ├── pin.py                    # PIN protection
│   ├── metrics.py                # Metric registry
│   ├── compare.py                # Score comparison
│   ├── equations.py              # PDE equation registry
│   ├── analyze.py                # Competition intelligence
│   ├── audit.py                  # Compliance validation
│   ├── config.py                 # YAML + .env loader
│   ├── worktree.py               # Git worktree isolation
│   ├── snowflake.py              # ID generator
│   ├── status.py                 # Health checks
│   ├── board.py                  # TensorBoard launcher
│   ├── mcp.py / mcp_server.py    # MCP server
│   ├── cli_*.py (8 files)        # CLI command groups
│   ├── init.py                   # Interactive config wizard
│   └── skills/                   # Agent skills (for Hermes Agent)
│       ├── expflow-pipeline-hpo.md
│       ├── experiment-lifecycle-governance.md
│       ├── clearml-metrics-logging-pattern.md
│       └── competition-task-intelligence.md
├── tests/                        # 17 test files, 329+ tests
│   ├── conftest.py               # Shared fixtures
│   ├── test_*.py                 # Per-module tests
│   └── test_entry_point.py       # Package build verification
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   ├── DEVELOPMENT.md
│   ├── DATA_LAYER.md
│   ├── COMPETITION.md
│   └── cn/                       # Chinese translations
├── pyproject.toml                # Package config
├── README.md                     # Project README
├── AGENTS.md                     # AI Agent instructions
├── PLAN.md                       # Roadmap
└── .gitignore
```

## Building for Release

```bash
# 1. Format & lint
ruff format .
ruff check --fix .

# 2. Type check
pyright .

# 3. Test
python -m pytest tests/ -v

# 4. Verify version alignment
grep __version__ expflow_pde/__init__.py   # e.g. '0.3.0'
grep ^version pyproject.toml               # Must match

# 5. Build + verify
python -m build
twine check dist/*

# 6. Tag
git tag v0.3.0
git push --tags

# 7. Publish
twine upload dist/*
```

## Pre-Release Checklist

- [ ] `ruff format .` — no changes
- [ ] `ruff check --fix .` — zero errors
- [ ] `pyright .` — zero errors in `expflow_pde/`
- [ ] `python -m pytest tests/ -v` — 329+ passed
- [ ] `python -m build` — success
- [ ] `twine check dist/*` — PASSED
- [ ] Version match: `__init__.py` == `pyproject.toml`
- [ ] Updated `docs/` if API changed
- [ ] Updated `AGENTS.md` if CLI structure changed
