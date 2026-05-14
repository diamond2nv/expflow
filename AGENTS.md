# expflow — AI Agent Development Guide

This file is for AI coding assistants (Hermes Agent, OpenCode, Claude Code, etc.)
working on this project. It describes the project structure, key patterns, pitfalls, and constraints.

## Quick Navigation

```
~/Gitlab/Agentic4Sci/expflow/
├── expflow/             # Main Python package
│   ├── __init__.py      # Version, exports
│   ├── config.py        # YAML + .env config loader
│   └── cli.py           # Typer CLI
├── tests/               # pytest tests
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
config.py        — Config loader (YAML + .env merge, dot-separated access)
       ↓
expflow/         — Future: clearml/optuna/langfuse/experiment modules
       ↓
cli.py           — Typer CLI (top-level commands)
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
