# expflow Phase 2: optuna Integration — Implementation Plan

> **For Hermes:** 按序执行。先写单元测试（mock optuna），再实现功能，然后 CLI+entry point 测试。

**Goal:** 实现 expflow 的 optuna 集成模块，支持 study/trial CRUD、ask/tell 优化循环、可视化命令

**Architecture:** `expflow/optuna.py` 封装核心逻辑，`cli_optuna.py` 提供 CLI 子命令，懒加载 optuna SDK

**Tech Stack:** Python 3.11+, optuna>=4.0, typer>=0.9, pytest (mock optuna)

---

## 执行顺序

```
Phase A: 单元测试 — mock optuna
   A1: Study CRUD (create, list, get, delete)
   A2: Trial management (ask, tell)
   A3: Visualization (plot -> save to file)
   A4: Error handling

Phase B: 功能实现
   B1: expflow/optuna.py — 核心模块
   B2: expflow/cli_optuna.py — CLI 子命令
   B3: expflow/cli.py 注册 optuna 子命令组

Phase C: CLI 测试
   C1: CliRunner mock optuna 测试
   C2: 第三方 entry point 测试
```

## API 设计

```python
def create_study(study_name: str, direction: str = "minimize",
                 storage: str | None = None) -> dict:
    """Create optuna study. Returns study_id, name, direction."""

def list_studies(storage: str | None = None) -> list[dict]:
    """List all studies."""

def get_study(study_name: str, storage: str | None = None) -> dict:
    """Get study details (best trial, params, value)."""

def delete_study(study_name: str, storage: str | None = None) -> dict:
    """Delete a study."""

def ask_trial(study_name: str, storage: str | None = None) -> dict:
    """Ask for next trial parameters (distributed optimization)."""

def tell_trial(study_name: str, trial_number: int, value: float,
               storage: str | None = None) -> dict:
    """Report trial result to optuna."""

def plot_study(study_name: str, plot_type: str = "history",
               output_path: str | None = None,
               storage: str | None = None) -> dict:
    """Generate optimization visualization."""
```

## CLI 子命令

```
expflow optuna create-study <name> [--direction minimize|maximize]
expflow optuna studies
expflow optuna study <name>
expflow optuna delete-study <name>
expflow optuna ask <study_name>
expflow optuna tell <study_name> <trial_number> <value>
expflow optuna plot <study_name> [--type history|parallel_coordinate|slice|contour]
```
