# expflow Code Review — 2026-05-15

> Review triggered by PDEBench Phase 1 needs: SCALARS/metrics, MCP, Optuna↔ClearML bridge, auto-trajectory.

## Review Scope

33 Python files, ~6,600 LOC, 98 tests, 6 phases + PEP8.

## Findings

### 1. Critical Gaps

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **No SCALARS/metrics support** — `Logger.report_scalar()` missing. PDEBench manually calls `cl_logger.report_scalar()` in every training script. | `clearml.py` | Blocks all experiment tracking |
| 2 | **MCP server: 5/13 documented tools implemented** — AGENTS.md lists `exp_get_metrics`, `exp_start_hpo`, `exp_hpo_plot`, `exp_compare_runs`, `exp_board_url`, `exp_check_compliance`, `exp_create_study` — none exist. | `mcp_server.py` | Blocks autonomous agent workflows |
| 3 | **No Optuna↔ClearML bridge** — `hpo.py` (49 lines) creates a study but never runs trials, logs to clearml, or tracks per-epoch SCALARS. | `hpo.py` | Blocks HPO auto-tracking |
| 4 | **`init_tracking()` doesn't return logger handle** — Returns task dict but no way to report scalars. User must `Task.get_task(id).get_logger()` manually. | `clearml.py:940-1003` | Boilerplate in every script |

### 2. Bugs

| # | Issue | Location | Severity |
|---|-------|----------|----------|
| 1 | **`scheduler_create()` is a pure stub** — returns `{"status": "created"}` without creating anything. | `clearml.py:780-796` | Broken |
| 2 | **`scheduler_start()` blocks forever** — `scheduler.start()` is a daemon loop, return unreachable. | `clearml.py:915-928` | Broken |
| 3 | **Version mismatch** — `cli.py:_EXPFLOW_VERSION = "0.1.0"` vs `__init__.py:__version__ = "0.2.0"`. | `cli.py:10`, `__init__.py:5` | Confusing |
| 4 | **Git commit not auto-detected** — PDEBench had to manually add `os.chdir()` before `Task.init()`. | `clearml.py:940-1003` | Minor |

### 3. MCP Tool Gap: AGENTS.md vs Actual Code

| Tool (AGENTS.md) | In `mcp_server.py`? | Code line |
|:---|:---|:---|
| `exp_list_runs` | ✅ | L45 |
| `exp_get_run` | ✅ | L49 |
| `exp_enqueue_run` | ✅ | L54 |
| `exp_dequeue_run` | ✅ | L59 |
| `exp_list_studies` | ✅ | L71 |
| `exp_get_study` | ✅ | L76 |
| `exp_dataset_upload` | ✅ | L93 |
| `exp_dataset_download` | ✅ | L112 |
| `exp_dataset_lineage` | ✅ | L131 |
| `exp_list_datasets` | ✅ | L136 |
| `exp_model_list` | ✅ | L153 |
| `exp_model_upload` | ✅ | L166 |
| `exp_pipeline_*` (5) | ✅ | L194-266 |
| `exp_generate_report` | ✅ | L276 |
| `exp_config_status` | ✅ | L296 |
| `exp_get_metrics` | ❌ | — |
| `exp_compare_runs` | ❌ | — |
| `exp_start_hpo` | ❌ | — |
| `exp_hpo_plot` | ❌ | — |
| `exp_check_compliance` | ❌ | — |
| `exp_board_url` | ❌ | — |
| `exp_create_study` | ❌ | — |
| `exp_ask_trial` | ❌ | — |
| `exp_tell_trial` | ❌ | — |
| `exp_register_dataset` | ❌ | — |

### 4. PDEBench Integration Gaps

Current PDEBench boilerplate (from `hpo_ft_optuna.py`) that expflow should absorb:

```python
from clearml import Task as CLTask
cl_task = CLTask.init(project_name='PDEBench', task_name=f'HPO_{tag}',
                       tags=['hpo', tag], output_uri=True)
cl_task.connect({'lr': lr, 'epochs': epochs, 'batch_size': batch_size, 'n_train': n_train})
cl_logger = cl_task.get_logger()
for epoch in range(epochs):
    cl_logger.report_scalar('Loss', 'Train', avg_loss, iteration=ep+1)
    cl_logger.report_scalar('Score', 'Seg Total', seg_total, iteration=ep+1)
    cl_logger.report_scalar('Score', 'Seg1', seg['seg1_score'], iteration=ep+1)
    cl_logger.report_scalar('Score', 'Seg2', seg['seg2_score'], iteration=ep+1)
    cl_logger.report_scalar('Score', 'Seg3', seg['seg3_score'], iteration=ep+1)
    cl_logger.report_scalar('Loss', 'Val MSE', mse, iteration=ep+1)
cl_task.close()
```

Should become:

```python
from expflow.clearml import init_tracking, report_scalars_batch
info = init_tracking(task_name=f'HPO_{tag}', task_params={'lr': lr, ...})
for epoch in range(epochs):
    report_scalars_batch(info['task_id'], epoch, {
        'Loss': {'Train': loss, 'Val MSE': mse},
        'Score': {'Seg Total': seg_total, 'Seg1': seg1, 'Seg2': seg2, 'Seg3': seg3},
    })
```

### 5. Code Health

| Metric | Value |
|--------|-------|
| ruff errors | 0 (pre-edit) |
| pyright errors | 0 (pre-edit) |
| test count | 98 |
| test coverage | Units only (all mocked), no integration tests |
| Version | `__init__.py` 0.2.0, `cli.py` 0.1.0 (mismatch) |

---

## Phase 1 Implementation Plan (2026-05-15)

### A. Bug Fixes (4 items)

| # | Fix | File | Lines |
|---|-----|------|-------|
| A1 | Version sync — import from `expflow.__init__` | `cli.py:10` | 1 |
| A2 | `scheduler_create()` — actual TaskScheduler creation | `clearml.py:780-796` | ~12 |
| A3 | `scheduler_start()` — add `blocking` param, daemon thread | `clearml.py:915-928` | ~10 |
| A4 | Git auto-detection — `os.chdir()` + `_get_git_commit()` | `clearml.py:940-1003` | ~20 |

### B. SCALARS/Metrics Layer

| # | Function | Description |
|---|----------|-------------|
| B1 | `report_scalar(task_id, title, series, value, iteration)` | Single scalar to clearml Logger |
| B2 | `report_scalars_batch(task_id, iteration, metrics)` | Batch: `{title: {series: value}}` |
| B3 | `get_task_metrics(task_id, filter, max_samples)` | Query all scalars from task |
| B4 | `get_task_extended(task_id, include_metrics)` | Enhanced get_task with optional metrics |

### C. init_tracking() Enhancement

| # | Change |
|---|--------|
| C1 | Add `project_root` parameter |
| C2 | Auto-detect git commit |
| C3 | Return `git_commit` and `logger_ready` in result |
| C4 | Update docstring with report_scalar usage example |

### D. MCP Additions

| # | Tool | Wraps |
|---|------|-------|
| D1 | `exp_get_metrics(task_id, metric_filter, limit)` | `get_task_metrics()` |
| D2 | `exp_compare_runs(task_a, task_b)` | Enhanced `compare_tasks()` |

### E. Exports Update

| File | New Exports |
|------|-------------|
| `__init__.py` | `report_scalar`, `report_scalars_batch`, `get_task_metrics`, `get_task_extended` |

### F. Tests

| Module | New Tests | Total |
|--------|:---:|:---:|
| `test_clearml.py` | +14 | 38→52 |
| `test_dispatcher.py` | +4 (MCP compare) | 13→17 |
| **Total** | **+18** | **98→116** |

### G. Verification

```bash
ruff format . && ruff check --fix .
pyright expflow/
python -m pytest tests/ -v
```

---

## Phase 2 Plan (Future): Optuna↔ClearML Bridge

### Goals

1. **HPOTracker class** — wraps `Task.init()` per trial + per-epoch SCALARS + optuna tell
2. **`objective_factory(config)`** — generates clearml-integrated objective function for `study.optimize()`
3. **`study_to_clearml()`** — sync optuna study results to clearml for visualization
4. **Config-driven HPO** — YAML search space definition, auto-expflow study creation
5. **Resumable HPO** — save/restore from checkpoint, merge partial results

### MCP Tools for Phase 2

```
exp_start_hpo       — Start HPO study with clearml tracking
exp_hpo_plot        — Generate optimization plot via MCP
exp_create_study    — Create optuna study via MCP
exp_ask_trial       — Ask for next trial params (distributed HPO)
exp_tell_trial      — Report trial result (distributed HPO)
```

## Phase 3 Plan (Future): Auto-Trajectory + Observability

1. **TrajectoryCollector** — auto-capture per-epoch metrics for any training script
2. **Auto-merge** partial runs into single clearml task
3. **Langfuse trace ↔ clearml task linking**
4. **7x24 autonomous loop** with cron + scheduler
5. **MCP tools**: `exp_board_url`, `exp_check_compliance`, `exp_register_dataset`
