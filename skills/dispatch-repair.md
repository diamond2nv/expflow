# expflow Dispatch & Repair — Hermes Agent Skill

> **Domain**: expflow v0.6.0 — SQLite-backed experiment dispatch database + three-level auto-repair
> **Target**: LLM agents (Hermes, OpenCode, Claude Code) using expflow for PDEBench experiment orchestration

---

## Overview

expflow v0.6.0 introduces two major subsystems:

### 1. DispatchDB — SQLite Experiment Scheduler

Replaces the volatile in-memory + `.jsonl` experiment registry with a proper SQLite database.

**Architecture** (borrowed from hfpapers-crawler PaperStore):
- `sqlite3` stdlib only (zero extra dependencies)
- WAL mode + `synchronous=NORMAL` + `foreign_keys=ON`
- Write transactions: `threading.Lock()` + `BEGIN IMMEDIATE`
- Read transactions: no lock
- `row_factory = sqlite3.Row` → `dict(row)` for JSON output

**5 tables + 8 indexes:**
| Table | Purpose | Key Fields |
|-------|---------|------------|
| `experiments` | Experiment definitions + status FSM | id, parent_id, root_id, status, fsm_state, script, args_json, clearml_task_id, best_value |
| `branches` | Tree hierarchy (parent→child) | parent_exp_id, child_exp_id, strategy, depth |
| `artifacts` | Output tracking | experiment_id, type (checkpoint/plot/log), path, checksum |
| `metrics` | Standardized numeric scores | experiment_id, name, value, iteration, group_name |
| `audit_log` | Immutable event trail | experiment_id, event_type, detail_json |

### 2. Repair Stage — Three-Level Auto-Repair

Auto-fixes failed clearml experiments without manual intervention.

| Level | Cost | Coverage | Method |
|-------|:----:|:--------:|--------|
| L0 | **0 token** | ~80% common failures | Rule engine: git clone fail, ModuleNotFoundError, pip conflict |
| L1 | Low | Remaining 20% | Traceback extraction + error localization |
| L2 | High (subagent) | Stubborn failures | Reflection subagent with expert context |

---

## CLI Reference

### Pipeline with auto-repair

```bash
# Submit and wait for completion, auto-repair on failure
expflow pipeline submit train_task1.py --queue default --repair --wait

# Full HPO pipeline with L2 reflection
expflow pipeline submit-full train_task1.py --queue default \
    --trials 50 --parallel 4 \
    --repair --repair-reflection --wait
```

### Dispatch database management

```bash
# List recent experiments
expflow dispatch list --limit 20

# Filter by status/project
expflow dispatch list --status failed --project PDEBench

# Show experiment details
expflow dispatch status exp:snow_1234567890

# Show experiment tree
expflow dispatch tree exp:snow_1234567890

# Archive old experiments
expflow dispatch archive 2025-06-01                      # move to archive/
expflow dispatch archive 2025-06-01 --dry-run             # preview only

# Database stats
expflow dispatch stats

# Audit log
expflow dispatch audit-log --experiment exp:snow_1234567890
expflow dispatch audit-log --event-type repair --limit 20
```

### One-shot iteration with auto-repair

```bash
expflow iterate run --task <task_id>
# If pipeline fails → auto-triggers RepairStage → pipe_result["repair"]
```

---

## MCP Tools (24 total)

New in v0.6.0:

| Tool | Function |
|------|----------|
| `exp_db_stats` | Dispatch database statistics |
| `exp_db_tree` | Experiment tree by root_id |
| `exp_db_archive` | Archive old experiments |
| `exp_db_audit_log` | Query audit entries |
| `exp_db_metrics` | Get metrics for an experiment |

---

## Key Files

```
expflow_pde/
├── dispatch_db.py     — DispatchDB SQLite class (869 lines)
├── repair_rules.py    — L0 rule engine (3 rules)
├── repair.py          — RepairStage (L0→L1→L2)
├── pipeline.py        — ExperimentPipeline.repair_task()
├── cli_dispatch.py    — dispatch CLI (6 sub-commands)
├── cli_pipeline.py    — --repair flag
├── iterate.py         — auto-repair on submit failure
└── mcp_server.py      — 6 new MCP tools (total 24)
```

## Design Decisions

- **SQLite over clearml as scheduler**: clearml-server 断网时调度不受影响。SQLite 做调度决策，clearml agent 做远程执行（互补非竞争）。
- **L0 over LLM first**: ~80% 的 clearml-agent 失败是配置问题（git/module/pip），纯规则解决，0 token。
- **L2 subagent only for reflection**: 子 agent 只出方案不执行代码（leaf agent 限制 + 防止并发冲突）。
- **Snowflake ID**: 64-bit yitter drift algorithm, worker_id=1, aligned with hfpapers-crawler (base_time=2024-10-04).
- **Cold-hot archive**: `archive` moves completed experiments to separate .db files by date prefix. Queries default to hot only; archive only on explicit `--include-archive`.
