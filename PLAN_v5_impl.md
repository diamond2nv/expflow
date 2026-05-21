# expflow v0.6 迭代执行计划 — 基于 PLAN_v4 + Repair Stage + SQLite 学习

> **当前基线**: f12d337, 382 tests, 0 failed (physicsnemo-cpu env)
> **总工作估时**: ~6-8h

---

## 任务清单（按执行顺序）

### Phase 1 — DispatchDB (核心基础设施) [~3h]
TDD for each sub-task, since these are the foundation of all future dispatch logic.

| # | 任务 | 文件变更 | 估时 |
|:--:|------|----------|:----:|
| 1.1 | DispatchDB 类骨架 + 连接管理 | Create `expflow_pde/dispatch_db.py` | 30min |
| 1.2 | Schema 初始化 + 迁移 (幂等) | `dispatch_db.py` | 20min |
| 1.3 | 实验 CRUD (register, update_status, get) | `dispatch_db.py` | 40min |
| 1.4 | 分支追踪 (branches 表 + 树查询) | `dispatch_db.py` | 20min |
| 1.5 | Metrics + Artifacts + AuditLog 表 | `dispatch_db.py` | 20min |
| 1.6 | 集成: dispatch.py 从 jsonl → DispatchDB | `dispatch.py` | 30min |

### Phase 2 — Repair Stage [~2h]
Build on DispatchDB (each repair creates child exp in tree).

| # | 任务 | 文件变更 | 估时 |
|:--:|------|----------|:----:|
| 2.1 | L0 规则引擎 + 3 条规则 | Create `expflow_pde/repair_rules.py` | 30min |
| 2.2 | RepairStage 类 (L1 快速修复脚手架) | Create `expflow_pde/repair.py` | 30min |
| 2.3 | L2 反思 subagent 入口 | `repair.py` | 20min |
| 2.4 | Pipeline 集成: `--repair` flag | `pipeline.py`, `cli_pipeline.py` | 20min |
| 2.5 | iterate.py 集成: 失败自动 repair | `iterate.py` | 20min |

### Phase 3 — 冷热归档 [~0.5h]
Minimal — only needed when DB > 500 MB.

| # | 任务 | 文件变更 | 估时 |
|:--:|------|----------|:----:|
| 3.1 | `dispatch_db.archive()` 方法 | `dispatch_db.py` | 15min |
| 3.2 | CLI: `expflow archive --before <date>` | `cli_dispatch.py` | 15min |

### Phase 4 — CLI + MCP 增强 [~0.5h]

| # | 任务 | 文件变更 | 估时 |
|:--:|------|----------|:----:|
| 4.1 | `expflow status --tree` 树形输出 | `cli_run.py` | 15min |
| 4.2 | MCP tools: 新增 dispatch 相关工具 | `mcp.py` | 15min |

---

## 执行策略

### 执行顺序
1. Phase 1 → Phase 2 → Phase 4 → Phase 3（归档可延迟）
2. 每子任务严格 TDD（先写测试→验证失败→实现→验证通过）
3. 每子任务完成后 git commit（精细粒度）

### 分派模式
- **Phase 1**（DispatchDB 基础设施）由我直接写——这是整个系统的新基石，需要最多的设计和连贯性判断
- **Phase 2.1**（repair_rules.py L0 规则引擎）可交给 subagent——纯规则匹配，独立性强
- **Phase 2.2-2.5**（repair.py + 集成）由我直接写——需要理解 pipeline/iterate 的现有调用链
- **Phase 3-4**（归档 + CLI）可交给 subagent——独立性强

### 不要做的
- ❌ 不改现有 clearml/optuna/langfuse 模块的功能逻辑
- ❌ 不改现有 test_*.py 的已有测试（只新增）
- ❌ 不改 pyproject.toml 的依赖
- ❌ 不涉及 clearml-server 的修改
