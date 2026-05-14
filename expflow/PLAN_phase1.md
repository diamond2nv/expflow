# expflow Phase 1: clearml Integration — Implementation Plan

> **For Hermes:** 按序执行此计划。先实现功能（TDD），再写测试三部曲：单元→集成→CLI

**Goal:** 实现 expflow 的 clearml 集成模块，包含 task CRUD、queue 管理、dataset compliance API，以及对应的 CLI 子命令

**Architecture:** 模块化设计：核心逻辑在 `expflow/clearml.py`，MCP 工具在 `expflow/clearml_mcp.py`，CLI 通过 typer 子命令树暴露

**Tech Stack:** Python 3.11+, clearml>=1.17, typer>=0.9, pytest(with mocked clearml)

---

## 执行顺序

```
Phase A: 单元测试层    — 测试用 mock clearml 的纯逻辑
   A1: clearml 模块骨架 + Task 列表查询
   A2: Queue 管理（list, enqueue, dequeue, status）
   A3: Dataset compliance（register, list, audit）

Phase B: 功能实现层    — TDD 顺序写真实 clearml 代码
   B1: expflow/clearml.py — 核心模块
   B2: 对应 cli.py clearml 子命令

Phase C: 集成测试层    — 需要 clearml server 连接
   C1: task CRUD 集成
   C2: queue 生命周期集成
   C3: dataset compliance 集成

Phase D: CLI 第三方测试  — entry point 脚本 + --no-deps 验证
   D1: test_entry_point_script()
   D2: test_entry_point_missing_dep_shows_error()
   D3: CliRunner 全命令覆盖测试
```

## 关键设计决策

### 1. clearml SDK 的使用模式

```python
# expflow 不直接调用 clearml SDK 的 OOP API，而是封装为简洁的函数接口：
def list_tasks(
    project_name: str | None = None,
    task_name: str | None = None,
    tags: list[str] | None = None,
    status: list[str] | None = None,
) -> list[dict]:
    """List clearml tasks, return serializable dicts."""
    ...

def enqueue_task(task_id: str, queue_name: str = "default") -> dict:
    """Enqueue a task to a clearml queue."""
    ...

def register_dataset(
    name: str,
    version: str,
    path: str,
    compliance: Literal["allowed", "forbidden"],
    **metadata,
) -> dict:
    """Register a PDEBench dataset with compliance annotation."""
    ...
```

### 2. Dataset Compliance 设计

```python
# 每个 dataset 注册时强制标注合规性
# 存储为 clearml dataset 的 metadata（便于 audit 查询）
DATASET_COMPLIANCE_TAG = "expflow:compliance"  # value: "allowed" | "forbidden"
```

### 3. MCP 工具预留

clearml_mcp.py 暂不实现（Phase 1 只做 CLI，MCP 留到后续 Phase），但保留空文件作为接口占位。

---

## Phase A: 单元测试（先写测试）

### Task A1: expflow/clearml 模块骨架 + Task 列表

```
RED:   test_clearml_list_tasks_with_mock
GREEN: expflow/clearml.py — 实现 list_tasks()
```

### Task A2: Queue 管理

```
RED:   test_clearml_queue_operations
GREEN: expflow/clearml.py — enqueue_task(), get_queue_status()
```

### Task A3: Dataset compliance

```
RED:   test_clearml_dataset_compliance
GREEN: expflow/clearml.py — register_dataset(), list_datasets()
```

---

## Phase B: 功能实现

### Task B1: 实现 expflow/clearml.py

**API 函数清单：**
- `list_tasks(project_name, task_name, tags, status)` → list[dict]
- `get_task(task_id)` → dict
- `enqueue_task(task_id, queue_name)` → dict
- `dequeue_task(task_id)` → dict
- `list_queues()` → list[dict]
- `get_queue_status(queue_name)` → dict
- `register_dataset(name, version, path, compliance, **metadata)` → dict
- `list_datasets(name_filter, compliance_filter)` → list[dict]

### Task B2: CLI 子命令

在 expflow/cli.py 里添加 expflow clearml 子命令组：

```
expflow clearml tasks              # 列任务
expflow clearml task <id>          # 查任务详情
expflow clearml enqueue <id>       # 入队
expflow clearml dequeue <id>       # 出队
expflow clearml queues             # 列队列
expflow clearml dataset register   # 注册数据集
expflow clearml dataset list       # 列数据集
```

---

## Phase C: 集成测试

使用 `@pytest.mark.integration` 标记，默认跳过。
需要 connected clearml server（通过 api_server 环境变量配置）。

---

## Phase D: CLI 第三方测试

对标 hfpclawer 的测试模式：
1. `test_entry_point_script()` — subprocess 模拟 entry point
2. `test_entry_point_missing_dep_show_error()` — temp venv + `--no-deps`
3. `test_cli_commands()` — CliRunner 全覆盖
