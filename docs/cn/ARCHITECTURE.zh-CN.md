# expflow-pde 系统架构

> **包名**：`expflow-pde`（PyPI: `expflow-pde`，CLI: `expflow`）
> **仓库**：[github.com/diamond2nv/expflow](https://github.com/diamond2nv/expflow)

## 项目概述

expflow-pde 是一个面向 PDEBench/Agentic4Sci 的实验工作流编排工具包。
它在 PDE 训练脚本和生产级 MLOps 之间架起桥梁，通过统一的 CLI 整合
ClearML（实验追踪）、Optuna（超参优化）和 Langfuse（LLM 可观测性）。

工具包采用**可选依赖**设计——核心 CLI 无需安装任何外部 SDK 即可运行，
特定 SDK 的功能按需加载。

## 架构分层

```
┌──────────────────────────────────────────────────────────────────┐
│                       CLI (Typer)                                 │
│  expflow version | info | clearml | optuna | langfuse | run      │
│  expflow audit | system | pin | analyze | pipeline | mcp | init  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    MCP Server (stdio)                              │
│  18+ MCP 工具: exp_compare_scores, exp_list_workers,             │
│  exp_list_tasks, exp_trace_experiment, exp_submit_experiment...   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    延迟导入层                                       │
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
│  │  (SHA-256 PIN) │  │  (标准度量注册  │  │  (7状态实验        │  │
│  │                │  │   表)          │  │   有限状态机)      │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  equations.py  │  │  analyze.py    │  │  compare.py        │  │
│  │  (11 个 PDE    │  │  (竞赛智能分析) │  │  (分数对比)        │  │
│  │   方程注册表)   │  │                │  │                    │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                     支撑模块                                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  config.py     │  │  dispatcher.py │  │  pipeline.py       │  │
│  │  (YAML + .env) │  │  (内存实验注册  │  │  (3模式 Pipeline)  │  │
│  │                │  │   表)          │  │                    │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  hpo.py        │  │  worktree.py   │  │  snowflake.py     │  │
│  │  (3模式 HPO:   │  │  (Git Worktree │  │  (ID 生成器)       │  │
│  │   本地/分布式/  │  │   实验隔离)    │  │                    │  │
│  │   优化器)       │  │                │  │                    │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐                           │
│  │  status.py     │  │  board.py      │                           │
│  │  (组件健康检查)  │  │  (TensorBoard  │                          │
│  │                 │  │   启动器)      │                           │
│  └────────────────┘  └────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

## 模块职责

| 模块 | 职责 | 依赖 | CLI 分组 |
|------|------|:----:|----------|
| `cli.py` | Typer CLI，8 个命令组，5 个顶级命令 | typer | `expflow` |
| `clearml.py` | Task/Queue/Dataset/Worker/Scheduler CRUD（全部延迟导入） | clearml (可选) | `expflow clearml` |
| `optuna.py` | Study/Trial/Plot 管理（延迟导入） | optuna (可选) | `expflow optuna` |
| `langfuse.py` | Trace/Session/Cost/Metrics（延迟导入） | langfuse (可选) | `expflow langfuse` |
| `hpo.py` | 3 模式超参优化运行器 | optuna, clearml (可选) | `expflow optuna run` |
| `pipeline.py` | 竞赛流水线：训练→评估→提交 | clearml (可选) | `expflow pipeline` |
| `dispatcher.py` | 内存实验注册表+生命周期 | 无 | `expflow run` |
| `fsm.py` | 实验的 7 状态有限状态机 | fysom | — |
| `pin.py` | PIN 保护（SHA-256），用于破坏性操作 | 无 | `expflow pin` |
| `metrics.py` | 标准化度量注册表（含阈值） | 无 | — |
| `compare.py` | 多模型分数排名+门控 | clearml (可选) | `expflow clearml compare` |
| `equations.py` | PDE 方程注册表（11 个方程） | 无 | — |
| `analyze.py` | 竞赛任务智能分析与策略 | 无 | `expflow analyze` |
| `audit.py` | 竞赛规则验证与合规检查 | 无 | `expflow audit` |
| `config.py` | YAML + .env 配置加载器 | pyyaml, python-dotenv | — |
| `worktree.py` | Git worktree 实验隔离 | git | — |
| `snowflake.py` | Snowflake ID 生成器 | 无 | — |
| `status.py` | 组件健康检查 | clearml/optuna/langfuse (可选) | `expflow system status` |
| `board.py` | TensorBoard 启动器 | tensorboard (可选) | `expflow system board` |
| `mcp.py` | MCP 服务器入口 | clearml, optuna, langfuse (可选) | `expflow mcp` |
| `mcp_server.py` | 18+ 个 MCP 工具定义 | clearml, optuna, langfuse, mcp (可选) | — |
| `cli_clearml.py` | clearml CLI 分组（14 命令） | clearml (可选) | `expflow clearml` |
| `cli_optuna.py` | optuna CLI 分组（8 命令） | optuna (可选) | `expflow optuna` |
| `cli_langfuse.py` | langfuse CLI 分组（6 命令） | langfuse (可选) | `expflow langfuse` |
| `cli_audit.py` | audit CLI 分组（3 命令） | clearml (可选) | `expflow audit` |
| `cli_analyze.py` | analyze CLI 分组（4 命令） | 无 | `expflow analyze` |
| `cli_pin.py` | PIN CLI 分组（4 命令） | 无 | `expflow pin` |
| `cli_pipeline.py` | pipeline CLI 分组（1 命令） | clearml (可选) | `expflow pipeline` |
| `cli_run.py` | 实验调度 CLI 分组（4 命令） | 无 | `expflow run` |
| `cli_system.py` | system CLI 分组（2 命令） | clearml/optuna/langfuse (可选) | `expflow system` |

## 模块依赖链

```
cli.py (Typer) — 8 个命令组 + 5 个顶级命令
  ├── clearml.py        → Task/Queue/Dataset SDK 封装
  ├── optuna.py          → Study/Trial/Plot SDK 封装
  ├── langfuse.py        → Trace/Session/Cost SDK 封装
  ├── dispatcher.py      → 实验调度（内存注册表）
  ├── audit.py           → 验证 + 合规 + 报告
  ├── system.py          → 组件健康检查 + board
  └── mcp.py             → MCP Server 入口

所有 SDK 导入都是 LAZY 的——避免模块级 import 时触发依赖加载，
只有在相应命令组被调用时才实际导入。
```

## 配置加载

```python
from expflow_pde.config import load_config, get

cfg = load_config()            # 加载 YAML + .env
val = get("clearml.api")       # 点分隔访问
```

配置搜索顺序：当前目录 `config.yaml` → 父目录 → `.env`（仅覆盖 API 密钥）。

## 数据流

### 实验生命周期

```
expflow run submit <script.py>       # 注册实验
         │
         ▼
dispatcher.py: add_experiment()      # 内存注册表
         │
         ▼
fsm.py: EXPERIMENT_FSM               # DRAFT → ENQUEUED → RUNNING
         │                               → COMPLETED / FAILED / CANCELLED
         ▼
expflow run cancel <id>              # FSM: CANCEL_PENDING → CANCELLED
         │                              （PIN 保护，--force 可跳过）
         ▼
expflow clearml compare-scores       # 获取 clearml 指标
         │                              （需配置 clearml）
         ▼
expflow audit validate <id>          # 按竞赛规则合规性检查
```

### 流水线流程（训练→评估→提交）

```
expflow pipeline submit-full <script.py>
         │
         ▼
  阶段 1: HPO (Optuna 试验 — 可选)
         ├── 本地: 子进程串行执行
         ├── 分布式: ask/tell + clearml Task clone
         └── 优化器: clearml HyperParameterOptimizer
         │
         ▼
  阶段 2: 训练（最佳参数）
         └── clearml-agent 队列 → GPU 节点
         │
         ▼
  阶段 3: 评估（生成 pred.hdf5）
         └── clearml-agent 队列 → GPU 节点
         │
         ▼
  结果: JSON 摘要 + clearml task ID
```

### HPO 执行模式

```
expflow optuna run <script.py>
         │
    ┌────┴──────────┐
    │               │
    ▼               ▼
  本地            分布式
  (子进程)        (clearml 队列)
    │               │
    ▼               ▼
  SQLite DB      clearml Task per trial
  ~/.expflow/    parent study = controller task
  optuna_<name>.db
```

## CLI 命令树

```
expflow
├── version / info / mcp / init / config         ← 顶级命令（无 SDK 依赖）
├── clearml     (14 子命令)                       ← 延迟导入 clearml
│   ├── tasks / task / enqueue / dequeue / queues / compare / workers
│   ├── dataset-register / dataset-list / dataset-upload / dataset-download
│   ├── pipeline-create / pipeline-add-step / pipeline-start / pipeline-stop / pipeline-list
│   └── scheduler-create / scheduler-start / scheduler-add-task / scheduler-list / scheduler-remove-task
├── optuna      (8 子命令)                        ← 延迟导入 optuna
│   ├── create-study / studies / study / delete-study / ask / tell / plot / run
├── langfuse    (6 子命令)                        ← 延迟导入 langfuse
│   ├── traces / trace / trace-cost / sessions / session / metrics
├── run         (4 子命令)                        ← 无 SDK 依赖
│   ├── submit / list / status / cancel
├── audit       (3 子命令)                        ← 无 SDK 依赖
│   ├── validate / check-dataset / report
├── system      (2 子命令)                        ← 按需延迟导入
│   ├── status (健康检查) / board (TensorBoard)
├── pin         (4 子命令)                        ← 无 SDK 依赖
│   ├── init / check / clear / status
├── analyze     (4 子命令)                        ← 无 SDK 依赖
│   ├── task / equations / status / advise
└── pipeline    (1 子命令)                        ← 延迟导入 clearml
    └── submit (训练→评估→提交 流水线)
```

## 关键设计决策

### 1. 可选 SDK 依赖

三个主要 SDK（clearml, optuna, langfuse）都是**可选扩展**。
`__init__.py` 使用 `__getattr__` 延迟解析 + `.pyi` 类型存根：

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

### 2. FSM 驱动的实验生命周期

实验遵循 7 状态有限状态机（fysom）：
- `DRAFT` → `ENQUEUED` → `RUNNING` → `COMPLETED`
- `RUNNING` → `FAILED`
- `CANCEL_PENDING` → `CANCELLED`
- 取消操作受 PIN 保护（除非使用 `--force`）

### 3. 破坏性操作的 PIN 保护

SHA-256 哈希的 PIN 存储在 `~/.expflow/pin.hash`（不在 config.yaml 中）。
支持 `--force` 绕过（用于 CI/自动化）。

### 4. 标准化度量注册表

指标采用 `分组/指标名` 命名方式（如 `Score/Seg Total`、`Loss/Val MSE`）
与 `expflow clearml compare-scores` 门控系统兼容。

### 5. Snowflake ID

线程安全的 yitter snowflake M1 实现，位于 `snowflake.py`。
worker_id=1 预留给 expflow。格式：`exp:snow_<19位整数>`。

## 相关文档

- [USAGE.zh-CN.md](USAGE.zh-CN.md) — 安装和 CLI 参考
- [DEVELOPMENT.zh-CN.md](DEVELOPMENT.zh-CN.md) — 开发者指南和测试
- [DATA_LAYER.zh-CN.md](DATA_LAYER.zh-CN.md) — ClearML 数据层设计
- [COMPETITION.zh-CN.md](COMPETITION.zh-CN.md) — PDE 竞赛集成
