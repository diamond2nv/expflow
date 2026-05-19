# expflow-pde 使用指南

## 安装

```bash
# 核心 CLI（无需外部 SDK）
pip install expflow-pde

# 安装全部 SDK 集成
pip install "expflow-pde[all]"

# 单独安装扩展
pip install "expflow-pde[clearml]"   # Task/queue/dataset 管理
pip install "expflow-pde[optuna]"    # 超参优化
pip install "expflow-pde[langfuse]"  # LLM 可观测性追踪
pip install "expflow-pde[mcp]"       # MCP 服务器 + 全部 SDK
pip install "expflow-pde[pipeline]"  # 流水线模式（需要 clearml）
```

### 本地开发

```bash
git clone https://github.com/diamond2nv/expflow.git
cd expflow
python -m venv venv
source venv/bin/activate
pip install -e ".[all,dev]"
```

## 配置

首次运行 `expflow init` 进行配置：

```bash
expflow init                        # 交互式向导
expflow init --quick                # 快速模式（使用默认值）
```

或在项目根目录手动创建 `config.yaml`：

```yaml
# ~/my_project/config.yaml
clearml:
  api_server: http://localhost:8008
  web_server: http://localhost:8080
  files_server: http://localhost:8081

langfuse:
  host: http://localhost:3000
  public_key: "pk-..."
  secret_key: "sk-..."
```

敏感值（API 密钥）使用 `.env`：

```env
LANGFLUSE_PUBLIC_KEY=pk-xxx
LANGFLUSE_SECRET_KEY=sk-xxx
```

配置搜索顺序：`CWD/config.yaml` → 父目录 → `.env`。

## CLI 命令

### 顶级命令（无 SDK 依赖）

```bash
expflow --help                           # 显示帮助
expflow version                          # 显示版本
expflow version --verbose                # 显示版本 + 构建信息
expflow info                             # 显示系统信息 + SDK 版本
expflow config                           # 显示当前配置
expflow init                             # 交互式配置
```

### ClearML 集成 (`expflow clearml`)

**要求**：`pip install "expflow-pde[clearml]"`

```bash
# 任务管理
expflow clearml tasks                    # 列出所有任务
expflow clearml task abc123              # 查看任务详情
expflow clearml enqueue abc123           # 入队任务
expflow clearml dequeue abc123           # 出队任务
expflow clearml queues                   # 列出队列
expflow clearml workers                  # 列出工作节点（含 GPU 信息）
expflow clearml compare-scores           # 比较实验分数
expflow clearml compare-scores \
    --project PDEBench --tags task1 \
    --sort-by seg_total --gate pde_mean:lt:18.09

# 数据集管理
expflow clearml dataset-list             # 列出数据集
expflow clearml dataset-register data/   # 注册数据集
expflow clearml dataset-upload data/     # 上传数据集
expflow clearml dataset-download abc123  # 下载数据集

# 流水线管理
expflow clearml pipeline-list            # 列出流水线
expflow clearml pipeline-create          # 创建流水线
expflow clearml pipeline-start           # 启动流水线

# 调度器
expflow clearml scheduler-create         # 创建调度器
expflow clearml scheduler-list           # 列出调度器
expflow clearml scheduler-start          # 启动调度器
```

### Optuna 集成 (`expflow optuna`)

**要求**：`pip install "expflow-pde[optuna]"`

```bash
# 研究管理
expflow optuna create-study my_study     # 创建研究
expflow optuna studies                   # 列出研究
expflow optuna study my_study            # 查看研究详情
expflow optuna delete-study my_study     # 删除研究

# HPO 运行（三种模式）
expflow optuna run train_task1.py \
    --trials 20                          # 本地模式（默认）

expflow optuna run train_task1.py \
    --trials 50 --parallel 4 \
    --distributed --queue default        # 分布式模式

expflow optuna run train_task1.py \
    --trials 50 --optimizer -O           # ClearML HyperParameterOptimizer

# 试验交互
expflow optuna ask my_study              # 请求下一个试验
expflow optuna tell my_study trial_id    # 报告结果
expflow optuna plot my_study             # 绘制研究图表
```

### Langfuse 集成 (`expflow langfuse`)

**要求**：`pip install "expflow-pde[langfuse]"`

```bash
expflow langfuse traces                  # 列出追踪
expflow langfuse trace lf_abc123         # 查看追踪详情
expflow langfuse trace-cost lf_abc123    # 查看追踪费用
expflow langfuse sessions                # 列出会话
expflow langfuse session my_session      # 查看会话详情
expflow langfuse metrics                 # 获取会话指标
```

### 实验调度 (`expflow run`)

**无 SDK 依赖**——使用内存实验注册表。

```bash
expflow run submit train.py              # 提交实验
expflow run list                         # 列出实验
expflow run status abc123                # 查看实验状态
expflow run cancel abc123                # 取消（PIN 保护）
expflow run cancel abc123 --force        # 取消（跳过 PIN）
```

### 流水线 (`expflow pipeline`)

**要求**：`pip install "expflow-pde[pipeline]"` 或 `"expflow-pde[clearml]"`

```bash
# 快速模式（训练→评估，跳过 HPO）
expflow pipeline submit train_task1.py \
    --queue default \
    --train-param lr=0.001 --train-param epochs=80 \
    --eval-script eval_task1.py

# 完整模式（HPO → 训练 → 评估）
expflow pipeline submit-full train_task1.py \
    --queue default \
    --trials 50 --parallel 4 \
    --eval-script eval_task1.py \
    --metric seg_total --direction maximize

# 灵活跳过
expflow pipeline submit-full train_task1.py --skip hpo --skip eval
expflow pipeline submit-full train_task1.py --skip train --skip eval
```

### 审计 (`expflow audit`)

核心验证无 SDK 依赖。`--task-id` 模式需要 clearml。

```bash
# 按竞赛规则验证实验
expflow audit validate <exp_id> \
    --competition-rules --task-id abc123

# 检查数据集合规性
expflow audit check-dataset <path>

# 生成报告
expflow audit report <exp_id>
```

### 系统 (`expflow system`)

```bash
expflow system status                    # 所有组件的健康检查
expflow system board                     # 启动 TensorBoard
```

### PIN 保护 (`expflow pin`)

**无 SDK 依赖**。保护破坏性操作。

```bash
expflow pin init 1234                    # 设置 PIN（SHA-256 存储）
expflow pin check                        # 交互式验证 PIN
expflow pin clear                        # 移除 PIN（需当前 PIN）
expflow pin clear --force                # 移除 PIN（跳过验证）
expflow pin status                       # 检查 PIN 是否已设置
```

### 竞赛分析 (`expflow analyze`)

**无 SDK 依赖**。

```bash
# 战略建议（主要入口）
expflow analyze advise

# 按任务分析
expflow analyze task task1               # Task 1 详情
expflow analyze task task3               # Task 3（Kuramoto-Sivashinsky）

# 方程参考
expflow analyze equations                # 所有方程
expflow analyze equations --task competition  # 仅竞赛方程
expflow analyze equations kuramoto_sivashinsky  # 单个方程

# 竞赛概览
expflow analyze status
```

### MCP 服务器

```bash
expflow mcp                              # 启动 MCP 服务器（stdio）
```

在 Hermes Agent `~/.hermes/config.yaml` 中注册以供 Agent 集成：

```yaml
mcp:
  servers:
    expflow:
      command: "expflow"
      args: ["mcp"]
注册后，Agent 可以直接在聊天中执行：列出任务、入队实验、比较分数等操作。

### Agent 技能安装

expflow-pde 内置了 4 个 Hermes Agent 技能，位于仓库的 `skills/` 目录：

```bash
# 通过 URL 安装
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/expflow-pipeline-hpo/SKILL.md
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/experiment-lifecycle-governance/SKILL.md
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/clearml-metrics-logging-pattern/SKILL.md
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/competition-task-intelligence/SKILL.md

# 或 tap 仓库后直接安装
hermes skills tap add diamond2nv/expflow
hermes skills install expflow-pipeline-hpo
```

安装后 Agent 会自动加载以下技能：
- **expflow-pipeline-hpo** — 竞赛流水线编排（HPO → 训练 → 评估）
- **experiment-lifecycle-governance** — PIN 保护、度量注册表、分数对比
- **clearml-metrics-logging-pattern** — 标准化 ClearML 度量命名与上报
- **competition-task-intelligence** — PDE 方程注册表、任务分析、战略建议

技能文件位于包内的 `expflow_pde/skills/` 目录。

---

## 流水线模式

### 完整流水线

```
HPO (Optuna) ──► 训练（最佳参数）──► 评估（生成提交文件）
   │                    │                        │
   ▼                    ▼                        ▼
 clearml trials    clearml task              clearml task
```

适用于：竞赛探索阶段。需要寻找最佳超参。

### 快速流水线

```
训练（固定参数）──► 评估（生成提交文件）
       │                        │
       ▼                        ▼
   clearml task             clearml task
```

适用于：竞赛冲刺阶段。已经知道最佳参数。

### 流水线参数

| 参数 | 适用模式 | 说明 |
|------|:--------:|------|
| `--queue <name>` | 全部 | clearml-agent 队列，用于 GPU 调度的队列名称 |
| `--skip hpo` | full | 跳过 HPO 步骤 |
| `--skip eval` | 全部 | 跳过评估步骤 |
| `--train-param key=val` | 全部 | 传递给训练脚本的额外参数 |
| `--eval-param key=val` | 全部 | 传递给评估脚本的额外参数 |
| `--trials N` | full | HPO 试验次数 |
| `--parallel M` | full | 最大并发试验数 |

## MCP 工具

MCP 服务器运行时，Hermes Agent 可以访问 18+ 个工具：

| 工具 | 说明 |
|------|------|
| `exp_list_tasks` | 列出 ClearML 任务 |
| `exp_enqueue_task` | 入队一个任务 |
| `exp_dequeue_task` | 出队一个任务 |
| `exp_list_queues` | 列出队列 |
| `exp_list_workers` | 列出工作节点 |
| `exp_compare_scores` | 比较实验分数 |
| `exp_dataset_list` | 列出数据集 |
| `exp_dataset_upload` | 上传数据集 |
| `exp_trace_experiment` | 创建 Langfuse 追踪 |
| `exp_submit_experiment` | 提交实验 |
| `exp_get_status` | 获取系统状态 |

## 脚本要求

训练和评估脚本需要遵循以下约定才能与 expflow 兼容：

```python
# 1. 接受 --key=value 格式的 CLI 参数作为超参
# 2. 以 expflow 可捕获的方式上报指标：
#    - 本地模式：向 stdout 打印 "METRIC:<name>=<value>"
#    - 分布式模式：clearml Task.current_task().report_scalar(...)
# 3. 接受标准参数：--epochs, --lr, --batch_size, --tag

# HPO 捕获的输出示例：
# METRIC:seg_total=57.09
```

## 参考文档

- [ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md) — 系统架构
- [DEVELOPMENT.zh-CN.md](DEVELOPMENT.zh-CN.md) — 开发者指南
- [DATA_LAYER.zh-CN.md](DATA_LAYER.zh-CN.md) — ClearML 数据层
- [COMPETITION.zh-CN.md](COMPETITION.zh-CN.md) — 竞赛集成
