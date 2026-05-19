# 数据层：ClearML Fileserver

> **设计决策**：放弃 DVC，改用 ClearML Server 内置的 Fileserver
> 作为 expflow 实验的唯一数据后端。
>
> **调研验证**：经 576+ 篇 clearml 官方文档全面学习，
> 设计方向完全正确。ClearML 的 Dataset class (`clearml-data`)
> 提供了版本管理、血缘追踪、差异化存储、本地缓存和元数据标注——
> 与 expflow 数据层需求高度一致。

## 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        expflow 服务器                              │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐    ┌────────────────────┐   │
│  │ PDEBench      │   │ expflow CLI  │    │ clearml-agent      │   │
│  │ 训练脚本      │──→│ (编排)       │──→ │ (GPU 调度)         │   │
│  │               │   │              │    │                    │   │
│  └──────────────┘   └──────────────┘    └────────┬───────────┘   │
│                                                   │                │
│  ┌─────────────────────────────────────────────────┴──────────┐   │
│  │           Docker 网络：clearml-server                        │   │
│  │                                                              │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │   │
│  │  │ apiserver    │   │ webserver    │   │ fileserver   │    │   │
│  │  │ (8008)       │   │ (8082 → 80)  │   │ (8081)       │    │   │
│  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘    │   │
│  │         │                  │                    │            │   │
│  │  ┌──────┴───────┐ ┌──────┴───────┐ ┌──────────┴──────┐    │   │
│  │  │ mongo        │ │ elasticsearch│ │ /mnt/fileserver │    │   │
│  │  │ (元数据)     │ │ (搜索索引)   │ │ (文件存储)      │    │   │
│  │  └──────────────┘ └──────────────┘ └─────────────────┘    │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 核心原则

1. **ClearML Fileserver 是唯一数据底座** — 所有实验数据、模型权重、数据集都通过 clearml SDK 的 Dataset/Model API 上传到 Fileserver。
2. **不引入 DVC** — clearml 的 Dataset class (`clearml-data`) 已内置版本管理、血缘追踪、元数据标注和差异化存储。
3. **基于文件服务器**（非独立 MinIO）— 当 clearml.conf 的 `api.api_server` 指向 clearml-apiserver 时，所有数据自动通过 API 存储到 fileserver。

## clearml-data 能力分析

### 完整度评估

| 需求维度 | clearml-data 支持度 | 细节 |
|---------|:------------------:|------|
| 版本管理 | ✅ 原生 | `Dataset.create(version='1.0')`，语义版本自动递增 |
| 血缘追踪 | ✅ 原生 | `parent_datasets` 参数，`Dataset.get(id).parent` 追溯 |
| 差异化存储 | ✅ 原生 | 子版本只存储与父版本的差异 |
| 本地缓存 | ✅ 原生 | 自动缓存到 `~/.clearml/cache/` |
| 并行上传/下载 | ✅ 原生 | `max_workers` 参数，默认逻辑核数 |
| 元数据标注 | ✅ 原生 | `set_metadata()` / `dataset_tags` |
| 文件验证 | ✅ 原生 | `clearml-data verify`（hash 和 filesize 两种模式） |
| 多分片下载 | ✅ 原生 | `part` / `num_parts` 多节点分片 |
| 离线模式 | ✅ 原生 | `CLEARML_OFFLINE_MODE=1` 创建本地 zip，事后批量上传 |
| 多种存储后端 | ✅ 原生 | 默认 fileserver，支持 S3/GS/Azure/共享目录 |
| S3 兼容性 | ✅ 原生 | `output_uri='s3://host:port/bucket'`（端口必填） |

**结论**：clearml-data 的数据层非常完善。expflow 不需要在数据层做任何额外实现——
所有文件传输、版本管理、缓存、验证都是 clearml SDK 开箱即用。

### 核心工作流

```
────── CLI ──────                          ────── SDK ──────

# 创建 + 添加文件                      dataset = Dataset.create(
clearml-data create \                       dataset_name='name',
  --project PDEBench \                      dataset_project='PDEBench',
  --name 1D_Burgers_v1 \                   parent_datasets=[PARENT_ID],
  --parents <PARENT_ID>                    version='1.0',
                                           )

clearml-data add --files data/            dataset.add_files(path='data/')

clearml-data close                        dataset.upload()
                                          dataset.finalize()
                                          print(dataset.id)
                                          print(dataset.url)

# 下载（任意版本）                    dataset_v2 = Dataset.get(
clearml-data get --id <ID> \                dataset_id=ID,
  --local-copy ~/expflow/data/              alias='burgers_v2',
                                           )
                                           dataset_v2.get_local_copy(
                                             ~/expflow/data/
                                           )
```

## 流水线数据流

### 训练脚本数据访问

```python
from clearml import Dataset

# Dataset v2.0: Nu=0.001 Burgers — 10,000 训练样本
ds = Dataset.get(dataset_id="abc123")

# 获取本地副本（自动缓存，差异更新）
local_path = ds.get_local_copy(
    local_cache_dir="~/.clearml/cache/",
)

# 通过 expflow pipeline 使用
from expflow_pde.pipeline import ExperimentPipeline
ep = ExperimentPipeline()
result = ep.train_hpo_val_submit(
    train_script="train_task1.py",
    dataset_id="abc123",
    n_trials=50,
)
```

### 实验产物流

```
训练脚本    ───► 模型权重 ──────────► Fileserver（通过 Model.upload_model()）
            └──► 指标 ─────────────► Task.report_scalar()
                                      （在 Web UI 中可见）

评估脚本    ───► pred.hdf5 ────────► Fileserver（通过 Dataset.create()）
                                      （提交产物）

expflow audit──► 指标获取 ────────► compare-scores 门控
            （从 clearml Task）       （通过 Task.get_last_scalar_metrics()）
```

## 数据集命名约定

为与 `expflow clearml compare-scores` 和度量注册表兼容，使用一致的命名：

```yaml
# 命名：<dataset_name>_v<major>.<minor>
# 示例：
project: PDEBench
datasets:
  1D_Burgers_Nu0.001: v2.0    # 训练：10k 样本
  1D_Burgers_Nu0.001_val: v1.0 # 验证：100 样本
  KS_Nu1.0-1.5: v1.0           # 训练：2000 样本
```

## 配置参考

### 最小 clearml.conf

```bash
api:
  api_server: http://localhost:8008
  web_server: http://localhost:8080
  files_server: http://localhost:8081
```

### 环境变量

```bash
CLEARML_API_HOST=http://localhost:8008
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_FILES_HOST=http://localhost:8081
CLEARML_OFFLINE_MODE=0
```

### 检测是否远程运行

```python
from clearml.config import running_remotely
if running_remotely():
    # clearml-agent 已挂载数据集
    local_path = Dataset.get(dataset_id=...).get_local_copy()
```

## 相关文档

- [ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md) — 整体系统架构
- [USAGE.zh-CN.md](USAGE.zh-CN.md) — 数据集命令的 CLI 参考
- [DEVELOPMENT.zh-CN.md](DEVELOPMENT.zh-CN.md) — 测试数据集操作
