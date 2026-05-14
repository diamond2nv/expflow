# expflow 数据层设计：基于 ClearML Fileserver 的实验数据管理

> **设计决策：** 放弃 DVC，改用 ClearML Server 内置的 Fileserver
> 作为 Agentic4Sci 的唯一实验数据层。
>
> **调研验证：** 经 clearml 官方文档（576 篇 .md）全面学习，验证设计方向完全正确。
> clearml 的 Dataset class (`clearml-data`) 提供了版本管理、血缘追踪、
> 差异化存储、缓存机制、元数据标注，与 expflow 数据层需求高度一致。
>
> **实现状态：** clearml-data CLI + SDK 已就绪（clearml 包自带，无需额外安装）。
> expflow Phase 7（dataset_upload/download/lineage 封装）规划中。
> 前提：需先部署 clearml-server（docker compose port 8082）。

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agentic4Sci 服务器                         │
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ PDEBench     │    │ Hermes Agent │    │ clearml-agent      │  │
│  │ 训练脚本     │ ──→ │ expflow CLI  │ ──→ │ (expflow dispatcher)│  │
│  └─────────────┘    └──────────────┘    └────────┬───────────┘  │
│                                                   │               │
│  ┌─────────────────────────────────────────────────┴─────────┐  │
│  │           Docker 网络：clearml-server                        │  │
│  │                                                             │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │  │
│  │  │ apiserver    │    │ webserver    │    │ fileserver   │ │  │
│  │  │ (8008)       │    │ (8082 → 80)  │    │ (8081)       │ │  │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘ │  │
│  │         │                  │                    │          │  │
│  │  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────────┴──────┐ │  │
│  │  │ mongo        │  │ elasticsearch│  │ /mnt/fileserver │ │  │
│  │  │ (元数据)     │  │ (搜索索引)   │  │ (文件系统存储)  │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心原则

1. **ClearML Fileserver 是唯一数据底座** — 所有实验数据、模型权重、数据集都通过 clearml SDK 的 Dataset/Model API 上传到 Fileserver
2. **不引入 DVC** — clearml 的 Dataset class (`clearml-data`) 已内置版本管理、血缘追踪、元数据标注、差异化存储
3. **Fileserver 是文件系统存储**（不是独立的 MinIO）— 当 clearml.conf 的 `api.api_server` 指向 clearml-apiserver 时，所有数据自动通过 API 存储到 fileserver

---

## 2. clearml-data 机制分析

### 2.1 数据层完整度评估

| 需求维度 | clearml-data 支持度 | 细节 |
|---------|--------------------|------|
| 版本管理 | ✅ 原生 | `Dataset.create(version='1.0')`，语义版本自动递增 |
| 血缘追踪 | ✅ 原生 | `parent_datasets` 参数，`Dataset.get(id).parent` 追溯 |
| 差异化存储 | ✅ 原生 | 子版本只存储与父版本的 change-set |
| 本地缓存 | ✅ 原生 | 自动缓存到 `~/.clearml/cache/` |
| 并行上传/下载 | ✅ 原生 | `max_workers` 参数，默认逻辑核数 |
| 元数据标注 | ✅ 原生 | `set_metadata()` / `dataset_tags` |
| 文件验证 | ✅ 原生 | `clearml-data verify` 支持 hash 和 filesize 两种 |
| 多分片下载 | ✅ 原生 | `part` / `num_parts` 多节点分片 |
| 离线模式 | ✅ 原生 | `CLEARML_OFFLINE_MODE=1` 创建本地 zip，事后批量上传 |
| 多种存储后端 | ✅ 原生 | 默认 fileserver，支持 S3/GS/Azure/共享目录 |
| S3 兼容（MinIO） | ✅ 原生 | `output_uri='s3://host:port/bucket'`，端口必填 |

**结论：clearml-data 的数据层非常完善，expflow 不需要在数据层做任何额外实现。** 所有文件传输、版本管理、缓存、验证都是 clearml SDK 开箱即用。

### 2.2 clearml-data 核心工作流

```
────── CLI ──────                          ────── SDK ──────

# 创建 + 添加文件                      dataset = Dataset.create(
clearml-data create \                    dataset_name='name',
  --project PDEBench \                   dataset_project='PDEBench',
  --name 1D_Burgers_v1 \                parent_datasets=[PARENT_ID],
  --parents <PARENT_ID>                  version='1.0',
  --version 1.0                          description='...'
                                      )
clearml-data add \                     dataset.add_files(path='/data/')
  --files /data/*.hdf5                 dataset.upload()
                                      dataset.finalize()
clearml-data close

# 下载到本地                            path = Dataset.get(
clearml-data get \                        dataset_id='<ID>'
  --id <ID> \                          ).get_mutable_local_copy(
  --copy /path/to/output                  target_folder='/path/to',
                                          overwrite=True
                                      )

# 文件夹同步                             dataset = Dataset.create(
clearml-data sync \                       parent_datasets=[PARENT_ID]
  --folder /data/ \                    )
  --parents <PARENT_ID>                 dataset.sync_folder(local_path='/data/')
                                      dataset.finalize()
```

---

## 3. expflow API 设计（更新版）

基于 clearml-data SDK 调研后的修正方案。

### 3.1 核心实现策略

原设计 `dataset_upload()` 和 `dataset_download()` 手写文件传输逻辑。修正为：**expflow 直接封装 clearml SDK Dataset 类**，所有文件传输由 clearml 处理。

**为什么这样设计：**
- clearml 已经处理了 hash 计算（不需要我们写 MD5）
- clearml 已经处理了分片上传、断点续传
- clearml 自动缓存管理
- clearml 自动差异化存储（只上传 diff）
- 代码量从~100行降为~20行

### 3.2 函数定义

#### `clearml.dataset_upload()`

```python
def dataset_upload(
    local_path: str,
    dataset_name: str,
    dataset_project: str = "PDEBench",
    version: str | None = None,
    parent_dataset_ids: list[str] | None = None,
    compliance: Literal["allowed", "forbidden"] | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> dict:
    """
    上传本地 HDF5 文件到 clearml fileserver。
    内部调用：Dataset.create() → add_files() → upload() → finalize()
    
    返回：{id, name, version, compliance, uri, file_count, total_bytes}
    
    Note:
        - 不需要手动计算 hash（clearml 自动处理）
        - 默认上传到 clearml fileserver（不需要配 S3）
        - 版本不指定则自动递增
    """
```

#### `clearml.dataset_download()`

```python
def dataset_download(
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    dataset_project: str | None = "PDEBench",
    target_folder: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    从 clearml fileserver 下载 Dataset 到本地。
    内部调用：Dataset.get() → get_mutable_local_copy()
    
    返回：{id, name, version, local_path, file_count}
    
    Note:
        - 缓存到 ~/.clearml/cache/ 后复制到 target_folder
        - 如果 target_folder 已有内容且 overwrite=False 则报错
        - 支持多分片并行下载
    """
```

#### `clearml.dataset_lineage()`

```python
def dataset_lineage(
    dataset_id: str,
    depth: int = 10,
) -> list[dict]:
    """
    沿 parent 链追溯 Dataset 血缘。
    内部调用：递归 parent 属性
    
    返回：[oldest → newest] sorted lineage
    每项：{id, name, version, compliance, created_at, parent_id}
    """
```

#### `clearml.model_list()`

```python
def model_list(
    project_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    only_published: bool = False,
    max_results: int = 20,
) -> list[dict]:
    """
    列出已注册的模型 checkpoint。
    内部调用：Model.query_models()
    
    返回：[{id, name, project, tags, created, uri, task_id, framework}]
    """
```

### 3.3 与现有模块的集成

现有的 `register_dataset()`（仅 metadata 标注模式）将被重构：

```
register_dataset(old)                             → dataset_upload(new)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset.create()                                      Dataset.create() 
set_metadata(compliance)                              set_metadata(compliance)
set_metadata(source_path)                      +      add_files(path)
                                                     +
[不上传文件，只做注册]                                 upload()
                                                     +
                                                     finalize()
```

旧 `register_dataset` 改为只对已上传到 fileserver 的 Dataset 做合规 tag 更新：
```python
def annotate_compliance(
    dataset_id: str,
    compliance: Literal["allowed", "forbidden"],
) -> dict:
    """对已有 Dataset 标注或修改合规信息。"""
```

### 3.4 MCP 工具

```python
# expflow/mcp.py — FastMCP 服务
from fastmcp import FastMCP

mcp = FastMCP("expflow-mcp")

@mcp.tool()
def dataset_upload(
    local_path: str, name: str,
    compliance: str = None,
    parent_ids: list[str] = None,
) -> dict: ...

@mcp.tool()
def dataset_download(
    dataset_id: str, target_folder: str = None,
) -> dict: ...

@mcp.tool()
def dataset_list(
    project: str = None, compliance: str = None,
) -> list[dict]: ...

@mcp.tool()
def dataset_lineage(
    dataset_id: str = None, dataset_name: str = None,
    project: str = "PDEBench",
) -> list[dict]: ...

@mcp.tool()
def model_list(
    project: str = None, tags: list[str] = None,
    framework: str = None,
) -> list[dict]: ...
```

---

## 4. 部署说明

### 4.1 clearml docker-compose 端口调整

```yaml
webserver:
  ports:
    - "8082:80"     # 原 8080 → 改 8082，避免与 ai-suite caddy(80) 冲突
```

fileserver(8081)、apiserver(8008) 不冲突，保持不变。

### 4.2 clearml.conf 配置

```
api {
  web_server: http://localhost:8082
  api_server: http://localhost:8008
  files_server: http://localhost:8081
  credentials {
    "access_key" = "<from_webui>"
    "secret_key" = "<from_webui>"
  }
}
```

**数据目标（不需要额外配 S3）：**
clearml SDK 默认向 apiserver 注册数据。当 `api_server` 指向 clearml-apiserver 时，所有 Dataset/Model/Artifact 自动存储到 fileserver。
`CLEARML_DEFAULT_OUTPUT_URI` 不设即为 fileserver。

### 4.3 启动步骤

```bash
# 1. 启动 clearml-server
cd ~/Gitlab/Agentic4Sci/clearml-server/docker
sed -i 's/"8080:80"/"8082:80"/' docker-compose.yml
docker compose up -d

# 2. 配置 clearml SDK
pip install clearml
clearml-init  # 输入 server 地址和 credentials

# 3. 验证
python -c "from clearml import Task; print('OK')"
```

---

## 5. 与现有工作流的集成

### 5.1 PDEBench 竞赛合规检查

```
Hermes 发起实验
    ↓
expflow clearml dataset-upload /data/train.hdf5 \
    --name 1D_Burgers_Sols_Nu0.001 --version 1.0 \
    --compliance allowed --parent-ids <PARENT_ID>
    ↓  dataset_id = "abc123"

expflow run submit train_fno.py --dataset-id abc123
    ↓ clearml 自动记录 input dataset + output model

expflow audit validate --task-id <TASK_ID> --compliance-required allowed
    ↓ 检查 task 使用的 dataset 合规链

expflow clearml model-list --project PDEBench --tags fno
    ↓ 查看所有已注册的 checkpoint
```

### 5.2 数据集版本管理实战

```python
# 情景：新到一批合成数据，需要基于官方数据创建新版本
dataset_v1 = clearml.dataset_upload(
    local_path='/data/1D_Burgers_Nu0.001.hdf5',
    dataset_name='1D_Burgers_Sols_Nu0.001',
    dataset_project='PDEBench',
    version='1.0',
    compliance='allowed',
    description='Official PDEBench data, Nu=0.001',
)

dataset_v2 = clearml.dataset_upload(
    local_path='/data/1D_Burgers_longtime.hdf5',
    dataset_name='1D_Burgers_Sols_Nu0.001',
    parent_dataset_ids=[dataset_v1['id']],
    version='2.0',
    compliance='forbidden',  # 合成数据不能用于 Task1
    description='Synthesized long-time data via Cole-Hopf',
    tags=['synthetic', 'long-time', '200-step'],
)
```

---

## 6. 为什么不选 DVC + 独立 MinIO

| 维度 | DVC + MinIO | ClearML Fileserver |
|------|------------|-------------------|
| 额外服务 | 另起 minio docker | clearml server 内置，0 额外 |
| 数据关联 | `dvc run` + `dvc push` 手动关联 | clearml Task 自动记录 InputModel/OutputModel |
| 版本回溯 | `git checkout dvc.lock` | clearml UI / Dataset lineage API |
| 合规审计 | 需自建 pipeline | clearml Dataset tags + metadata 原生支持 |
| Hermes 集成 | 需写 DVC CLI wrapper | expflow MCP 直接调 clearml API |
| 学习曲线 | DVC 概念多 | clearml Dataset/Model/Artifact 三对象 |
| 生态系统 | 纯数据版本管理 | ML 全链路（实验追踪+数据+模型+调度） |
| 差异化存储 | DVC 支持 | clearml 原生（子版本只存 diff） |
| 缓存管理 | 手动 | clearml 自动（~/.clearml/cache/） |

---

## 7. 实现路线图（更新版）

### Phase 7 — 数据层实现 [PLANNED]

**前提：** 必须先部署 clearml-server，因为 clearml-data 的 Dataset API 需要
clearml apiserver 才能工作。

| 子阶段 | 内容 | 状态 |
|--------|------|------|
| P7.0 | 启动 clearml-server（docker compose port 8082） | ⏳ |
| P7.1 | `dataset_upload()` / `dataset_download()` / `dataset_lineage()` | 📝 设计完成 |
| P7.2 | `Model.query_models()` 封装 → `model_list()` | 📝 |
| P7.3 | 旧 `register_dataset()` → `annotate_compliance()` 重构 | 📝 |
| P7.4 | MCP 工具注册（6 个 tool） | 📝 |
| P7.5 | 单元测试（mock clearml SDK Dataset 类） | 📝 |
| P7.6 | 端到端测试（Hermes → MCP → clearml → fileserver） | ⏳ 需部署后 |

### Phase 8 — Pipeline 集成（基于 clearml Pipeline）

clearml Pipeline 已经提供了成熟的实验编排方案：

```python
pipe = PipelineController(project='PDEBench', name='train_validate')
pipe.add_step(name='train', base_task_id=TRAIN_TASK_ID, ...)
pipe.add_step(name='validate', parents=['train'], base_task_id=VAL_TASK_ID, ...)
```

expflow 在此基础封装：
- 自动注入 dataset ID 到 pipeline 参数
- 自动注入 compliance 检查 step
- Hermes MCP 暴露 `pipeline_submit` / `pipeline_list` / `pipeline_status`

---

## 8. 设计决策记录

### ADR-001：放弃 DVC

- **时间：** 2026-05-13
- **决策：** 不使用 DVC，所有数据管理通过 clearml Dataset API + Fileserver
- **理由：** clearml 已提供完整的版本管理 + 血缘追踪 + 合规标注
- **后果：** 所有 experiment 的 input/output 数据都在 clearml 体系内可追溯

### ADR-002：Dataset 命名规范

- **时间：** 2026-05-13
- **决策：** Dataset name 使用 `{PDE名称}_{参数}/v{版本}` 格式
- **例子：** `1D_Burgers_Sols_Nu0.001/v1.0`
- **理由：** 与 PDEBench 官方命名一致

### ADR-003：expflow 不重复实现 clearml-data 的文件传输逻辑

- **时间：** 2026-05-13（clearml 文档调研后更新）
- **决策：** `dataset_upload()` / `dataset_download()` 内部直接调 clearml Dataset SDK，不手写 MD5/hash/分片逻辑
- **理由：** clearml 已提供完整的 hash 计算、差异化上传、缓存管理、并行下载
- **后果：** 实现代码量减为~20 行/函数，clearml-data 的升级自动带来功能增强

### ADR-004：先在 guides/ 模式中使用 clearml Pipeline 替代 expflow dispatcher

- **时间：** 2026-05-13
- **决策：** expflow 作为 clearml Pipeline 的"合规增强层"，而非替代品
- **理由：** clearml Pipeline 已经是成熟的实验编排方案（step 依赖、缓存、并行），expflow 不做重复轮子
- **后果：** expflow `run` 组改为 Pipeline 封装 + 合规注入

---

## 9. 调研总结：clearml 完善度评估

对 Agentic4Sci 场景的 clearml 各子系统的完善度（1-5星）：

| 子系统 | 完善度 | 评价 |
|--------|:----:|------|
| **clearml-data**（Dataset） | ★★★★★ | 文件版本管理、血缘追踪、差异化存储——完全满足需求 |
| **Task 实验追踪** | ★★★★★ | 自动捕获超参/模型/日志，`Task.init()` 一行搞定 |
| **Model 注册** | ★★★★☆ | `query_models(metadata=...)` 支持合规过滤，但需要 clearml-server 运行 |
| **Pipeline** | ★★★★☆ | 成熟的实验编排，但 remote 模式需要队列+agent 配合 |
| **HyperParameterOptimizer** | ★★★★☆ | 内置 Optuna/BOHB，但需要 base_task 已注册 |
| **Agent 调度** | ★★★★★ | `clearml-agent daemon` 非常成熟，Docker/conda/git 全自动 |
| **Serving** | ★★★☆☆ | 我们暂时不需要模型部署，略过 |
| **Scheduler/Trigger** | ★★★★☆ | TaskScheduler/TriggerScheduler 可以替代大部分 cron job |

**总体评价：clearml 非常完善，expflow 的核心价值不在"替代 clearml 功能"，而在于：**

1. **合规审计层** — 竞赛约束标注 + lineage 验证（clearml 没有）
2. **Hermes MCP 接口** — Agent 友好的工具暴露
3. **统一 CLI** — 跨 clearml/optuna/langfuse 三系的统一入口
