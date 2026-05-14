# expflow 数据层设计：基于 ClearML Fileserver 的实验数据管理

> **设计决策：** 放弃 DVC，改用 ClearML Server 内置的 Fileserver（基于 MinIO）
> 作为 Agentic4Sci 的唯一实验数据层。
>
> **原因：** 无需独立部署 MinIO 服务 —— clearml docker compose 已内置 fileserver，
> 且 clearml SDK 的 Dataset/Model 体系天然提供版本管理、血缘追踪、合规标注能力。

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agentic4Sci 服务器                         │
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ PDEBench     │    │ Hermes Agent │    │ clearml-agent (本机) │  │
│  │ 训练脚本     │ ──→ │ expflow CLI  │ ──→ │ SDK API            │  │
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
│  │  │ (元数据)     │  │ (搜索索引)   │  │ (MinIO 兼容)   │ │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心原则

1. **ClearML Fileserver 是唯一数据底座** — 所有实验数据、模型权重、数据集都通过 clearml SDK 的 Dataset/Model API 上传到 Fileserver
2. **不引入 DVC** — clearml 的 Dataset class 已内置版本管理、血缘追踪、元数据标注能力
3. **Fileserver 即 MinIO** — 支持 S3 兼容协议，可通过 clearml.conf 配置为外置 S3/MinIO，无需改代码

---

## 2. 存储模型

### 2.1 三类存储对象

| 类别 | clearml API | Fileserver 路径 | 用途 |
|------|-------------|----------------|------|
| **Dataset** | `clearml.Dataset` | `/mnt/fileserver/datasets/<id>/` | PDEBench 数据集原始 HDF5 |
| **Model** | `Task.get_output_model()` | `/mnt/fileserver/models/<id>/` | FNO/DeepONet checkpoint |
| **Artifact** | `Task.upload_artifact()` | `/mnt/fileserver/artifacts/<task_id>/` | 评估结果、JSON、PNG |

### 2.2 数据集版本模型

```
Dataset(name="1D_Burgers_Sols_Nu0.001", project="PDEBench")
  ├── v1.0   ← 原始官方数据 (MD5: a1b2c3...)
  ├── v1.1   ← 裁剪后竞赛用 (MD5: d4e5f6...)
  └── v2.0   ← 数值求解器合成的长时数据 (MD5: g7h8i9...)
```

每个版本通过 `clearml.Dataset.create()` 创建，自动继承父 Dataset 的血缘：
- `dataset_v2 = Dataset.create(parent_datasets=[dataset_v1])`
- lineage 可通过 `Dataset.get(dataset_id).parent` 追溯

### 2.3 数据集合规标注

使用 clearml Dataset 的 tags 和 metadata 存储竞赛合规信息：

```python
dataset.set_metadata("expflow:compliance", "allowed")    # competition-legal
dataset.set_metadata("expflow:compliance", "forbidden")   # not allowed
dataset.set_metadata("expflow:md5", "a1b2c3d4...")       # file integrity
dataset.set_metadata("expflow:source", "official")        # official | synthetic
```

---

## 3. API 设计

### 3.1 expflow 新增函数

#### `clearml.dataset_upload()`

```python
def dataset_upload(
    name: str,
    version: str,
    local_path: str,
    parent_dataset_id: str | None,
    compliance: Literal["allowed", "forbidden"],
    md5: str | None,
    description: str | None,
) -> dict:
    """
    上传本地 HDF5 到 clearml Fileserver，注册为 Dataset。
    - 自动计算 MD5（如未提供）
    - 通过 Dataset.create() 继承血缘
    - 同步文件到 Fileserver
    """
```

#### `clearml.dataset_download()`

```python
def dataset_download(
    dataset_id: str,
    local_path: str,
) -> dict:
    """
    从 clearml Fileserver 下载 Dataset 到本地。
    - 自动 MD5 校验
    - 缓存到 ~/.clearml/cache/ 后 symlink 到目标
    """
```

#### `clearml.dataset_lineage()`

```python
def dataset_lineage(
    dataset_id: str,
    depth: int = 3,
) -> list[dict]:
    """
    沿 parent 链追溯 Dataset 血缘。
    返回顺序列表：[oldest → newest]
    """
```

#### `clearml.model_list()`

```python
def model_list(
    task_id: str | None,
    project: str | None,
) -> list[dict]:
    """
    列出已注册的 Model（checkpoint）。
    - 可按 Task ID 或 project 过滤
    - 返回 model ID、URI、task_id、tags
    """
```

### 3.2 MCP 工具

在 `expflow/mcp.py` 中注册以下工具供 Hermes Agent 使用：

| 工具名 | 功能 | 对应 expflow 函数 |
|--------|------|------------------|
| `dataset_upload` | 上传数据集并注册 | `clearml.dataset_upload()` |
| `dataset_download` | 下载数据集到本地 | `clearml.dataset_download()` |
| `dataset_list` | 列出所有数据集（可过滤 compliance） | `clearml.list_datasets()` |
| `dataset_lineage` | 追溯数据集血缘 | `clearml.dataset_lineage()` |
| `model_list` | 列出模型 checkpoint | `clearml.model_list()` |
| `model_upload` | 上传模型 checkpoint | `clearml.model_upload()` |

### 3.3 CLI 命令

在 `expflow clearml` 组中新增：

```
expflow clearml
  dataset-register   ← 已实现（metadata 标注模式）
  dataset-upload     ← NEW：上传文件到 Fileserver
  dataset-download   ← NEW：从 Fileserver 下载
  dataset-lineage    ← NEW：追溯血缘链
  dataset-list       ← 已实现
  model-list         ← NEW：列出 checkpoint
  model-upload       ← NEW：上传 checkpoint
```

---

## 4. 部署说明

### 4.1 clearml docker-compose 端口调整

官方 compose 的 webserver 使用 8080:80，与 ai-suite 的 caddy（占 80）冲突。
调整为 `8082:80`：

```yaml
webserver:
  ports:
    - "8082:80"     # 原 8080 → 改 8082
```

同时 fileserver（8081）、apiserver（8008）端口不冲突，保持不变。

### 4.2 clearml.conf 配置

**场景 A：AI Suite 内部 MinIO（通过内部网络访问）：**

```hocon
sdk {
  aws {
    s3 {
      credentials: [{
        host: "minio:9000"
        key: "minio"
        secret: "<MINIO_ROOT_PASSWORD>"
        multipart: true
        secure: false
      }]
    }
  }
}
```

**场景 B：ClearML 内置 Fileserver（默认）：**

clearml SDK 默认向 apiserver（8008）注册数据，fileserver 自动分配存储路径。
当 clearml.conf 的 `api.api_server` 指向 clearml-apiserver 时，所有数据自动存储到 fileserver。
无需额外 S3 配置。

### 4.3 数据流向

```
上传：本地 HDF5 → clearml API → fileserver /mnt/fileserver/
下载：fileserver /mnt/fileserver/ → clearml API → 本地缓存 → 目标路径
```

clearml 自动处理分片上传和缓存机制。对大数据集（> 1GB），clearml 使用 multipart upload
批量传输，支持断点续传。

---

## 5. 与现有工作流的集成

### 5.1 PDEBench 竞赛合规检查

竞赛规则要求：
1. Task1 只准用官方 Nu=0.001 数据 → 标注 `compliance=allowed`
2. 合成数据只能用于 Task2 → 标注 `compliance=forbidden`（Task1 不可用）

使用 expflow 审计模块中的 `check_dataset_provenance()` 自动验证
Task1 实验使用的数据集是否合规：

```python
audit.check_dataset_provenance(
    task_id="...",
    compliance_required="allowed",
)
```

### 5.2 Hermes Agent 实验流程

```
Hermes 发起实验
    ↓
expflow clearml dataset-download <dataset_id>
    ↓ 下载到本地 training 目录
expflow run submit <script> --dataset_id <dataset_id>
    ↓ clearml 自动记录输入数据 + output model
expflow audit validate <task_id>
    ↓ 检查 dataset lineage 是否合规
expflow clearml model-upload <local_model_path>
    ↓ checkpoint 持久化到 Fileserver
```

---

## 6. 为什么不选 DVC + 独立 MinIO

| 维度 | DVC + MinIO | ClearML Fileserver |
|------|------------|-------------------|
| 额外服务 | 另起 minio docker | clearml server 内置，0 额外 |
| 数据关联 | `dvc run` + `dvc push` 手动关联 | clearml Task 自动记录 InputModel/OutputModel |
| 版本回溯 | `git checkout dvc.lock` 或 `dvc checkout` | clearml UI / API 直接查 Dataset lineage |
| 合规审计 | 需自建 pipeline 维护 metadata | clearml Dataset tags + metadata 原生支持 |
| Hermes 集成 | 需写 DVC CLI wrapper | expflow MCP 工具直接调 clearml API |
| 学习曲线 | DVC 概念（.dvc/cache/remote/run） | clearml Dataset/Model/Artifact 三对象 |
| 生态系统 | 纯数据版本管理 | ML 全链路（实验追踪 + 数据 + 模型 + 调度） |

**结论：** 在已有 clearml 全套服务的场景下，DVC 没有增量价值。

---

## 7. 实现路线图

| Phase | 内容 | 依赖 |
|-------|------|------|
| P0 | clearml-server docker-compose 启动 | docker-compose.yml 修改端口 |
| P1 | `clearml.dataset_upload()` + `dataset_download()` | clearml SDK Dataset API |
| P2 | `clearml.dataset_lineage()` + 合规链检查 | clearml SDK parent chain |
| P3 | `clearml.model_list()` + `model_upload()` | clearml SDK Task.get_output_model() |
| P4 | MCP 工具注册 + Hermes 集成测试 | expflow/mcp.py |
| P5 | 端到端测试：Hermes → expflow → clearml → fileserver | GPU 服务器部署 |

---

## 8. 设计决策记录

### ADR-001：放弃 DVC

- **时间：** 2026-05-13
- **决策：** 不使用 DVC，所有数据管理通过 clearml Dataset API + Fileserver
- **理由：** clearml 已提供完整的版本管理 + 血缘追踪 + 合规标注，引入 DVC 只是增加一层不必要的抽象
- **后果：** 所有 experiment 的 input/output 数据都在 clearml 体系内可追溯

### ADR-002：Dataset 命名规范

- **时间：** 2026-05-13
- **决策：** Dataset name 使用 `{PDE名称}_{参数}/v{版本}` 格式
- **例子：** `1D_Burgers_Sols_Nu0.001/v1.0`
- **理由：** 与 PDEBench 官方命名一致，版本号标识数据来源（原始/裁剪/合成）

