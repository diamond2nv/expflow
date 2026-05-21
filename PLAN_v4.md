# PLAN_v4 — SQLite 调度中心：分布式实验编排的鲁棒基石

> **核心辩证**: 当前 expflow 的 clearml-Pipeline 依赖网络 (clearml-server:8080/8008/8081) 是不安全点。
> SQLite 作为调度入口解决此问题，同时与 clearml-agent 的 pull-based 分发、Optuna RDB storage、Langfuse 追踪形成完整的"本地调度→远程分发→审计追踪"三层架构。

**版本号**: v0.6.0 (当前已验证的 architecture，部分已实现)
**核心变更**: 引入 sqlite3 作为调度持久化引擎 → 替换当前 `.jsonl` + 新增树形实验分支追踪
**依赖**: `sqlite3`（stdlib，零额外依赖）、clearml-server 已有、Optuna SQLite RDB 已有

---

## 一、核心认知升级：从 JSONL 到 SQLite

当前 `~/.expflow/experiments.jsonl` 的问题：

| 问题 | 当前 .jsonl | SQLite |
|------|:-----------:|:------:|
| 并发安全 | ❌ append 追加可能交错 | ✅ `BEGIN...COMMIT` 事务 |
| 查询效率 | ❌ O(n) 全量扫描 | ✅ O(log n) 索引 |
| 树形查询（parent_id） | ❌ 全量加载后 filter | ✅ `WHERE parent_id=?` |
| 状态更新 | ❌ 重写全部 | ✅ `UPDATE ... WHERE id=?` |
| 字段类型 | ❌ 全 string | ✅ INTEGER/REAL/TEXT/BOOL |
| 完整性约束 | ❌ 无 | ✅ UNIQUE, FOREIGN KEY, CHECK |
| 多进程访问 | ❌ 不安全 | ✅ WAL 模式支持 |
| **子系统间共享** | ❌ 每个模块自带 registry | ✅ 统一 dispatch DB |

### 关键洞察：SQLite 不仅是存储，而是调度引擎

```
SQLite dispatch.db
  ├─ experiments    → 实验定义 + 状态机
  ├─ branches       → 树形实验分支（parent_id 实现 AI Scientist 风格树搜索）
  ├─ artifacts      → 产出的 checkpoint/plot/metrics 引用
  └─ metrics        → 数值指标（避免反范式化争议，独立表更灵活）
  
Hermes Agent ──读写──→ dispatch.db ──触发──→ clearml-agent queue
  (本地决策)               (调度决策)             (远程执行)
                              │
                              ├──→ Optuna RDB (trial 历史)
                              │
                              └──→ Langfuse (审计追踪)
```

---

## 二、架构全景

```
┌──────────────────────────────────────────────────────────┐
│                      Hermes Agent                         │
│  ┌─ diagnose ─→ suggest ─→ iterate ─→ submit ─→ collect ┐ │
│  │  (L1 规则)   (LLM 推理)   (自动化)   (调度)   (审计)   │ │
│  └─────────────────────────────────────────────────────┘ │
│                            │                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              expflow CLI / MCP                       │ │
│  │  submit | status | compare | iterate | audit | mcp  │ │
│  └─────────────────────────────────────────────────────┘ │
│                            │                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          expflow_pde 核心层 (7个模块)                 │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐  │ │
│  │  │ dispatch │ │ pipeline│ │  optuna  │ │ langfuse │  │ │
│  │  │  .py     │ │  .py    │ │  .py     │ │  .py     │  │ │
│  │  └────┬─────┘ └────┬────┘ └────┬────┘ └────┬─────┘  │ │
│  └───────┼────────────┼───────────┼───────────┼────────┘ │
└──────────┼────────────┼───────────┼───────────┼──────────┘
           │            │           │           │
           ▼            ▼           ▼           ▼
┌──────────────────────────────────────────────────────────┐
│               Layer 3 — SQLite dispatch.db                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  experiments    │  branches    │  artifacts        │  │
│  │  ├─ id PK       │  ├─ id PK   │  ├─ exp_id FK     │  │
│  │  ├─ parent_id FK│  ├─ exp_id  │  ├─ type (ckpt/   │  │
│  │  ├─ status FSM  │  ├─ depth   │  │  plot/metric)  │  │
│  │  ├─ script/args │  └─ status  │  ├─ path           │  │
│  │  ├─ result_json │             │  ├─ checksum       │  │
│  │  ├─ created/    │             │  └─ created_at     │  │
│  │  │  updated     │             │                     │  │
│  │  └─ clearml_id  │             │                     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│ clearml-svr  │ │Optuna RDB│ │ Langfuse │
│ (queue+exec) │ │ (trial)  │ │ (trace)  │
└──────┬───────┘ └──────────┘ └──────────┘
       │ pull-based
       ▼
┌──────────────┐
│ 5090 GPU     │
│ clearml-agent│
└──────────────┘
```

---

## 三、辩证分析：为什么要用 SQLite，而不是……

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **SQLite** (本次推选) | 零依赖、事务安全、树查询、跨进程 | 单写限制(WAL缓解)、无权限管理 | ✅ **最佳匹配** |
| 保持 `.jsonl` | 已有、简单 | 并发不安全、无类型、查询O(n) | ❌ 无法支持分布式 |
| PostgreSQL (RDS) | 强并发、权限管理 | 需安装运维、国内网络不稳定 | ❌ 过度杀伤 |
| clearml 做唯一调度入口 | 集中管理 | **断网就 dead** (已验证挂死) | ❌ 违背鲁棒原则 |
| Redis | 快 | 内存型、持久化弱 | ❌ 实验数据不适合 |
| **Hybrid**：SQLite 做调度决策，clearml 做执行 | 本地不依赖网络+远程分发 | 两套状态需同步 | ✅ **最佳实践** |

### 核心辩证思考

1. **clearml 作为执行层而非调度层**
   - clearml 的 Pipeline/HPO 功能都是通过 clearml-server 编排的——这意味着**网络不可用时 Pipeline 无法工作**
   - 让 clearml 只负责"发送 Task 到 agent + 收集 artifact"，不做调度决策
   - SQLite 做调度决策，生成 Task 后通过 clearml SDK enqueue——网络不可用时至少本地决策不受影响

2. **clearml 已有状态 vs SQLite 状态的双存问题**
   - SQLite 存的是**调度意图**和**实验树关系**（parent_id、分支策略）
   - clearml 存的是**执行状态**（queued/running/completed/failed）
   - 二者通过 `clearml_task_id` 关联，互为主备查询
   - Hermes 优先读 SQLite（本地、快），fallback 到 clearml API（当 SQLite 状态 stale 时）

3. **SQLite 的并发边界在哪？**
   - 多进程写入：WAL 模式支持多读一写
   - 写冲突概率低：一次 submit 才写一次，不是高频写
   - 但 cron 任务轮询调度（每 15 分钟查 pending）和 Hermes 会话提交可能同时写——需 `BEGIN IMMEDIATE` + retry

---

## 四、Dispatch DB Schema 设计（基于 hfpapers-paper_store 优化）

### 4.1 连接管理（采用 paper_store 工厂模式）

```python
# expflow_pde/dispatch.py — 新增 SQLite 连接管理
import sqlite3
import threading
from contextlib import contextmanager

class DispatchDB:
    """SQLite-backed experiment dispatch database.
    
    连接管理借鉴 hfpapers-crawler PaperStore 的成熟模式：
    - 每次 CRUD 创建新连接（SQLite 开销极低 ~0.1ms）
    - 写操作加 threading.Lock() 串行化
    - 读操作不加锁（WAL 模式支持并发读）
    - row_factory=sqlite3.Row
    """
    
    def __init__(self, db_path: str | None = None):
        self.path = db_path or os.path.expanduser("~/.expflow/dispatch.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()
    
    def _conn(self) -> sqlite3.Connection:
        """创建新连接（非线程安全，每次 CRUD 新建）。"""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row          # ← 命名列访问
        conn.execute("PRAGMA journal_mode=WAL") # ← WAL 模式
        conn.execute("PRAGMA synchronous=NORMAL") # ← 性能与安全平衡
        conn.execute("PRAGMA foreign_keys=ON")   # ← 外键约束
        return conn
    
    @contextmanager
    def _write_tx(self):
        """写事务：加锁 + 自动 commit/rollback。"""
        with self._lock, self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    @contextmanager
    def _read_tx(self):
        """读事务：不加锁，只读。"""
        with self._conn() as conn:
            yield conn
```

### 4.2 Schema 定义（在 _init_schema 中执行）

```sql
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    parent_id       TEXT REFERENCES experiments(id),
    root_id         TEXT NOT NULL REFERENCES experiments(id),
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN (
                        'pending','queued','running','completed',
                        'failed','cancelled','pruned'
                    )),
    fsm_state       TEXT DEFAULT 'ideation',
    script          TEXT NOT NULL,
    args_json       TEXT NOT NULL DEFAULT '{}',
    search_space_json TEXT,
    tags_json       TEXT DEFAULT '[]',
    queue           TEXT DEFAULT 'default',
    project         TEXT DEFAULT 'expflow',
    branch          TEXT,
    commit_hash     TEXT,
    clearml_task_id TEXT,
    
    best_value      REAL,
    best_params_json TEXT,
    result_summary  TEXT,
    error_message   TEXT,
    
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    started_at      TEXT,
    completed_at    TEXT,
    
    source          TEXT DEFAULT 'hermes',
    version         TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_parent  ON experiments(parent_id);
CREATE INDEX IF NOT EXISTS idx_experiments_root    ON experiments(root_id);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_ctask   ON experiments(clearml_task_id);

CREATE TABLE IF NOT EXISTS branches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_exp_id   TEXT NOT NULL REFERENCES experiments(id),
    child_exp_id    TEXT NOT NULL REFERENCES experiments(id),
    strategy        TEXT,
    condition_json  TEXT,
    depth           INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(child_exp_id)
);

CREATE INDEX IF NOT EXISTS idx_branches_parent ON branches(parent_exp_id);
CREATE INDEX IF NOT EXISTS idx_branches_child  ON branches(child_exp_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id),
    type            TEXT NOT NULL CHECK(type IN (
                        'checkpoint','plot','dataset','log','submission','report'
                    )),
    name            TEXT NOT NULL,
    path            TEXT NOT NULL,
    checksum        TEXT,
    size_bytes      INTEGER,
    metadata_json   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_artifacts_exp ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(type);

CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id),
    name            TEXT NOT NULL,
    value           REAL NOT NULL,
    iteration       INTEGER,
    group_name      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id, name, iteration)
);

CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT,
    event_type      TEXT NOT NULL,
    detail_json     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_audit_exp ON audit_log(experiment_id);
```

### 4.3 迁移函数（幂等，采用 hfpapers 检查模式）

```python
def _ensure_migration(self):
    """幂等迁移：检查列是否存在，不存在才 ALTER TABLE。
    
    借鉴 hfpapers-download_queue 的 ensure_migration 模式。
    """
    with self._conn() as conn:
        has_col = conn.execute(
            "SELECT COUNT(*) FROM pragma_table_info('experiments') "
            "WHERE name = 'source'"
        ).fetchone()[0]
        if has_col:
            return
        
        migrations = [
            "ALTER TABLE experiments ADD COLUMN source TEXT DEFAULT 'hermes'",
            "ALTER TABLE experiments ADD COLUMN version TEXT",
        ]
        for stmt in migrations:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        conn.commit()
```

### 4.4 典型 CRUD 方法（写加锁，读不加锁）

```python
def register_experiment(self, script, args=None, parent_id=None,
                         queue='default', project='expflow') -> dict:
    """注册新实验（写操作）。"""
    exp_id = _snowflake_id()
    now = datetime.utcnow().isoformat()
    root_id = parent_id if parent_id else exp_id
    args_json = json.dumps(args or {})
    
    with self._write_tx() as conn:
        conn.execute(
            """INSERT INTO experiments 
               (id, parent_id, root_id, script, args_json, queue, project,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, parent_id, root_id, script, args_json,
             queue, project, now, now),
        )
        
        if parent_id:
            # 同时写入 branches 表
            depth = self._get_depth(conn, parent_id) + 1
            conn.execute(
                "INSERT INTO branches (parent_exp_id, child_exp_id, depth) "
                "VALUES (?, ?, ?)",
                (parent_id, exp_id, depth),
            )
    
    return {"experiment_id": exp_id, "root_id": root_id, 
            "status": "pending", "created_at": now}

def query_recent(self, limit=20, status=None) -> list[dict]:
    """查询实验（读操作，不加锁）。"""
    with self._read_tx() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]  # ← sqlite3.Row → dict

def record_metric(self, exp_id, name, value, iteration=None):
    """记录指标（写操作）。"""
    with self._write_tx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO metrics "
            "(experiment_id, name, value, iteration, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (exp_id, name, value, iteration),
        )

def get_experiment_tree(self, root_id) -> list[dict]:
    """获取整棵实验树（读操作）。"""
    with self._read_tx() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments WHERE root_id=? "
            "ORDER BY created_at ASC",
            (root_id,),
        ).fetchall()
        return [dict(r) for r in rows]

def _get_depth(self, conn, parent_id) -> int:
    """递归查询深度（仅供内部写事务使用）。"""
    row = conn.execute(
        "SELECT depth FROM branches WHERE child_exp_id=?",
        (parent_id,),
    ).fetchone()
    if row:
        return row["depth"]
    return 0
```

### 4.5 与 hfpapers paper_store 的关键差异

| 方面 | paper_store | DispatchDB (新) | 原因 |
|:----|:-----------|:----------------|:-----|
| 锁粒度 | 全局 `self._lock` | **`_write_tx` 写锁**，`_read_tx` 无锁 | 实验调度读多写少，锁放开读性能更好 |
| 写入 | 每操作 `with self._lock, self._conn()` | **显式 `BEGIN IMMEDIATE`** | 防止多个写事务交叉阻塞 |
| 返回类型 | `dict(row)` 或 `PaperRecord` | **统一返回 `dict`** | JSON 序列化兼容 clearml/Langfuse MCP |
| 批处理 | `executemany` | `executemany`（相同） | 一致 |
| 迁移 | 启动时自动检查 | 启动时自动检查（相同） | 一致 |
| 每个 table | 独立 .db | 5 表 + 8 索引在同一 .db（外键关联） | 实验调度比论文存储表间关联更密 |


## 五、与现有 expflow 模块的集成
## 五、与现有 expflow 模块的集成

### 5.1 调度内核 (dispatch.py) — 增强

当前 dispatch.py 已有 `_load_registry()` / `_save_to_registry()` 操作 `.jsonl`。改为线程安全的 `DispatchDB` 类：

```python
# expflow_pde/dispatch.py — 新增 DispatchDB 类

import sqlite3
from contextlib import contextmanager

_DB: 'DispatchDB | None' = None

class DispatchDB:
    """SQLite-backed experiment dispatch database.
    
    Thread-safe (WAL mode). Handles all CRUD for experiments/branches/artifacts/metrics.
    Zero external dependencies — uses Python stdlib sqlite3.
    """
    
    def __init__(self, db_path: str | None = None):
        self.path = db_path or os.path.expanduser("~/.expflow/dispatch.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
    
    @contextmanager
    def transaction(self):
        """Yield a cursor in an active transaction."""
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")  # Blocks other writers
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def register_experiment(self, ...) -> dict: ...
    def update_status(self, exp_id: str, status: str, **fields) -> dict: ...
    def query_experiments(self, status=None, project=None, tag=None,
                          parent_id=None, root_id=None, limit=50) -> list[dict]: ...
    def add_artifact(self, experiment_id, type, name, path, ...): ...
    def record_metric(self, experiment_id, name, value, iteration=None): ...
    def compare_scores(self, metric_name, gates=None, sort_by='value',
                       direction='desc', limit=20) -> list[dict]: ...
    def get_experiment_tree(self, root_id) -> list[dict]: ...
    def close(self): ...
```

### 5.2 Pipeline 集成 — 增强当前 pipeline.py

当前 pipeline.py 已有 `ExperimentPipeline.train_val_submit()` / `train_hpo_val_submit()`。增强：

```python
# pipelin模式 A (full) : HPO → Train → Eval
def train_hpo_val_submit(self, ...):
    # 1. 写 DispatchDB → experiments 表 (status='pending')
    # 2. 创建 clearml HPO Task → enqueue(queue)
    # 3. clearml 自动创建子 Task（每个 Trial 一个）
    # 4. 每 Trial 完成 → Hermes / cron 轮询 clearml status → update DispatchDB
    # 5. 收集 best_params → 写回 experiments.best_params_json
    # 6. 发起 Train step (parent_id 指向 HPO experiment)
    # 7. Train 完成 → Eval step
    pass
```

### 5.3 FSM 状态机 — 重审 7-state

当前 `fsm.py` 定义的 7 状态应映射到 DispatchDB 的 `fsm_state` 字段：

```
ideation → hpo_tuning → training → evaluation → submission → review → archived
    ↑           │            │           │            │          │
    └───────────┴────────────┴───────────┴────────────┴──────────┘  (any → failed)
```

每个状态转换写入 `audit_log`。

### 5.4 Hermes cron 轮询 — 分布式实验采集

```python
# cron: 每 15 分钟执行
# 1. 连接 DispatchDB
# 2. SELECT experiments WHERE status='pending' OR status='running'
# 3. 对于 pending:
#    - 检查 queue 是否有空间 → 若无，跳过
#    - 若有 → update status='queued', create clearml Task, enqueue
# 4. 对于 running:
#    - 通过 clearml API 查最近状态
#    - 若 completed → update status + 提取 metrics (best_value)
#    - 若 failed → update status + 错误记录
#    - 若 prunable → update status='pruned' (Optuna pruner)
# 5. 对 newly completed:
#    - 检查 root_id → 若还有兄弟/子实验 pending，继续
#    - 若全完成 → 触发下一阶段（如 train→eval, hpo→train）
# 6. 生成摘要 → 写入 audit_log → 可推送 QQ
```

### 5.5 Langfuse 双轨链接

每次状态转换同时写入 Langfuse：

```python
def _sync_to_langfuse(exp_id: str, db: DispatchDB, lf: langfuse.Langfuse):
    exp = db.get_experiment(exp_id)
    # clearml_task_id → clearml 的 scalars/params
    # Langfuse trace name = exp_id
    # session_id = root_id (整棵树共享一个 session)
    lf.trace(...)
```

---

## 六、AI Scientist 风格的树搜索映射

AI Scientist-v2 的 Agentic Tree Search 在 expflow 的 SQLite 架构中可自然表达：

| AI Scientist 概念 | expflow SQLite 映射 |
|---|---|
| 根节点（初始实验） | `root_id = id` (self-referencing) |
| 子实验分支（消融/调参） | `parent_id = <父实验>`, `branches.strategy='ablation'` |
| 超参数节点 | `experiments.search_space_json` + Optuna RDB 联动 |
| 报错→自调试 | `status='failed'` → `error_message` → Hermes 重试（新实验 parent=原实验） |
| VLM 图表质量门禁 | `artifacts.type='plot'` + `artifacts.metadata_json` 存 VLM 评分 |
| 最佳优先搜索（best-first search） | `SELECT * FROM experiments WHERE root_id=? AND status='completed' ORDER BY best_value DESC LIMIT 1` |
| 节点分配预算（每阶段 N 个节点） | `branches` 表 + `branches.strategy` 按策略过滤 |
| 剪枝（pruning） | `status='pruned'` |

**关键差异**：AI Scientist 的树搜索是 LLM 驱动的实验设计决策（决定什么实验值得做），expflow 的树是**超参/消融分支**的自动化编排。前者需要 LLM 推理，后者可规则驱动。但在 schema 层面，SQLite 统一支持两种模式。

---

## 七、与现有系统集成关系

```
                    SQLite dispatch.db
                    (调度意志 + 树关系)
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Hermes Agent    clearml Server   Langfuse Server
    (读+写调度)     (执行 + artifact)  (审计追踪)
          │              │
          │              │ pull-based
          │              ▼
          │         clearml-agent
          │         (5090/3080 GPU)
          │
          ▼
    ~/.expflow/task_meta.yaml
    (策略记忆, LLM reasoning)
```

**数据流完整示例**：

```
1. Hermes: expflow pipeline submit-full train_task1.py --trials 30
2. DispatchDB: INSERT experiments (id=exp:a1b2, status='pending', root_id=exp:a1b2)
3. DispatchDB: INSERT audit_log (event='submit')
4. Hermes cron: SELECT pending → create clearml HPO Task → enqueue → update status='queued'
5. clearml-agent: pull Task → train → report scalar → complete
6. Hermes cron: SELECT running → check clearml API → get best_value=57.09
   → UPDATE experiments SET status='completed', best_value=57.09
   → INSERT metrics (name='seg_total', value=57.09)
   → INSERT audit_log (event='status_change')
7. DispatchDB: 自动触发下一阶段——root 实验下创建 child experiment
   INSERT experiments (id=exp:c3d4, parent_id=exp:a1b2, root_id=exp:a1b2,
                       script='eval_task1.py', status='pending',
                       args_json='{"checkpoint": "best_ckpt.pt"}')
8. Langfuse: trace.create(name='exp:a1b2', session_id='exp:a1b2')
9. Hermes: expflow status exp:a1b2 → 读取 DispatchDB + clearml 拉回最新 → 展示树
```

---

## 八、执行路线

| Phase | 内容 | 估时 | 优先级 | 当前状态 |
|:-----:|------|:----:|:-----:|:-------:|
| **1** | `DispatchDB` 类：创建 schema + 核心 CRUD + 事务 + WAL + 测试 | 2h | 🔴 | ❌ 待实现 |
| **2** | 迁移 dispatch.py：从 `.jsonl` + `_experiments` dict → `DispatchDB` | 1h | 🔴 | ❌ 待实现 |
| **3** | cron 轮询：`pending/running` → clearml API 同步 → 状态更新 | 1.5h | 🔴 | ❌ 待实现 |
| **4** | pipeline.py 增强：写 DispatchDB + 树形分支追踪 | 1h | 🟡 | 部分 `jsonl` 已 |
| **5** | metrics 表 + compare-scores 从 clearml 改为查询 SQLite | 1h | 🟡 | 已有 clearml 版 |
| **6** | `expflow audit` + 审计日志 + Langfuse 同步 | 1h | 🟡 | 已有 langfuse 模块 |
| **7** | MCP 工具增强：13 个工具全覆盖 (11 已有) | 1h | 🟢 | ✅ 11/13 已有 |
| **8** | Wiki 更新 + AGENTS.md + skills 同步 | 0.5h | 🟢 | ❌ |

---

## 九、非目标（明确不做）

- ❌ 不替换 clearml 的执行层——只增强调度层
- ❌ 不重写现有 pipeline.py 的 3 种模式——只增加 DispatchDB 作为数据后端
- ❌ 不清理旧 `.jsonl` 数据——可迁移脚本，但非紧急
- ❌ 不引入 Redis/PostgreSQL——SQLite 已足够且更鲁棒
- ❌ 不实现 AI Scientist 风格的 LLM 驱动的实验设计——保持规则引擎 + Hermes 推理的混合模式

---

## 十、价值验证清单

- [x] 网络不可用时本地调度不中断（DispatchDB 本地 sqlite）
- [x] 树形实验分支追踪（`parent_id` + `branches` 表）
- [x] 多状态机支持（`status` 调度 + `fsm_state` 生命周期）
- [x] 指标标准化查询（`metrics` 表索引 + `compare-scores`）
- [x] 产物追踪（`artifacts` 表，checkpoint/plot 一对一关联）
- [x] 审计日志（`audit_log` 表，不可变追加）
- [x] 与现有 clearml/Optuna/Langfuse 双向关联
- [x] 零额外依赖（Python stdlib `sqlite3` 即可）
- [x] cron 友好（WAL 模式多进程安全）
