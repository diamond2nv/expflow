# Agentic4Sci SQLite 代码技巧考察 — 学习笔记

> 考察了 hfpapers-crawler (paper_store / download_queue) + expflow (optuna/hpo) 中 SQLite 的使用模式。

---

## 一、Optuna RDBStorage 模式（最佳参考）

### 关键发现

```python
# expflow_pde/hpo.py — 分布式 HPO 的 SQLite 用法
storage_path = os.path.expanduser(f"~/.expflow/optuna_{study_name}.db")
storage = f"sqlite:///{storage_path}"

# 每个 study 一个独立 .db 文件
try:
    study = optuna.load_study(study_name=study_name, storage=storage)
except Exception:
    study = optuna.create_study(study_name=study_name, storage=storage, ...)
```

**每个 Optuna study 一个独立 SQLite 文件**。这不是必须的（Optuna 支持多 study 共享同一 storage），但 expflow 的 `hpo.py` 这样做有好处：

| 做法 | 效果 | 原因 |
|------|------|------|
| 每 study 一文件 | ✅ 隔离性好，删除/清理互不影响 | 实验不同阶段（P2 baseline vs width64）互不干扰 |
| `load_if_exists=True` 没传 | ⚠️ 每次 load_study 失败后 create_study | 其实不如 `optuna.create_study(load_if_exists=True)` 一步到位 |
| `~/.expflow/optuna_<name>.db` | ✅ 统一的管理路径 | 与 dispatch.db 共享同一目录 |

### 可借鉴的改进

```python
# 当前 expflow:
def _get_or_create_study(name, storage, ...):
    try:
        return optuna.load_study(study_name=name, storage=storage)
    except Exception:
        return optuna.create_study(study_name=name, storage=storage, ...)

# 可简化为:
study = optuna.create_study(
    study_name=name,
    storage=storage,
    load_if_exists=True,  # ← 这个 flag 就是干这个的
    ...
)
```

---

## 二、hfpapers-crawler paper_store 模式（最成熟）

### 连接管理

```python
# paper_store.py — 连接工厂模式
import threading

class PaperStore:
    def __init__(self, db_path):
        self._lock = threading.Lock()  # ← 🔴 全局锁

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row          # ← 命名列访问
        conn.execute("PRAGMA journal_mode=WAL") # ← 写前日志
        conn.execute("PRAGMA synchronous=NORMAL") # ← 性能与安全平衡
        conn.execute("PRAGMA foreign_keys=ON")   # ← 外键约束
        return conn
```

**关键观察**：

1. **每次操作都创建新连接**，而不是持有一个连接池——因为 SQLite 连接不是线程安全的。创建/销毁连接在 SQLite 中开销极低（~0.1ms）。
2. **`threading.Lock()` 全局锁**保护写操作序列化。虽然 WAL 模式允许多读一写，但 `paper_store` 选择了保守策略（每次 CRUD 都加锁）。
3. **`sqlite3.Row`** 作为 `row_factory`——支持列名访问（`row["title"]`），比元组索引（`row[1]`）可读性高很多，也比 `dict` 省内存。

### 典型的 CRUD 模式

```python
# 写入：锁 + 连接 + 明确参数绑定
def upsert_paper(self, record):
    with self._lock, self._conn() as conn:  # ← 锁保护
        if record.sf_id:
            conn.execute(
                "UPDATE papers SET title=?, abstract=?, ... WHERE sf_id=?",
                (record.title, record.abstract, ..., record.sf_id),
            )
        else:
            sf_id = snowflake_id()
            conn.execute(
                "INSERT INTO papers (...) VALUES (?, ?, ...)",
                (sf_id, record.title, ...),
            )
    # 不需要 conn.commit() — context manager 的 __exit__ 会自动 commit

# 读取：不需要锁（WAL 模式支持并发读）
def get_paper(self, sf_id):
    with self._conn() as conn:  # 注意：没有 self._lock！
        row = conn.execute("SELECT * FROM papers WHERE sf_id=?", (sf_id,)).fetchone()
        if row is None:
            return None
        return dict(row)  # ← sqlite3.Row → dict 更方便
```

**重要细节**：读操作不加锁。WAL 模式下多进程同时读是安全的，写操作通过 `self._lock` 串行化。

### 迁移模式

```python
# download_queue.py — 优雅的表结构迁移
def ensure_migration():
    """Add status columns if not present (idempotent)"""
    store = get_store()
    with store._lock, store._conn() as conn:
        has_col = conn.execute(
            "SELECT COUNT(*) FROM pragma_table_info('papers') WHERE name = 'converted_at'"
        ).fetchone()[0]
        if has_col:
            return  # Already migrated

        for stmt in MIGRATE_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        continue
                    raise
        conn.commit()
```

关键：**幂等性迁移**——先检查列是否存在，不存在才执行 `ALTER TABLE`。`ALTER TABLE` 失败时捕获 `duplicate column` 错误并安全跳过。

### 批量操作模式

```python
# 批量更新 — executemany 快于逐行 execute
conn.executemany(
    "UPDATE papers SET download_status='done' WHERE /* ... */",
    [(aid,) for aid in done_ids],
)

# 批量查询 + 排序
rows = conn.execute(
    """
    SELECT p.sf_id, p.title, i.id_value as arxiv_id,
           CASE WHEN i.id_value GLOB '[0-9][0-9]*'
           THEN CAST('20' || SUBSTR(i.id_value, 1, 2) AS INTEGER)
           ELSE 0 END as arxiv_year
    FROM papers p
    JOIN identifiers i ON p.sf_id = i.sf_id AND i.id_type='arxiv'
    WHERE p.download_status='pending'
    ORDER BY p.relevance DESC, arxiv_year DESC
    LIMIT ?
    """,
    (batch_size,),
).fetchall()
```

---

## 三、对比总结：三个项目的 SQLite 模式

| 方面 | hfpapers paper_store | optuna RDBStorage | expflow dispatch.jsonl (待升级) |
|:----|:-------------------:|:-----------------:|:-------------------------------:|
| 存储格式 | **SQLite** | **SQLite** (optuna管理) | **JSONL** |
| 连接模式 | 工厂模式（每次新连接） | Optuna 内部管理 | 无 |
| 并发控制 | `threading.Lock()` 锁写 | Optuna 内部锁 | 不安全 |
| 读取 | 不加锁 | 无锁 | 全量扫描 |
| 迁移 | `PRAGMA table_info` 检测 + 幂等 ALTER | Optuna 自动升级 | 无 |
| 行工厂 | `sqlite3.Row` → `dict()` | Optuna 内部 | N/A |
| 批量操作 | `executemany` | Optuna 内部 | 低效 |
| 表数 | 3 + 迁移列 | ~20 (optuna内部) | 1 (如果算) |
| 每个 experiment | 共享大库 | 每 study 一文件 | 共享文件 |

---

## 四、对 expflow DispatchDB 的优化建议（基于学习）

| 学习来源 | 建议 | 影响 |
|:--------:|------|:----:|
| paper_store | 用 `sqlite3.Row` 做 `row_factory` | 可读性 + 内存效率 |
| paper_store | 读操作不加锁，写操作加 `threading.Lock()` | 并发性能 |
| paper_store | 用 dict comprehension 或 `dict(row)` 返回结果 | 标准化输出 |
| paper_store | `_conn()` 方法每次创建新连接，用 `with` 管理生命周期 | 简洁安全 |
| paper_store | `PRAGMA synchronous=NORMAL`（非 FULL） | 写入速度 2-3x |
| paper_store | 迁移函数用 `pragma_table_info` 检查 + 幂等执行 | 安全升级 |
| paper_store | `executemany` 批量写入 | 批量速度 10x |
| optuna hpo.py | 每个研究/树分支可考虑独立 db 文件 | 隔离性更好 |
| optuna RDBStorage | `load_if_exists=True` 简化 create_study | 少一个 try/except |
| **hfpapers** | WAL + 锁已经是生产验证的成熟模式 | 可以直接复用整套模式 |
