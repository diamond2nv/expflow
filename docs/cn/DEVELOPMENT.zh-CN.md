# 开发者指南

## 开发环境

```bash
# 克隆并安装
git clone https://github.com/diamond2nv/expflow.git
cd expflow
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 验证
expflow --help
```

## 代码规范

| 标准 | 值 |
|------|-----|
| **语言** | Python 3.11+ |
| **格式化** | Ruff（行长度=100，双引号） |
| **Lint** | Ruff（E, W, F, I, N） |
| **类型检查** | Pyright（`expflow_pde/` 严格模式） |
| **测试框架** | Pytest（329+ 测试） |
| **包管理** | pip + setuptools |

### PEP8 国际化

所有 Python 文件必须为纯英文：
- **注释**：仅英文（文档字符串、行内注释、块注释）
- **字符串**：仅英文（print、log、错误信息、CLI 输出）
- **变量/函数/类名**：仅英文（PEP8 命名）
- **`.py`/`.yaml`/`.sh` 文件中禁止中文字符、emoji、制表符

原因：`conda` 环境若设 `LC_ALL=C`，非 ASCII 输出会触发 `UnicodeEncodeError`。

例外情况（允许中文）：
| 位置 | 内容 | 原因 |
|------|------|------|
| `docs/cn/` | 中文文档 | 面向中文读者 |
| `README.md` | `简体中文` 导航链接 | 仅一行标签 |
| `.hermes/` | Hermes agent 计划 | 内部工具 |

### 中文文档约定

遵循 hfpapers-crawler 项目确立的惯例：中文文档必须是英文原版的**逐行翻译**。
这支持差异跟踪、并排编辑和自动同步检查。
始终先更新英文版，然后将编辑内容镜像到中文版。

## 运行命令

```bash
# 格式化代码
ruff format .

# Lint + 自动修复
ruff check --fix .

# 类型检查
pyright .

# 运行测试
python -m pytest tests/ -v            # 全部测试，详细输出
python -m pytest tests/ -q            # 静默模式
python -m pytest tests/test_pin.py    # 单个测试文件
python -m pytest tests/ -x -v         # 首次失败时停止

# 运行测试带覆盖率
python -m pytest tests/ --cov=expflow_pde

# 构建包
python -m build

# 验证包
twine check dist/*
```

## 测试指南

### 测试策略

| 类别 | 描述 | 外部依赖 |
|------|------|----------|
| 单元 | 配置 CRUD、CLI 解析 | 无 |
| 单元 | 模块级逻辑（pin, metrics, compare, equations, analyze） | 无 |
| 单元 | FSM（fysom 状态机） | fysom |
| 集成 | 配置↔文件系统 | 文件系统 |
| 集成 | 测试入口点（`built_wheel` fixture） | 构建隔离 |
| E2E（标记） | clearml/optuna/langfuse 交互 | 外部服务 |

### 创建新测试

```bash
# 1. 创建测试文件
touch tests/test_<module>.py

# 2. 使用 tmp_path 进行文件系统隔离
# 3. 使用 monkeypatch 模拟外部 SDK 导入
# 4. 集成测试标记 @pytest.mark.integration
```

### 关键 Fixture 模式

```python
# PIN 测试，使用隔离目录
@pytest.fixture(autouse=True)
def setup(self, tmp_path, monkeypatch):
    monkeypatch.setattr("expflow_pde.pin._PIN_DIR", str(tmp_path))

# FSM 测试
@pytest.fixture
def fsm():
    from expflow_pde.fsm import create_experiment_fsm
    return create_experiment_fsm()

# 入口点测试（使用 conftest.py 中的 built_wheel fixture）
def test_entry_point(built_wheel):
    result = subprocess.run(...)
    assert result.returncode == 0
```

## 延迟导入模式

所有 clearml/optuna/langfuse SDK 导入必须**延迟**——在函数体内部，不在模块级：

```python
# ✅ 正确——函数内部的延迟导入
def list_tasks(project=None, tags=None):
    from clearml import Task
    tasks = Task.get_tasks(project_name=project or "PDEBench")
    return [_serialize_task(t) for t in tasks]

# ❌ 错误——模块级导入会在 import 时触发 SDK 依赖
from clearml import Task  # 永远不要这样做
```

`__init__.py` 也使用 `__getattr__` 进行延迟重新导出：

```python
# expflow_pde/__init__.py
def __getattr__(name: str):
    _lazy_map = {
        "list_tasks": ("expflow_pde.clearml", "list_tasks"),
    }
    if name in _lazy_map:
        mod_path, attr = _lazy_map[name]
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(...)
```

## 包结构

```
expflow/
├── expflow_pde/                  # 主 Python 包（33 个模块）
│   ├── __init__.py               # 版本 + 延迟重新导出
│   ├── __init__.pyi              # IDE/类型检查器的类型存根
│   ├── cli.py                    # Typer CLI（8 个命令组）
│   ├── clearml.py                # ClearML SDK 封装（~1K 行）
│   ├── optuna.py                 # Optuna SDK 封装
│   ├── langfuse.py               # Langfuse SDK 封装
│   ├── hpo.py                    # 3 模式 HPO 运行器
│   ├── pipeline.py               # 竞赛流水线
│   ├── dispatcher.py             # 实验注册表
│   ├── fsm.py                    # 7 状态 FSM
│   ├── pin.py                    # PIN 保护
│   ├── metrics.py                # 度量注册表
│   ├── compare.py                # 分数对比
│   ├── equations.py              # PDE 方程注册表
│   ├── analyze.py                # 竞赛智能分析
│   ├── audit.py                  # 合规性验证
│   ├── config.py                 # YAML + .env 加载器
│   ├── worktree.py               # Git worktree 隔离
│   ├── snowflake.py              # ID 生成器
│   ├── status.py                 # 健康检查
│   ├── board.py                  # TensorBoard 启动器
│   ├── mcp.py / mcp_server.py    # MCP 服务器
│   ├── cli_*.py (8 个文件)       # CLI 命令组
│   ├── init.py                   # 交互式配置向导
│   └── skills/                   # Agent 技能（用于 Hermes Agent）
│       ├── expflow-pipeline-hpo.md
│       ├── experiment-lifecycle-governance.md
│       ├── clearml-metrics-logging-pattern.md
│       └── competition-task-intelligence.md
├── tests/                        # 17 个测试文件，329+ 测试
│   ├── conftest.py               # 共享 fixtures
│   ├── test_*.py                 # 按模块测试
│   └── test_entry_point.py       # 包构建验证
├── docs/                         # 文档
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   ├── DEVELOPMENT.md
│   ├── DATA_LAYER.md
│   ├── COMPETITION.md
│   └── cn/                       # 中文翻译
├── pyproject.toml                # 包配置
├── README.md                     # 项目 README
├── AGENTS.md                     # AI Agent 说明
├── PLAN.md                       # 路线图
└── .gitignore
```

## 构建发布

```bash
# 1. 格式化 & Lint
ruff format .
ruff check --fix .

# 2. 类型检查
pyright .

# 3. 测试
python -m pytest tests/ -v

# 4. 验证版本一致性
grep __version__ expflow_pde/__init__.py   # 例如 '0.3.0'
grep ^version pyproject.toml               # 必须匹配

# 5. 构建 + 验证
python -m build
twine check dist/*

# 6. 打标签
git tag v0.3.0
git push --tags

# 7. 发布
twine upload dist/*
```

## 发布前检查清单

- [ ] `ruff format .` — 无更改
- [ ] `ruff check --fix .` — 零错误
- [ ] `pyright .` — `expflow_pde/` 内零错误
- [ ] `python -m pytest tests/ -v` — 329+ 通过
- [ ] `python -m build` — 成功
- [ ] `twine check dist/*` — PASSED
- [ ] 版本匹配：`__init__.py` == `pyproject.toml`
- [ ] 如 API 有变更则更新 `docs/`
- [ ] 如 CLI 结构有变更则更新 `AGENTS.md`
