# expflow-pde 虚拟实验游戏

> **无需 GPU、ClearML、Optuna 或 torch，即可测试完整的实验生命周期——诊断、建议、提交、失败、修复、迭代。**

## 概述

虚拟实验游戏是 expflow 实验循环的**零依赖、完全自包含模拟系统**。它将真实 ML 训练（需要 GPU、ClearML 服务器和 Optuna 数据库）替换为一个合成模型——该模型从超参数变化中产生合理的 seg 分数、注入真实的故障模式，并像真实实验一样将每一步记录到 `DispatchDB` 中。

这服务于三个目的：

| 目的 | 证明什么 |
|------|---------|
| **系统集成测试** | diagnose → suggest → submit → fail → repair → iterate：完整流水线能否端到端工作？ |
| **修复验证** | L0 能否捕捉 `git_not_found`？L1 能否提取 CUDA OOM 的追踪信息？L2 能否生成反思上下文？ |
| **入门 / 演示** | 无需任何基础设施即可向新用户展示 expflow 的工作原理——只需 `pip install` 即可运行。 |

## 快速开始

```bash
# 启动新游戏（task1，Burgers 场景）
expflow dummy start --task task1 --seed 42

# 运行一步迭代
expflow dummy step

# 注入故障运行（测试修复系统）
expflow dummy step --inject git_not_found

# 模拟 Hermes 的"建议修复"后运行一步
expflow dummy step --params '{"n_modes": 20, "num_sub_steps": 5}'

# 查看游戏状态
expflow dummy status

# 运行全自动循环（诊断 → 建议 → 执行 → 修复）
expflow dummy auto --max-steps 10 --repair
```

## 工作原理

### 模拟模型

游戏维护每步的 `{seg1, seg2, seg3}` 分数内部状态。

- **基线**：task1（Burgers FNO）从 `{55, 30, 20}` 开始。
- **效果**：每个超参数变化对 seg 分数有加性影响：

| 参数变化 | seg1 | seg2 | seg3 | 建模含义 |
|---------|:----:|:----:|:----:|---------|
| `n_modes +4` | 0 | +3 | +8 | 更多傅里叶模式捕捉高频动力学 |
| `num_sub_steps=5` | +2 | +3 | +5 | 更细的时间分辨率修复 dt 不匹配 |
| `lr ×2` | +5 | -2 | -1 | 高学习率有助于短程但损害长程稳定性 |
| `stability_lambda=0.001` | -1 | +6 | 0 | 稳定性惩罚抑制中程漂移 |
| `width +16` | +2 | +2 | +3 | 更宽的网络增加容量 |
| `weight_decay=1e-4` | 0 | +1 | +3 | 正则化改善长程泛化 |
| `epochs +40` | +1 | +1 | +2 | 更多训练有助于收敛 |

- **噪声**：每步添加高斯噪声（±2 标准差）以模拟实验方差。
- **天花板**：分数被按任务上限截断（task1 为 `{70, 60, 45}`，task3 为 `{25, 18, 12}`），建模真实 ML 改进的收益递减特性。

### 任务配置

| 任务 | 基线 | 天花板 | 特征 |
|------|:----:|:------:|------|
| `task1`（Burgers） | 55/30/20 | 70/60/45 | seg 段之间差距适中 |
| `task2` | 40/20/10 | 55/45/30 | 更难，差距更大 |
| `task3`（KS） | 15/8/4 | 25/18/12 | 混沌动力学，基线低得多 |

### 故障注入

游戏可以注入逼真的故障来测试修复流水线：

| 故障模式 | 退出码 | 期望层级 | 日志内容 |
|---------|:------:|:--------:|---------|
| `git_not_found` | 128 | **L0** — 规则匹配 | `"ERROR: Repository not found"` |
| `module_not_found` | 1 | **L0** — 规则匹配 | `"ModuleNotFoundError: No module named 'torch'"` |
| `cuda_oom` | 1 | **L1** — 追踪 | `"torch.cuda.OutOfMemoryError"` 在 `train.py:42` |
| `data_not_found` | 1 | **L1** — 追踪 | `"FileNotFoundError: dataset.hdf5"` 在 `eval.py:15` |
| `unknown_error` | 1 | **L2** — 深度反思 | 无追踪信息，不透明错误码 |

默认情况下，约 30% 的步骤会随机失败（游戏从 5 种模式中选一个）。使用 `--inject <名称>` 强制特定故障，或 `--inject none` 强制成功。

## 所有 CLI 命令

| 命令 | 说明 |
|------|------|
| `expflow dummy start` | 启动新游戏会话。选项：`--task`, `--seed` |
| `expflow dummy step` | 运行一次迭代。选项：`--params`, `--strategy`, `--inject` |
| `expflow dummy status` | 显示当前游戏状态、到达天花板的剩余步数 |
| `expflow dummy reset` | 重置到基线。选项：`--seed` |
| `expflow dummy auto` | 运行全自动循环。选项：`--max-steps`, `--repair/--no-repair` |
| `expflow dummy list-failures` | 列出所有可注入的故障模式 |

## 查询游戏历史

因为每一步都会在 `DispatchDB` 中创建真实记录，你可以检查实验树：

```bash
# 显示完整实验树
expflow dispatch tree <root_experiment_id>

# 获取数据库统计信息
expflow dispatch stats

# 查看修复事件的审计日志
expflow dispatch audit-log --event-type repair

# 列出最近的实验
expflow dispatch list --limit 20
```

## 自动化测试

虚拟游戏附带 **20 个 pytest 测试**，涵盖：

- 基本生命周期（`start`, `step`, `status`, `reset`）
- 全部 5 种故障模式正确注入
- L0 修复匹配 `git_not_found` 和 `module_not_found`
- L1 提取 `cuda_oom` 和 `data_not_found` 的追踪信息
- 天花板收敛：分数在预期最大值处趋于平稳
- `diagnose_experiment()` + `suggest_next_params()` 集成

运行测试：

```bash
python -m pytest tests/test_dummy_game.py -v
```

全部 20 个测试零外部依赖通过（无 torch、无 clearml、无 GPU）。

## 与真实实验对比

| 方面 | 真实实验 | 虚拟游戏 |
|------|:-------:|:--------:|
| 需要 GPU | ✅ 是 | ❌ 否 |
| ClearML 服务器 | ✅ 需要 | ❌ 不需要 |
| Optuna 数据库 | ✅ 需要 | ❌ 不需要 |
| torch / CUDA | ✅ 需要 | ❌ 不需要 |
| 数据文件（HDF5） | ✅ 需要 | ❌ 不需要 |
| 训练时间 | 数小时 | 毫秒级 |
| seg 分数 | 真实评估 | 合成（效果 + 噪声 + 天花板） |
| DispatchDB 记录 | ✅ 相同 | ✅ 相同 |
| 修复流水线 | ✅ 相同 | ✅ 相同 |
| diagnose → suggest | ✅ 相同 | ✅ 相同 |

虚拟游戏产生**与真实实验相同的 DispatchDB 模式、相同的审计追踪、相同的分支树和相同的修复接口**。唯一的区别在于 seg 分数的来源。

## 相关文档

- [USAGE.zh-CN.md — `expflow dummy` CLI 章节](USAGE.zh-CN.md#虚拟实验游戏)
- [ARCHITECTURE.zh-CN.md — 系统分层](ARCHITECTURE.zh-CN.md)
- [DEVELOPMENT.zh-CN.md — 测试指南](DEVELOPMENT.zh-CN.md)
- [DispatchDB 设计 — `expflow_pde/dispatch_db.py`](../expflow_pde/dispatch_db.py)
