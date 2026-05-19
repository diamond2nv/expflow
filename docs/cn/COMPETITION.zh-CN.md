# 竞赛集成指南

## 概述

expflow-pde 提供了专门针对 **AI4S 神经算子 PDE 竞赛**
（[competition.ai4s.com.cn](https://competition.ai4s.com.cn/race/7) Race 7）设计的工具。
它编排完整的实验生命周期：训练、HPO、评估、提交打包、竞赛规则审计和战略规划。

## 竞赛任务

| 任务 | PDE | 满分 | 状态 |
|:----:|-----|:----:|:----:|
| 1 | Burgers (nu=0.001) | 150 | `进行中` |
| 2 | Burgers (多 nu) | 150 | `未开始` |
| 3 | Kuramoto-Sivashinsky | 350 | `未开始` |

## 竞赛流水线

推荐的工作流使用 expflow 的三种流水线模式：

### 阶段 1：探索（寻找最佳超参）

```bash
expflow pipeline submit-full train_task1.py \
    --queue default \
    --trials 50 --parallel 4 \
    --eval-script eval_task1.py \
    --metric seg_total --direction maximize \
    --pruner hyperband
```

### 阶段 2：冲刺（已知参数，快速迭代）

```bash
expflow pipeline submit train_task1.py \
    --queue default \
    --train-param lr=0.001 --train-param epochs=80 \
    --train-param sub_step=5 \
    --eval-script eval_task1.py
```

### 阶段 3：竞赛审计

```bash
# 按竞赛规则验证指标
expflow audit validate exp-001 \
    --competition-rules \
    --task-id <clearml_task_id>

# 使用门控比较所有运行
expflow clearml compare-scores \
    --project PDEBench --tags task1 \
    --sort-by seg_total --gate pde_mean:lt:18.09
```

## 标准化度量注册表

所有竞赛指标遵循 `分组/指标名` 命名约定：

| 分组 | 指标 | 越高越好 | 竞赛阈值 |
|------|------|:-------:|:--------:|
| `Score` | `Seg Total` | ✅ | — |
| `Score` | `Seg1` | ✅ | — |
| `Score` | `Seg2` | ✅ | — |
| `Score` | `Seg3` | ✅ | — |
| `Loss` | `Val MSE` | ❌ | — |
| `Loss` | `Val RelMSE` | ❌ | — |
| `PDE` | `Mean Residual` | ❌ | 18.09 |
| `Time` | `Train Time Min` | ❌ | 60.0 |
| `Time` | `Inference Time Min` | ❌ | 2.0 |

将这些添加到训练脚本中（完整模式见 [USAGE.zh-CN.md](USAGE.zh-CN.md)）：

```python
if clearml_logger is not None:
    clearml_logger.report_scalar('Score', 'Seg Total', seg_total, iteration=epoch)
    clearml_logger.report_scalar('Score', 'Seg1', seg1, iteration=epoch)
    clearml_logger.report_scalar('Score', 'Seg2', seg2, iteration=epoch)
    clearml_logger.report_scalar('Score', 'Seg3', seg3, iteration=epoch)
    clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=epoch)
    clearml_logger.report_scalar('Loss', 'Val RelMSE', val_rel, iteration=epoch)
    clearml_logger.report_scalar('PDE', 'Mean Residual', pde_mean, iteration=epoch)
```

## 竞赛规则审计

`expflow audit validate --competition-rules` 命令检查：

| 检查项 | 规则 | 评分说明 |
|-------|------|---------|
| `seg_total` | 必须上报（无门控） | 竞赛主要评分指标 |
| `pde_mean` | 必须 < 18.09 | PDE 残差合规性 |
| `train_time_min` | 必须 < 60 | 训练时间在 60 分钟限制内 |
| `sub_step` | 必须存在且 > 0 | dt 不匹配修复要求 |

### Python API

```python
from expflow_pde.audit import validate_competition_rules

result = validate_competition_rules(
    task_metrics={
        "seg_total": 57.09,
        "pde_mean": 15.0,
        "train_time_min": 45.5,
    },
    task_params={"Args/--sub_step": "5"},
)

print(f"全通过: {result['all_pass']}")
for check in result["checks"]:
    print(f"  {check['name']}: {'✓' if check['passed'] else '✗'} ({check['detail']})")
```

## 任务智能分析

`expflow analyze` 命令组提供竞赛策略建议：

```bash
# 整体状态
expflow analyze status

# Task 1 深度分析
expflow analyze task task1

# 战略推荐
expflow analyze advise

# 方程参考
expflow analyze equations --task competition
```

### 分数预估

```python
from expflow_pde.analyze import estimate_score_potential, get_strategic_recommendation

# 按任务的分数预估
estimates = estimate_score_potential("task1")
# 返回：
# {
#   "optimistic": 148,
#   "expected": 145,
#   "conservative": 140,
#   "confidence": "high",
# }

# 完整策略
rec = get_strategic_recommendation()
# 返回：
# {
#   "primary_focus": "task1",
#   "remaining_headroom": {...},
#   "suggested_schedule": {
#     "day_1_2": "Task 1: HPO on lambda_stab + longer epochs",
#     "day_3_4": "...",
#   },
# }
```

## 已验证策略（Task 1）

来自 `token_arena/PDEBench` 的实际实验记录：

| 策略 | Seg 提升 | 详情 |
|------|:--------:|------|
| `sub_step=5` | +11.37 | dt 不匹配修复（dt=0.01 vs 0.05） |
| Stability FT | +23.45 | 步间方差惩罚（3 行代码） |
| P2 架构 (16/32, 50K params) | — | 最优模型大小 |
| FT lr≈1e-7 | — | 保留预训练特征 |

## 竞赛约束汇总

| 约束 | Task 1 | Task 2 | Task 3 |
|------|:------:|:------:|:------:|
| 训练时间 | < 60 min | < 60 min | N/A |
| 推理时间 | < 2 min | < 2 min | < 2 min |
| 总时间 | < 12 h | < 12 h | < 12 h |
| 观测步数 | 10 | 10 | 20 |
| 预测步数 | 190 | 190 | 380 |
| 测试时已知 nu? | ✅ | ❌ | N/A |
| 测试时已知 λ₂? | N/A | N/A | ❌ |
| 允许预训练? | ❌ | ❌ | ❌ |

## 相关文档

- [ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md) — 系统架构
- [USAGE.zh-CN.md](USAGE.zh-CN.md) — CLI 参考
- [DATA_LAYER.zh-CN.md](DATA_LAYER.zh-CN.md) — 数据层设计
- 包内技能 `competition-task-intelligence` 用于 Agent 指导
