---
name: clearml-metrics-logging-pattern
description: Standardized ClearML metrics logging patterns for PDEBench experiment scripts — train loss, validation metrics, competition scores, PDE residual, and TensorBoardX integration. Includes patterns for dist/expflow compatibility.
category: mlops
author: Hermes Agent
version: 1.0.0
---

# ClearML Metrics Logging Pattern

## When to Use

- 创建或修改 PDEBench 训练/评估脚本时
- 需要给 `train_task1.py` / `train_task1_phys.py` / `train_task1_ft.py` / `train_task1_unroll.py` 加 clearml 上报
- 需要确保 expflow（单机+分布式）能自动捕获 metrics
- 参考 `token_arena/PDEBench` 中已验证的上报模式

## 现有已上报的 metrics（token_arena/PDEBench）

| 分类 | Metric Name | Group | 来源文件 | 已有? |
|------|------------|-------|---------|:----:|
| Train | Train Data Loss | **Loss/Train Data** | train_task1_commut.py | ✅ |
| Train | Commutativity Loss | **Loss/Commut** | train_task1_commut.py | ✅ |
| Train | Stability Loss | **Loss/Stability** | train_task1_commut.py | ✅ |
| Val | Val MSE | **Loss/Val MSE** | eval_task1.py, train_task1_commut.py | ✅ |
| Val | Val RelMSE | **Loss/Val RelMSE** | eval_task1.py, train_task1_commut.py | ✅ |
| Score | Seg Total | **Score/Seg Total** | eval_task1.py, train_task1_commut.py | ✅ |
| Score | Seg1 | **Score/Seg1** | eval_task1.py, train_task1_commut.py | ✅ |
| Score | Seg2 | **Score/Seg2** | eval_task1.py, train_task1_commut.py | ✅ |
| Score | Seg3 | **Score/Seg3** | eval_task1.py, train_task1_commut.py | ✅ |
| PDE | Mean Residual | **PDE/Mean Residual** | eval_task1.py | ✅ |
| PDE | Seg1/2/3 Residual | **PDE/Seg1 Residual** etc. | eval_task1.py | ✅ |
| Kfold | Mean/Std Seg, CV Seg% | **Kfold/Mean Seg** etc. | kfold_eval.py | ✅ |
| ✅ Filled | Train MSE | **Loss/Train MSE** | 所有 train_task1*.py (commit 8582f59) | ✅ |
| ✅ Filled | Physics Loss | **Loss/Physics** | train_task1_phys.py via _phys_ema | ✅ |
| ✅ Filled | GPU Memory | **System/GPU Alloc MB** | train_task1.py, train_task1_phys.py, train_task1_unroll.py | ✅ |
| ✅ Filled | Learning Rate | **System/LR** | 所有 4 个训练脚本 | ✅ |

## 标准化 metric 命名约定

所有 clearml metric 使用 **Group/Metric** 两层结构，与 `expflow clearml compare-scores` 兼容：

```python
# Loss 分组 — 所有与误差/cost 相关的标量
clearml_logger.report_scalar('Loss', 'Train MSE',     float_val, iteration=epoch)
clearml_logger.report_scalar('Loss', 'Val MSE',       float_val, iteration=epoch)
clearml_logger.report_scalar('Loss', 'Val RelMSE',    float_val, iteration=epoch)
clearml_logger.report_scalar('Loss', 'Physics',       float_val, iteration=epoch)
clearml_logger.report_scalar('Loss', 'Commut',        float_val, iteration=epoch)
clearml_logger.report_scalar('Loss', 'Stability',     float_val, iteration=epoch)

# Score 分组 — 竞赛分段评分（100分制）
clearml_logger.report_scalar('Score', 'Seg Total',    float_val, iteration=epoch)
clearml_logger.report_scalar('Score', 'Seg1',         float_val, iteration=epoch)
clearml_logger.report_scalar('Score', 'Seg2',         float_val, iteration=epoch)
clearml_logger.report_scalar('Score', 'Seg3',         float_val, iteration=epoch)

# PDE 分组 — PDE 残差（按段统计）
clearml_logger.report_scalar('PDE', 'Mean Residual',  float_val, iteration=epoch)
clearml_logger.report_scalar('PDE', 'Seg1 Residual',  float_val, iteration=epoch)
clearml_logger.report_scalar('PDE', 'Seg2 Residual',  float_val, iteration=epoch)
clearml_logger.report_scalar('PDE', 'Seg3 Residual',  float_val, iteration=epoch)

# System 分组 — 系统监控指标
clearml_logger.report_scalar('System', 'GPU Alloc MB',   float_val, iteration=epoch)
clearml_logger.report_scalar('System', 'GPU Reserved MB', float_val, iteration=epoch)
clearml_logger.report_scalar('System', 'LR',              float_val, iteration=epoch)

# Kfold 分组 — K折交叉验证结果
clearml_logger.report_scalar('Kfold', 'Mean Seg',    float_val, iteration=0)
clearml_logger.report_scalar('Kfold', 'Std Seg',     float_val, iteration=0)
clearml_logger.report_scalar('Kfold', 'CV Seg%',     float_val, iteration=0)
```

## 代码模板

### 模板 A: 给训练循环加 clearml metric logging

在已有 `train_task1.py` / `train_task1_phys.py` / `train_task1_ft.py` / `train_task1_unroll.py` 的 epoch 循环中插入：

```python
# === 在训练循环开头（Task.init 之后），获取 logger ===
# task1.py / task1_ft.py 已经在 try 块中初始化了 clearml_task
# 需要额外获取 logger 实例：

clearml_logger = None
if clearml_task is not None:
    try:
        clearml_logger = clearml_task.get_logger()
    except Exception:
        pass


# === 在 epoch 循环末尾（计算出 avg_loss 之后）=== 
# 在 log.info(...) 或 log.metric(...) 旁边添加：

if clearml_logger is not None:
    clearml_logger.report_scalar('Loss', 'Train MSE', avg_loss, iteration=epoch + 1)
    clearml_logger.report_scalar('System', 'LR', scheduler.get_last_lr()[0], iteration=epoch + 1)
    if DEVICE.type == 'cuda':
        clearml_logger.report_scalar('System', 'GPU Alloc MB', round(gpu_alloc, 1), iteration=epoch + 1)
        clearml_logger.report_scalar('System', 'GPU Reserved MB', round(gpu_reserved, 1), iteration=epoch + 1)


# === 在 val 评估段（已计算出 val_mse, val_rel, seg 之后）===

if clearml_logger is not None:
    clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=epoch + 1)
    clearml_logger.report_scalar('Loss', 'Val RelMSE', val_rel, iteration=epoch + 1)
    clearml_logger.report_scalar('Score', 'Seg Total', seg['total_segmented_score'], iteration=epoch + 1)
    clearml_logger.report_scalar('Score', 'Seg1', seg['seg1_score'], iteration=epoch + 1)
    clearml_logger.report_scalar('Score', 'Seg2', seg['seg2_score'], iteration=epoch + 1)
    clearml_logger.report_scalar('Score', 'Seg3', seg['seg3_score'], iteration=epoch + 1)


# === 如果用了物理损失（train_task1_phys.py）===

if clearml_logger is not None and phys_loss is not None:
    clearml_logger.report_scalar('Loss', 'Physics', phys_loss.item(), iteration=epoch + 1)
```

### 模板 B: 给评估脚本加 clearml logging（参考 eval_task1.py 已有模式）

`utils/eval_task1.py` 已有完整的 clearml 上报，如需在自定义 eval 脚本中使用：

```python
def run_eval_and_log(model, val_data, cl_task, tag):
    """评估 + clearml 上报"""
    clearml_logger = cl_task.get_logger() if cl_task is not None else None

    # 自回归推理
    val_mse, val_rel, seg_scores = evaluate_autoregressive(model, val_data)

    # 上报到 clearml
    if clearml_logger is not None:
        iteration = 1  # 评估一般是单次，用 1
        clearml_logger.report_scalar('Score', 'Seg Total', seg_scores['total_segmented_score'], iteration=iteration)
        clearml_logger.report_scalar('Score', 'Seg1', seg_scores['seg1_score'], iteration=iteration)
        clearml_logger.report_scalar('Score', 'Seg2', seg_scores['seg2_score'], iteration=iteration)
        clearml_logger.report_scalar('Score', 'Seg3', seg_scores['seg3_score'], iteration=iteration)
        clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=iteration)
        clearml_logger.report_scalar('Loss', 'Val RelMSE', val_rel, iteration=iteration)

    return val_mse, val_rel, seg_scores
```

### 模板 C: TensorBoardX + DoubleLogger 模式

当训练脚本用 TensorBoardX（`from torch.utils.tensorboard import SummaryWriter`），clearml 的 `capture_tensorboard=True` 会自动捕获。但更可靠的方式是**双写**：

```python
# 初始化
tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard', tag))

# 每个 epoch 双写
if tb_writer is not None:
    tb_writer.add_scalar('Loss/Train', avg_loss, epoch + 1)
    tb_writer.add_scalar('Loss/Val_MSE', val_mse, epoch + 1)
    tb_writer.add_scalar('Score/Seg_Total', seg['total_segmented_score'], epoch + 1)

if clearml_logger is not None:
    clearml_logger.report_scalar('Loss', 'Train MSE', avg_loss, iteration=epoch + 1)
    clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=epoch + 1)
    clearml_logger.report_scalar('Score', 'Seg Total', seg['total_segmented_score'], iteration=epoch + 1)

# 更简洁的封装 — DoubleLogger
class DoubleLogger:
    """同时写 TensorBoard 和 ClearML"""
    def __init__(self, tb_writer=None, cl_logger=None):
        self.tb = tb_writer
        self.cl = cl_logger

    def scalar(self, group, name, value, iteration):
        if self.tb is not None:
            self.tb.add_scalar(f'{group}/{name}', value, iteration)
        if self.cl is not None:
            self.cl.report_scalar(group, name, value, iteration=iteration)

# 使用
log = DoubleLogger(tb_writer, clearml_logger)
log.scalar('Loss', 'Train MSE', avg_loss, epoch+1)
log.scalar('Score', 'Seg Total', seg_total, epoch+1)
```

### 模板 D: 带 physics loss 的完整训练循环

`train_task1_phys.py` 的典型上报模式（加上缺失的 clearml logging）：

```python
# 在 Task.init() 之后获取 logger
clearml_logger = clearml_task.get_logger() if clearml_task is not None else None

for epoch in range(args.epochs):
    epoch_loss = 0.0
    for batch in train_loader:
        # ... 正常训练 ...
        data_loss = loss_fn(pred, yb)
        if args.physics_weight > 0:
            phys = pde_burgers_residual(u_stack, nu=0.001)
            loss = data_loss + args.physics_weight * phys
        else:
            loss = data_loss

        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / n_batches

    # === 上报 ===
    if clearml_logger is not None:
        clearml_logger.report_scalar('Loss', 'Train MSE', avg_loss, iteration=epoch + 1)
        if args.physics_weight > 0 and phys is not None:
            clearml_logger.report_scalar('Loss', 'Physics', phys.item(), iteration=epoch + 1)

    # Val 评估（每 eval_every epoch）
    if (epoch + 1) % args.eval_every == 0:
        val_mse, val_rel, seg = evaluate_autoregressive(model, comp_val_data, ...)

        if clearml_logger is not None:
            clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=epoch + 1)
            clearml_logger.report_scalar('Loss', 'Val RelMSE', val_rel, iteration=epoch + 1)
            clearml_logger.report_scalar('Score', 'Seg Total', seg['total_segmented_score'], iteration=epoch + 1)
            clearml_logger.report_scalar('Score', 'Seg1', seg['seg1_score'], iteration=epoch + 1)
            clearml_logger.report_scalar('Score', 'Seg2', seg['seg2_score'], iteration=epoch + 1)
            clearml_logger.report_scalar('Score', 'Seg3', seg['seg3_score'], iteration=epoch + 1)
```

## expflow 兼容性

### 单机模式（`expflow run submit`）

`expflow` 的 `init_tracking()` 默认开启 `capture_tensorboard=True`，所以：
- 如果训练脚本使用 `SummaryWriter` → clearml 自动捕获 TensorBoard scalars
- 如果训练脚本直接调 `report_scalar` → clearml 原生上报 ✅
- 两种方式都可以用 `expflow clearml compare-scores` 查询

### 分布式模式（`expflow optuna run --distributed --queue`）

每个 trial 独立在 GPU 节点上执行，`Task.init()` + `report_scalar` 在每个 trial 内独立上报：
- trial 脚本必须包含 Task.init()（expflow 不自动注入）
- trial 的 clearml task 会被自动关联到 parent optuna study task
- 所有 metrics 在 clearml Web UI 的 experiments table 中可按 trial 过滤

## 验证方法

新增 clearml logging 后：

```bash
# 1. 本地验证（单epoch）— 确认 clearml task 创建成功
cd ~/Gitlab/token_arena/PDEBench
source ~/miniconda3/bin/activate physicsnemo-cpu
python utils/train_task1.py --epochs 2 --n_train 30 --tag clearml_test

# 2. 检查 clearml Web UI 能否看到新 metrics
#    打开 https://clearml-server:8080 → experiments → 搜索 clearml_test
#    确认 Scalars 下有 Loss/Train MSE、Score/Seg Total 等曲线

# 3. 用 expflow 验证读取
expflow clearml compare-scores --project PDEBench --sort-by "Seg Total" --limit 5
```

## 已知陷阱

1. **`Task.get_logger()` 必须在 `Task.init()` 之后调用**，否则返回 None
2. **`report_scalar` 的 iteration 必须递增** — clearml 会用 iteration 作为 x-axis。从 1 开始，不要跳值
3. **`capture_tensorboard=True` 时**，TensorBoardX 和 clearml 双写不会冲突，但 clearml 看到的指标会在 Group 名上加 TensorBoard 的路径前缀
4. **分布式下 metrics 存储在 trial 级别** — parent optuna study task 只存 optuna 优化的 `user_objective`，不自动 aggregate trial metrics。需要用 `expflow clearml compare-scores` 手动查询
5. **`report_scalar` 的 group + name 组合必须一致** — 否则 compare-scores 的 `--sort-by` 找不到对应 metric。例如总是 `Score/Seg Total`，不要有时 `Score/Seg Total` 有时 `Score/Seg_Total`
