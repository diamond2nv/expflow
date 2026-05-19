# PLAN_v3 — Neural Operator Loss Suite + Distributed HPO

> 将 HyperNOs 的 PDE 专用损失函数和 PDEBench 的评估指标移植到 expflow，形成一个统一的`expflow_pde/losses.py` + 增强的 `expflow_pde/metrics.py`，并用 optuna + clearml agent 实现跨 GPU 的分布式 HPO 与全指标监控。

**版本号**: v0.4.0 (当前 0.3.0)
**估时**: 3-4h
**依赖**: clearml-server 已部署, clearml-agent已在远程GPU机器运行

---

## 动机

PDEBench 现有训练全部使用 `nn.MSELoss()`，缺乏 PDE 领域专用的相对误差和 Sobolev 范数损失。HyperNOs (`~/Gitlab/Agentic4Sci/HyperNOs/`) 实现了完整的损失函数家族，但其架构老旧(依赖 Ray Tune)，仅需抽取其损失函数核心。

## 架构图

```
expflow_pde/
├── losses.py          ← [NEW] 15+ PDE专用损失函数（移植自HyperNOs）
├── metrics.py         ← [增强] 合并PDEBench metrics + 竞赛评分
├── equations.py       ← [增强] 新增损失推荐字段
├── hpo.py             ← [增强] 支持分布式HPO + 自定义损失作为Trial指标
├── clearml.py         ← [增强] 支持多GPU worker 监控增强
├── compare.py         ← [增强] 支持新指标排序
└── tests/
    ├── test_losses.py    ← [NEW]
    └── test_metrics.py   ← [增强]
```

## Phase 1 — Losses 模块移植 (核心)

**目标**: 从 HyperNOs 抽取关键损失函数，清理依赖（去除 beartype/jaxtyped），适配 expflow 风格（PEP8, type hints, top-level `__all__`）。

### 1A. 移植哪几个类（按优先级）

| 优先级 | 类 | 来源行数 | 说明 |
|:------:|----|:--------:|------|
| 🔴 P0 | `LprelLoss(p, size_mean)` | ~30 | **相对 Lᵖ 核心** — 竞赛用 Rel-MSE 的通用版 |
| 🔴 P0 | `H1relLoss_1D(beta, size_mean, alpha)` | ~80 | **H¹ Sobolev 1D** — FFT频域加权，抑制高频震荡 |
| 🔴 P0 | `H1relLoss(beta, size_mean, alpha)` | ~80 | **H¹ Sobolev 2D** — 同上，2D版本 |
| 🟡 P1 | `MSELoss_rel` | ~15 | 相对 MSE — 简单有效的归一化损失 |
| 🟡 P1 | `SmoothL1Loss_rel` | ~20 | 相对 Smooth L1 — 对大误差鲁棒 |
| 🟡 P1 | `lpLoss(p, size_mean)` | ~15 | 绝对 Lᵖ — 参考基线 |
| 🟢 P2 | `ChebyshevLprelLoss` | ~60 | Chebyshev 网格加权 |
| 🟢 P2 | `H1relLoss_cheb_mp` | ~100 | Chebyshev 多patch H¹ |

### 1B. 关键实现细节

```python
# expflow_pde/losses.py

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal

# ── Relative Lp Loss ──

class LprelLoss(nn.Module):
    """Relative Lp loss for N-D functions.

    loss = ||pred - target||_p / ||target||_p
    Averaged over output channels.

    Args:
        p: Norm order (1 or 2 for L1/L2).
        size_mean: True=mean, False=sum, None=per-sample tensor.
    """
    def __init__(self, p: int = 2, size_mean: bool = True):
        super().__init__()
        self.p = p
        self.size_mean = size_mean

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (n_samples, *spatial, out_dim)
        # Flatten spatial dims, keep out_dim
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1, x.shape[-1])  # (N, D, C)
        y_flat = y.reshape(batch, -1, y.shape[-1])
        diff_norm = torch.norm(x_flat - y_flat, p=self.p, dim=1)   # (N, C)
        y_norm = torch.norm(y_flat, p=self.p, dim=1)               # (N, C)
        rel = diff_norm / (y_norm + 1e-7)
        # Average over output channels
        loss = rel.mean(dim=-1)  # (N,)
        if self.size_mean:
            return loss.mean()
        elif self.size_mean is False:
            return loss.sum()
        return loss


class H1relLoss(nn.Module):
    """Relative H1 (Sobolev) loss for 2D — FFT frequency-weighted.

    loss = ||alpha*(x-y) + beta*grad(x-y)||_2 / ||alpha*y + beta*grad(y)||_2
    Uses FFT: weight = alpha + beta * (k_x^2 + k_y^2)

    Args:
        beta: Derivative weight (default: 1.0).
        size_mean: Reduction mode.
        alpha: Identity weight (default: 1.0).
    """
    def __init__(self, beta: float = 1.0, size_mean: bool = True, alpha: float = 1.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.size_mean = size_mean

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (N, H, W, C) or (N, C, H, W)
        # 1. FFT along spatial dims [1, 2] or [2, 3]
        # 2. Create frequency grid weights
        # 3. Weighted L2 norm
        ...


class H1relLoss_1D(nn.Module):
    """Relative H1 loss for 1D — FFT frequency-weighted.
    Same as H1relLoss but FFT dim=1 only.
    """
    ...
```

### 1C. loss_selector 函数

```python
def loss_selector(name: str, **kwargs) -> nn.Module:
    """Select and configure a loss function by name.

    Names: 'l1_rel', 'l2_rel', 'h1_1d', 'h1_2d', 'mse_rel',
           'smoothl1_rel', 'l2_abs', 'mse_abs'
    """
    registry = {
        'l1_rel': lambda: LprelLoss(p=1, **kwargs),
        'l2_rel': lambda: LprelLoss(p=2, **kwargs),
        'h1_1d': lambda: H1relLoss_1D(**kwargs),
        'h1_2d': lambda: H1relLoss(**kwargs),
        'mse_rel': lambda: MSELoss_rel(**kwargs),
        'mse_abs': lambda: nn.MSELoss(),
    }
    ...
```

## Phase 2 — Metrics 模块增强

**目标**: 将 PDEBench 的 `metric_func()` 六个指标和竞赛评分整合进 expflow 的 metrics 注册表。

### 2A. 新增指标

```python
# expflow_pde/metrics.py — 新增
STANDARD_METRICS.update({
    # HyperNOs-style training loss
    "train_lprel":     {"type": "scalar", "group": "Loss", "higher_is_better": False},
    "train_h1rel":     {"type": "scalar", "group": "Loss", "higher_is_better": False},
    # PDEBench 6-metric suite
    "val_rmse":        {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_nrmse":       {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_max_err":     {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_bd_err":      {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_csv_err":     {"type": "scalar", "group": "Error", "higher_is_better": False},
    "val_fourier_low": {"type": "scalar", "group": "Fourier", "higher_is_better": False},
    "val_fourier_mid": {"type": "scalar", "group": "Fourier", "higher_is_better": False},
    "val_fourier_high":{"type": "scalar", "group": "Fourier", "higher_is_better": False},
})
```

### 2B. 新增 `compute_all_metrics()` 函数

包装 PDEBench 的 `metric_func()` 六合一接口：

```python
def compute_pdebench_metrics(pred: torch.Tensor, target: torch.Tensor,
                              initial_step: int = 10) -> dict[str, float]:
    """Compute all PDEBench metrics (RMSE, nRMSE, CSV, Max, Boundary, Fourier).

    Wraps PDEBench's metric_func() for direct use in expflow training scripts.
    """
```

## Phase 3 — 分布式 HPO 增强

**目标**: `expflow optuna run` 支持将每个 Trial 作为独立 clearml Task 分发到 GPU queue。

### 3A. 搜索空间与损失联动

```
expflow optuna run train.py \
    --study burgers_hpo --trials 50 --parallel 4 \
    --queue gpu_queue \
    --loss l2_rel  # ← NEW: 使用 expflow 的损失函数
    --search-space '{
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "loss.beta": {"type": "float", "low": 0.1, "high": 5.0}
    }'
```

每个 Trial 脚本自动注入：
```python
# clearml-agent 执行时的脚本自动补丁
from expflow_pde.losses import loss_selector
loss_fn = loss_selector("l2_rel")
# 训练中使用 loss_fn(pred, target)
```

### 3B. 分布式 Trial 架构

```python
def run_hpo_distributed(
    study_name: str, script: str, n_trials: int, parallel: int,
    queue: str, search_space: dict, loss: str | None = None
) -> dict:
    """
    1. Create/get Optuna study (RDB storage, SQLite for now)
    2. For each trial: ask Optuna for params
    3. Create clearml Task.clone from template script
    4. Inject params + loss function choice as clearml task params
    5. Enqueue to queue
    6. Wait/collect results via clearml Task API
    """
```

## Phase 4 — CLI 命令

| 命令 | 功能 |
|------|------|
| `expflow analyze losses` | 列出所有可用损失函数及其参数 |
| `expflow optuna run --loss l2_rel` | HPO 时指定损失函数 |
| `expflow clearml compare-scores --sort-by val_h1rel` | 新指标排序 |

## Phase 5 — 测试

| 测试文件 | 测试数 | 覆盖内容 |
|----------|:------:|----------|
| `tests/test_losses.py` | ~20 | 每类loss: perfect pred=0, noisy pred≈expected, batch/reduction, grad flow |
| `tests/test_metrics.py` | ~15 | compute_pdebench_metrics vs PDEBench reference, metric registry |

## 执行路线

```
Phase 1 (2h) — loss_fun.py 移植: LprelLoss → H1relLoss → loss_selector
Phase 2 (0.5h) — metrics 增强: 6 new metric + compute_pdebench_metrics
Phase 3 (1h) — HPO 增强: --loss flag + trial param injection
Phase 4 (0.5h) — CLI: analyze losses, compare-scores 增强
Phase 5 (0.5h) — tests
```

## 非目标 (明确不做)

- ❌ 不移植 HyperNOs 的 Physics Residuals（Poisson/Darcy/Helmholtz）— 对 Burgers FNO 无用
- ❌ 不移植 HyperNOs 的 Chebyshev 变体 — 除非验证需要
- ❌ 不移植 Ray Tune — expflow 用 optuna
- ❌ 不改写 PDEBench 训练脚本 — 只提供 expflow 工具，让用户脚本自行选择
