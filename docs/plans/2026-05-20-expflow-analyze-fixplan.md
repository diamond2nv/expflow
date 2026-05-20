# Expflow Analyze 体系修复计划

> 基于深度学习反思的七个系统性问题，逐一修复

**核心原则：**
1. 所有新功能通过 TDD 实现（先写失败测试 → 再实现）
2. 每次改动后跑完整测试套件
3. 每次改动后 commit
4. 不破坏 expflow-pde 作为 PyPI 包的可安装性

---

## 修复清单（按优先级排序）

| 优先级 | 问题 | 改动量 | 影响范围 |
|--------|------|--------|---------|
| 🔴 P0 | 问题5：clearml 连接失败静默退出 | 小 | `analyze.py` `_load_experiment_metrics` |
| 🔴 P0 | 问题1：阈值刚性 + 诊断规则改进 | 中 | `analyze.py` `diagnose_experiment()` + `suggest_next_params()` |
| 🟡 P1 | 问题3：置信度数据驱动 | 中 | `analyze.py` `estimate_score_potential()` + clearml `get_task_scalars()` |
| 🟡 P1 | 问题2：时间感知战略建议 | 中 | `analyze.py` `get_strategic_recommendation()` |
| 🟡 P1 | 问题4：元数据自动同步 | 大 | `analyze.py` `sync_task_meta()` + clearml 集成 |
| 🟢 P2 | 问题6：跨 PDE 策略迁移约束标记 | 小 | `task_meta.yaml` + `analyze.py` |
| 🟢 P2 | 问题7：假说验证记录（负结果日志） | 大 | 新文件 `hypothesis_registry.py` + YAML |

---

## Task 1: 修复 clearml 连接失败静默退出

**Objective:** `_load_experiment_metrics()` 的 clearml 异常时，返回带 error_info 的 dict 而非 None，诊断输出清晰的错误信息。

**Files:**
- Modify: `expflow_pde/analyze.py:348-376` — `_load_experiment_metrics()`
- Modify: `expflow_pde/analyze.py:378-474` — `diagnose_experiment()` 返回信息性错误
- Test: `tests/test_analyze.py` — 新增测试

**Step 1: 修改 `_load_experiment_metrics`**

在 clearml 的 `try` 块中捕获异常并返回带 `_error` 标记的 dict：

```python
def _load_experiment_metrics(
    task_id: str | None = None,
    json_path: str | None = None,
) -> dict | None:
    import json
    import os

    if json_path:
        if not os.path.exists(json_path):
            return None
        with open(json_path) as f:
            return json.load(f)

    if task_id:
        try:
            from expflow_pde.clearml import get_task_scalars
            result = get_task_scalars(task_id)
            if result is None:
                return {"_error": f"clearml task {task_id} returned no scalars (may not exist or have no data)"}
            return result
        except ImportError:
            return {"_error": "clearml SDK not installed — cannot fetch task scalars"}
        except ConnectionError as e:
            return {"_error": f"clearml server connection failed: {e}"}
        except Exception as e:
            return {"_error": f"clearml error for task {task_id}: {e}"}

    return None
```

**Step 2: 修改 `diagnose_experiment()`**

```python
def diagnose_experiment(...):
    metrics = _load_experiment_metrics(task_id, json_path)
    if metrics is None:
        return None
    
    # Check for clearml error
    if "_error" in metrics:
        return {
            "seg1": 0, "seg2": 0, "seg3": 0, "total": 0,
            "total_mse": 0.0,
            "diagnosis": [f"CLEARML ERROR: {metrics['_error']}"],
            "degradation_pattern": "error",
            "_connection_error": metrics["_error"],
        }
    ...
```

**Step 3: 写测试**

```python
def test_diagnose_clearml_connection_error(self):
    """诊断返回 clearml 连接错误信息，而非静默 None。"""
    # mock _load_experiment_metrics to return error
    ...
```

---

## Task 2: 修复阈值刚性 + 诊断规则改进

**Objective:** 诊断规则从绝对值阈值改为相对比较 + 多模式覆盖。

**关键改进：**
1. 不再以 Seg1 < 70 作为唯一"short_term"判据——如果 Seg1 < 70 但 Seg2 接近 Seg1，说明是弹性天花板而非训练不足
2. 同时识别 mid_term + long_term 的复合模式（不再只返回最后一个 match）
3. 引入 Seg 衰减率指标：`decay_rate = max(Seg1-Seg2, Seg2-Seg3) / max(1e-8, max(Seg1,Seg2))`

**Files:**
- Modify: `expflow_pde/analyze.py:378-474` — `diagnose_experiment()`
- Modify: `expflow_pde/analyze.py:476-558` — `suggest_next_params()`
- Test: `tests/test_analyze.py` — 新增测试

**Step 1: 修改 diagnose_experiment 诊断规则**

新规则引擎：

```python
    # ── Compute composite signals ──
    decay_rate = 0.0
    if max(seg1, seg2, segmented) > 0:
        d1 = (seg1 - seg2) / max(1e-8, max(seg1, seg2))
        d2 = (seg2 - seg3) / max(1e-8, max(seg2, seg3))
        decay_rate = max(d1, d2)
    
    # Detect ceiling: Seg1 < 70 but Seg2 close to Seg1 and Seg3 not collapsing
    is_ceiling = (
        isinstance(seg1, (int, float)) and seg1 < 70
        and isinstance(seg2, (int, float)) and seg2 > 0
        and isinstance(seg3, (int, float)) and seg3 > 0
        and (seg1 - seg2) < 10
        and seg3 > seg2 * 0.7
    )
    
    if is_ceiling:
        diagnosis.append("Score ceiling detected — model capacity may be limiting (Seg uniformly low)")
        degradation_pattern = "ceiling"
    
    elif isinstance(seg1, (int, float)) and seg1 < 70:
        diagnosis.append("Short-term prediction is weak (Seg1 low)")
        degradation_pattern = "short_term"
    
    # mid_term and long_term are now ADDITIVE, not mutually exclusive
    mid_term_flag = False
    long_term_flag = False
    
    if (isinstance(seg1, (int, float)) and seg1 > 0
        and isinstance(seg2, (int, float)) and seg2 > 0
        and (seg1 - seg2) > 25):
        diagnosis.append("Medium-term stability degraded (Seg2 drops >25 from Seg1)")
        mid_term_flag = True
    
    if isinstance(seg3, (int, float)) and (
        seg3 < 35 or (isinstance(seg2, (int, float)) and seg2 > 0 and seg3 < seg2 * 0.6)):
        diagnosis.append("Long-term autoregressive collapse (Seg3 collapse)")
        long_term_flag = True
    
    if mid_term_flag and long_term_flag:
        degradation_pattern = "compound_mid_long"
    elif long_term_flag:
        degradation_pattern = "long_term"
    elif mid_term_flag:
        degradation_pattern = "mid_term"
```

**Step 2: 修改 suggest_next_params 适配新模式**

```python
    if pattern == "long_term" or pattern == "compound_mid_long":
        # 原有 long_term 策略
        ...
    elif pattern == "ceiling":
        suggestions["tag"] = "auto_ceiling_fix"
        # 宽度或模式数
        suggestions["n_modes"] = ...  
        rationale.append(
            f"Ceiling at Seg1={seg1:.1f}: try wider architecture or more data "
            f"to break through score plateau"
        )
```

---

## Task 3: 置信度数据驱动化

**Objective:** `estimate_score_potential()` 接受可选的实验历史，计算实际收敛趋势，输出基于数据的置信度。

**Files:**
- Modify: `expflow_pde/analyze.py:186-219` — `estimate_score_potential()`

**设计：**
- 新增 `estimate_score_potential_from_history(task_id, seg_history: list[float])` 
- `seg_history` 是最近 N 次提交的 Seg Total 序列
- 用指数衰减拟合：`future ≈ latest + Δ_last × (1 + r + r² + ...)` 其中 `r = Δ_this / Δ_prev`
- `confidence` 变为浮点数：`1 - (收敛过程的不确定性 / 当前分)`

```python
def _compute_convergence_estimate(seg_history: list[float]) -> dict[str, Any]:
    """Given ordered Seg total history [s0, s1, ..., sn], estimate ceiling."""
    if len(seg_history) < 3:
        return {"optimistic": seg_history[-1] + 5, "expected": seg_history[-1] + 2,
                "conservative": seg_history[-1], "confidence": "low"}
    
    # Compute incremental gains
    gains = [seg_history[i+1] - seg_history[i] for i in range(len(seg_history)-1)]
    gain_decay = 0.5  # default halving
    
    if len(gains) >= 2:
        decay_ratios = [gains[i+1] / max(1e-8, gains[i]) for i in range(len(gains)-1)]
        # 求最小值以免乐观
        gain_decay = min(decay_ratios)
    
    last_gain = gains[-1]
    asymptotic_gain = last_gain / max(1 - gain_decay, 0.1)
    projected = round(seg_history[-1] + asymptotic_gain, 1)
    
    # confidence based on gain_decay stability
    if len(gains) >= 3:
        stable = all(0.3 <= gains[i+1] / max(1e-8, gains[i]) <= 2.0 for i in range(len(gains)-1))
        conf = "high" if stable else "medium"
    else:
        conf = "medium"
    
    return {
        "optimistic": round(projected + 1, 1),
        "expected": projected,
        "conservative": round(projected - 2, 1),
        "confidence": conf,
        "gain_decay": round(gain_decay, 3),
        "seg_history": seg_history,
    }
```

**变更：** `estimate_score_potential()` 仍然保留无参形式（向后兼容），但新增可选参数 `seg_history`。

---

## Task 4: 时间感知战略建议

**Objective:** `get_strategic_recommendation()` 根据剩余天数改变输出，T≤2 时不再建议启动 Task2。

**Files:**
- Modify: `expflow_pde/analyze.py:222-264` — `get_strategic_recommendation()`

**逻辑：**
```python
    remaining_days = ...
    
    if remaining_days <= 2:
        # 冲刺模式
        recommendations = ...
    elif remaining_days <= 5:
        # 中期模式
        ...
    else:
        # 正常模式（当前逻辑）
        ...
```

---

## Task 5: 元数据自动同步（`expflow analyze sync`）

**Objective:** 新增命令从 clearml 自动拉取最新实验成绩并更新 `task_meta.yaml`。

**Files:**
- Create: `expflow_pde/analyze.py` — 新增 `sync_task_meta_from_clearml()`
- Modify: `expflow_pde/cli_analyze.py` — 新增 `analyze sync` 命令

**设计：**
```python
def sync_task_meta_from_clearml(
    project_name: str = "PDEBench",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch latest completed task metrics from clearml, update task_meta.yaml."""
    from expflow_pde.clearml import list_tasks
    
    tasks = list_tasks(project_name=project_name, tags=tags, status=["completed"])
    # Sort by last_iteration desc, group by tag (task1/task2/task3)
    # For each task group, find max seg_total
    # Update task_meta.yaml with current_best_seg, current_best_total
```

---

## Task 6: 跨 PDE 策略迁移约束标记

**Objective:** `proven_strategies` 和 `suggest_next_params` 都加上适用 PDE 标记。

**Files:**
- Modify: `~/.expflow/task_meta.yaml` — 每个 strategy 加 `applicable_tasks`
- Modify: `expflow_pde/analyze.py` — `suggest_next_params()` 检查适用性

**YAML 格式：**
```yaml
task1:
  proven_strategies:
    - text: 'P2 architecture (16/32, 50K params): optimal size'
      applied_to: [task1]    # 仅 Task1 验证
      seg_gain: 0            # 这是架构选择不是增量改进
      experiment_id: null
    - text: 'sub_step=5: +11.37 Seg (dt mismatch fix)'
      applied_to: [task1]    # dt mismatch 对所有任务的 Burgers 有效
      seg_gain: 11.37
      experiment_id: cm_abc123
    - text: 'Stability FT (rollout_stability_penalty): +23.45 Seg'
      applied_to: [task1, task3]  # 步间方差惩罚对 AR 类问题都有效
      seg_gain: 23.45
      experiment_id: cm_abc124
```

---

## Task 7: 假说验证记录（负结果日志）

**Objective:** 新增 `hypothesis_registry` 模块，记录每次实验的假设和被排除的方向。

**Files:**
- Create: `expflow_pde/hypothesis.py` — HypothesisRegistry 类
- Create: `expflow_pde/cli_hypothesis.py` — CLI 命令
- Data: `~/.expflow/hypotheses.yaml`

**设计：**
```yaml
hypotheses:
  - id: hyp_20260520_001
    created: 2026-05-20T14:00:00
    # 假设的内容
    hypothesis: "Increasing n_modes from 16 to 24 will improve Seg3"
    # 理论的动机
    rationale: "Seg3 collapse suggests missing high frequencies"
    # 建议参数
    suggested_params:
      n_modes: 24
    # 参考的实验
    origin_task_id: cm_task123
    # 验证状态
    status: proposed  # proposed | accepted | rejected | inconclusive
    
  - id: hyp_20260520_002
    created: 2026-05-20T15:00:00
    hypothesis: "Larger n_modes (24) improves Seg3 on Task1"
    rationale: "Hyp_001 executed — Seg3 dropped 5 points"
    status: rejected
    evidence_task_id: cm_task124
    evidence: "Seg: 57/33/22 -> 55/30/17"
    rejected_by: experiment
```

**CLI：**
```bash
expflow hypothesis list                 # 列出所有假说
expflow hypothesis show <id>            # 查看详情
expflow hypothesis close <id> --status rejected --evidence "..."  # 关闭假说
```

---

## 执行顺序

```
Task 1 (P0: clearml静默失败) 
  → Task 2 (P0: 阈值刚性)
    → Task 3 (P1: 置信度数据驱动)
      → Task 4 (P1: 时间感知)
        → Task 5 (P1: 元数据同步)
          → Task 6 (P2: 策略标记)
            → Task 7 (P2: 假说日志)
```

每步 commit 后编译检查 + 测试。
