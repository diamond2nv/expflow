# Expflow Iterate — 实验闭环引擎实现计划

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task.

**Goal:** 为 expflow 添加 `expflow analyze` → `expflow iterate` 闭环能力——赛后自动分析实验结果 → 生成诊断 → 自动构造下一轮实验参数 → 提交 clearml queue。这是 Robin 的 Finch（数据分析）+ Falcon（深度综述）+ Crow（文献搜索）的 PDE 版本。

**架构：**
```
expflow_pde/iterate.py     — 闭环引擎（analyze → diagnosis → suggest → execute）
expflow_pde/analyze.py     — 扩展：增加 diagnose_experiment(), suggest_next_params()
expflow_pde/cli_analyze.py — 新增 diagnose, suggest, iterate 子命令
```

**Tech Stack:** expflow_pde 现有体系（analyze.py, pipeline.py, clearml.py, compare.py）
——全部使用纯 Python + stdlib，不引入新依赖。

**原理**：参考 Robin 的三个环节——Crow（文献搜索）由 hfpclawer + wiki 知识库替代；Falcon（深度评审）由 analyze.py 的 `_TASK_META` + 实验历史对比替代；Finch（数据分析）由 `expflow iterate` 自身替代。

---

## 阶段 1：`expflow analyze diagnose` — 赛后自动诊断

在 analyze.py 中新增 `diagnose_experiment()` 函数，读取一个实验的 clearml 任务 ID 或本地 eval JSON，输出退化模式诊断。

### Task 1: 创建 diagnose_experiment() 函数体

**Objective:** 从 clearml task 或本地 JSON 加载 eval 指标，输出结构化诊断

**Files:**
- Modify: `expflow_pde/analyze.py`
- Test: `tests/test_analyze.py`

**Step 1: 写测试文件**

```python
# tests/test_analyze.py
def test_diagnose_experiment_from_json():
    """可以读取 eval JSON 并输出诊断。"""
    from expflow_pde.analyze import diagnose_experiment
    result = diagnose_experiment(json_path="test_fixtures/eval_task1_sample.json")
    assert result is not None
    assert "seg1" in result
    assert "seg2" in result
    assert "seg3" in result
    assert "diagnosis" in result
```

**Step 2: 跑一下确认失败**
```bash
cd ~/Gitlab/Agentic4Sci/expflow
source venv/bin/activate
python -m pytest tests/test_analyze.py -v 2>&1 | head -5
```
预期：Test not found（test 文件还不存在）或 import error。

**Step 3: 在 analyze.py 末尾添加 diagnose_experiment()**

```python
def diagnose_experiment(
    task_id: str | None = None,
    json_path: str | None = None,
) -> dict[str, Any] | None:
    """Diagnose an experiment's results and identify degradation patterns.
    
    Args:
        task_id: ClearML task ID (optional, alternative to json_path).
        json_path: Path to local eval JSON file (optional).
    
    Returns:
        Dict with seg scores, diagnosis, and degradation pattern.
    """
    # Load metrics
    metrics = _load_experiment_metrics(task_id, json_path)
    if metrics is None:
        return None
    
    seg = metrics.get("segmented_scores", {}) if isinstance(metrics, dict) else {}
    seg1 = seg.get("seg1_score", seg.get("seg1", 0)) if isinstance(seg, dict) else 0
    seg2 = seg.get("seg2_score", seg.get("seg2", 0)) if isinstance(seg, dict) else 0
    seg3 = seg.get("seg3_score", seg.get("seg3", 0)) if isinstance(seg, dict) else 0
    total = seg.get("total_segmented_score", seg.get("total", 0)) if isinstance(seg, dict) else 0
    total_mse = metrics.get("total_mse", metrics.get("results", {}).get("total_mse")) if isinstance(metrics, dict) else 0
    if isinstance(total_mse, dict):
        total_mse = total_mse.get("total_mse", 0)
    
    # Diagnosis rules
    diagnosis: list[str] = []
    degradation_pattern = "stable"
    
    # Seg1 < 70 → short-term accuracy issue
    if seg1 < 70:
        diagnosis.append("Short-term prediction is weak (Seg1 low)")
        degradation_pattern = "short_term"
    
    # Seg2 - Seg1 decline > 20 → medium-term stability issue
    if seg1 > 0 and seg2 > 0 and (seg1 - seg2) > 20:
        diagnosis.append("Medium-term stability degraded (Seg2 drops >20 from Seg1)")
        degradation_pattern = "mid_term"
    
    # Seg3 < 30 or Seg3 < Seg2 * 0.5 → long-term collapse
    if seg3 < 30 or (seg2 > 0 and seg3 < seg2 * 0.5):
        diagnosis.append("Long-term autoregressive collapse (Seg3 collapse)")
        degradation_pattern = "long_term"
    
    # All seg low but MSE moderate → distribution shift
    if all(s > 0 for s in [seg1, seg2, seg3]) and max(seg1, seg2, seg3) < 40 and total_mse < 0.1:
        diagnosis.append("Consistent underperformance — possible IC distribution mismatch")
        degradation_pattern = "distribution_shift"
    
    if not diagnosis:
        diagnosis.append("No critical degradation detected")
    
    return {
        "seg1": round(seg1, 2),
        "seg2": round(seg2, 2),
        "seg3": round(seg3, 2),
        "total": round(total, 2),
        "diagnosis": diagnosis,
        "degradation_pattern": degradation_pattern,
        "total_mse": round(total_mse, 6) if isinstance(total_mse, (int, float)) else 0,
    }
```

**Step 4: 创建辅助函数 _load_experiment_metrics**

```python
def _load_experiment_metrics(
    task_id: str | None = None,
    json_path: str | None = None,
) -> dict[str, Any] | None:
    """Load experiment metrics from clearml task or local JSON."""
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    
    if task_id:
        try:
            from expflow_pde.clearml import get_task_scalars
            return get_task_scalars(task_id)
        except Exception:
            return None
    
    return None
```

**Step 5: 写 fixture JSON 和测试**

```python
# tests/conftest.py 或单独 fixture
@pytest.fixture
def sample_eval_json(tmp_path):
    """Create a sample eval_task1 JSON fixture."""
    data = {
        "experiment_id": "eval_task1_test",
        "results": {
            "total_mse": 0.0035,
            "mean_rel_mse": 0.067,
        },
        "segmented_scores": {
            "seg1_score": 85.3,
            "seg2_score": 55.1,
            "seg3_score": 22.7,
            "total_segmented_score": 46.4,
        },
    }
    path = tmp_path / "eval_result.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)

def test_diagnose_detects_long_term_collapse(sample_eval_json):
    from expflow_pde.analyze import diagnose_experiment
    result = diagnose_experiment(json_path=sample_eval_json)
    assert result["degradation_pattern"] == "long_term"
    assert any("collapse" in d for d in result["diagnosis"])
```

**Run:**
```bash
cd ~/Gitlab/Agentic4Sci/expflow && source venv/bin/activate
python -m pytest tests/test_analyze.py -v --tb=short
```

**Step 6: Commit**
```bash
git add tests/test_analyze.py expflow_pde/analyze.py
git commit -m "feat: add diagnose_experiment() with degradation pattern detection"
```

### Task 2: 添加 `expflow analyze diagnose` CLI 命令

**Objective:** CLI 入口 `expflow analyze diagnose --task <id> --json <path>`

**Files:**
- Modify: `expflow_pde/cli_analyze.py`

**添加命令：**
```python
@analyze_app.command("diagnose")
def diagnose_cmd(
    task_id: Optional[str] = typer.Option(None, "--task", "-t",
        help="ClearML task ID to diagnose"),
    json_path: Optional[str] = typer.Option(None, "--json", "-j",
        help="Path to eval JSON file"),
) -> None:
    """Analyze experiment results and identify degradation patterns."""
    from expflow_pde.analyze import diagnose_experiment

    if not task_id and not json_path:
        print("ERROR: Provide --task or --json")
        raise typer.Exit(code=1)

    result = diagnose_experiment(task_id=task_id, json_path=json_path)
    if result is None:
        print(f"Could not load experiment (task_id={task_id}, json={json_path})")
        raise typer.Exit(code=1)

    print(f"  Seg1: {result['seg1']:>6.2f}  | Seg2: {result['seg2']:>6.2f}  "
          f"| Seg3: {result['seg3']:>6.2f}  | Total: {result['total']:>6.2f}")
    print(f"  MSE:  {result['total_mse']:.6f}")
    print(f"  Pattern: {result['degradation_pattern']}")
    print(f"  Diagnosis:")
    for d in result["diagnosis"]:
        print(f"    - {d}")
```

**测试 CLI：**
```python
from typer.testing import CliRunner
runner = CliRunner()

def test_diagnose_cli(sample_eval_json):
    from expflow_pde.cli import app
    result = runner.invoke(app, ["analyze", "diagnose", "--json", sample_eval_json])
    assert result.exit_code == 0
    assert "Seg1" in result.stdout
    assert "collapse" in result.stdout or "degradation" in result.stdout
```

**Commit:**
```bash
git add tests/test_cli_analyze.py expflow_pde/cli_analyze.py
git commit -m "feat: add analyze diagnose CLI command"
```

## 阶段 2：`expflow analyze suggest` — 自动生成下一轮建议

### Task 3: 创建 suggest_next_params()

**Objective:** 根据诊断结果 + wiki 知识 + 历史实验，自动提议下一组超参

**Files:**
- Modify: `expflow_pde/analyze.py`

**Step 1: 测试**
```python
def test_suggest_based_on_long_term_collapse():
    """Seg3 collapse → suggest more modes or sub_steps."""
    from expflow_pde.analyze import suggest_next_params
    diagnosis = {
        "degradation_pattern": "long_term",
        "seg1": 85.3, "seg2": 55.1, "seg3": 22.7, "total": 46.4,
    }
    current = {"n_modes": 12, "hidden_channels": 20, "n_layers": 4}
    result = suggest_next_params(diagnosis, current_hparams=current)
    assert result is not None
    assert "suggested_params" in result
    assert "rationale" in result
```

**Step 2: 实现**
```python
def suggest_next_params(
    diagnosis: dict[str, Any],
    current_hparams: dict[str, Any] | None = None,
    task_id: str = "task1",
) -> dict[str, Any]:
    """Suggest the next set of hyperparameters based on diagnosis.
    
    Uses rule-based suggestions derived from proven strategies.
    
    Args:
        diagnosis: Output from diagnose_experiment()
        current_hparams: Current experiment's hyperparameters.
        task_id: Competition task ID.
    
    Returns:
        Dict with suggested_params, rationale, next_command.
    """
    pattern = diagnosis.get("degradation_pattern", "stable")
    seg1 = diagnosis.get("seg1", 0)
    seg3 = diagnosis.get("seg3", 0)
    
    hp = current_hparams or {}
    suggestions: dict[str, Any] = {}
    rationale: list[str] = []
    
    if pattern == "long_term" or (seg3 < 30 and seg1 > 60):
        # Seg3 collapse → modes + sub_step + stability FT
        current_modes = hp.get("n_modes", 12)
        suggestions["n_modes"] = min(current_modes + 4, 24)
        suggestions["num_sub_steps"] = 5
        suggestions["tag"] = "auto_seg3_fix"
        rationale.append(
            f"Seg3 collapse ({seg3:.1f}): Increase n_modes {current_modes}→{suggestions['n_modes']} "
            f"to capture more frequencies"
        )
        rationale.append(
            "Add sub_step=5 to fix dt mismatch between training (0.01) and inference (0.05)"
        )
        if hp.get("weight_decay", 0) == 0:
            suggestions["weight_decay"] = 1e-4
            rationale.append("Add weight_decay=1e-4 (HyperNOs Burgers best practice)")
    
    elif pattern == "mid_term":
        # Medium-term drop → stability FT
        suggestions["tag"] = "auto_mid_fix"
        if "stability_lambda" not in hp or hp.get("stability_lambda", 0) == 0:
            suggestions["stability_lambda"] = 0.001
            rationale.append(
                "Seg2 drop: add step-wise stability penalty (stability_lambda=0.001)"
            )
    
    elif pattern == "short_term":
        # Short-term weak → more epochs or higher lr
        current_lr = hp.get("lr", 0.001)
        suggestions["lr"] = min(current_lr * 2, 0.005)
        suggestions["epochs"] = max(hp.get("epochs", 80), 100)
        rationale.append(
            f"Seg1 low ({seg1:.1f}): increase LR {current_lr}→{suggestions['lr']} "
            f"and extend training"
        )
    
    else:
        # Stable but could improve → HPO on proven strategies
        suggestions["tag"] = "auto_hpo_round"
        rationale.append("Experiment stable. Run targeted HPO on remainder strategies.")
    
    return {
        "suggested_params": suggestions,
        "rationale": rationale,
        "task_id": task_id,
    }
```

**Step 3: 测试**
```python
def test_suggest_based_on_mid_term():
    """Medium-term drop → stability penalty."""
    from expflow_pde.analyze import suggest_next_params
    diagnosis = {
        "degradation_pattern": "mid_term",
        "seg1": 87.0, "seg2": 45.0, "seg3": 38.0, "total": 54.0,
    }
    result = suggest_next_params(diagnosis, current_hparams={"lr": 0.001})
    assert result["suggested_params"].get("stability_lambda") == 0.001

def test_suggest_stable():
    """Stable → recommends HPO round."""
    from expflow_pde.analyze import suggest_next_params
    diagnosis = {"degradation_pattern": "stable", "seg1": 90, "seg2": 85, "seg3": 80, "total": 85}
    result = suggest_next_params(diagnosis, current_hparams={})
    assert "hpo" in result["suggested_params"].get("tag", "")
```

**Run & Commit:**
```bash
cd ~/Gitlab/Agentic4Sci/expflow && source venv/bin/activate
python -m pytest tests/test_analyze.py -v --tb=short -k "suggest"
git add expflow_pde/analyze.py tests/test_analyze.py
git commit -m "feat: add suggest_next_params() for automated experiment iteration"
```

### Task 4: CLI `expflow analyze suggest`

添加到 cli_analyze.py：
```python
@analyze_app.command("suggest")
def suggest_cmd(
    task_id: Optional[str] = typer.Option(None, "--task", "-t"),
    json_path: Optional[str] = typer.Option(None, "--json", "-j"),
    n_modes: Optional[int] = typer.Option(None, "--n-modes"),
    hidden_channels: Optional[int] = typer.Option(None, "--hidden"),
    lr: Optional[float] = typer.Option(None, "--lr"),
) -> None:
    """Analyze and suggest next hyperparameters."""
    from expflow_pde.analyze import diagnose_experiment, suggest_next_params
    ...
```

## 阶段 3：`expflow iterate` — 全自动闭环

### Task 5: 创建 iterate.py 模块

**Objective:** 闭环引擎——diagnose → suggest → submit

**Files:**
- Create: `expflow_pde/iterate.py`
- Create: `expflow_pde/cli_iterate.py`

```python
# expflow_pde/iterate.py
"""Iterate engine — automatic experiment loop: diagnose → suggest → submit."""

from typing import Any


def run_iteration(
    task_id: str | None = None,
    json_path: str | None = None,
    current_hparams: dict[str, Any] | None = None,
    train_script: str = "train_task1.py",
    eval_script: str = "eval_task1.py",
    project: str = "PDEBench",
    queue: str = "default",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one complete iteration: diagnose → suggest → submit.
    
    Args:
        task_id: ClearML task ID of the experiment to iterate from.
        json_path: Local eval JSON path (alternative to task_id).
        current_hparams: Current experiment's hyperparameters.
        train_script: Training script for the next iteration.
        eval_script: Eval script for the next iteration.
        project: ClearML project name.
        queue: Queue to submit the next iteration.
        dry_run: If True, print what would happen without executing.
    
    Returns:
        Dict with diagnosis, suggestion, and (if not dry_run) pipeline result.
    """
    from expflow_pde.analyze import diagnose_experiment, suggest_next_params
    
    # Step 1: Diagnose
    diagnosis = diagnose_experiment(task_id=task_id, json_path=json_path)
    if diagnosis is None:
        return {"error": "Could not load experiment metrics", "step": "diagnose"}
    
    # Step 2: Suggest
    suggestion = suggest_next_params(diagnosis, current_hparams=current_hparams)
    
    # Step 3: Submit next iteration
    if dry_run:
        return {
            "diagnosis": diagnosis,
            "suggestion": suggestion,
            "submission": {"status": "dry_run"},
        }
    
    from expflow_pde.pipeline import ExperimentPipeline
    ep = ExperimentPipeline(project=project, queue=queue)
    
    train_params = {**suggestion.get("suggested_params", {})}
    # Remove non-param keys
    train_params.pop("tag", None)
    
    pipeline_result = ep.train_val_submit(
        train_script=train_script,
        train_params=train_params,
        eval_script=eval_script,
    )
    
    return {
        "diagnosis": diagnosis,
        "suggestion": suggestion,
        "submission": pipeline_result,
    }
```

**CLI:**
```python
# expflow_pde/cli_iterate.py
@iterate_app.command("run")
def iterate_run_cmd(...): ...
```

**测试:**
```python
def test_iterate_dry_run(sample_eval_json):
    from expflow_pde.iterate import run_iteration
    result = run_iteration(
        json_path=sample_eval_json,
        current_hparams={"n_modes": 12, "lr": 0.001},
        dry_run=True,
    )
    assert result["diagnosis"]["degradation_pattern"] == "long_term"
    assert result["suggestion"]["suggested_params"]["n_modes"] >= 16
    assert result["submission"]["status"] == "dry_run"
```

## 阶段 4：CLI 集成 + 注册到主 CLI

### Task 6: 注册 `iterate` 命令组到 cli.py

```python
def _lazy_register_iterate():
    from expflow_pde.cli_iterate import iterate_app
    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "iterate":
            return iterate_app
    app.add_typer(
        iterate_app, name="iterate",
        help="Automatic experiment iteration: diagnose → suggest → submit",
    )
    return iterate_app

_ = _lazy_register_iterate()
```

## 完整执行流程

```bash
# 0. 跑完一个实验
expflow clearml task submit train_task1.py --args epochs=80 lr=0.001

# 1. 赛后诊断
expflow analyze diagnose --task <task_id>
# Seg1: 85.30  | Seg2: 55.10  | Seg3: 22.70  | Total: 46.40
# Pattern: long_term
# Diagnosis:
#   - Long-term autoregressive collapse (Seg3 collapse)

# 2. 看建议
expflow analyze suggest --task <task_id> --n-modes 12
# Suggested: n_modes=16, sub_step=5, weight_decay=1e-4
# Rationale: Seg3 collapsed → increase modes, add sub-stepping

# 3. 全自动闭环
expflow iterate run --task <task_id> --train-params '{"n_modes":12,"lr":0.001}'
# [diagnose] → [suggest: n_modes=16, sub_step=5] → [submitted as pipeline_auto_v2]
```

## 延续性（后续可能的扩展）

- `expflow iterate loop` — 持续循环直到收敛（max_iterations, target_seg）
- `expflow analyze compare` — 多实验自动对比诊断
- hfpclawer 集成：建议中引用相关论文证据（"arXiv:2503.18087 推荐 weight_decay=1e-4"）
- `expflow iterate batch` — 一次建议多组参数（并行探索不同方向）
