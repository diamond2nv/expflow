# Expflow Experiment Loop — Plan

> **For Hermes:** Build in two phases: first the analyze tools (diagnose + suggest), then the iterate engine.

**Goal:** Close the experiment loop for all deployment modes (local PID, remote clearml, mixed, goal-driven). After a training run completes, automatically: read metrics → diagnose degradation → suggest next params → submit next experiment.

**Three deployment modes this serves:**
- A) **Local PID** — `taskctl` monitors local process, triggers chain command on completion
- B) **Remote clearml** — user or Hermes polls task status, calls diagnose on completed task
- C) **Mixed (current daily use)** — 3080 analyzes, 5090 executes, handoff via clearml server
- D) **Goal-driven** — Hermes `/goal` loop calls diagnose/suggest/iterate automatically

All modes share the same core: `diagnose_experiment()` + `suggest_next_params()`.

**Key integration point with existing reverse-pipeline:** The `expflow-reverse-pipeline` skill already has taskctl (PID monitoring) + chain commands. But it lacks the actual analysis functions to put in those chains. This plan builds those functions. The reverse-pipeline's on-success hooks will call `expflow analyze diagnose --task <clearml_id>` and `expflow analyze suggest` once they exist.

**Tech Stack:** expflow_pde existing modules (analyze.py, pipeline.py, clearml.py, compare.py) — zero new dependencies.

---

## Phase 0: Check existing test infrastructure

### Task 0: Verify test setup

**Objective:** Confirm expflow tests pass and understand test patterns before adding new code.

```bash
cd ~/Gitlab/Agentic4Sci/expflow
source venv/bin/activate
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass (or at least confirm baseline). Note test file layout.

**Commit:** only after checking — no code change needed.

---

## Phase 1: `diagnose_experiment()` — Read clearml task or local JSON, output structured diagnosis

### Task 1: Create fixture + unit test for diagnose_experiment

**Objective:** Test diagnostic rules using a local eval JSON fixture (no clearml dependency).

**Files:**
- Modify: `expflow_pde/analyze.py` — add `diagnose_experiment()`, `_load_experiment_metrics()`
- Create: `tests/test_diagnose.py` — dedicated test file for diagnose + suggest

**Step 1: Write fixture and tests**

```python
# tests/test_diagnose.py
"""Tests for experiment diagnosis engine."""

import json
import pytest


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


def test_diagnose_basic_output_shape(sample_eval_json):
    """Returns dict with seg keys, diagnosis list, degradation pattern."""
    from expflow_pde.analyze import diagnose_experiment
    result = diagnose_experiment(json_path=sample_eval_json)
    assert result is not None
    assert "seg1" in result
    assert "seg2" in result
    assert "seg3" in result
    assert "total" in result
    assert "diagnosis" in result
    assert isinstance(result["diagnosis"], list)
    assert "degradation_pattern" in result
    assert "total_mse" in result


def test_diagnose_detects_long_term_collapse(sample_eval_json):
    """seg1=85, seg2=55, seg3=23 → long_term pattern."""
    from expflow_pde.analyze import diagnose_experiment
    result = diagnose_experiment(json_path=sample_eval_json)
    assert result["degradation_pattern"] == "long_term"
    assert any("collapse" in d.lower() for d in result["diagnosis"])


def test_diagnose_unknown_path():
    """Non-existent file returns None."""
    from expflow_pde.analyze import diagnose_experiment
    result = diagnose_experiment(json_path="/nonexistent/file.json")
    assert result is None


@pytest.fixture
def stable_eval_json(tmp_path):
    """Fixture with good scores — no degradation."""
    data = {
        "experiment_id": "stable_test",
        "segmented_scores": {
            "seg1_score": 92.0,
            "seg2_score": 88.0,
            "seg3_score": 85.0,
            "total_segmented_score": 87.5,
        },
    }
    path = tmp_path / "stable_result.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


def test_diagnose_stable(stable_eval_json):
    """All seg >80 → stable pattern, no critical diagnosis."""
    from expflow_pde.analyze import diagnose_experiment
    result = diagnose_experiment(json_path=stable_eval_json)
    assert result["degradation_pattern"] == "stable"
    assert any("No critical" in d for d in result["diagnosis"])
```

Run:
```bash
cd ~/Gitlab/Agentic4Sci/expflow && source venv/bin/activate
python -m pytest tests/test_diagnose.py -v --tb=short
```
Expected: 4/4 FAILED (import error — diagnose_experiment doesn't exist yet).

**Step 2: Run to verify failures**
```bash
python -m pytest tests/test_diagnose.py -v --tb=short
```

**Step 3: Implement in analyze.py**

Add to `expflow_pde/analyze.py` (after imports, before public API functions):

```python
import json
import os


def _load_experiment_metrics(
    task_id: str | None = None,
    json_path: str | None = None,
) -> dict | None:
    """Load experiment metrics from clearml task or local JSON.
    
    Supports two input sources:
    - json_path: local eval_task1_*.json file (for unit tests / offline use)
    - task_id: clearml task ID (fetches metrics from clearml server)
    """
    if json_path:
        if not os.path.exists(json_path):
            return None
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

Then add `diagnose_experiment()` at the end of the file, before the `if __name__` guard:

```python
def diagnose_experiment(
    task_id: str | None = None,
    json_path: str | None = None,
) -> dict | None:
    """Analyze experiment results and identify degradation patterns.
    
    Reads metrics from a clearml task or local eval JSON, then applies
    rule-based diagnosis to classify degradation mode.
    
    Args:
        task_id: ClearML task ID (alternative to json_path).
        json_path: Path to local eval JSON file (alternative to task_id).
    
    Returns:
        Dict with seg scores, diagnosis list, and degradation pattern string.
        Returns None if metrics cannot be loaded.
    """
    metrics = _load_experiment_metrics(task_id, json_path)
    if metrics is None:
        return None

    # Extract seg scores — handles both flat and nested JSON shapes
    seg = metrics.get("segmented_scores", metrics)
    if isinstance(seg, dict):
        seg1 = seg.get("seg1_score", seg.get("seg1", seg.get("Seg1", 0)))
        seg2 = seg.get("seg2_score", seg.get("seg2", seg.get("Seg2", 0)))
        seg3 = seg.get("seg3_score", seg.get("seg3", seg.get("Seg3", 0)))
        total = seg.get("total_segmented_score", seg.get("total", seg.get("Total", 0)))
    else:
        seg1 = seg2 = seg3 = total = 0

    # Extract total_mse from possibly nested results
    results = metrics.get("results", {})
    total_mse = results.get("total_mse", metrics.get("total_mse", metrics.get("Total MSE", 0)))
    if not isinstance(total_mse, (int, float)):
        total_mse = 0.0

    # Diagnosis rules
    diagnosis = []
    degradation_pattern = "stable"

    # Seg1 < 70 → short-term accuracy issue
    if isinstance(seg1, (int, float)) and seg1 < 70:
        diagnosis.append("Short-term prediction is weak (Seg1 low)")
        degradation_pattern = "short_term"

    # Seg2 - Seg1 decline > 25 pts → medium-term stability issue
    if (isinstance(seg1, (int, float)) and seg1 > 0
            and isinstance(seg2, (int, float)) and seg2 > 0
            and (seg1 - seg2) > 25):
        diagnosis.append("Medium-term stability degraded (Seg2 drops >25 from Seg1)")
        if degradation_pattern == "stable":
            degradation_pattern = "mid_term"

    # Seg3 < 35 or Seg3 < 60% of Seg2 → long-term collapse
    if (isinstance(seg3, (int, float))
            and (seg3 < 35 or (isinstance(seg2, (int, float)) and seg2 > 0 and seg3 < seg2 * 0.6))):
        diagnosis.append("Long-term autoregressive collapse (Seg3 collapse)")
        degradation_pattern = "long_term"

    # All seg low but MSE moderate → distribution shift
    if (all(isinstance(s, (int, float)) and s > 0 for s in [seg1, seg2, seg3])
            and max(seg1, seg2, seg3) < 40
            and isinstance(total_mse, (int, float)) and total_mse < 0.1):
        diagnosis.append("Consistent underperformance — possible IC distribution mismatch")
        degradation_pattern = "distribution_shift"

    if not diagnosis:
        diagnosis.append("No critical degradation detected")

    return {
        "seg1": round(float(seg1), 2) if isinstance(seg1, (int, float)) else 0,
        "seg2": round(float(seg2), 2) if isinstance(seg2, (int, float)) else 0,
        "seg3": round(float(seg3), 2) if isinstance(seg3, (int, float)) else 0,
        "total": round(float(total), 2) if isinstance(total, (int, float)) else 0,
        "total_mse": round(float(total_mse), 6) if isinstance(total_mse, (int, float)) else 0,
        "diagnosis": diagnosis,
        "degradation_pattern": degradation_pattern,
    }
```

**Step 4: Run tests to verify they pass**
```bash
python -m pytest tests/test_diagnose.py -v --tb=short
```
Expected: 4/4 PASSED.

**Step 5: Commit**
```bash
cd ~/Gitlab/Agentic4Sci/expflow
git add tests/test_diagnose.py expflow_pde/analyze.py
git commit -m "feat: add diagnose_experiment() with degradation pattern detection"
```

---

### Task 2: CLI `expflow analyze diagnose`

**Objective:** CLI entry point so reverse-pipeline on-success hooks and Hermes can call it.

**Files:**
- Modify: `expflow_pde/cli_analyze.py`

Add after the `advise` command:

```python
@analyze_app.command("diagnose")
def diagnose_cmd(
    task_id: str | None = typer.Option(None, "--task", "-t",
        help="ClearML task ID to diagnose"),
    json_path: str | None = typer.Option(None, "--json", "-j",
        help="Path to local eval JSON file"),
) -> None:
    """Analyze experiment results and identify degradation patterns.
    
    Provide either --task (clearml task ID) or --json (local file path).
    Outputs seg scores, degradation pattern, and actionable diagnosis.
    """
    if not task_id and not json_path:
        print("ERROR: Provide --task <id> or --json <path>")
        raise typer.Exit(code=1)

    from expflow_pde.analyze import diagnose_experiment
    result = diagnose_experiment(task_id=task_id, json_path=json_path)
    if result is None:
        src = f"task_id={task_id}" if task_id else f"json={json_path}"
        print(f"Could not load experiment metrics ({src})")
        raise typer.Exit(code=1)

    print(f"  Seg1: {result['seg1']:>6.2f}  | Seg2: {result['seg2']:>6.2f}  "
          f"| Seg3: {result['seg3']:>6.2f}  | Total: {result['total']:>6.2f}")
    print(f"  MSE:  {result['total_mse']:.6f}")
    print(f"  Pattern: {result['degradation_pattern']}")
    print(f"  Diagnosis:")
    for d in result['diagnosis']:
        print(f"    - {d}")
```

**Test CLI:**
Add to `tests/test_diagnose.py`:

```python
from typer.testing import CliRunner


def test_diagnose_cli_json(sample_eval_json):
    from expflow_pde.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "diagnose", "--json", sample_eval_json])
    assert result.exit_code == 0, result.output
    assert "Seg1" in result.output
    assert "Seg3" in result.output or "collapse" in result.output
    assert "Pattern" in result.output


def test_diagnose_cli_no_args():
    from expflow_pde.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "diagnose"])
    assert result.exit_code != 0
    assert "ERROR" in result.output
```

Run:
```bash
python -m pytest tests/test_diagnose.py -v --tb=short -k cli
```

**Commit:**
```bash
git add tests/test_diagnose.py expflow_pde/cli_analyze.py
git commit -m "feat: add analyze diagnose CLI command"
```

---

## Phase 2: `suggest_next_params()` — Generate next experiment parameters from diagnosis

### Task 3: Create suggest_next_params() with tests

**Objective:** Given diagnosis + current hparams, produce suggestion. Rule-based (0 token, deterministic).

**Files:**
- Modify: `expflow_pde/analyze.py`
- Modify: `tests/test_diagnose.py`

**Tests:**
```python
def test_suggest_long_term_collapse():
    """Seg3 collapse → n_modes + sub_step + wd."""
    from expflow_pde.analyze import suggest_next_params
    diagnosis = {
        "degradation_pattern": "long_term",
        "seg1": 85.3, "seg2": 55.1, "seg3": 22.7, "total": 46.4,
    }
    hp = {"n_modes": 12, "hidden_channels": 20, "lr": 0.001}
    result = suggest_next_params(diagnosis, current_hparams=hp)
    assert result["suggested_params"]["n_modes"] == 16  # 12+4
    assert result["suggested_params"]["num_sub_steps"] == 5
    assert "rationale" in result


def test_suggest_mid_term():
    """Medium-term drop → stability_lambda."""
    from expflow_pde.analyze import suggest_next_params
    diagnosis = {
        "degradation_pattern": "mid_term",
        "seg1": 87.0, "seg2": 45.0, "seg3": 38.0, "total": 54.0,
    }
    result = suggest_next_params(diagnosis, current_hparams={"lr": 0.001})
    assert result["suggested_params"].get("stability_lambda") == 0.001


def test_suggest_stable():
    """Stable → HPO round recommended."""
    from expflow_pde.analyze import suggest_next_params
    diagnosis = {
        "degradation_pattern": "stable",
        "seg1": 90, "seg2": 85, "seg3": 80, "total": 85,
    }
    result = suggest_next_params(diagnosis, current_hparams={})
    assert "hpo" in result["suggested_params"].get("tag", "")


def test_suggest_short_term():
    """Short-term weak → higher lr + more epochs."""
    from expflow_pde.analyze import suggest_next_params
    diagnosis = {
        "degradation_pattern": "short_term",
        "seg1": 55.0, "seg2": 50.0, "seg3": 45.0, "total": 50.0,
    }
    result = suggest_next_params(diagnosis, current_hparams={"lr": 0.001, "epochs": 80})
    assert result["suggested_params"]["lr"] == 0.002  # 0.001*2
    assert result["suggested_params"]["epochs"] == 100
```

**Implementation:**
Add after `diagnose_experiment()` in analyze.py:

```python
def suggest_next_params(
    diagnosis: dict,
    current_hparams: dict | None = None,
    task_id: str = "task1",
) -> dict:
    """Suggest next experiment parameters based on diagnosis.
    
    Uses rule-based suggestions derived from proven strategies
    (sub_step, stability FT, HyperNOs best practices).
    Zero token cost — deterministic rules only.
    
    Args:
        diagnosis: Output from diagnose_experiment().
        current_hparams: Current experiment's hyperparameters.
        task_id: Competition task ID (for context).
    
    Returns:
        Dict with suggested_params, rationale list.
    """
    pattern = diagnosis.get("degradation_pattern", "stable")
    seg1 = diagnosis.get("seg1", 0)
    seg3 = diagnosis.get("seg3", 0)
    
    hp = dict(current_hparams) if current_hparams else {}
    suggestions = {}
    rationale = []
    
    if pattern == "long_term" or (
        isinstance(seg3, (int, float)) and seg3 < 30
        and isinstance(seg1, (int, float)) and seg1 > 60
    ):
        # Long-term collapse → modes + sub_step + weight_decay
        current_modes = int(hp.get("n_modes", 12))
        suggestions["n_modes"] = min(current_modes + 4, 24)
        suggestions["num_sub_steps"] = 5
        suggestions["tag"] = "auto_seg3_fix"
        rationale.append(
            f"Seg3 collapse ({seg3:.1f}): Increase n_modes "
            f"{current_modes}->{suggestions['n_modes']} "
            "to capture more spatial frequencies"
        )
        rationale.append(
            "Add sub_step=5 to fix dt mismatch between "
            "training (0.01) and inference (0.05)"
        )
        if not hp.get("weight_decay"):
            suggestions["weight_decay"] = 1e-4
            rationale.append(
                "Add weight_decay=1e-4 (HyperNOs Burgers best practice)"
            )
    
    elif pattern == "mid_term":
        # Medium-term drop → stability penalty
        suggestions["tag"] = "auto_mid_fix"
        if "stability_lambda" not in hp or not hp.get("stability_lambda"):
            suggestions["stability_lambda"] = 0.001
            rationale.append(
                "Seg2 drop: add step-wise stability penalty (stability_lambda=0.001)"
            )
    
    elif pattern == "short_term":
        # Short-term weak → higher lr + more epochs
        current_lr = float(hp.get("lr", 0.001))
        suggestions["lr"] = min(current_lr * 2, 0.005)
        suggestions["epochs"] = max(int(hp.get("epochs", 80)), 100)
        suggestions["tag"] = "auto_short_fix"
        rationale.append(
            f"Seg1 low ({seg1:.1f}): increase LR "
            f"{current_lr}->{suggestions['lr']} and extend training"
        )
    
    else:
        # Stable — HPO round
        suggestions["tag"] = "auto_hpo_round"
        rationale.append(
            "Experiment stable. Run targeted HPO on remaining strategies."
        )
    
    return {
        "suggested_params": suggestions,
        "rationale": rationale,
        "task_id": task_id,
        "degradation_pattern": pattern,
    }
```

**Run tests:**
```bash
python -m pytest tests/test_diagnose.py -v --tb=short -k suggest
```

**Commit:**
```bash
git add expflow_pde/analyze.py tests/test_diagnose.py
git commit -m "feat: add suggest_next_params() for automated experiment iteration"
```

---

### Task 4: CLI `expflow analyze suggest`

**Objective:** CLI wrapping diagnose + suggest. Quick preview before iterate.

Add to `cli_analyze.py`:

```python
@analyze_app.command("suggest")
def suggest_cmd(
    task_id: str | None = typer.Option(None, "--task", "-t",
        help="ClearML task ID (reads metrics from clearml)"),
    json_path: str | None = typer.Option(None, "--json", "-j",
        help="Path to eval JSON file"),
) -> None:
    """Analyze experiment and suggest next hyperparameters.
    
    Combines diagnose + suggest. Shows current diagnosis and
    recommended next experiment parameters.
    """
    from expflow_pde.analyze import diagnose_experiment, suggest_next_params
    
    # Step 1: Build current hparams from CLI overrides
    hp = {}
    # (In future: read from clearml task params)
    
    # Step 2: Diagnose
    diagnosis = diagnose_experiment(task_id=task_id, json_path=json_path)
    if diagnosis is None:
        src = f"task_id={task_id}" if task_id else f"json={json_path}"
        print(f"Cannot load experiment ({src})")
        raise typer.Exit(code=1)
    
    # Step 3: Suggest
    suggestion = suggest_next_params(diagnosis, current_hparams=hp)
    
    # Output
    print(f"\n  Diagnosis:")
    print(f"    Pattern: {diagnosis['degradation_pattern']}")
    for d in diagnosis['diagnosis']:
        print(f"    - {d}")
    
    params = suggestion.get("suggested_params", {})
    rationale = suggestion.get("rationale", [])
    
    print(f"\n  Suggested next params:")
    for k, v in params.items():
        if k == "tag":
            continue
        print(f"    --{k}={v}")
    
    print(f"\n  Rationale:")
    for r in rationale:
        print(f"    - {r}")
```

**Test:**
```python
def test_suggest_cli(sample_eval_json):
    from expflow_pde.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "suggest", "--json", sample_eval_json])
    assert result.exit_code == 0, result.output
    assert "n_modes" in result.output
    assert "Rationale" in result.output
```

**Commit:**
```bash
git add expflow_pde/cli_analyze.py tests/test_diagnose.py
git commit -m "feat: add analyze suggest CLI command"
```

---

## Phase 3: `expflow iterate` — One-shot automated iteration

### Task 5: Create iterate.py module

**Objective:** diagnose → suggest → submit (one command).

**Files:**
- Create: `expflow_pde/iterate.py`
- Create: `expflow_pde/cli_iterate.py`

```python
# expflow_pde/iterate.py
"""Iterate engine: diagnose -> suggest -> submit as a one-shot pipeline."""

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
    """Run one complete iteration: diagnose -> suggest -> submit.
    
    Args:
        task_id: Clearml task ID of the experiment to iterate from.
        json_path: Local eval JSON path (alternative).
        current_hparams: Current experiment's hyperparameters.
        train_script: Base clearml task name for training.
        eval_script: Base clearml task name for evaluation.
        project: Clearml project name.
        queue: Queue to submit the next iteration.
        dry_run: Print what would happen, don't execute.
    
    Returns:
        Dict with diagnosis, suggestion, and (if not dry_run) pipeline result.
    """
    from expflow_pde.analyze import diagnose_experiment, suggest_next_params
    
    diagnosis = diagnose_experiment(task_id=task_id, json_path=json_path)
    if diagnosis is None:
        return {"error": "Cannot load experiment metrics", "step": "diagnose"}
    
    suggestion = suggest_next_params(
        diagnosis,
        current_hparams=current_hparams or {},
    )
    
    if dry_run:
        return {
            "diagnosis": diagnosis,
            "suggestion": suggestion,
            "submitted": False,
        }
    
    from expflow_pde.pipeline import ExperimentPipeline
    ep = ExperimentPipeline(project=project, queue=queue)
    
    suggested = dict(suggestion.get("suggested_params", {}))
    suggested.pop("tag", None)
    
    pipe_result = ep.train_val_submit(
        train_script=train_script,
        train_params=suggested,
        eval_script=eval_script,
    )
    
    return {
        "diagnosis": diagnosis,
        "suggestion": suggestion,
        "submitted": True,
        "pipeline": pipe_result,
    }
```

```python
# expflow_pde/cli_iterate.py
"""expflow iterate CLI — one-shot automate experiment iteration."""

from typing import Optional

import typer

iterate_app = typer.Typer(
    name="iterate",
    help="One-shot experiment iteration: diagnose -> suggest -> submit",
    no_args_is_help=True,
)


@iterate_app.command("run")
def iterate_run_cmd(
    task_id: Optional[str] = typer.Option(None, "--task", "-t",
        help="Clearml task ID of completed experiment"),
    json_path: Optional[str] = typer.Option(None, "--json", "-j",
        help="Path to eval JSON file (alternative)"),
    train_script: str = typer.Option("train_task1.py", "--train", "-T",
        help="Base clearml task name for training"),
    queue: str = typer.Option("default", "--queue", "-q",
        help="Clearml queue"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n",
        help="Preview suggestion without submitting"),
) -> None:
    """Diagnose -> suggest -> submit next experiment iteration."""
    from expflow_pde.iterate import run_iteration
    
    result = run_iteration(
        task_id=task_id,
        json_path=json_path,
        train_script=train_script,
        queue=queue,
        dry_run=dry_run,
    )
    
    if "error" in result:
        print(f"ERROR: {result['error']} (step: {result.get('step', '?')})")
        raise typer.Exit(code=1)
    
    diag = result.get("diagnosis", {})
    sugg = result.get("suggestion", {})
    
    print(f"\n  Diagnosis:")
    print(f"    Pattern: {diag.get('degradation_pattern', '?')}")
    for d in diag.get("diagnosis", []):
        print(f"    - {d}")
    
    params = sugg.get("suggested_params", {})
    print(f"\n  Suggested params:")
    for k, v in params.items():
        if k == "tag":
            continue
        print(f"    --{k}={v}")
    
    if dry_run:
        print(f"\n  [dry-run] Would submit to queue '{queue}'")
    else:
        pipe = result.get("pipeline", {})
        print(f"\n  Submitted: pipeline_id={pipe.get('pipeline_id', '?')}")
        print(f"  Queue: {queue}")
```

### Task 6: Register iterate command group in cli.py

Add to `expflow_pde/cli.py` after `_lazy_register_pipeline()`:

```python
def _lazy_register_iterate():
    from expflow_pde.cli_iterate import iterate_app
    for cmd in app.registered_commands:
        if getattr(cmd, "name", None) == "iterate":
            return iterate_app
    app.add_typer(
        iterate_app,
        name="iterate",
        help="One-shot experiment iteration: diagnose -> suggest -> submit",
    )
    return iterate_app


_ = _lazy_register_iterate()
```

**Test iterate:**
```python
def test_iterate_dry_run(sample_eval_json):
    from expflow_pde.iterate import run_iteration
    result = run_iteration(
        json_path=sample_eval_json,
        current_hparams={"n_modes": 12, "lr": 0.001},
        dry_run=True,
    )
    assert not result.get("submitted", True)
    assert "diagnosis" in result
    assert "suggestion" in result
    assert result["diagnosis"]["degradation_pattern"] == "long_term"
```

**Commit:**
```bash
git add expflow_pde/iterate.py expflow_pde/cli_iterate.py expflow_pde/cli.py tests/test_diagnose.py
git commit -m "feat: add iterate command group (diagnose -> suggest -> submit)"
```

---

## Usage Flow (after completion)

```bash
# Mode B/C (current daily use — remote clearml execution):
expflow pipeline submit train_task1.py --queue default
# ... wait for 5090 to finish ...
# On 3080, with clearml task ID:
expflow analyze diagnose --task abc123
#   Seg1: 85.30  | Seg2: 55.10  | Seg3: 22.70  | Pattern: long_term
expflow analyze suggest --task abc123
#   Suggested: --n_modes=16 --num_sub_steps=5 --weight_decay=1e-4
# One-shot:
expflow iterate run --task abc123 --queue default --dry-run
expflow iterate run --task abc123 --queue default

# Mode A (local PID, with reverse-pipeline taskctl):
taskctl.py add --id exp_fno --pid $PID --duration 7200 \
  --on-success "expflow analyze diagnose --task abc123 && expflow analyze suggest --task abc123"

# Mode D (Hermes /goal):
/goal reach seg_total 140 on Task1
# Hermes -> expflow analyze advise -> expflow submit -> wait
# -> expflow analyze diagnose --task <id> -> expflow iterate run -> ...
```

---

## Test Summary

| Test file | Tests | What they cover |
|-----------|:-----:|-----------------|
| `tests/test_diagnose.py` | ~12 | diagnose (4 tests), suggest (4), CLI (2), iterate dry-run (1), stable fixture (1) |
