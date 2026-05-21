---
name: clearml-metrics-logging-pattern
description: Generated from skills/clearml-metrics-logging-pattern/SKILL.md (package reference copy)
---

     1|---
     2|name: clearml-metrics-logging-pattern
     3|description: Standardized ClearML metrics logging patterns for PDEBench experiment scripts — train loss, validation metrics, competition scores, PDE residual, and TensorBoardX integration. Includes patterns for dist/expflow compatibility.
     4|category: mlops
     5|author: Li Shen
     6|version: 1.0.0
     7|metadata:
     8|  hermes:
     9|    tags: [mlops, pde, clearml, metrics, logging, experiment, competition]
    10|    homepage: https://github.com/diamond2nv/expflow
    11|    related_skills: [expflow-pipeline-hpo, experiment-lifecycle-governance, competition-task-intelligence]
    12|---
    13|
    14|# ClearML Metrics Logging Pattern
    15|
    16|## When to Use
    17|
    18|- Creating or modifying PDEBench training/evaluation scripts
    19|- Adding clearml logging to `train_task1.py`, `train_task1_phys.py`, `train_task1_ft.py`, `train_task1_unroll.py`
    20|- Ensuring expflow (single-node + distributed) can auto-capture metrics
    21|- Standardizing metric naming for compare-scores and gating
    22|
    23|## Installation
    24|
    25|```bash
    26|pip install "expflow-pde[clearml]"
    27|```
    28|
    29|## Standardized Metric Naming Convention
    30|
    31|All clearml metrics use **Group/Metric** naming, compatible with `expflow clearml compare-scores`:
    32|
    33|```python
    34|# Loss group — error/cost related scalars
    35|clearml_logger.report_scalar('Loss', 'Train MSE',     float_val, iteration=epoch)
    36|clearml_logger.report_scalar('Loss', 'Val MSE',       float_val, iteration=epoch)
    37|clearml_logger.report_scalar('Loss', 'Val RelMSE',    float_val, iteration=epoch)
    38|clearml_logger.report_scalar('Loss', 'Physics',       float_val, iteration=epoch)
    39|clearml_logger.report_scalar('Loss', 'Commut',        float_val, iteration=epoch)
    40|clearml_logger.report_scalar('Loss', 'Stability',     float_val, iteration=epoch)
    41|
    42|# Score group — competition segment scores (100-point scale)
    43|clearml_logger.report_scalar('Score', 'Seg Total',    float_val, iteration=epoch)
    44|clearml_logger.report_scalar('Score', 'Seg1',         float_val, iteration=epoch)
    45|clearml_logger.report_scalar('Score', 'Seg2',         float_val, iteration=epoch)
    46|clearml_logger.report_scalar('Score', 'Seg3',         float_val, iteration=epoch)
    47|
    48|# PDE group — PDE residuals (per-segment)
    49|clearml_logger.report_scalar('PDE', 'Mean Residual',  float_val, iteration=epoch)
    50|clearml_logger.report_scalar('PDE', 'Seg1 Residual',  float_val, iteration=epoch)
    51|clearml_logger.report_scalar('PDE', 'Seg2 Residual',  float_val, iteration=epoch)
    52|clearml_logger.report_scalar('PDE', 'Seg3 Residual',  float_val, iteration=epoch)
    53|
    54|# System group — system monitoring
    55|clearml_logger.report_scalar('System', 'GPU Alloc MB',   float_val, iteration=epoch)
    56|clearml_logger.report_scalar('System', 'GPU Reserved MB', float_val, iteration=epoch)
    57|clearml_logger.report_scalar('System', 'LR',              float_val, iteration=epoch)
    58|
    59|# Kfold group — k-fold cross-validation results
    60|clearml_logger.report_scalar('Kfold', 'Mean Seg',    float_val, iteration=0)
    61|clearml_logger.report_scalar('Kfold', 'Std Seg',     float_val, iteration=0)
    62|clearml_logger.report_scalar('Kfold', 'CV Seg%',     float_val, iteration=0)
    63|```
    64|
    65|## Code Templates
    66|
    67|### Template A: Add clearml logging to training loop
    68|
    69|Insert into existing `train_task1.py` / `train_task1_phys.py` / `train_task1_ft.py` / `train_task1_unroll.py`:
    70|
    71|```python
    72|# After Task.init(), get logger
    73|clearml_logger = None
    74|if clearml_task is not None:
    75|    try:
    76|        clearml_logger = clearml_task.get_logger()
    77|    except Exception:
    78|        pass
    79|
    80|# At end of epoch loop (after avg_loss is computed)
    81|if clearml_logger is not None:
    82|    clearml_logger.report_scalar('Loss', 'Train MSE', avg_loss, iteration=epoch + 1)
    83|    clearml_logger.report_scalar('System', 'LR', scheduler.get_last_lr()[0], iteration=epoch + 1)
    84|    if DEVICE.type == 'cuda':
    85|        clearml_logger.report_scalar('System', 'GPU Alloc MB', round(gpu_alloc, 1), iteration=epoch + 1)
    86|
    87|# After validation (after val_mse, val_rel, seg are computed)
    88|if clearml_logger is not None:
    89|    clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=epoch + 1)
    90|    clearml_logger.report_scalar('Loss', 'Val RelMSE', val_rel, iteration=epoch + 1)
    91|    clearml_logger.report_scalar('Score', 'Seg Total', seg['total_segmented_score'], iteration=epoch + 1)
    92|    clearml_logger.report_scalar('Score', 'Seg1', seg['seg1_score'], iteration=epoch + 1)
    93|    clearml_logger.report_scalar('Score', 'Seg2', seg['seg2_score'], iteration=epoch + 1)
    94|    clearml_logger.report_scalar('Score', 'Seg3', seg['seg3_score'], iteration=epoch + 1)
    95|
    96|# For physics loss (train_task1_phys.py)
    97|if clearml_logger is not None and phys_loss is not None:
    98|    clearml_logger.report_scalar('Loss', 'Physics', phys_loss.item(), iteration=epoch + 1)
    99|```
   100|
   101|### Template B: Eval script clearml logging
   102|
   103|```python
   104|def run_eval_and_log(model, val_data, cl_task, tag):
   105|    clearml_logger = cl_task.get_logger() if cl_task is not None else None
   106|    val_mse, val_rel, seg_scores = evaluate_autoregressive(model, val_data)
   107|
   108|    if clearml_logger is not None:
   109|        clearml_logger.report_scalar('Score', 'Seg Total', seg_scores['total_segmented_score'], iteration=1)
   110|        clearml_logger.report_scalar('Score', 'Seg1', seg_scores['seg1_score'], iteration=1)
   111|        clearml_logger.report_scalar('Score', 'Seg2', seg_scores['seg2_score'], iteration=1)
   112|        clearml_logger.report_scalar('Score', 'Seg3', seg_scores['seg3_score'], iteration=1)
   113|        clearml_logger.report_scalar('Loss', 'Val MSE', val_mse, iteration=1)
   114|        clearml_logger.report_scalar('Loss', 'Val RelMSE', val_rel, iteration=1)
   115|
   116|    return val_mse, val_rel, seg_scores
   117|```
   118|
   119|### Template C: Double Logger (TensorBoardX + ClearML)
   120|
   121|```python
   122|class DoubleLogger:
   123|    def __init__(self, tb_writer=None, cl_logger=None):
   124|        self.tb = tb_writer
   125|        self.cl = cl_logger
   126|
   127|    def scalar(self, group, name, value, iteration):
   128|        if self.tb is not None:
   129|            self.tb.add_scalar(f'{group}/{name}', value, iteration)
   130|        if self.cl is not None:
   131|            self.cl.report_scalar(group, name, value, iteration=iteration)
   132|```
   133|
   134|## Consistency with expflow
   135|
   136|- Group names match `compare-scores` display names
   137|- Metric names match `STANDARD_METRICS` keys (via underscore)
   138|- `iteration` must increment monotonically (clearml x-axis requirement)
   139|- Single-value eval metrics use `iteration=1`
   140|
   141|## Known Pitfalls
   142|
   143|1. **`Task.get_logger()` must be called after `Task.init()`**, otherwise returns None
   144|2. **`capture_tensorboard=True`** — TensorBoardX and clearml dual-write works, but clearml adds TensorBoard path prefix to group names
   145|3. **Distributed metrics** are stored per-trial — parent optuna study only stores `user_objective`, not aggregated trial metrics
   146|4. **Group + Metric name must be consistent** — always `Score/Seg Total`, never `Score/Seg_Total`
   147|
