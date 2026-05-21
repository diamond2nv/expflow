---
name: expflow-pipeline-hpo
description: Generated from skills/expflow-pipeline-hpo/SKILL.md (package reference copy)
---

     1|---
     2|name: expflow-pipeline-hpo
     3|description: >
     4|  PDEBench competition workflow orchestration with expflow —
     5|  three pipeline modes (full/fast/skip), distributed HPO, pruner integration,
     6|  and ClearML HyperParameterOptimizer native mode.
     7|category: mlops
     8|author: Li Shen
     9|version: 1.0.0
    10|metadata:
    11|  hermes:
    12|    tags: [mlops, pde, hpo, clearml, optuna, pipeline, competition]
    13|    homepage: https://github.com/diamond2nv/expflow
    14|    related_skills: [experiment-lifecycle-governance, clearml-metrics-logging-pattern, competition-task-intelligence]
    15|---
    16|
    17|# expflow PDEBench Pipeline & HPO
    18|
    19|Orchestrate experiment workflows for the AI4S PDE competition using expflow.
    20|Three modes for three competition phases.
    21|
    22|## Triggers
    23|
    24|- User says "run HPO", "submit pipeline", "distributed experiment"
    25|- User says "competition sprint" or "fast iterate"
    26|- User asks about automating the train→eval→submit loop
    27|- User mentions needing to find best hyperparams
    28|
    29|## Installation
    30|
    31|```bash
    32|pip install "expflow-pde[pipeline]"
    33|```
    34|
    35|## Available Pipeline Modes
    36|
    37|Three pipeline modes, each mapped to a CLI command:
    38|
    39|### Mode A — Full (HPO → Train → Eval)
    40|
    41|For the **exploration phase** of a competition task. Optuna finds best params
    42|via distributed clearml-agent trials, trains with best, then evaluates.
    43|
    44|```bash
    45|expflow pipeline submit-full train_task1.py \
    46|    --queue default \
    47|    --trials 50 --parallel 4 \
    48|    --eval-script eval_task1.py \
    49|    --metric seg_total --direction maximize
    50|```
    51|
    52|Flags used:
    53|- `--trials N`: total HPO trials
    54|- `--parallel M`: max concurrent trials (use GPU node count)
    55|- `--metric`: objective metric name prefixed `METRIC:` in script stdout
    56|- `--pruner hyperband|median|percentile`: early-stop poor trials
    57|- `--study-name`: Optuna study name (auto if omitted; persists to SQLite)
    58|- `--skip hpo --skip eval`: run train only within full skeleton
    59|
    60|### Mode B — Fast (Train → Eval)
    61|
    62|For the **competition sprint** phase. You already know best params. Skip HPO,
    63|run directly with fixed args.
    64|
    65|```bash
    66|expflow pipeline submit train_task1.py \
    67|    --queue default \
    68|    --train-param lr=0.001 --train-param epochs=80 \
    69|    --eval-script eval_task1.py \
    70|    --eval-param sub_step=5
    71|```
    72|
    73|Flags:
    74|- `--skip eval`: train-only (just submit checkpoint)
    75|- `--train-param key=val`: injected as `--key=val` to training script
    76|- `--eval-param key=val`: injected as `--key=val` to eval script
    77|
    78|### Mode C — Flexible Skip
    79|
    80|Override step inclusion on either mode:
    81|
    82|```bash
    83|expflow pipeline submit-full train_task1.py \
    84|    --skip hpo --skip eval          # = train only
    85|expflow pipeline submit-full train_task1.py \
    86|    --skip train --skip eval         # = HPO only
    87|```
    88|
    89|## HPO: Three Execution Modes
    90|
    91|HPO (`expflow optuna run`) has three backends:
    92|
    93|| Mode | Flag | Description | Best for |
    94||------|------|-------------|----------|
    95|| Local | (default) | subprocess serial on CPU | ≤20 trials, quick test |
    96|| Distributed | `--distributed` | ask/tell + clearml Task clone| Multi-GPU, custom control|
    97|| Optimizer | `--optimizer -O` | Clearml `HyperParameterOptimizer` | Production, 50-200+ trials |
    98|
    99|### Key flags across all HPO modes:
   100|- `--pruner hyperband|median|percentile|none`: ASHA pruner saves ~40% GPU time
   101|- `--metric <name>`: reads `METRIC:<name>=<value>` from script stdout
   102|- `--direction maximize|minimize`
   103|- `--timeout <min>`: safety cutoff
   104|
   105|## Script Requirements
   106|
   107|The training/eval script must:
   108|1. Accept hyperparams as `--key=value` CLI arguments
   109|2. Output `METRIC:<name>=<value>` to stdout for objective capture (local mode)
   110|3. Report clearml scalars for distributed/optimizer mode:
   111|   ```python
   112|   Task.current_task().report_scalar("Score", "seg_total", value, iteration=epoch)
   113|   ```
   114|
   115|## Pitfalls
   116|
   117|- **Pruner needs `trial.report()` calls during training.** If the script only reports at the end, the pruner has nothing to prune on. Call `trial.report(val_loss, epoch)` at least every 10 epochs.
   118|- **HyperParameterOptimizer needs the metric name in `Title/Series` format.** If your metric is `seg_total`, it becomes `title=seg_total, series=seg_total`. If your clearml report_scalar is `report_scalar("Score", "seg_total", v)`, pass `--metric Score/seg_total`.
   119|- **Clearml-agent must be running on GPU nodes** before submitting. Verify with `expflow clearml workers` or check Web UI.
   120|- **`_collect_one_trial` polls every 5s** — waits up to 60min per trial. If trials are expected to run longer, increase `timeout_minutes`.
   121|
   122|## Architecture Reference
   123|
   124|Key files in `expflow_pde/`:
   125|- `hpo.py` — 3-mode HPO runner (local/distributed/optimizer)
   126|- `pipeline.py` — ExperimentPipeline class (fast/full modes)
   127|- `cli_pipeline.py` — `pipeline submit` + `pipeline submit-full`
   128|- `cli_optuna.py` — `optuna run` with all three backends
   129|
   130|## Related
   131|
   132|- `experiment-lifecycle-governance` — PIN, metrics registry, compare-scores, competition rules audit
   133|- `pde-experiment-hyperparameters` — PDEBench-specific hyperparameter reference
   134|- `multi-agent-distributed-experiment-workflow` — Hermes → OpenCode → clearml
   135|
