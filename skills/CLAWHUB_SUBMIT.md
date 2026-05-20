# ClawHub 技能提交清单

ClawHub 网站需要 GitHub 登录后才能看到提交表单。登录后到以下 URL 提交：

> https://clawhub.ai/publish/skill

每个技能需要以下信息。直接在浏览器中粘贴提交：

---

## Skill 1: expflow-pipeline-hpo

| 字段 | 值 |
|------|-----|
| **Name** | `expflow-pipeline-hpo` |
| **Description** | PDEBench competition workflow orchestration with expflow — three pipeline modes (full/fast/skip), distributed HPO, pruner integration, and ClearML HyperParameterOptimizer native mode. |
| **Category** | mlops |
| **Tags** | mlops, pde, hpo, clearml, optuna, pipeline, competition |
| **Homepage** | https://github.com/diamond2nv/expflow |
| **SKILL.md URL** | `https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/expflow-pipeline-hpo/SKILL.md` |

---

## Skill 2: experiment-lifecycle-governance

| 字段 | 值 |
|------|-----|
| **Name** | `experiment-lifecycle-governance` |
| **Description** | Add governance to experiment workflows — PIN-protected destructive ops, standardized metrics registry with thresholds, compare-scores ranking with gating, and competition rules audit. Builds on clearml-agent-dispatch and fysom-fsm-integration. |
| **Category** | mlops |
| **Tags** | governance, pin, metrics, compare, audit, guard, competition, safety |
| **Homepage** | https://github.com/diamond2nv/expflow |
| **SKILL.md URL** | `https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/experiment-lifecycle-governance/SKILL.md` |

---

## Skill 3: clearml-metrics-logging-pattern

| 字段 | 值 |
|------|-----|
| **Name** | `clearml-metrics-logging-pattern` |
| **Description** | Standardized ClearML metrics logging patterns for PDEBench experiment scripts — train loss, validation metrics, competition scores, PDE residual, and TensorBoardX integration. Includes patterns for dist/expflow compatibility. |
| **Category** | mlops |
| **Tags** | mlops, pde, clearml, metrics, logging, experiment, competition |
| **Homepage** | https://github.com/diamond2nv/expflow |
| **SKILL.md URL** | `https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/clearml-metrics-logging-pattern/SKILL.md` |

---

## Skill 4: competition-task-intelligence

| 字段 | 值 |
|------|-----|
| **Name** | `competition-task-intelligence` |
| **Description** | Build and maintain a structured PDE equation registry, analyze competition tasks (difficulty, bottlenecks, score projections), generate strategic recommendations for research focus, and expose this intelligence via CLI and MCP tools. |
| **Category** | mlops |
| **Tags** | mlops, competition, strategy, equations, analysis, planning, pde, task-intelligence |
| **Homepage** | https://github.com/diamond2nv/expflow |
| **SKILL.md URL** | `https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/competition-task-intelligence/SKILL.md` |

---

## Skill 5: expflow-reverse-pipeline

| 字段 | 值 |
|------|-----|
| **Name** | `expflow-reverse-pipeline` |
| **Description** | Zero-token background task monitor for the reverse pipeline pattern. Register PID-based tasks; crontab polls completion/timeout every 15min; auto-sends QQ notification and triggers chain commands (expflow analyze, hfpclawer search) to close the data-experiment-feedback loop. |
| **Category** | devops |
| **Tags** | monitor, reverse-pipeline, cron, qq, no-llm, expflow, hfpclawer, experiment |
| **Homepage** | https://github.com/diamond2nv/expflow |
| **SKILL.md URL** | `https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/expflow-reverse-pipeline/SKILL.md` |
