---
name: expflow-reverse-pipeline
description: >
  Zero-token background task monitor for the reverse pipeline pattern.
  Register PID-based tasks; crontab polls completion/timeout every 15min;
  auto-sends QQ notification and triggers chain commands (expflow analyze,
  hfpclawer search) to close the data-experiment-feedback loop.
category: devops
author: Li Shen
version: 1.0.0
metadata:
  hermes:
    tags: [monitor, reverse-pipeline, cron, qq, no-llm, expflow, hfpclawer, experiment]
    related_skills:
      - expflow-pipeline-hpo
      - experiment-lifecycle-governance
      - competition-task-intelligence
    requires:
      - qq_send.py (standalone REST client, no gateway)
---

# expflow Reverse Pipeline

Zero-token background task monitor built on crontab + PID polling. Detects
experiment completion or timeout, sends notifications, and runs chain
commands to close the **data -> experiment -> analysis -> feedback** loop.

> **Who this is for**: Researchers running experiments on remote/headless
> servers who want automatic notification and auto-analysis when a training
> run finishes, without keeping Hermes Agent running 24/7.

## Overview

```
You start a long process (training / download / build)
                    |
                    v
taskctl add --pid X --duration 3600 --on-success "analyze.sh"
                    |
          +---------+---------+
          v                   v
    Cron (15min)        Cron (15min)
    -----------         -----------
    PID alive?          PID died?
      |  YES              |  YES
      v                   v
    Check timeout      [OK] QQ notification
    > 1.5x?            Run --on-success chain
      |  NO      YES      (REVERSE PIPELINE)
      v         v
    Wait       [WARN] QQ notification
               Run --on-fail chain
```

## File Structure

```
expflow/skills/expflow-reverse-pipeline/
+-- SKILL.md              # This file
+-- scripts/
|   +-- taskctl.py        # CLI: add / check / list / remove / clear / status
|   +-- qq_send.py        # QQ Bot REST API sender (standalone)
|   +-- setup.sh          # Installer: symlinks scripts, creates config template
+-- references/
    +-- reverse-pipeline-design.md  # Architecture doc (SJTU Science inspired)
```

## Installation

```bash
# 1. Install the skill in Hermes Agent
hermes skills install https://raw.githubusercontent.com/diamond2nv/expflow/main/skills/expflow-reverse-pipeline/SKILL.md

# 2. Run the setup script to symlink scripts and create config
bash path/to/expflow/skills/expflow-reverse-pipeline/scripts/setup.sh

# 3. Add crontab entries
crontab -e
# Add:
#   */15 * * * * cd ~/.hermes/task_monitor && python3 taskctl.py check >/dev/null 2>&1
#   0 4 * * * cd ~/.hermes/task_monitor && python3 taskctl.py clear >/dev/null 2>&1
```

## Prerequisites

- Python 3.10+
- `requests` (optional, falls back to `urllib` stdlib)
- **QQ Bot credentials** for notification (set in `~/.hermes/.env`):
  - `QQ_APP_ID`
  - `QQ_CLIENT_SECRET`
  - `QQBOT_HOME_CHANNEL`

> Without QQ credentials, notifications fall back to console. The reverse
> pipeline chain commands still run.

## Quick Start

### 1. Register an Experiment

```bash
# Start a training script in the background
expflow pipeline submit train_task1.py --queue default &
PID=$!

# Register with the monitor
python3 ~/.hermes/task_monitor/taskctl.py add \
  --id "fno_task1_$(date +%s)" \
  --pid $PID \
  --ctx "FNO Task1 training, sub_step=5" \
  --duration 7200 \
  --on-success "
    expflow analyze advise --task task1 &&
    expflow clearml compare-scores \
      --project PDEBench --tags task1 \
      --gate pde_mean:lt:18.09 &&
    hfpclawer search --query 'FNO stability autoregressive' --max-pages 2
  "
```

### 2. Check Status

```bash
python3 ~/.hermes/task_monitor/taskctl.py list
python3 ~/.hermes/task_monitor/taskctl.py status
```

### 3. Manual Check (for testing)

```bash
python3 ~/.hermes/task_monitor/taskctl.py check
```

### 4. Clean Up

```bash
python3 ~/.hermes/task_monitor/taskctl.py remove my_task
python3 ~/.hermes/task_monitor/taskctl.py clear
```

## Commands

| Command | Flags | Description |
|---------|-------|-------------|
| `add` | `--id`, `--pid`, `--ctx`, `--duration`, `--on-success`, `--on-fail` | Register a task |
| `check` | — | Check all running tasks (called by crontab) |
| `list` | — | List all tasks with progress |
| `remove` | `id` | Remove a task by ID |
| `clear` | — | Purge old completed tasks (keep running + last 10 done) |
| `status` | — | Quick overview: counts by status |

## Notification Format

On completion:
```
[OK] Task Complete
  ID: fno_task1_1715000000
  Context: FNO Task1 training, sub_step=5
  Duration: 3600s
  Time: 14:30
```

On timeout (>1.5x expected):
```
[WARN] Task Timeout
  ID: fno_task1_1715000000
  Context: FNO Task1 training, sub_step=5
  Elapsed: 5400s
  Expected: 3600s
  Time: 15:00
```

## The Reverse Pipeline Pattern

The `--on-success` chain enables the key feature: **closing the loop** from
experiment results back into the decision engine.

### Use Case 1: Experiment Feedback (PDE Competition)

```bash
# After training finishes, auto-analyze and search for related papers
python3 ~/.hermes/task_monitor/taskctl.py add \
  --id "exp_$(date +%s)" --pid $PID --duration 7200 \
  --on-success "
    expflow analyze advise --task task1 &&
    expflow clearml compare-scores \\
      --project PDEBench --tags task1 \\
      --gate pde_mean:lt:18.09 &&
    hfpclawer search --query 'FNO autoregressive training' \\
      --max-pages 2
  "
```

### Use Case 2: Cascading Pipeline Steps

```bash
# Step 1: Train (auto-starts step 2 when done)
python3 train_task1.py &
taskctl.py add --id step1_train --pid $! --duration 7200 \
  --on-success "
    python3 eval_task1.py &
    taskctl.py add --id step2_eval --pid $! --duration 3600
  "
```

### Use Case 3: Paper Download -> Convert

```bash
hfpclawer download --background &
taskctl.py add --id paper_dl --pid $! --duration 1800 \
  --on-success "hfpclawer convert --to-wiki"
```

## Design Principles

- **Zero token**: Pure Python + crontab. No LLM calls, no Hermes Agent overhead.
- **Privacy safe**: No hardcoded paths or credentials. All config through
  `~/.hermes/.env` or `~/.hermes/task_monitor/taskctl.conf`.
- **Hermes independent**: Notification via standalone qq_send.py REST client.
  Hermes Agent can restart without affecting monitors.
- **Fail-open**: QQ send failure does NOT block status update. Logs warning,
  continues.
- **Persistent FIFO**: JSON in `~/.hermes/task_monitor/tasks.json`, max 50,
  oldest auto-trimmed.
- **Timeout (1.5x)**: Process is NOT killed on timeout — only notified.
  You decide what to do.
- **Rotation logging**: 5MB auto-rotate, 3 backups.

## Common Pitfalls

1. **QQ_TARGET_USER not set**: Notifications silently skipped. Set in
   `~/.hermes/.env` or `taskctl.conf`.
2. **Shell quoting**: Multi-command chains with `&&` or `|` must be inside
   double quotes: `--on-success "cmd1 && cmd2"`.
3. **PID reuse**: Very rare (<15min window). If a process exits and its PID
   is reused before the next cron check, the task may be falsely marked
   completed.
4. **Crontab PATH**: Crontab has restricted PATH. Use absolute paths:
   `python3 ~/.hermes/task_monitor/taskctl.py`.
5. **Chinese/emoji in output**: All output uses `[OK]`/`[WARN]`/`[NF]`/`[ERR]`
   labels. No emoji. Python code is 100% English (PEP8 internationalization).

## Verification Checklist

- [ ] `taskctl.py` has PEP8 shebang + coding header
- [ ] All output uses `[OK]`/`[WARN]`/`[ERR]`/`[NF]` labels only
- [ ] Registration works: `taskctl.py add --id test --pid $$ --duration 60`
- [ ] Check works: `taskctl.py check` detects the running test process
- [ ] `--on-success` chain runs when process exits
- [ ] `--on-fail` chain runs on timeout
- [ ] QQ notification arrives (if credentials configured)
- [ ] No hardcoded paths or user IDs in any file
- [ ] Crontab entries present: `crontab -l | grep taskctl`
- [ ] `setup.sh` can be run by a fresh user
