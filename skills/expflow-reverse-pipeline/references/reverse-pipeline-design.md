# Reverse Pipeline: Experiment-to-Auto-Analysis Feedback Loop

## Motivation

Inspired by the SJTU Science AI4S multi-agent platform (Zhao et al., 2026),
the **reverse pipeline** closes the gap between "running experiments" and
"using results to inform the next decision."

### Current Flow (forward only)

```
hfpclawer (paper search)
    -> Human decides next experiment
    -> expflow pipeline submit (train script + eval)
    -> ClearML executes experiment
    -> Human checks results
    -> Human searches for new papers
    ^^^^^^^^^^^^^^^^^^^^^^^^
    This loop is manual. It breaks at night, during weekends,
    and whenever the user is doing something else.
```

### Desired Flow (closed loop)

```
hfpclawer (paper search) ──┐
                           ├──> expflow analyze advise
expflow analyze (task intel) ┘         |
    |                                   v
    v                          taskctl --on-success chain
expflow pipeline submit ──> ClearML experiment
                                   |
                                   v (completes)
                              taskctl detects PID exit
                                   |
                    ┌──────────────┴──────────────┐
                    v                              v
            [OK] QQ notification           on-success chain runs:
                                              1. expflow analyze advise
                                              2. hfpclawer search
                                              3. Schedule next experiment
                    │                              │
                    └──────────────┬───────────────┘
                                   v
                          Knowledge base updated
                          (wiki entities, _TASK_META)
                                   │
                                   v
                          Next experiment is smarter
```

## Architecture

### Three Layers

| Layer | Technology | Role |
|-------|-----------|------|
| **Trigger** | taskctl + crontab | Detect completion/timeout |
| **Notification** | qq_send.py | Zero-token QQ alert |
| **Action** | `--on-success` chain | Arbitrary shell commands |

### No New Infrastructure

The reverse pipeline adds **zero infrastructure**:
- No ZeroMQ broker needed (though ZeroMQ is available for real-time upgrades)
- No Hermes gateway dependency (notification via standalone REST API)
- No LLM calls (notification text is template-based)
- No database changes (state in `tasks.json`, FIFO max 50)

## Using the Reverse Pipeline

### Pattern 1: Experiment Feedback

```bash
expflow pipeline submit train_task1.py --queue default ... &
PID=$!

# Register with reverse pipeline
python3 ~/.hermes/task_monitor/taskctl.py add \
  --id "fno_train_$(date +%s)" \
  --pid $PID \
  --ctx "FNO Task1 training, sub_step=5" \
  --duration 7200 \
  --on-success "
    expflow analyze advise --task task1 &&
    expflow clearml compare-scores \
      --project PDEBench --tags task1 \
      --gate pde_mean:lt:18.09 &&
    hfpclawer search --query 'FNO autoregressive stability 2026' --max-pages 2
  "
```

### Pattern 2: Cascading Steps

```bash
# Step 1: Train
python3 train_task1.py &
taskctl.py add --id step1_train --pid $! --duration 7200 \
  --on-success "step2_eval.sh"

# step2_eval.sh kicks off step 2 automatically
```

### Pattern 3: Download-and-Convert

```bash
hfpclawer download --background &
taskctl.py add --id paper_dl --pid $! --duration 1800 \
  --on-success "hfpclawer convert --to-wiki"
```

## Future: ZeroMQ Upgrade Path

The current architecture uses **polling** (crontab every 15min). For
**real-time** event response, ZeroMQ can be added as a second trigger:

```
Current: Cron (15min) -> taskctl check -> PID probe
Future:  ClearML Task.completed -> ZMQ PUB -> Hermes subscriber
         GPU OOM -> ZMQ PUB -> auto cleanup
         Training epoch anomaly -> ZMQ PUB -> checkpoint rollback
```

The two patterns coexist:
- **Polling** for simple PID monitoring (0 infra, 0 tokens)
- **Event-driven** for time-sensitive events (requires ZMQ broker)

See the original architecture design doc for ZeroMQ port assignments
(15556/15557, PUB-SUB fanout) and topic schemas.

## Relation to Hermes Goal

When Hermes Agent supports a `goal` command (persistent objective driving),
the reverse pipeline becomes the execution layer:

```
User: goal: reach 140 on Task1

Hermes goal engine:
  while seg_total < 140:
    expflow analyze advise         -> next strategy
    taskctl submit(...)            -> run experiment
    taskctl check (cron)           -> wait for completion
    on-success chain: evaluate     -> check seg_total
    loop
```

The reverse pipeline provides the **imperative execution** (run, detect,
react) while the goal engine provides the **declarative planning** (what to
achieve, how to decompose, when to stop).
