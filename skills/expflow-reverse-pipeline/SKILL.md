---
name: expflow-reverse-pipeline
description: >
  Zero-token background task monitor with ZMQ event broker for the reverse
  pipeline pattern. Three-layer trigger: crontab PID polling (reliable
  fallback) + ZMQ PUB-SUB real-time events + optional Hermes goal
  integration. Auto-detects experiment completion/timeout, sends QQ
  notification, and runs chain commands (expflow analyze, hfpclawer
  search) to close the data-experiment-feedback loop.
category: devops
author: Li Shen
version: 0.5.2
metadata:
  hermes:
    tags:
      - monitor
      - zeromq
      - reverse-pipeline
      - cron
      - qq
      - no-llm
      - expflow
      - hfpclawer
      - experiment
      - goal
    related_skills:
      - expflow-pipeline-hpo
      - experiment-lifecycle-governance
      - competition-task-intelligence
    requires:
      - qq_send.py (standalone REST client, no gateway)
      - pyzmq>=24.0 (optional, cron-only mode without it)
---

# expflow Reverse Pipeline v2

Three-layer background task monitor for the reverse pipeline pattern:

| Layer | Trigger | Latency | Infra | Status |
|-------|---------|---------|-------|--------|
| **L1** | Cron PID polling | 15min | None (crontab) | Always-on |
| **L2** | ZMQ PUB-SUB event | ~1ms | pyzmq | Optional |
| **L3 (Goal)** | Hermes Agent /goal (built-in slash command) | Variable (judge-driven loop) | Hermes Agent | Active — see below |

> **Who this is for**: Researchers running experiments who want **automatic
> notification + auto-analysis** when training finishes, with optional
> **real-time event response** and **persistent goal tracking**.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │       taskctl v2 (daemon)         │
                    │   ┌──────────┐  ┌─────────────┐  │
User:               │   │ PID Poll │  │ ZMQ PUB     │  │
expflow submit ─────┼──▶│ (cron)   │  │ (port 15556)│  │
                    │   └────┬─────┘  └──────┬──────┘  │
                    │        │               │         │
                    └────────┼───────────────┼─────────┘
                             │               │
                    ┌────────┴───────┐  ┌────┴──────────┐
                    │  tasks.json    │  │  ZMQ SUB      │
                    │  (FIFO+WAL)    │  │  (port 15557) │
                    └────────────────┘  └────┬──────────┘
                                             │
                                    ┌────────┴────────┐
                                    │  Hermes Agent    │
                                    │  (goal engine)   │
                                    └─────────────────┘
                                        │         │
                                        ▼         ▼
                                  expflow     hfpclawer
                                  analyze     search
```

### Layer 1: Cron PID Polling (always-on)

```
*/15 * * * * taskctl.py check
```

- **Reliable**: works without ZeroMQ
- **Resilient**: survives Hermes Agent restart, venv changes
- **Resource-aware**: auto-drops to 1h frequency after 4 consecutive empty checks
- **Lock-safe**: WAL-based dedup lock prevents double transitions from overlapping cron + ZMQ

### Layer 2: ZMQ Event Broker (optional, real-time)

```
taskctl.py daemon --port 15556
```

- **PUB-SUB fanout** via IPC + TCP (dual transport)
- **Topic-based routing**: `taskctl/<id>/complete`, `taskctl/<id>/timeout`, `system/*`
- **HWM=1000**: drops oldest if subscriber is too slow
- **QoS 1**: JSONL fallback file for offline subscribers
- **LINGER=0**: publisher never blocks
- **Graceful degradation**: falls back to cron-only if ZMQ import fails

### Layer 3: Hermes /goal (active — persistent objective driving)

Hermes Agent has a built-in `/goal` slash command that persists an objective
across turns. After each response, a light judge model checks "is this goal
satisfied?" If not, it auto-injects a continuation prompt and keeps going.

```
User:  /goal reach seg_total 140 on Task1

Hermes session loop (max 20 auto-continuation turns):
  Turn 1: expflow analyze advise
          -> "Try sub_step=5 with P2(16/32)"
  Turn 2: taskctl add --pid $PID --duration 7200
          -> --on-success "evaluate_seg.py"
  Judge:  "Not yet (task just started)"
          |
  (cron detects completion -> on-success chain runs)
          |
  Turn 3: "seg_total=116, need 140. Try Stability FT"
  Judge:  "Not yet (116 < 140)"
          |
  ... loop until seg_total >= 140 or blocked ...
```

`/goal` sub-commands: `/goal <text>`, `/goal pause`, `/goal resume`,
`/goal clear`, `/goal status`. State persists across sessions.

## File Structure

```
expflow/skills/expflow-reverse-pipeline/
+-- SKILL.md                       # This file
+-- scripts/
|   +-- taskctl.py                 # CLI + daemon (cron + ZMQ)
|   +-- zmq_broker.py              # ZMQ PUB-SUB broker module
|   +-- qq_send.py                 # QQ Bot REST API sender
|   +-- setup.sh                   # Installer
+-- references/
    +-- reverse-pipeline-design.md # Architecture doc
```

## Installation

```bash
# 1. Install the skill
hermes skills install https://raw.githubusercontent.com/.../expflow-reverse-pipeline/SKILL.md

# 2. Run setup (symlinks scripts, creates config)
bash path/to/scripts/setup.sh

# 3. Install ZMQ (optional, for real-time events)
pip install pyzmq

# 4. Add crontab entries
crontab -e
#   */15 * * * * cd ~/.hermes/task_monitor && python3 taskctl.py check >/dev/null 2>&1
#   0 4 * * * cd ~/.hermes/task_monitor && python3 taskctl.py clear >/dev/null 2>&1

# 5. (Optional) Start ZMQ daemon
#    Add to crontab @reboot:
#   @reboot cd ~/.hermes/task_monitor && python3 taskctl.py daemon --port 15556 >/dev/null 2>&1
```

## Commands (v2.0)

| Command | Flags | Description |
|---------|-------|-------------|
| `add` | `--id`, `--pid`, `--ctx`, `--duration`, `--on-success`, `--on-fail` | Register a task |
| `daemon` | `--port` | Start ZMQ event broker daemon |
| `check` | — | Check all running tasks (cron or manual) |
| `list` | — | List tasks with progress |
| `remove` | `id` | Remove a task by ID |
| `clear` | — | Purge old completed tasks |
| `status` | — | Quick overview: counts by status |

## Quick Start

### 1. Register an Experiment (with reverse pipeline)

```bash
expflow pipeline submit train_task1.py --queue default &
PID=$!

python3 ~/.hermes/task_monitor/taskctl.py add \
  --id "fno_task1_$(date +%s)" \
  --pid $PID \
  --ctx "FNO Task1 training, sub_step=5" \
  --duration 7200 \
  --on-success "
    expflow analyze advise --task task1 &&
    expflow clearml compare-scores \\
      --project PDEBench --tags task1 \\
      --gate pde_mean:lt:18.09 &&
    hfpclawer search --query 'FNO stability' --max-pages 2
  "
```

### 2. Monitor in Real-Time (with ZMQ)

```bash
# Start daemon (background)
python3 ~/.hermes/task_monitor/taskctl.py daemon --port 15556 &

# Hermes Agent subscribes:
#   topic: taskctl/<id>/complete  ->  triggers expflow analyze
#   topic: taskctl/<id>/timeout   ->  triggers alert
#   topic: system/heartbeat       ->  daemon alive check
```

### 3. Check Status

```bash
python3 ~/.hermes/task_monitor/taskctl.py list
python3 ~/.hermes/task_monitor/taskctl.py status
```

## Robustness Features (v2.0)

| Feature | Mechanism | Why |
|---------|-----------|-----|
| **Dedup lock** | WAL-based `taskctl.lock`, 30s TTL | Prevents double transition from cron + ZMQ |
| **Chain timeout** | 300s per subprocess | Chain command cannot hang forever |
| **Parallel chain** | `subprocess.Popen` + `wait` | Multi-command chains run simultaneously |
| **Graceful shutdown** | SIGTERM/SIGINT handler | ZMQ LINGER=0, PID file cleanup |
| **Idle frequency scaling** | 15min → 1h after 4 empty checks | Zero CPU waste when no active tasks |
| **Dual transport** | IPC + TCP ZMQ sockets | IPC for same-host, TCP for Hermes |
| **QoS 1** | JSONL fallback | No event loss if subscriber offline |
| **Structured JSON log** | `.jsonl` rotation (5MB x3) | Machine-parseable event history |
| **Rotating text log** | `.log` rotation (5MB x3) | Human-readable debugging |
| **Fail-open QQ** | Falls back to console | Notification failure never blocks workflow |

## Application: Reverse Pipeline in Practice

### Use Case 1: Experiment Feedback Loop

```bash
python3 ~/.hermes/task_monitor/taskctl.py add \
  --id "exp_$(date +%s)" --pid $PID --duration 7200 \
  --on-success "
    expflow analyze advise --task task1 &&
    expflow clearml compare-scores \\
      --project PDEBench --tags task1 --gate pde_mean:lt:18.09 &&
    hfpclawer search --query 'FNO autoregressive' --max-pages 2
  "
```

### Use Case 2: Cascading Steps

```bash
python3 train_task1.py &
taskctl.py add --id step1_train --pid $! --duration 7200 \
  --on-success "
    python3 eval_task1.py &
    taskctl.py add --id step2_eval --pid $! --duration 3600
  "
```

### Use Case 3: ZMQ-Triggered Auto-Analysis

```bash
# Hermes Agent subscribes to taskctl/<id>/complete
# On event -> auto-run:
#   1. expflow analyze advise --task task1
#   2. expflow clearml compare-scores --project PDEBench
#   3. hfpclawer search --query 'latest results'
#   4. QQ: "Experiment complete! seg_total=57.09 Check recommendation."
```

## Common Pitfalls

1. **ZMQ not installed**: Taskctl falls back to cron-only. pip install pyzmq.
2. **Chain command timeout**: Default 300s per command. Change via CHAIN_TIMEOUT in taskctl.py.
3. **Lock contention**: cron + ZMQ cannot collide (30s TTL lock). If cron fires during ZMQ processing, one gets deferred.
4. **QQ_TARGET_USER not set**: Notification silently skipped.
5. **PID reuse**: Very rare (<15min window). WAL lock helps but cannot fully prevent.

## Design Principles

- **Zero token priority**: Cron layer requires 0 LLM calls.
- **Graceful degradation**: ZMQ unavailable → cron-only. qq_send missing → console fallback.
- **Privacy safe**: No hardcoded paths or user IDs.
- **Hermes independent**: All notifications via standalone REST API.
- **Resource conservative**: Idle scaling, HWM caps, rotation logs.
- **PEP8 internationalization**: English-only, no emoji, no Chinese.

## Verification Checklist (v2.0)

- [ ] `taskctl.py add` registers a task in tasks.json
- [ ] `taskctl.py check` detects running/completed processes
- [ ] `taskctl.py daemon` starts ZMQ broker without error
- [ ] ZMQ publish: `taskctl/test/complete` event emitted via IPC + TCP
- [ ] ZMQ subscribe: taskctl subscriber receives the event
- [ ] Chain command isolation: timeout 300s does not block other chains
- [ ] Graceful shutdown: SIGTERM cleans up PID file and ZMQ sockets
- [ ] Idle scaling: 4 empty checks → frequency drops to 1h
- [ ] Structured JSON log: taskctl.log.jsonl contains parseable events
- [ ] No hardcoded paths or user IDs in any file
- [ ] `setup.sh` works for a fresh user

## Usage Notes by Competition Phase

### Exploration Phase (long-term R&D)

Use the **full three-layer reverse pipeline** to drive iterative
data-experiment-feedback loops:

```bash
/goal explore optimal architecture for Task3 KS equation

Hermes + taskctl + ZMQ + ClearML:
  loop:
    expflow analyze advise -> decide strategy
    taskctl register ...   -> run experiment
    cron/ZMQ completion    -> trigger analysis
    chain: expflow analyze -> update _TASK_META
```

### Fast Competition Phase (sprint, current priority)

In fast comp sprint mode (days, not weeks), running multiple GPU experiment
iterations is not the bottleneck — **information retrieval and knowledge
consolidation** are. The highest-ROI usage is:

```bash
/goal reach seg_total 140 on PDEBench Task1

Hermes /goal + llm-wiki iteration:
  loop (max 20 turns):
    search hfpclawer wiki     -> find relevant entries
    cross-reference memory    -> connect past findings
    update llm-wiki            -> consolidate new knowledge
    judge: "goal met?"         -> continue or stop

  No GPU, no ClearML, no taskctl needed.
  Pure knowledge work: search -> reason -> write -> verify.
```

Key difference:

| Aspect | Exploration Loop | Fast Sprint |
|--------|-----------------|-------------|
| Core engine | taskctl + ZMQ + ClearML | Hermes + llm-wiki |
| Latency | Hours (experiment) | Minutes (retrieval) |
| Resource | GPU + queue | 0 GPU (search only) |
| Output | Experiment results | Structured knowledge |
| Best for | Architecture search, HPO | Comp strategy, sprint decisions |

The reverse pipeline (taskctl + ZMQ) is always available for when you do
run experiments — but during fast comp sprint, invest tokens in **wiki
iteration** instead of experiment orchestration.
