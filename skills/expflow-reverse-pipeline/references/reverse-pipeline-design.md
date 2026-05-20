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

### Three Layers (Current: v0.5.2)

| Layer | Technology | Latency | Status |
|-------|-----------|---------|--------|
| **L1 (Trigger)** | taskctl + crontab PID polling | 15min | Active |
| **L2 (Event)** | ZeroMQ PUB-SUB (port 15556/15557) | ~1ms | Active |
| **L3 (Goal)** | Hermes Agent goal engine (external) | Variable | Planning |

### ZeroMQ Event Broker (NEW)

The broker runs inside `taskctl.py daemon` as a PUB socket. Events flow:

```
taskctl daemon (PUB on port 15556)
    |
    +-- IPC: ipc:///tmp/taskctl_pub.ipc  (local processes)
    +-- TCP: tcp://127.0.0.1:15556       (Hermes Agent, cross-process)
    |
    +-- Topic: taskctl/<id>/complete     (experiment done)
    +-- Topic: taskctl/<id>/timeout      (experiment timeout)
    +-- Topic: taskctl/<id>/register     (new task registered)
    +-- Topic: system/heartbeat          (daemon alive every 60s)
```

Subscriber pattern (e.g. Hermes Agent):

```python
from zmq_broker import Subscriber
sub = Subscriber(port=15557)
sub.start(topics=["taskctl/", "system/"])
events = sub.poll(timeout_ms=5000)
for e in events:
    if e["topic"].endswith("/complete"):
        # Auto-trigger reverse pipeline
```

### QoS Semantics

| QoS | Mechanism | Guarantee |
|-----|-----------|-----------|
| **0** (fire-and-forget) | ZMQ PUB (no ack) | Best effort |
| **1** (at-least-once) | ZMQ PUB + JSONL fallback | Replayable on reconnect |
| **2** (exactly-once) | Not implemented (requires ROUTER-DEALER) | Future |

### No New Infrastructure (still true)

The reverse pipeline adds **minimal infrastructure**:
- pyzmq is the only new dependency
- No broker daemon needed in cron-only mode
- No new ports to open (all loopback)

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

## Integration with Hermes /goal (active)

Hermes Agent has a built-in `/goal` slash command that persists an objective
across turns. After each response, a light **judge** model checks: "is this
goal satisfied?" If not, a continuation prompt is auto-injected. The reverse
pipeline feeds the goal loop with execution results.

```
User:  /goal reach seg_total 140 on Task1

Hermes session loop (max 20 auto-continuation turns):
  ┌─────────────────────────────────────────────────┐
  │ Turn 1: expflow analyze advise                  │
  │     -> "Try sub_step=5 with P2(16/32)"          │
  │ Turn 2: taskctl add --pid $PID --duration 7200  │
  │     -> --on-success "evaluate_seg.py && echo"    │
  │ Judge: "Not yet (task just started)"            │
  ├─────────────────────────────────────────────────┤
  │ (cron detects completion)                       │
  │ on-success chain: evaluate_seg.py               │
  │ output: seg_total=116, still need improvement   │
  ├─────────────────────────────────────────────────┤
  │ Turn 3: agent reads result from chain output    │
  │     -> "seg_total=116, need 140. Try Stability  │
  │        FT (step variance) to push past plateau" │
  │ Judge: "Not yet (116 < 140)"                    │
  ├─────────────────────────────────────────────────┤
  │ ... loop until seg_total >= 140 or blocked ...  │
  └─────────────────────────────────────────────────┘
```

`/goal` sub-commands:
- `/goal <text>` — Set a new standing goal
- `/goal pause` — Pause goal loop (keep goal)
- `/goal resume` — Resume paused goal
- `/goal clear` — Clear active goal
- `/goal status` — Show current goal + progress

Key integration points:
- **State persistence**: Goals survive `/resume` (stored in SessionDB `goal:<id>`)
- **User message preemption**: If user sends a message, it takes priority and pauses
  the goal loop for that turn (goal is NOT cleared)
- **Judge fail-open**: If the judge fails to parse, the loop continues (turn budget
  is the backstop). After 3 consecutive parse failures, auto-pauses.
- **Continuation prompt**: Is just a normal user message — no system prompt changes,
  no toolset swap, prompt caching stays intact
- **Reverse pipeline**: taskctl's `--on-success` chain bridges the gap between
  "agent decided what to do" and "agent knows the result" — the chain output
  feeds back into the next conversation turn
